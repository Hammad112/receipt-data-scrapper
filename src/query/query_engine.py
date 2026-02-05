"""
Orchestrator for the Receipt Intelligence Query Engine.

This module provides the QueryEngine class which coordinates query parsing, 
vector retrieval, and answer generation.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from decimal import Decimal
from openai import OpenAI

try:
    from ..models import QueryResult
    from ..vectorstore import VectorManager
    from .query_parser import QueryParser
    from .answer_generator import AnswerGenerator
    from ..utils.logging_config import logger
except ImportError:
    from models import QueryResult
    from vectorstore import VectorManager
    from query.query_parser import QueryParser
    from query.answer_generator import AnswerGenerator
    from utils.logging_config import logger


class QueryEngine:
    """
    Central orchestrator for natural language receipt queries.
    
    Coordinates the RAG (Retrieval-Augmented Generation) pipeline:
    1. Parsing intent and filters from user input.
    2. Executing hybrid search in Pinecone.
    3. Aggregating results and calculating metrics.
    4. Generating a natural language answer via LLM.
    """
    
    def __init__(self, vector_manager: VectorManager):
        """
        Initializes the QueryEngine with its dependencies.
        
        Args:
            vector_manager: Optimized manager for Pinecone operations.
        """
        self.vector_manager = vector_manager
        self.openai_client = OpenAI()
        
        # Modular Sub-components
        self.parser = QueryParser()
        self.generator = AnswerGenerator(self.openai_client)
        
        logger.info("QueryEngine successfully initialized with modular components.")

    def process_query(self, query: str) -> QueryResult:
        """
        Executes a full query pipeline from string to QueryResult.
        
        Args:
            query: The raw user query.
            
        Returns:
            QueryResult: Object containing the answer, raw items, and metadata.
        """
        start_time = datetime.utcnow()
        logger.info(f"Processing new query: '{query}'")
        
        # 1. Parsing
        query_params = self.parser.parse(query)
        logger.debug(f"Extracted Params: {query_params}")
        
        # 2. Search Preparation
        search_filters = self._build_search_filters(query_params)
        expanded_query = query
        if 'semantic_categories' in query_params:
            expanded_query = f"{query} {' '.join(query_params['semantic_categories'])}"
            
        # 3. Execution
        search_results = self.vector_manager.hybrid_search(
            query=expanded_query,
            filters=search_filters,
            top_k=20
        )
        
        # 4. Aggregation
        processed_results = self._process_search_results(search_results, query_params)
        
        # 5. Language Generation
        answer = self.generator.generate(query, processed_results, query_params)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Query processed in {processing_time:.2f}s")
        
        return QueryResult(
            answer=answer,
            receipts=processed_results.get('receipts', []),
            items=processed_results.get('items', []),
            confidence=processed_results.get('confidence', 0.0),
            query_type=query_params.get('query_type', 'general'),
            processing_time=processing_time,
            metadata={
                'search_results_count': len(search_results),
                'query_params': query_params,
                'search_filters': search_filters,
            }
        )

    def _build_search_filters(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Maps query parameters to Pinecone-specific filter syntax."""
        filters = {}
        
        # Dates
        if 'date_filter' in query_params:
            filters.update(query_params['date_filter'])
        if 'date_range' in query_params:
            dr = query_params['date_range']
            filters['transaction_date'] = {'$gte': dr['start'], '$lte': dr['end']}
        
        # Merchants
        if 'merchants' in query_params:
            extracted = query_params['merchants']
            variations = {
                'Walmart': ['Walmart', 'Walmart Supercenter'],
                'Whole Foods': ['Whole Foods', 'Whole Foods Market'],
                'Target': ['Target', 'Target Store'],
                'Best Buy': ['Best Buy', 'Best Buy Store'],
                'Costco': ['Costco', 'Costco Wholesale'],
            }
            final_list = []
            for m in extracted:
                if m in variations: final_list.extend(variations[m])
                else: final_list.append(m)
            
            if len(final_list) == 1: filters['merchant_name'] = final_list[0]
            else: filters['merchant_name'] = {'$in': final_list}
        
        # Categories
        if 'categories' in query_params:
            cats = query_params['categories']
            if len(cats) == 1: filters['item_category'] = cats[0]
            else: filters['item_category'] = {'$in': cats}
        
        # Amounts
        target_field = 'total_amount'
        if 'categories' in query_params or 'items' in query_params.get('original_query', '').lower():
            target_field = 'item_price'
            
        if 'min_amount' in query_params:
            filters[target_field] = {'$gte': query_params['min_amount']}
        if 'max_amount' in query_params:
            if target_field in filters: filters[target_field]['$lte'] = query_params['max_amount']
            else: filters[target_field] = {'$lte': query_params['max_amount']}
        
        return filters

    def _process_search_results(self, search_results: List[Dict], query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregates raw vector results into helpful business insights."""
        receipts = []
        items = []
        receipt_ids = set()
        item_keys = set()
        
        for result in search_results:
            meta = result.get('metadata', {})
            rid = meta.get('receipt_id')
            if not rid: continue
            
            # 1. Process Items (Always capture if it's an item, regardless of if we've seen the receipt)
            chunk_type = meta.get('chunk_type')
            if chunk_type == 'item_detail':
                item_name = meta.get('item_name', 'Unknown')
                item_price = meta.get('item_price', 0)
                # Differentiate items by receipt and name to avoid showing the same item twice from different chunks
                item_key = f"{rid}_{item_name}_{item_price}"
                if item_key not in item_keys:
                    item_keys.add(item_key)
                    items.append({
                        'name': item_name,
                        'price': item_price,
                        'category': meta.get('item_category', 'other'),
                        'merchant': meta.get('merchant_name', 'Unknown'),
                        'date': meta.get('transaction_date', ''),
                        'payment_method': meta.get('payment_method', 'other'),
                        'filename': meta.get('filename'),
                    })
            
            # 2. Process Receipts (Ensure we capture high-level info)
            if rid not in receipt_ids:
                # Capture receipt summary OR fallback to metadata from ANY chunk
                # If it's a summary, we definitely want it. 
                # If it's something else, we take it as a placeholder until we see a summary (if ever)
                receipt_ids.add(rid)
                receipts.append({
                    'receipt_id': rid,
                    'merchant_name': meta.get('merchant_name', 'Unknown'),
                    'transaction_date': meta.get('transaction_date'),
                    'total_amount': meta.get('total_amount', 0),
                    'content': meta.get('content', '') if chunk_type == 'receipt_summary' else f"Record found in {meta.get('filename', 'receipt')}",
                    'payment_method': meta.get('payment_method', 'other'),
                    'merchant_address': meta.get('merchant_address'),
                    'merchant_phone': meta.get('merchant_phone'),
                    'filename': meta.get('filename'),
                    'is_summary': chunk_type == 'receipt_summary'
                })
            elif chunk_type == 'receipt_summary':
                # If we previously added this receipt using a non-summary chunk, upgrade it to the summary
                for i, r in enumerate(receipts):
                    if r['receipt_id'] == rid and not r.get('is_summary', False):
                        receipts[i] = {
                            'receipt_id': rid,
                            'merchant_name': meta.get('merchant_name', 'Unknown'),
                            'transaction_date': meta.get('transaction_date'),
                            'total_amount': meta.get('total_amount', 0),
                            'content': meta.get('content', ''),
                            'payment_method': meta.get('payment_method', 'other'),
                            'merchant_address': meta.get('merchant_address'),
                            'merchant_phone': meta.get('merchant_phone'),
                            'filename': meta.get('filename'),
                            'is_summary': True
                        }
                        break
        
        # Math & Stats
        total_val = sum(Decimal(str(item.get('price', 0))) for item in items)
        if total_val == 0 and receipts:
             total_val = sum(Decimal(str(r.get('total_amount', 0))) for r in receipts)
        
        agg = {}
        if query_params.get('aggregation') == 'sum': agg['total'] = float(total_val)
        elif query_params.get('aggregation') == 'average':
            count = len(items)
            agg['average'] = float(total_val / count) if count > 0 else 0
        elif query_params.get('aggregation') == 'count':
            agg['count'] = len(items)
        
        confidence = min(len(search_results) / 10.0, 1.0)
        
        return {
            'receipts': receipts,
            'items': items,
            'aggregations': agg,
            'total_amount': float(total_val),
            'confidence': confidence,
        }

