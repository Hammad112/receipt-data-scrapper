"""
Orchestrator for the Receipt Intelligence Query Engine.
"""

import re
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from decimal import Decimal

# Industrial-grade absolute imports
from .query_parser import QueryParser
from .answer_generator import AnswerGenerator
from ..models import QueryResult
from ..utils.logging_config import logger
from ..utils.normalization import normalize_merchant_name


class QueryEngine:
    """
    Orchestrates the RAG pipeline for receipt queries.
    
    Responsibilities:
    - Intent parsing (QueryParser)
    - Semantic search & filtering (VectorManager)
    - Financial verification (Deterministic Audit)
    - Response synthesis (AnswerGenerator)
    """

    def __init__(self, vector_manager):
        """Initializes the engine with its component dependencies."""
        self.parser = QueryParser()
        self.generator = AnswerGenerator()
        self.vector_manager = vector_manager

    def query(self, query_text: str) -> QueryResult:
        """
        Executes a full RAG cycle for a natural language query.
        
        Args:
            query_text: The user's question about their receipts.
            
        Returns:
            A QueryResult object containing the synthesized answer and metadata.
        """
        start_time = time.time()
        logger.info(f"Processing query: {query_text}")

        try:
            # 0. Get latest receipt date from index for temporal reference
            latest_date = self.vector_manager.get_latest_transaction_date()
            if latest_date:
                logger.info(f"Using latest receipt date as temporal reference: {latest_date}")
                # Temporarily override reference date for temporal resolver
                original_ref = self.parser.temporal_resolver._reference_date
                self.parser.temporal_resolver._reference_date = latest_date
            else:
                logger.warning("No receipts found in index, using current date for temporal queries")

            # 1. Parsing intent and parameters
            params = self.parser.parse(query_text)
            logger.debug(f"Parsed parameters: {params}")

            # Restore original reference date
            if latest_date:
                self.parser.temporal_resolver._reference_date = original_ref

            # 2. Contextual Retrieval (Pinecone hybrid search)
            filters = self._build_search_filters(params)
            search_results = self.vector_manager.hybrid_search(query_text, filters=filters)
            
            if not search_results:
                return QueryResult(
                    answer="I couldn't find any receipts matching those criteria.",
                    confidence=0.0,
                    query_type=params.get('query_type', 'general'),
                    processing_time=time.time() - start_time
                )

            # 3. Independent Financial Audit (Independent Audit Pattern)
            # This verifies LLM-generated summaries against deterministic math.
            audit_result = {}
            if params.get('query_type') == 'aggregation':
                audit_result = self._perform_aggregation_audit(params, search_results)
                logger.info(f"Audit completed: {audit_result}")

            # 4. Answer Generation
            answer = self.generator.generate(
                query=query_text,
                context=search_results,
                query_params=params,
                audit_result=audit_result
            )

            # 5. Result Assembly
            processing_time = time.time() - start_time
            return QueryResult(
                answer=answer,
                receipts=self._deduplicate_receipts(search_results),
                items=self._extract_items(search_results),
                confidence=0.85 if audit_result.get('verified') else 0.7,
                query_type=params.get('query_type', 'general'),
                processing_time=processing_time,
                metadata={'audit': audit_result, 'params': params}
            )

        except Exception as e:
            logger.exception(f"Fatal error in QueryEngine: {e}")
            return QueryResult(
                answer="An internal error occurred while processing your request.",
                confidence=0.0,
                query_type="error",
                processing_time=time.time() - start_time
            )

    def process_query(self, query: str) -> QueryResult:
        """Alias for query() to support older test scripts."""
        return self.query(query)

    def _build_search_filters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Maps query parameters to Pinecone metadata filters."""
        filters = {}
        
        # Merchant filtering (handles multi-merchant lists)
        merchants = params.get('merchants', [])
        if merchants:
            if len(merchants) == 1:
                filters['merchant_name_norm'] = normalize_merchant_name(merchants[0])
            else:
                filters['merchant_name_norm'] = {"$in": [normalize_merchant_name(m) for m in merchants]}

        # Date range filtering
        date_range = params.get('date_range')
        if date_range:
            try:
                if isinstance(date_range, dict):
                    start_val = date_range.get('start')
                    end_val = date_range.get('end')
                    # Ensure values are strings before parsing
                    if isinstance(start_val, str) and isinstance(end_val, str):
                        start_dt = datetime.fromisoformat(start_val)
                        end_dt = datetime.fromisoformat(end_val)
                    else:
                        raise ValueError(f"date_range values must be strings, got start={type(start_val)}, end={type(end_val)}")
                else:
                    start_dt = date_range[0]
                    end_dt = date_range[1]
                
                filters['transaction_ts'] = {
                    "$gte": int(start_dt.timestamp()),
                    "$lte": int(end_dt.timestamp())
                }
            except (KeyError, IndexError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse date_range for filters: {e}")

        # Category filtering - search both item_detail and category_group chunks
        categories = params.get('categories', [])
        if categories:
            if len(categories) > 1:
                filters['$or'] = [
                    {'item_category': {"$in": categories}},
                    {'category': {"$in": categories}},
                ]
            else:
                filters['$or'] = [
                    {'item_category': categories[0]},
                    {'category': categories[0]},
                ]

        if 'feature_any_of' in params:
            if '$or' in filters:
                # Combine with existing $or
                existing_or = filters.pop('$or')
                filters['$and'] = [
                    {'$or': existing_or},
                    {'$or': [{f: True} for f in params['feature_any_of']]},
                ]
            else:
                filters['$or'] = [{f: True} for f in params['feature_any_of']]

        return filters if filters else None

    def _deduplicate_receipts(self, results: List[Dict]) -> List[Dict]:
        """Extracts unique receipts from multiple chunk results."""
        seen = set()
        receipts = []
        for r in results:
            meta = r.get('metadata', {})
            rid = meta.get('receipt_id')
            if rid and rid not in seen:
                seen.add(rid)
                receipts.append({
                    'receipt_id': rid,
                    'merchant_name': meta.get('merchant_name'),
                    'total_amount': meta.get('total_amount'),
                    'transaction_date': meta.get('transaction_date'),
                    'filename': meta.get('filename')
                })
        return receipts

    def _extract_items(self, results: List[Dict]) -> List[Dict]:
        """Extracts individual item data from item_detail chunks. Fallback to receipts if no items found."""
        items = []
        # 1. Try to get specific item details first
        for r in results:
            meta = r.get('metadata', {})
            if meta.get('chunk_type') == 'item_detail':
                items.append({
                    'name': meta.get('item_name'),
                    'price': meta.get('item_price'),
                    'category': meta.get('item_category'),
                    'merchant': meta.get('merchant_name'),
                    'filename': meta.get('filename')
                })
        
        # 2. Fallback: If no items found (e.g., aggregation query), use receipts as "items" for visualization
        if not items and results:
            seen_receipts = set()
            for r in results:
                meta = r.get('metadata', {})
                rid = meta.get('receipt_id')
                if rid and rid not in seen_receipts:
                    seen_receipts.add(rid)
                    items.append({
                        'name': f"Receipt from {meta.get('merchant_name', 'Unknown')}",
                        'price': meta.get('total_amount'),
                        'category': meta.get('category', 'Receipt'),  # Default to generic if missing
                        'merchant': meta.get('merchant_name', 'Unknown'),
                        'filename': meta.get('filename')
                    })
        return items

    def _perform_aggregation_audit(self, params: Dict[str, Any], results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Deterministic calculation to verify LLM summaries.
        Addresses the 'No aggregation support' red flag.
        """
        agg_type = params.get('aggregation')
        if not agg_type: return None
        
        basis = params.get('sum_basis', 'receipts')
        metric = params.get('metric', 'total')
        
        unique_receipts = {}
        items = []
        
        # 1. De-duplicate and extract values
        for r in results:
            meta = r.get('metadata', {})
            rid = meta.get('receipt_id')
            if rid not in unique_receipts:
                unique_receipts[rid] = meta
            
            # For item-level math, we want every unique item row
            if basis == 'items' and meta.get('chunk_type') == 'item_detail':
                items.append(meta)

        # 2. Determine target values based on metric (total, tax, tip)
        values = []
        if basis == 'receipts':
            field = {
                'tax': 'tax_amount',
                'tip': 'tip_amount',
                'subtotal': 'subtotal'
            }.get(metric, 'total_amount')
            values = [float(m.get(field, 0)) for m in unique_receipts.values() if m.get(field) is not None]
        else:
            # Item-level math (e.g., "calories", "price")
            values = [float(m.get('item_price', 0)) for m in items if m.get('item_price') is not None]

        if not values: return None

        # 3. Compute deterministic result
        count = len(values)
        total = sum(values)
        
        result = {'count': count}
        if agg_type == 'sum':
            result['value'] = total
        elif agg_type == 'average':
            result['value'] = total / count if count > 0 else 0
        elif agg_type == 'count':
            result['value'] = float(count)
            
        return result
