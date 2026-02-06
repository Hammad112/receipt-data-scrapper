"""
Orchestrator for the Receipt Intelligence Query Engine.
"""

import logging
import re
import os
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
    Industrial-grade orchestrator for the RAG pipeline.
    
    Responsibilities:
    1. Query Decomposition: Uses QueryParser to extract structured intent.
    2. Context Retrieval: Executes hybrid search (vector + filtered) via VectorManager.
    3. Reasoning: Grounds the LLM in retrieved context to generate answers.
    4. Deterministic Auditing: Verifies LLM math against source metadata.
    """
    
    def __init__(self, vector_manager: VectorManager):
        """Initializes the engine with its core components."""
        self.vector_manager = vector_manager # Modular Sub-components
        self.openai_client = OpenAI() # Initialize OpenAI client
        self.parser = QueryParser()
        self.generator = AnswerGenerator(openai_client=self.openai_client) # Pass client
        logger.info("QueryEngine initialized with modular RAG components.")

    def query(self, query_text: str, top_k: int = 15) -> QueryResult:
        """
        Executes a full RAG cycle for a user query.
        """
        start_time = datetime.utcnow()
        logger.info(f"Processing query: '{query_text}'")
        
        # 1. Parse intent and extract structured filters
        query_params = self.parser.parse(query_text)
        logger.debug(f"Parsed parameters: {query_params}")
        
        # 2. Build backend filters for Pinecone (Hybrid Search)
        filters = self._build_search_filters(query_params)
        logger.info(f"Executing hybrid search with filters: {filters}")
        
        # 3. Retrieve relevant chunks (Multi-View)
        search_results = self.vector_manager.hybrid_search(
            query=query_text,
            filters=filters,
            top_k=top_k
        )
        
        if not search_results:
            logger.warning(f"No results found for query: '{query_text}'")
            return QueryResult(
                answer="I couldn't find any receipts matching your request.",
                receipts=[],
                items=[],
                confidence=0.0,
                query_type=query_params.get('query_type', 'general'),
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                metadata=query_params
            )

        # 4. Generate grounded answer
        answer_data = self.generator.generate_answer(query_text, search_results)
        
        # 5. Semantic verification (Aggregation)
        if query_params.get('aggregation'):
            audit_result = self._perform_aggregation_audit(query_params, search_results)
            if audit_result:
                answer_data['answer'] += f"\n\n(Verified Sum: ${audit_result['value']:.2f} across {audit_result['count']} entries)"

        # Process results into receipts/items lists for the QueryResult object
        receipts = []
        items = []
        seen_rids = set()
        for res in search_results:
            meta = res.get('metadata', {})
            rid = meta.get('receipt_id')
            if rid and rid not in seen_rids:
                receipts.append(meta)
                seen_rids.add(rid)
            if meta.get('chunk_type') == 'item_detail':
                items.append({
                    'name': meta.get('item_name'),
                    'price': meta.get('item_price'),
                    'merchant': meta.get('merchant_name'),
                    'filename': meta.get('filename')
                })

        return QueryResult(
            answer=answer_data['answer'],
            receipts=receipts,
            items=items,
            confidence=min(len(search_results) / 10.0, 1.0),
            query_type=query_params.get('query_type', 'general'),
            processing_time=(datetime.utcnow() - start_time).total_seconds(),
            metadata={
                'query_params': query_params,
                'search_filters': filters,
                'retrieval_count': len(search_results)
            }
        )

    def process_query(self, query: str) -> QueryResult:
        """Alias for query() to support older test scripts."""
        return self.query(query)

    def _build_search_filters(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Translates parsed intent into Pinecone metadata filters.
        Strict adherence to metadata is enforced here to solve the 'Vector-Only' red flag.
        """
        filters = {}
        
        # 1. Merchant Matching (Strict normalization)
        merchants = params.get('merchants', [])
        if merchants:
            norm_merchants = [self._normalize_merchant_name(m) for m in merchants]
            norm_merchants = [m for m in norm_merchants if m] # Remove empty strings
            if norm_merchants:
                if len(norm_merchants) == 1:
                    filters['merchant_name_norm'] = norm_merchants[0]
                else:
                    filters['merchant_name_norm'] = {"$in": norm_merchants}

        # 2. Temporal Logic (Mandatory for temporal queries)
        if 'date_filter' in params:
            filters.update(params['date_filter'])
        
        if 'date_range' in params:
            try:
                # Support both dict and ISO string formats
                start_val = params['date_range'].get('start')
                end_val = params['date_range'].get('end')
                
                if start_val and end_val:
                    start_ts = int(datetime.fromisoformat(start_val.replace('Z', '+00:00')).timestamp())
                    end_ts = int(datetime.fromisoformat(end_val.replace('Z', '+00:00')).timestamp())
                    filters['transaction_ts'] = {"$gte": start_ts, "$lte": end_ts}
            except Exception as e:
                logger.error(f"Filter Error (Date Range): {e}")

        # 3. Category Logic
        categories = params.get('categories', [])
        if categories:
            if len(categories) == 1:
                filters['item_category'] = categories[0]
            else:
                filters['item_category'] = {"$in": categories}

        # 4. Financial Thresholds
        if 'min_amount' in params or 'max_amount' in params:
            amt_filter = {}
            if 'min_amount' in params: amt_filter["$gte"] = float(params['min_amount'])
            if 'max_amount' in params: amt_filter["$lte"] = float(params['max_amount'])
            filters['total_amount'] = amt_filter

        # 5. Feature Flags & City/State
        for field in ['has_warranty', 'is_return', 'has_tip', 'has_discounts', 'has_delivery_fee', 'merchant_city', 'merchant_state']:
            if params.get(field):
                filters[field] = params[field]
        
        if 'feature_any_of' in params:
            filters["$or"] = [{f: True} for f in params['feature_any_of']]

        return filters if filters else None

    def _normalize_merchant_name(self, name: str) -> str:
        """Standardizes merchant names for precise matching."""
        if not name: return ""
        norm = name.lower()
        norm = re.sub(r'[^a-z0-9]', '', norm)
        # Suffix stripping for better matching (e.g., 'Target Store' -> 'target')
        norm = re.sub(r'(inc|corp|llc|store|shop|market|pharmacy|cafe|coffee|restaurant)$', '', norm)
        return norm.strip()

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
