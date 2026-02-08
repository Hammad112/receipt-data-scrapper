"""
Refactored QueryParser utilizing modular components for patterns and date resolution.
"""

import re
from typing import List, Optional, Dict, Any

# Industrial-grade absolute imports
from .patterns import (
    QUERY_PATTERNS, SEMANTIC_MAPPINGS, METRIC_PATTERNS, 
    AGGREGATION_PATTERNS
)
from .advanced_date_resolver import TemporalQueryResolver
from .semantic_merchant_matcher import SemanticMerchantMatcher
from ..models import PaymentMethod, ItemCategory
from ..utils.logging_config import logger

class QueryParser:
    """
    Lean orchestrator for parsing natural language receipt queries.
    Delegates specific extraction tasks to specialized modules.
    """

    def __init__(self, openai_client=None):
        """Initializes the parser and compiles patterns for reuse."""
        self.openai_client = openai_client
        self.metric_re = [re.compile(p, re.I) for p in METRIC_PATTERNS]
        self.query_pattern_compiled = {
            q_type: [re.compile(p, re.I) for p in patterns]
            for q_type, patterns in QUERY_PATTERNS.items()
        }
        self.aggregation_pattern_compiled = {
            agg_type: [re.compile(p, re.I) for p in patterns]
            for agg_type, patterns in AGGREGATION_PATTERNS.items()
        }
        
        # Initialize specialized resolvers
        self.temporal_resolver = TemporalQueryResolver(openai_client)
        self.merchant_matcher = SemanticMerchantMatcher(openai_client)

    def parse(self, query: str) -> Dict[str, Any]:
        """Entry point for query decomposition."""
        params = {
            'original_query': query,
            'query_type': self._classify_query(query)
        }

        # 1. Metric & Date Resolution
        metric = self._extract_metric(query)
        if metric: params['metric'] = metric
        
        # Use advanced temporal resolver
        date_range = self.temporal_resolver.resolve_date_range(query)
        if date_range:
            params.update(date_range)

        # 2. Entity Extraction
        # Use semantic merchant matcher
        merchants = self.merchant_matcher.extract_merchants(query)
        
        # Filter out patterns that look like dates (e.g., "December 2023")
        
        merchants = self._filter_merchants(merchants)
        
        if merchants: params['merchants'] = merchants
        
        categories = self._extract_categories(query)
        if categories: params['categories'] = categories

        # 3. Attributes & Flags
        params.update(self._extract_payment_details(query))
        params.update(self._extract_feature_flags(query))
        params.update(self._extract_amounts(query))
        
        # 4. Semantic & Mathematical Intent
        semantic_cats = self._extract_semantic_categories(query)
        if semantic_cats: params['semantic_categories'] = semantic_cats
        
        agg = self._extract_aggregation_type(query)
        if agg: params['aggregation'] = agg

        # 5. LLM Fallback (if critical fields missing)
        if not params.get('merchants') or not params.get('date_range'):
            params.update(self._get_llm_fallback(query, params))

        # 6. Final Derivations
        params['sum_basis'] = self._derive_sum_basis(params)
        # 6. Final Derivations
        params['sum_basis'] = self._derive_sum_basis(params)
        return params

    def _filter_merchants(self, merchants: List[str]) -> List[str]:
        """
        Applies rigorous filtering to remove temporal terms, categories, 
        and date-like patterns from merchant lists.
        """
        if not merchants: return []
        
        # Filter out temporal terms that might be mistaken as merchants
        temporal_terms = {
            'january', 'jan', 'february', 'feb', 'march', 'mar', 'april', 'apr',
            'may', 'june', 'jun', 'july', 'jul', 'august', 'aug', 'september', 
            'sep', 'sept', 'october', 'oct', 'november', 'nov', 'december', 'dec',
            'today', 'yesterday', 'tomorrow', 'week', 'month', 'year'
        }
        
        # Filter out category terms that should not be treated as merchants
        category_terms = {
            'coffee shops', 'coffee shop', 'restaurants', 'restaurant',
            'groceries', 'grocery', 'electronics', 'pharmacy', 'pharmacies',
            'treats', 'desserts', 'fast food', 'health', 'shopping', 'store'
        }
        
        # Filter out patterns that look like dates
        import re
        date_pattern = re.compile(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+\d{4}\b', re.I)
        
        filtered_merchants = []
        for m in merchants:
            m_lower = m.lower().strip()
            if not m_lower: continue
            
            # Skip if it's a temporal term
            if m_lower in temporal_terms:
                continue
            # Skip if it's a category term
            if m_lower in category_terms:
                continue
            # Skip if it looks like a date
            if date_pattern.match(m):
                continue
                
            filtered_merchants.append(m)
            
        return filtered_merchants

    def _classify_query(self, query: str) -> str:
        """Categorizes the query intent."""
        ql = query.lower()
        for q_type, patterns in self.query_pattern_compiled.items():
            if any(p.search(ql) for p in patterns):
                return q_type
        return 'general'

    def _extract_metric(self, query: str) -> Optional[str]:
        """Identifies the numerical field (tax, tip, total)."""
        ql = query.lower()
        if 'tax' in ql: return 'tax_amount'
        if 'tip' in ql: return 'tip_amount'
        if any(p.search(ql) for p in self.metric_re):
            return 'total_amount'
        return None


    def _extract_categories(self, query: str) -> List[str]:
        """Maps query terms to system categories."""
        ql = query.lower()
        categories = []
        
        # Map user-friendly terms to actual category values
        if any(term in ql for term in ['coffee shops', 'coffee shop']):
            categories.extend(['coffee_shop', 'fast_food'])  # Both are coffee-related
        if any(term in ql for term in ['restaurants', 'restaurant']):
            categories.extend(['restaurant', 'fast_food'])  # Both are dining out
        if any(term in ql for term in ['groceries', 'grocery']):
            categories.append('groceries')
        if 'electronics' in ql:
            categories.append('electronics')
        if any(term in ql for term in ['pharmacy', 'health']):
            categories.append('pharmacy')
            
        return list(set(categories))  # Remove duplicates

    def _extract_payment_details(self, query: str) -> Dict[str, Any]:
        """Detects payment method and card network."""
        ql = query.lower()
        res = {}
        methods = {
            'apple pay': PaymentMethod.APPLE_PAY,
            'google pay': PaymentMethod.GOOGLE_PAY,
            'cash': PaymentMethod.CASH,
            'debit': PaymentMethod.DEBIT,
            'credit': PaymentMethod.CREDIT
        }
        for kw, method in methods.items():
            if kw in ql: res['payment_method'] = method.value
        
        networks = ['visa', 'mastercard', 'amex', 'discover']
        for n in networks:
            if n in ql: 
                res['card_network'] = n
                if 'payment_method' not in res: res['payment_method'] = PaymentMethod.CREDIT.value
        return res

    def _extract_feature_flags(self, query: str) -> Dict[str, Any]:
        """Detects boolean feature intent."""
        ql = query.lower()
        flags = {}
        if 'warranty' in ql: flags['has_warranty'] = True
        if re.search(r'\b(return|refund|returned)\b', ql): flags['is_return'] = True
        if 'discount' in ql: flags['has_discounts'] = True
        if 'delivery' in ql: flags['has_delivery_fee'] = True
        if 'tip' in ql: flags['has_tip'] = True
        return flags

    def _extract_amounts(self, query: str) -> Dict[str, Any]:
        """Extracts financial threshold filters."""
        res = {}
        matches = re.findall(r'\$(\d+(?:\.\d{2})?)', query)
        for val in matches:
            amt = float(val)
            if any(kw in query.lower() for kw in ['over', 'more', 'above']): res['min_amount'] = amt
            elif any(kw in query.lower() for kw in ['under', 'less', 'below']): res['max_amount'] = amt
        return res

    def _extract_semantic_categories(self, query: str) -> List[str]:
        """Expands descriptive terms for vector expansion."""
        res = []
        ql = query.lower()
        for cat, keywords in SEMANTIC_MAPPINGS.items():
            if cat.replace('_', ' ') in ql or any(kw in ql for kw in keywords):
                res.extend(keywords)
        return list(set(res))

    def _extract_aggregation_type(self, query: str) -> Optional[str]:
        """Identifies mathematical goal."""
        ql = query.lower()
        for agg, patterns in self.aggregation_pattern_compiled.items():
            if any(p.search(ql) for p in patterns):
                return agg
        return None

    def _get_llm_fallback(self, query: str, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """LLM enrichment for complex entity resolution."""
        try:
            from openai import OpenAI
            import json
            client = OpenAI()
            prompt = f"Extract financial parameters from: \"{query}\"\nReturn JSON: {{'merchants': [], 'date_range': {{'start':'ISO', 'end':'ISO'}}, 'aggregation': 'sum|avg|count|null'}}"
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            data = json.loads(resp.choices[0].message.content)
            # Only update missing fields
            res = {}
            if not current_params.get('merchants') and data.get('merchants'): 
                # CRITICAL: Apply the same rigorous filtering to LLM outputs
                filtered = self._filter_merchants(data['merchants'])
                if filtered:
                    res['merchants'] = filtered
                    
            if not current_params.get('date_range') and data.get('date_range'):
                # Validate date_range has actual values, not None
                dr = data.get('date_range')
                if dr and dr.get('start') and dr.get('end'):
                    res['date_range'] = dr
            if not current_params.get('aggregation') and data.get('aggregation') in ['sum', 'average', 'count']:
                res['aggregation'] = data['aggregation']
            return res
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return {}

    def _derive_sum_basis(self, params: Dict[str, Any]) -> str:
        """Determines if calculation should be item-based or receipt-based."""
        ql = params.get('original_query', '').lower()
        if params.get('metric') in ['tax_amount', 'tip_amount']: return 'receipts'
        if params.get('query_type') in ['category', 'item_specific'] or 'categories' in params: return 'items'
        if any(kw in ql for kw in ['items', 'buy', 'bought', 'purchase']): return 'items'
        return 'receipts'
