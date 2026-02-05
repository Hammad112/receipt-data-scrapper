"""
Query parsing logic for extracting structured parameters from natural language.
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal

try:
    from ..models import ItemCategory, PaymentMethod
    from ..utils.logging_config import logger
except ImportError:
    from models import ItemCategory, PaymentMethod
    from utils.logging_config import logger


class QueryParser:
    """
    Parses natural language queries to extract filters and intent parameters.
    """
    
    def __init__(self):
        """Initializes the parser with intent patterns and mappings."""
        self.query_patterns = {
            'temporal': [
                r'how much.* (?:in|during|for) (?:january|february|march|april|may|june|july|august|september|october|november|december)',
                r'how much.* (?:last|this|past) (?:week|month|year)',
                r'show me.* (?:january|february|march|april|may|june|july|august|september|october|november|december)',
                r'what did i buy (?:last|this|past) (?:week|month|year)',
                r'(?:in|during) (?:20\d{2})',
            ],
            'merchant': [
                r'show me.* (?:from|at) .*',
                r'find all.* receipts? (?:from|at) .*',
                r'how much.* (?:at|from) .*',
                r'(?:whole foods|target|walmart|cvs|walgreens|starbucks|amazon|best buy|costco|safeway)',
            ],
            'category': [
                r'how much.* (?:coffee shops|restaurants|groceries|electronics)',
                r'show me.* (?:electronics|groceries|pharmacy|health)',
                r'what(?:\'s|\'s).* (?:total|total spending) (?:at|in) (?:restaurants|coffee shops)',
                r'list all.* (?:groceries|electronics)',
            ],
            'amount': [
                r'over \$\d+',
                r'under \$\d+',
                r'between \$\d+ and \$\d+',
                r'more than \$\d+',
                r'less than \$\d+',
            ],
            'item_specific': [
                r'find.* with warranty',
                r'show me.* (?:phone|laptop|tv|tablet)',
                r'list all.* (?:vitamins|medicine|supplements)',
            ],
            'aggregation': [
                r'how much.* (?:total|sum)',
                r'what\'s my total',
                r'average',
                r'count',
            ]
        }
        
        self.semantic_mappings = {
            'health_related': ['pharmacy', 'health', 'medicine', 'vitamin', 'supplement'],
            'treats': ['candy', 'chocolate', 'ice cream', 'cake', 'cookie', 'donut', 'dessert', 'sweet'],
            'coffee_shops': ['coffee', 'starbucks', 'dunkin', 'cafe', 'latte', 'espresso'],
            'restaurants': ['restaurant', 'burger', 'pizza', 'sandwich', 'salad', 'pasta', 'steak'],
        }

    def parse(self, query: str) -> Dict[str, Any]:
        """
        Extracts all parameters and intent from a query.
        """
        query_type = self._classify_query(query)
        params = {'original_query': query, 'query_type': query_type}
        
        # Extraction pipeline
        params.update(self._extract_dates(query))
        
        merchants = self._extract_merchants(query)
        if merchants: params['merchants'] = merchants
        
        categories = self._extract_categories(query)
        if categories: params['categories'] = categories
        
        params.update(self._extract_amounts(query))
        
        semantic_cats = self._extract_semantic_categories(query)
        if semantic_cats: params['semantic_categories'] = semantic_cats
        
        agg = self._extract_aggregation_type(query)
        if agg: params['aggregation'] = agg
        
        return params

    def _classify_query(self, query: str) -> str:
        """Categorizes the query into high-level intent types."""
        query_lower = query.lower()
        for q_type, patterns in self.query_patterns.items():
            if any(re.search(p, query_lower) for p in patterns):
                return q_type
        return 'general'

    def _extract_dates(self, query: str) -> Dict[str, Any]:
        """Extracts absolute or relative dates."""
        query_lower = query.lower()
        months = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
            'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        for name, num in months.items():
            if re.search(r'\b' + name + r'\b', query_lower):
                year_match = re.search(r'20(\d{2})', query)
                date_filter = {'transaction_month': num}
                if year_match:
                    date_filter['transaction_year'] = int(year_match.group())
                return {'date_filter': date_filter}
        
        # Relative
        now = datetime.now()
        if 'last week' in query_lower:
            return {'date_range': {'start': (now - timedelta(days=7)).isoformat(), 'end': now.isoformat()}}
        if 'last month' in query_lower:
            start = (now.replace(day=1) - timedelta(days=30)).replace(day=1)
            return {'date_range': {'start': start.isoformat(), 'end': now.replace(day=1).isoformat()}}
        if 'yesterday' in query_lower:
            target = now - timedelta(days=1)
            return {'date_range': {'start': target.replace(hour=0, minute=0, second=0).isoformat(), 
                                   'end': target.replace(hour=23, minute=59, second=59).isoformat()}}
        if 'this week' in query_lower:
            start = now - timedelta(days=now.weekday())
            return {'date_range': {'start': start.replace(hour=0, minute=0, second=0).isoformat(), 'end': now.isoformat()}}
        if 'this month' in query_lower:
            return {'date_range': {'start': now.replace(day=1, hour=0, minute=0, second=0).isoformat(), 'end': now.isoformat()}}
            
        return {}

    def _extract_merchants(self, query: str) -> List[str]:
        """Extracts known merchant names."""
        known = ['whole foods', 'target', 'walmart', 'cvs', 'walgreens', 'starbucks', 
                 'amazon', 'best buy', 'costco', 'safeway', "trader joe's"]
        query_lower = query.lower()
        return [m.title() for m in known if m in query_lower]

    def _extract_categories(self, query: str) -> List[str]:
        """Maps query terms to system categories."""
        mappings = {
            'coffee shops': 'coffee_shop', 'restaurants': 'restaurant', 
            'groceries': 'groceries', 'electronics': 'electronics',
            'pharmacy': 'pharmacy', 'health': 'pharmacy', 'treats': 'treats'
        }
        query_lower = query.lower()
        return [cat for term, cat in mappings.items() if term in query_lower]

    def _extract_amounts(self, query: str) -> Dict[str, Any]:
        """Extracts min/max amount filters."""
        amounts = {}
        matches = re.findall(r'\$(\d+(?:\.\d{2})?)', query)
        for val in matches:
            try:
                amt = float(val)
                if any(kw in query.lower() for kw in ['over', 'more than', 'above']):
                    amounts['min_amount'] = amt
                elif any(kw in query.lower() for kw in ['under', 'less than', 'below']):
                    amounts['max_amount'] = amt
            except ValueError:
                continue
        return amounts

    def _extract_semantic_categories(self, query: str) -> List[str]:
        """Finds additional query keywords based on semantic groups."""
        semantic_cats = []
        ql = query.lower()
        for cat, keywords in self.semantic_mappings.items():
            if cat.replace('_', ' ') in ql or any(kw in ql for kw in keywords):
                semantic_cats.extend(keywords)
        return list(set(semantic_cats))

    def _extract_aggregation_type(self, query: str) -> Optional[str]:
        """Identifies math intent (sum, avg, count)."""
        ql = query.lower()
        if any(kw in ql for kw in ['total', 'sum', 'add up']): return 'sum'
        if any(kw in ql for kw in ['average', 'avg']): return 'average'
        if 'count' in ql or 'how many' in ql: return 'count'
        return None
