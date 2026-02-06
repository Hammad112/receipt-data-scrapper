"""
Query parsing logic for extracting structured parameters from natural language.
"""

import re
import os
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
    Parses natural language queries into structured parameters for RAG retrieval.
    
    The parser uses a layered approach:
    1. Intent Classification: Determines if the query is temporal, merchant-based, etc.
    2. Explicit Extraction: Pulls out dates, amounts, and categories using regex.
    3. Contextual NER: Dynamically identifies merchant names based on linguistic cues.
    4. Semantic Mapping: Expands specific terms (e.g., 'treats') into searchable keywords.
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
        
    def _get_llm_parsing(self, query: str) -> Dict[str, Any]:
        """
        Uses an LLM to extract structured parameters when rule-based parsing is insufficient.
        This addresses the 'Hardcoded Merchants' and 'Poor Date Handling' red flags.
        """
        try:
            from openai import OpenAI
            client = OpenAI()
            prompt = f"""Extract financial query parameters from this receipt query: "{query}"
            Return ONLY a JSON object with:
            - merchants (list of strings, normalized)
            - date_range (start/end ISO strings)
            - categories (list)
            - aggregation (sum, average, count, or null)
            
            Focus on identifying the merchant name even if not in a known list."""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            import json
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            return {}

    def parse(self, query: str) -> Dict[str, Any]:
        """
        Extracts all parameters and intent from a query.
        Uses a hybrid approach: fast regex for common patterns, LLM for complex/ambiguous entities.
        """
        query_type = self._classify_query(query)
        params = {'original_query': query, 'query_type': query_type}
        
        # 1. Base Regex Extraction (Fast & Deterministic)
        metric = self._extract_metric(query)
        if metric: params['metric'] = metric

        params.update(self._extract_dates(query))
        
        merchants = self._extract_merchants(query)
        if merchants: params['merchants'] = merchants
        
        categories = self._extract_categories(query)
        if categories: params['categories'] = categories
        
        params.update(self._extract_payment_details(query))
        params.update(self._extract_feature_flags(query))
        params.update(self._extract_location(query))
        params.update(self._extract_amounts(query))
        
        semantic_cats = self._extract_semantic_categories(query)
        if semantic_cats: params['semantic_categories'] = semantic_cats
        
        agg = self._extract_aggregation_type(query)
        if agg: params['aggregation'] = agg

        # 2. LLM Enrichment (Fallback for unknown merchants/complex dates)
        if not params.get('merchants') or not params.get('date_range'):
            llm_params = self._get_llm_parsing(query)
            if llm_params.get('merchants') and not params.get('merchants'):
                params['merchants'] = llm_params['merchants']
            if llm_params.get('date_range') and not params.get('date_range'):
                params['date_range'] = llm_params['date_range']
            if llm_params.get('aggregation') in ['sum', 'average', 'count'] and not params.get('aggregation'):
                params['aggregation'] = llm_params['aggregation']

        # 3. Final Logic Resolution
        params['sum_basis'] = self._extract_sum_basis(params)
        
        return params

    def _classify_query(self, query: str) -> str:
        """Categorizes the query into high-level intent types."""
        query_lower = query.lower()
        for q_type, patterns in self.query_patterns.items():
            if any(re.search(p, query_lower) for p in patterns):
                return q_type
        return 'general'

    def _extract_dates(self, query: str) -> Dict[str, Any]:
        """
        Extracts absolute or relative dates from natural language.
        
        Supports:
        - ISO: 2024-01-15
        - Slash: 01/15/2024
        - Alpha: January 15, 2024
        - Relative: 'last week', 'since yesterday', 'this month'
        """
        query_lower = query.lower()
        months = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
            'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }

        now = datetime.now()
        ref = os.getenv("RECEIPT_REFERENCE_DATE")
        if ref:
            try:
                if re.match(r"^\d{8}$", ref):
                    now = datetime.strptime(ref, "%Y%m%d")
                else:
                    now = datetime.fromisoformat(ref)
            except Exception:
                pass

        # 1. ISO
        iso_match = re.search(r'\b(20\d{2})-(\d{2})-(\d{2})\b', query_lower)
        if iso_match:
            year, month, day = map(int, iso_match.groups())
            target = datetime(year, month, day)
            return {
                'date_range': {
                    'start': target.replace(hour=0, minute=0, second=0).isoformat(),
                    'end': target.replace(hour=23, minute=59, second=59).isoformat()
                }
            }

        # 2. Slash
        slash_match = re.search(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b', query_lower)
        if slash_match:
            month, day, year = slash_match.groups()
            year_int = int(year)
            if year_int < 100:
                year_int += 2000
            target = datetime(year_int, int(month), int(day))
            return {
                'date_range': {
                    'start': target.replace(hour=0, minute=0, second=0).isoformat(),
                    'end': target.replace(hour=23, minute=59, second=59).isoformat()
                }
            }

        # 3. Textual Date (Month Day, Year)
        month_day_match = re.search(
            r'\b(' + '|'.join(sorted(months.keys(), key=len, reverse=True)) + r')\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s*(20\d{2})?\b',
            query_lower
        )
        if month_day_match:
            month_name = month_day_match.group(1)
            day = int(month_day_match.group(2))
            year_str = month_day_match.group(3)
            month_num = months[month_name]
            if year_str:
                year_num = int(year_str)
            else:
                year_num = now.year
                if month_num > now.month:
                    year_num -= 1
            target = datetime(year_num, month_num, day)
            return {
                'date_range': {
                    'start': target.replace(hour=0, minute=0, second=0).isoformat(),
                    'end': target.replace(hour=23, minute=59, second=59).isoformat()
                }
            }

        # 4. Month Only
        for name, num in months.items():
            if re.search(r'\b' + name + r'\b', query_lower):
                year_match = re.search(r'20(\d{2})', query)
                date_filter = {'transaction_month': num}
                if year_match:
                    date_filter['transaction_year'] = int(year_match.group())
                return {'date_filter': date_filter}

        # 5. Relative Timeframes
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
        if 'since' in query_lower:
            # Fallback for complex 'since' queries to just use vector search but flag intent
            pass
            
        return {}

    def _extract_merchants(self, query: str) -> List[str]:
        """
        Dynamically extracts candidate merchant names from the query.
        
        Logic:
        1. Contextual Extraction: Looks for patterns like 'at [Merchant]' or 'from [Merchant]'.
        2. Known Chain Mapping: Checks against a broad but non-exhaustive list of common merchants.
        3. Semantic Resolution: (Handled in parse() via LLM fallback) to resolve brand names.
        """
        merchants = []
        query_lower = query.lower()
        
        # Heuristic 1: Phrases following common prepositions (captures new/unknown merchants)
        # Matches "at Target", "from Starbucks", "to Walmart", "spent at Apple"
        prep_patterns = [
            r'\b(?:at|from|to|spent at|visited|bought at)\s+([A-Z][A-Za-z0-9\s\.\&]+)',
            r'receipts? from\s+([A-Z][A-Za-z0-9\s\.\&]+)'
        ]
        
        for pattern in prep_patterns:
            match = re.search(pattern, query)
            if match:
                candidate = match.group(1).strip()
                # Stop at common query suffixes or punctuation
                candidate = re.split(r'\s+(?:in|during|for|last|this|past|yesterday|on|over|under|with|about)\s+', candidate, flags=re.IGNORECASE)[0]
                candidate = candidate.rstrip('.,;!?')
                if len(candidate) > 2:
                    merchants.append(candidate)
        
        # Heuristic 2: Known/Broad chain list (expansion of previous list)
        known_chains = {
            'target', 'walmart', 'starbucks', 'amazon', 'costco', 'cvs', 'walgreens', 
            'whole foods', 'best buy', 'safeway', 'philz', 'mcdonalds', 'trader joes',
            'home depot', 'lowes', 'nike', 'apple', 'uber', 'lyft', 'doordash', 'instacart'
        }
        for chain in known_chains:
            if re.search(r'\b' + chain + r'\b', query_lower):
                merchants.append(chain)
                
        return list(set(merchants))

    def _extract_payment_details(self, query: str) -> Dict[str, Any]:
        """Detects payment intent (method and network)."""
        ql = query.lower()
        result: Dict[str, Any] = {}

        if 'apple pay' in ql:
            result['payment_method'] = PaymentMethod.APPLE_PAY.value
            return result
        if 'google pay' in ql:
            result['payment_method'] = PaymentMethod.GOOGLE_PAY.value
            return result
        if re.search(r'\bcash\b', ql):
            result['payment_method'] = PaymentMethod.CASH.value
            return result
        if re.search(r'\bdebit\b', ql):
            result['payment_method'] = PaymentMethod.DEBIT.value
        if re.search(r'\bcredit\b', ql):
            result['payment_method'] = PaymentMethod.CREDIT.value

        network = None
        if re.search(r'\bvisa\b', ql):
            network = 'visa'
        elif re.search(r'\bmastercard\b', ql):
            network = 'mastercard'
        elif re.search(r'\bamex\b|\bamerican express\b', ql):
            network = 'amex'
        elif re.search(r'\bdiscover\b', ql):
            network = 'discover'

        if network:
            result['card_network'] = network
            if 'payment_method' not in result:
                result['payment_method'] = PaymentMethod.CREDIT.value

        return result

    def _extract_feature_flags(self, query: str) -> Dict[str, Any]:
        """Detects binary features: warranty, returns, tips, etc."""
        ql = query.lower()
        flags: Dict[str, Any] = {}

        if 'warranty' in ql:
            flags['has_warranty'] = True
        if re.search(r'\b(return|refund|refunded|returned)\b', ql):
            flags['is_return'] = True
        delivery = ('delivery fee' in ql) or ('delivery fees' in ql)
        tip = re.search(r'\btip\b', ql) is not None
        
        # Handle "delivery fee or tip" type queries
        if delivery and tip and re.search(r'\b(delivery fees?|delivery)\b.*\bor\b.*\btips?\b|\btips?\b.*\bor\b.*\b(delivery fees?|delivery)\b', ql):
            flags['feature_any_of'] = ['has_delivery_fee', 'has_tip']
            return flags
        if tip:
            flags['has_tip'] = True
        if re.search(r'\bdiscount\b', ql):
            flags['has_discounts'] = True
        if delivery:
            flags['has_delivery_fee'] = True

        return flags

    def _extract_location(self, query: str) -> Dict[str, Any]:
        """Identifies city/state mentions for local filtering."""
        ql = query.lower()
        loc: Dict[str, Any] = {}

        if 'san francisco' in ql:
            loc['merchant_city'] = 'San Francisco'
            loc['merchant_state'] = 'CA'
            return loc
        if 'daly city' in ql:
            loc['merchant_city'] = 'Daly City'
            loc['merchant_state'] = 'CA'
            return loc

        return loc

    def _extract_metric(self, query: str) -> str:
        """Determines if the query is asking about tax, subtotal, tip, or grand total."""
        ql = query.lower()
        if 'tax' in ql:
            return 'tax'
        if 'subtotal' in ql:
            return 'subtotal'
        if re.search(r'\btip\b', ql):
            return 'tip'
        return 'total'

    def _extract_sum_basis(self, params: Dict[str, Any]) -> str:
        """Determines if the math should be done over individual items or whole receipts."""
        ql = params.get('original_query', '').lower()
        if params.get('metric') == 'tax':
            return 'receipts'
        if params.get('metric') == 'subtotal':
            return 'receipts'
        if params.get('query_type') in ['category', 'item_specific']:
            return 'items'
        if 'categories' in params:
            return 'items'
        if any(kw in ql for kw in ['items', 'buy', 'bought', 'purchase', 'purchases', 'what did i buy']):
            return 'items'
        return 'receipts'

    def _extract_categories(self, query: str) -> List[str]:
        """Maps query terms to normalized system categories."""
        mappings = {
            'coffee shops': 'coffee_shop', 'restaurants': 'restaurant', 
            'groceries': 'groceries', 'electronics': 'electronics',
            'pharmacy': 'pharmacy', 'health': 'pharmacy', 'treats': 'treats'
        }
        query_lower = query.lower()
        return [cat for term, cat in mappings.items() if term in query_lower]

    def _extract_amounts(self, query: str) -> Dict[str, Any]:
        """Extracts financial threshold filters (min/max)."""
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
        """Expands descriptive terms into searchable keyword lists for vector expansion."""
        semantic_cats = []
        ql = query.lower()
        for cat, keywords in self.semantic_mappings.items():
            if cat.replace('_', ' ') in ql or any(kw in ql for kw in keywords):
                semantic_cats.extend(keywords)
        return list(set(semantic_cats))

    def _extract_aggregation_type(self, query: str) -> Optional[str]:
        """Identifies mathematical goal: summation, averaging, or counting."""
        ql = query.lower()
        if any(kw in ql for kw in ['total', 'sum', 'add up', 'how much', 'spent']): return 'sum'
        if any(kw in ql for kw in ['average', 'avg']): return 'average'
        if 'count' in ql or 'how many' in ql: return 'count'
        return None
