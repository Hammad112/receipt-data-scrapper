"""
Centralized regex patterns and semantic mappings for the QueryParser.
"""

import re

# Patterns for classifying the overall query intent
QUERY_PATTERNS = {
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

# Semantic mappings for expanding general terms into specific keywords
SEMANTIC_MAPPINGS = {
    'health_related': ['pharmacy', 'health', 'medicine', 'vitamin', 'supplement'],
    'treats': ['candy', 'chocolate', 'ice cream', 'cake', 'cookie', 'donut', 'dessert', 'sweet', 'pastry', 'croissant', 'muffin', 'brownie', 'snack'],
    'coffee_shops': ['coffee', 'starbucks', 'dunkin', 'cafe', 'latte', 'espresso'],
    'restaurants': ['restaurant', 'burger', 'pizza', 'sandwich', 'salad', 'pasta', 'steak'],
}

# Patterns for identifying the primary metric (e.g., total, tax, tip)
METRIC_PATTERNS = [
    r'\btotal\b', r'\bsum\b', r'\bspent\b', r'\bcost\b', r'\bprice\b',
    r'\bamount\b', r'\bhow much\b', r'\btaxes\b', r'\btax\b', r'\btips?\b'
]

# Patterns for identifying aggregation types (sum, average, count)
AGGREGATION_PATTERNS = {
    'sum': [r'\bsum\b', r'\btotal\b', r'\badd up\b', r'\ball\b', r'\bhow much\b', r'\bspent\b'],
    'average': [r'\baverage\b', r'\bavg\b', r'\bmean\b'],
    'count': [r'\bcount\b', r'\bhow many\b', r'\bnumber of\b']
}
