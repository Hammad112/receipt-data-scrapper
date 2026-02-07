"""
Centralized normalization utilities for the Receipt Intelligence System.
"""

import re

def normalize_merchant_name(name: str) -> str:
    """
    Standardizes merchant names for precise matching and indexing.
    
    Transformation pipeline:
    1. Force lowercase
    2. Remove non-alphanumeric characters
    3. Strip common corporate suffixes (inc, corp, llc, etc.)
    4. Strip common store types (store, shop, market, etc.)
    """
    if not name:
        return ""
    
    # 1. Basic cleaning
    norm = name.lower()
    norm = re.sub(r'[^a-z0-9\s]', '', norm)
    norm = re.sub(r'\s+', ' ', norm).strip()
    
    # 2. Suffix stripping (e.g., 'Target Store' -> 'target', 'Walmart Inc' -> 'walmart')
    suffixes = [
        'inc', 'corp', 'llc', 'store', 'shop', 'market', 
        'pharmacy', 'cafe', 'coffee', 'restaurant', 'ltd'
    ]
    
    # Sort by length descending to match longest suffixes first
    suffixes_sorted = sorted(suffixes, key=len, reverse=True)
    pattern = r'\s+(?:' + '|'.join(suffixes_sorted) + r')$'
    
    norm = re.sub(pattern, '', norm)
    
    return norm.strip()
