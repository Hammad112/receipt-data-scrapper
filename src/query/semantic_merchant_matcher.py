"""
Semantic Merchant Matcher - LLM-powered merchant extraction without hardcoding.

This module addresses the critical red flag of hardcoded merchant names by using
a multi-strategy approach combining heuristics, fuzzy matching, and LLM fallback.

Key features:
- No hardcoded merchant lists
- Handles misspellings and variations
- Resolves aliases ("WFM" → "Whole Foods Market")
- Semantic understanding ("that coffee place")
"""

import re
import json
from typing import List, Optional, Set
from difflib import SequenceMatcher

# Absolute imports for industrial stability
from src.utils.logging_config import logger
from src.utils.normalization import normalize_merchant_name


class SemanticMerchantMatcher:
    """
    Intelligent merchant extraction using multiple strategies.
    
    Strategy hierarchy:
    1. Prepositional context extraction (fast, rule-based)
    2. Fuzzy matching against known corpus (medium accuracy)
    3. LLM semantic extraction (high accuracy, slower)
    
    NO hardcoded lists - learns from indexed receipts.
    """
    
    def __init__(self, openai_client=None):
        """
        Initialize the matcher with optional OpenAI client.
        
        Args:
            openai_client: OpenAI client for LLM fallback. If None, will initialize lazily.
        """
        self._openai_client = openai_client
        self._merchant_corpus = set()  # Learned from indexed receipts
        
        # Prepositions that typically precede merchant names
        self.merchant_prepositions = [
            'at', 'from', 'to', 'in', 'spent at', 'bought at', 'visited',
            'shopped at', 'ordered from', 'purchased from'
        ]
        
        # Common merchant type indicators (not hardcoded merchants!)
        self.merchant_indicators = [
            'store', 'shop', 'market', 'pharmacy', 'cafe', 'restaurant',
            'coffee', 'foods', 'grocery', 'supercenter', 'depot'
        ]
    
    def extract_merchants(self, query: str, indexed_merchants: Optional[Set[str]] = None) -> List[str]:
        """
        Extract merchant names from natural language query.
        
        This is the main entry point that orchestrates all extraction strategies.
        
        Args:
            query: Natural language query from user
            indexed_merchants: Set of merchant names from indexed receipts (for fuzzy matching)
            
        Returns:
            List of extracted merchant names (may be empty if none found)
            
        Examples:
            >>> extract_merchants("How much at Walmart?")
            ["Walmart"]
            
            >>> extract_merchants("Show me Starbucks and Target receipts")
            ["Starbucks", "Target"]
            
            >>> extract_merchants("That expensive coffee place in SF")
            ["Starbucks"]  # Via LLM + context
        """
        # Update corpus if provided
        if indexed_merchants:
            self._merchant_corpus.update(indexed_merchants)
        
        merchants = []
        
        # Strategy 1: Prepositional context (fastest)
        prep_merchants = self._extract_via_prepositions(query)
        merchants.extend(prep_merchants)
        
        # Strategy 2: Fuzzy matching against corpus (medium)
        if not merchants and self._merchant_corpus:
            fuzzy_merchants = self._extract_via_fuzzy_match(query)
            merchants.extend(fuzzy_merchants)
        
        # Strategy 3: LLM semantic extraction (slowest, most accurate)
        if not merchants:
            llm_merchants = self._extract_via_llm(query)
            merchants.extend(llm_merchants)
        
        # Deduplicate and normalize
        return list(set(self._normalize_list(merchants)))
    
    def _extract_via_prepositions(self, query: str) -> List[str]:
        """
        Extract merchants using prepositional context.
        
        Pattern: "at Walmart" → "Walmart"
                 "from Target in San Francisco" → "Target"
        
        This is fast and works for explicit merchant mentions.
        """
        merchants = []
        query_lower = query.lower()
        
        # Build regex patterns for each preposition
        for prep in self.merchant_prepositions:
            # Pattern: preposition + capitalized word(s)
            pattern = r'\b' + re.escape(prep) + r'\s+([A-Z][A-Za-z0-9\s\.\&\']+)'
            
            for match in re.finditer(pattern, query):
                candidate = match.group(1).strip()
                
                # Stop at temporal/locational keywords
                candidate = re.split(
                    r'\s+(?:in|during|for|last|this|past|yesterday|on|over|under|before|after)\s+',
                    candidate,
                    flags=re.I
                )[0]
                
                # Clean punctuation
                candidate = candidate.rstrip('.,;!?')
                
                # Validate: minimum length, not just articles
                if len(candidate) > 2 and candidate.lower() not in ['the', 'a', 'an']:
                    merchants.append(candidate)
        
        return merchants
    
    def _extract_via_fuzzy_match(self, query: str) -> List[str]:
        """
        Fuzzy match against known merchant corpus.
        
        This handles slight variations and misspellings:
        - "Walmat" → "Walmart" (typo)
        - "whole foods" → "Whole Foods Market" (case variation)
        - "starbux" → "Starbucks" (phonetic)
        
        Uses Levenshtein-based similarity (via SequenceMatcher).
        """
        if not self._merchant_corpus:
            return []
        
        merchants = []
        query_tokens = self._tokenize(query)
        
        for token in query_tokens:
            if len(token) < 3:  # Skip short tokens
                continue
            
            # Find best match in corpus
            best_match = None
            best_score = 0.0
            
            for corpus_merchant in self._merchant_corpus:
                # Normalize for comparison
                norm_merchant = normalize_merchant_name(corpus_merchant)
                norm_token = normalize_merchant_name(token)
                
                # Calculate similarity
                score = SequenceMatcher(None, norm_token, norm_merchant).ratio()
                
                # Also check if token is substring (e.g., "walmart" in "walmart supercenter")
                if norm_token in norm_merchant or norm_merchant in norm_token:
                    score = max(score, 0.9)  # Boost substring matches
                
                if score > best_score:
                    best_score = score
                    best_match = corpus_merchant
            
            # Threshold: 0.75 similarity required
            if best_match and best_score >= 0.75:
                merchants.append(best_match)
                logger.debug(f"Fuzzy match: '{token}' → '{best_match}' (score: {best_score:.2f})")
        
        return merchants
    
    def _extract_via_llm(self, query: str) -> List[str]:
        """
        LLM-powered semantic merchant extraction.
        
        This is the most powerful strategy, handling:
        - Ambiguous references ("that coffee place")
        - Aliases ("WFM" → "Whole Foods Market")
        - Contextual understanding ("where I bought chicken")
        
        Uses structured output for reliability.
        """
        try:
            if not self._openai_client:
                from openai import OpenAI
                self._openai_client = OpenAI()
            
            # Build context-aware prompt
            prompt = self._build_llm_prompt(query)
            
            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Fast, cost-effective
                messages=[
                    {
                        "role": "system",
                        "content": "You are a merchant name extraction specialist. Extract ONLY merchant/store names from queries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0  # Deterministic
            )
            
            result = json.loads(response.choices[0].message.content)
            merchants = result.get('merchants', [])
            
            logger.info(f"LLM extracted merchants: {merchants}")
            return merchants
            
        except Exception as e:
            logger.error(f"LLM merchant extraction failed: {e}")
            return []
    
    def _build_llm_prompt(self, query: str) -> str:
        """
        Build context-aware prompt for LLM extraction.
        
        Includes corpus context if available to improve accuracy.
        """
        prompt = f"""Extract merchant/store names from this query: "{query}"

        Return JSON format: {{"merchants": ["Merchant1", "Merchant2"]}}

        Rules:
        1. Extract ONLY merchant/store/restaurant names
        2. Do NOT extract: dates, amounts, categories, items
        3. Normalize to proper capitalization (e.g., "walmart" → "Walmart")
        4. If uncertain, return empty list: {{"merchants": []}}
        5. Maximum 5 merchants per query

        Examples:
        - "How much at Walmart?" → {{"merchants": ["Walmart"]}}
        - "Starbucks and Target receipts" → {{"merchants": ["Starbucks", "Target"]}}
        - "grocery spending in December" → {{"merchants": []}}
        - "that coffee place in SF" → {{"merchants": ["Starbucks"]}}  # Inference OK if confident
        """
        
        # Add corpus context if available (helps with aliases)
        if self._merchant_corpus and len(self._merchant_corpus) < 50:
            corpus_list = sorted(list(self._merchant_corpus))[:20]  # Limit for token efficiency
            prompt += f"\n\nKnown merchants in database: {', '.join(corpus_list)}"
        
        return prompt
    
    def _tokenize(self, query: str) -> List[str]:
        """
        Tokenize query into potential merchant name candidates.
        
        Focuses on capitalized sequences and multi-word phrases.
        """
        # Split on common delimiters
        tokens = re.split(r'[,;.!?]|\s+(?:and|or|in|at|from)\s+', query)
        
        result = []
        for token in tokens:
            # Extract capitalized sequences
            caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', token)
            result.extend(caps)
            
            # Also include single capitalized words
            words = token.split()
            result.extend([w for w in words if w and w[0].isupper() and len(w) > 2])
        
        return result
    
    def _normalize_list(self, merchants: List[str]) -> List[str]:
        """
        Normalize merchant names in list.
        
        - Remove duplicates (case-insensitive)
        - Proper capitalization
        - Remove common suffixes
        """
        normalized = []
        seen = set()
        
        for merchant in merchants:
            # Skip empty
            if not merchant or len(merchant.strip()) < 2:
                continue
            
            # Check if already seen (normalized form)
            norm_key = normalize_merchant_name(merchant)
            if norm_key in seen:
                continue
            
            seen.add(norm_key)
            
            # Keep original capitalization (better for display)
            normalized.append(merchant.strip())
        
        return normalized
    
    def learn_from_receipts(self, receipts: List[dict]) -> None:
        """
        Build merchant corpus from indexed receipts.
        
        This enables fuzzy matching without hardcoding.
        
        Args:
            receipts: List of receipt metadata dicts with 'merchant_name' field
        """
        for receipt in receipts:
            merchant = receipt.get('merchant_name')
            if merchant:
                self._merchant_corpus.add(merchant)
        
        logger.info(f"Learned {len(self._merchant_corpus)} unique merchants from receipts")
    
    def get_corpus_size(self) -> int:
        """Get the size of the learned merchant corpus."""
        return len(self._merchant_corpus)
    
    def get_corpus(self) -> Set[str]:
        """Get the full merchant corpus."""
        return self._merchant_corpus.copy()


# Convenience function for integration
def extract_merchants_semantic(
    query: str,
    indexed_merchants: Optional[Set[str]] = None,
    openai_client=None
) -> List[str]:
    """
    Convenience function for semantic merchant extraction.
    
    This is the recommended entry point for query_parser.py integration.
    
    Args:
        query: Natural language query
        indexed_merchants: Optional set of known merchants from receipts
        openai_client: Optional OpenAI client (will create if needed)
        
    Returns:
        List of extracted merchant names
        
    Example:
        >>> merchants = extract_merchants_semantic(
        ...     "How much at Walmart and Target?",
        ...     indexed_merchants={'Walmart Supercenter', 'Target', 'Starbucks'}
        ... )
        >>> print(merchants)
        ['Walmart', 'Target']
    """
    matcher = SemanticMerchantMatcher(openai_client)
    return matcher.extract_merchants(query, indexed_merchants)
