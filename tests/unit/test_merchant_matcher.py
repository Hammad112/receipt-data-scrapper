import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure project root in PATH is handled by conftest.py, but we use absolute imports
from src.query.semantic_merchant_matcher import SemanticMerchantMatcher

@pytest.fixture
def matcher():
    """Initializes matcher for unit testing."""
    return SemanticMerchantMatcher(openai_client=None)

def test_extract_via_prepositions(matcher):
    query = "Total at Walmart in SF"
    results = matcher._extract_via_prepositions(query)
    assert "Walmart" in results

def test_extract_via_fuzzy_match(matcher):
    # Setup learned corpus
    matcher._merchant_corpus = {"Starbucks", "Target", "Whole Foods Market"}
    
    # Test typo resolution
    assert "Starbucks" in matcher._extract_via_fuzzy_match("Starbux")
    # Test partial match
    assert "Whole Foods Market" in matcher._extract_via_fuzzy_match("Whole Foods")

@patch('src.query.semantic_merchant_matcher.SemanticMerchantMatcher._extract_via_llm')
def test_extract_merchants_orchestration(mock_llm, matcher):
    """Verifies strategy hierarchy: Prepositions -> Fuzzy -> LLM fallback."""
    # 1. Clear corpus and no prepositions in query
    matcher._merchant_corpus = set()
    mock_llm.return_value = ["CVS"]
    
    query = "Find my last pharmacy receipt"
    results = matcher.extract_merchants(query)
    
    assert results == ["CVS"]
    mock_llm.assert_called_once()

def test_normalize_list(matcher):
    raw = ["walmart", "Walmart ", "WALMART"]
    normalized = matcher._normalize_list(raw)
    assert len(normalized) == 1
    assert "walmart" == normalized[0].lower()
