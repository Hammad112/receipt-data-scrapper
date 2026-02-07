import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from decimal import Decimal

# Absolute imports from src
from src.parsers.receipt_parser import ReceiptParser
from src.models import ItemCategory, PaymentMethod

@pytest.fixture
def parser():
    """Initializes the parser without a real OpenAI client for unit testing."""
    return ReceiptParser(openai_client=None)

def test_extract_merchant_name(parser):
    lines = ["WALMART #1234", "123 Main St", "City, ST 12345"]
    assert "WALMART" in parser._extract_merchant_name(lines)

def test_extract_date(parser):
    lines = ["Date: 12/25/2023", "Time: 14:30"]
    dt = parser._extract_date(lines)
    assert dt.year == 2023
    assert dt.month == 12
    assert dt.day == 25

def test_parse_item_line_with_heuristic_cat(parser):
    """Verifies that common items are categorized via heuristics."""
    line = "Milk $4.50"
    item = parser._parse_item_line(line)
    assert item is not None
    assert item.name == "Milk"
    assert item.total_price == Decimal("4.50")
    assert item.category == ItemCategory.GROCERIES

@patch('src.parsers.receipt_parser.ReceiptParser._categorize_via_llm')
def test_categorize_item_llm_primary(mock_llm, parser):
    """Verifies that LLM categorization is attempted first as the primary strategy."""
    parser.openai_client = MagicMock()
    mock_llm.return_value = ItemCategory.ELECTRONICS
    
    # "Xenon Widget" is not in heuristics, but LLM is primary anyway
    result = parser._categorize_item("Xenon Widget")
    
    assert result == ItemCategory.ELECTRONICS
    mock_llm.assert_called_once_with("Xenon Widget")

def test_categorize_item_heuristic_fallback(parser):
    """Verifies that when LLM is unavailable, heuristics still work."""
    parser.openai_client = None # Force fallback
    assert parser._categorize_item("Starbucks Coffee") == ItemCategory.COFFEE_SHOP
