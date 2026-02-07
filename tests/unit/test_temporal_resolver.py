import pytest
import os
from datetime import datetime
from src.query.advanced_date_resolver import TemporalQueryResolver

@pytest.fixture
def resolver():
    # Fix reference date for deterministic testing via environment variable
    # This must be set BEFORE the resolver is instantiated
    os.environ["RECEIPT_REFERENCE_DATE"] = "2024-01-15T00:00:00Z"
    return TemporalQueryResolver()

def test_absolute_date_resolution(resolver):
    # Test YYYY-MM-DD
    response = resolver.resolve_date_range("Receipts from 2023-12-25")
    result = response.get('date_range', {})
    assert result.get('start', '').startswith("2023-12-25")
    assert result.get('end', '').startswith("2023-12-25")

def test_relative_range_resolution(resolver):
    # Test "last week" relative to 2024-01-15
    response = resolver.resolve_date_range("What did I buy last week?")
    result = response.get('date_range', {})
    assert "2024-01-08" in result.get('start', '')
    assert "2024-01-14" in result.get('end', '')
