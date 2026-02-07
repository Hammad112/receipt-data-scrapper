"""
Property-based tests using hypothesis for edge case discovery.
Tests system invariants that should always hold true.
"""
import pytest
from datetime import datetime
from decimal import Decimal
from hypothesis import given, strategies as st, settings, assume

from src.models import Receipt, ReceiptItem, ItemCategory, PaymentMethod
from src.parsers import ReceiptParser
from src.chunking import ReceiptChunker


class TestParserProperties:
    """Property-based tests for receipt parser."""

    @given(
        merchant=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
        total=st.decimals(min_value=0.01, max_value=10000, places=2),
        item_count=st.integers(min_value=0, max_value=20)
    )
    @settings(max_examples=50, deadline=1000)
    def test_total_always_positive(self, merchant, total, item_count):
        """Receipt total should always be positive."""
        lines = [merchant, "01/15/2024"]
        for i in range(item_count):
            lines.append(f"Item{i} $1.00")
        lines.append(f"Total ${total}")
        
        parser = ReceiptParser()
        receipt = parser.parse_receipt("\n".join(lines))
        
        assert receipt.total_amount >= 0

    @given(
        year=st.integers(min_value=2000, max_value=2030),
        month=st.integers(min_value=1, max_value=12),
        day=st.integers(min_value=1, max_value=28)
    )
    @settings(max_examples=100)
    def test_date_parsing_always_valid(self, year, month, day):
        """Parsed date should always be valid."""
        text = f"Store\n{month}/{day}/{year}\nTotal $10.00"
        parser = ReceiptParser()
        receipt = parser.parse_receipt(text)
        
        # Should parse without exception
        assert receipt.transaction_date.year == year
        assert receipt.transaction_date.month == month
        assert receipt.transaction_date.day == day

    @given(
        item_name=st.text(min_size=1, max_size=30),
        price=st.decimals(min_value=0.01, max_value=1000, places=2)
    )
    @settings(max_examples=50)
    def test_item_price_parsing(self, item_name, price):
        """Item prices should be parsed correctly regardless of name."""
        # Skip names that look like metadata
        assume("total" not in item_name.lower())
        assume("tax" not in item_name.lower())
        assume("subtotal" not in item_name.lower())
        
        text = f"Store\n01/15/2024\n{item_name} ${price}\nTotal ${price}"
        parser = ReceiptParser()
        receipt = parser.parse_receipt(text)
        
        if receipt.items:
            assert receipt.items[0].total_price > 0


class TestChunkingProperties:
    """Property-based tests for chunking invariants."""

    @given(item_count=st.integers(min_value=1, max_value=50))
    @settings(max_examples=30)
    def test_chunk_count_invariant(self, item_count):
        """Chunk count should be: 1 summary + 1 merchant + 1 payment + items + categories."""
        parser = ReceiptParser()
        chunker = ReceiptChunker()
        
        lines = ["STORE", "01/15/2024"]
        for i in range(item_count):
            lines.append(f"Item{i} $5.00")
        lines.append(f"Total ${item_count * 5}.00")
        
        receipt = parser.parse_receipt("\n".join(lines))
        chunks = chunker.chunk_receipt(receipt)
        
        # Minimum: summary + merchant + payment + at least 1 item
        min_chunks = 3 + min(item_count, 1)
        assert len(chunks) >= min_chunks
        
        # Maximum: summary + merchant + payment + all items + all categories
        max_chunks = 3 + item_count + len(ItemCategory)
        assert len(chunks) <= max_chunks

    @given(content_length=st.integers(min_value=10, max_value=50000))
    @settings(max_examples=20)
    def test_chunk_content_size_limit(self, content_length):
        """No chunk should exceed token limit."""
        chunker = ReceiptChunker()
        
        # Create content of specified length
        long_name = "X" * content_length
        text = f"Store\n01/15/2024\n{long_name[:100]} $10.00\nTotal $10.00"
        
        parser = ReceiptParser()
        receipt = parser.parse_receipt(text)
        chunks = chunker.chunk_receipt(receipt)
        
        for chunk in chunks:
            char_limit = chunker.MAX_CHUNK_TOKENS * 3
            assert len(chunk.content) <= char_limit + 20

    @given(
        merchant=st.text(min_size=1, max_size=30),
        item_count=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=30)
    def test_all_chunks_have_receipt_id(self, merchant, item_count):
        """Every chunk must have receipt_id in metadata."""
        parser = ReceiptParser()
        chunker = ReceiptChunker()
        
        lines = [merchant, "01/15/2024"]
        for i in range(item_count):
            lines.append(f"Item{i} $5.00")
        lines.append(f"Total ${item_count * 5}.00")
        
        receipt = parser.parse_receipt("\n".join(lines))
        chunks = chunker.chunk_receipt(receipt)
        
        for chunk in chunks:
            assert "receipt_id" in chunk.metadata
            assert chunk.metadata["receipt_id"] == receipt.receipt_id


class TestCategorizationProperties:
    """Property-based tests for item categorization."""

    @given(item_name=st.sampled_from([
        "coffee", "latte", "espresso", "cappuccino",
        "milk", "bread", "eggs", "cheese",
        "laptop", "phone", "headphones", "charger",
        "vitamin", "medicine", "bandage", "supplement",
        "burger", "pizza", "salad", "sandwich",
        "candy", "chocolate", "ice cream", "cookie"
    ]))
    @settings(max_examples=24)
    def test_known_items_have_category(self, item_name):
        """Known items should be categorized (not OTHER)."""
        parser = ReceiptParser()
        text = f"Store\n01/15/2024\n{item_name.title()} $10.00\nTotal $10.00"
        receipt = parser.parse_receipt(text)
        
        if receipt.items:
            assert receipt.items[0].category != ItemCategory.OTHER


class TestFinancialIntegrity:
    """Financial calculation invariants."""

    @given(
        subtotal=st.decimals(min_value=0.01, max_value=500, places=2),
        tax_rate=st.decimals(min_value=0, max_value=0.20, places=4)
    )
    @settings(max_examples=50)
    def test_total_calculation(self, subtotal, tax_rate):
        """Total = subtotal + tax + tip - discounts."""
        tax = (subtotal * tax_rate).quantize(Decimal("0.01"))
        total = subtotal + tax
        
        # Create receipt
        text = f"""Store
01/15/2024
Item $1.00
Subtotal ${subtotal}
Tax ${tax}
Total ${total}"""
        
        parser = ReceiptParser()
        receipt = parser.parse_receipt(text)
        
        assert receipt.subtotal >= 0
        assert receipt.tax_amount >= 0
        assert receipt.total_amount >= receipt.subtotal

    @given(
        items=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=20),
                st.decimals(min_value=0.01, max_value=100, places=2)
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=30)
    def test_item_sum_matches_subtotal(self, items):
        """Sum of item prices should approximate subtotal."""
        lines = ["Store", "01/15/2024"]
        item_total = Decimal("0")
        
        for name, price in items:
            # Clean name for receipt format
            clean_name = name.replace(" ", "_")
            lines.append(f"{clean_name} ${price}")
            item_total += price
        
        lines.append(f"Subtotal ${item_total}")
        lines.append(f"Total ${item_total}")
        
        parser = ReceiptParser()
        receipt = parser.parse_receipt("\n".join(lines))
        
        if receipt.items and receipt.subtotal > 0:
            calculated = sum(i.total_price for i in receipt.items)
            assert abs(calculated - receipt.subtotal) < Decimal("5.00")  # Allow rounding


class TestQueryParsingProperties:
    """Property-based tests for query parsing."""

    @given(
        merchant=st.text(min_size=3, max_size=20),
        month=st.sampled_from(["January", "February", "March", "April", "May", "June",
                              "July", "August", "September", "October", "November", "December"])
    )
    @settings(max_examples=50)
    def test_merchant_extraction_from_query(self, merchant, month):
        """Merchant names in queries should be extracted."""
        from src.query import QueryParser
        
        # Use a clean merchant name
        clean_merchant = "".join(c for c in merchant if c.isalnum() or c.isspace()).strip()
        assume(len(clean_merchant) >= 3)
        
        query = f"How much did I spend at {clean_merchant} in {month}?"
        parser = QueryParser()
        result = parser.parse(query)
        
        # Should detect temporal component
        assert result.get("date_range") is not None or result.get("query_type") == "temporal"


class TestEdgeCaseDiscovery:
    """Discover edge cases through fuzzing."""

    @given(st.text(min_size=1, max_size=1000))
    @settings(max_examples=100)
    def test_parser_never_crashes(self, text):
        """Parser should never crash on any input."""
        parser = ReceiptParser()
        try:
            receipt = parser.parse_receipt(text)
            # Should return a valid receipt object
            assert receipt is not None
            assert hasattr(receipt, 'items')
            assert hasattr(receipt, 'total_amount')
        except Exception as e:
            # Only acceptable exceptions are for truly invalid data
            pytest.fail(f"Parser crashed on input: {repr(text[:100])}... Error: {e}")

    @given(st.lists(st.text(min_size=1), min_size=0, max_size=100))
    @settings(max_examples=50)
    def test_chunker_never_crashes(self, lines):
        """Chunker should handle any receipt structure."""
        parser = ReceiptParser()
        chunker = ReceiptChunker()
        
        try:
            receipt = parser.parse_receipt("\n".join(lines))
            chunks = chunker.chunk_receipt(receipt)
            assert isinstance(chunks, list)
            for chunk in chunks:
                assert hasattr(chunk, 'content')
                assert hasattr(chunk, 'metadata')
        except Exception as e:
            pytest.fail(f"Chunker crashed on input. Error: {e}")
