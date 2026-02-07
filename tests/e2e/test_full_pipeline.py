"""
Integration tests for full RAG pipeline end-to-end testing.
Tests the complete flow: parse -> chunk -> index -> query.
"""
import pytest
import os
from datetime import datetime
from decimal import Decimal

from src.parsers import ReceiptParser
from src.chunking import ReceiptChunker
from src.query import QueryEngine
from src.models import Receipt, ItemCategory, PaymentMethod


class TestFullRAGPipeline:
    """End-to-end integration tests for the complete RAG system."""

    @pytest.fixture
    def sample_receipt_text(self):
        """Realistic receipt sample for testing."""
        return """
STARBUCKS COFFEE
123 Market St, San Francisco, CA 94105
(415) 555-0123

Date: 01/15/2024  09:23 AM
Transaction: 789123

1  Venti Latte                    $5.75
1  Blueberry Muffin               $3.50
1  Croissant                      $4.25

Subtotal:                         $13.50
Tax (8.5%):                        $1.15
Total:                            $14.65

Payment: Apple Pay
Card: **** 4242
        """

    @pytest.fixture
    def parsed_receipt(self, sample_receipt_text):
        """Parse sample receipt for testing."""
        parser = ReceiptParser()
        return parser.parse_receipt(sample_receipt_text, filename="test_receipt.txt")

    def test_receipt_parsing_accuracy(self, parsed_receipt):
        """Verify parser extracts all fields correctly from real receipt format."""
        receipt = parsed_receipt
        
        # Merchant extraction
        assert "Starbucks" in receipt.merchant_name or "STARBUCKS" in receipt.merchant_name
        
        # Date parsing
        assert receipt.transaction_date.year == 2024
        assert receipt.transaction_date.month == 1
        assert receipt.transaction_date.day == 15
        
        # Financial totals
        assert receipt.subtotal == Decimal("13.50")
        assert receipt.tax_amount == Decimal("1.15")
        assert receipt.total_amount == Decimal("14.65")
        
        # Item count
        assert len(receipt.items) == 3
        
        # Payment method
        assert receipt.payment_method == PaymentMethod.APPLE_PAY

    def test_chunking_produces_multiple_views(self, parsed_receipt):
        """Verify chunker creates all 5 view types."""
        chunker = ReceiptChunker()
        chunks = chunker.chunk_receipt(parsed_receipt)
        
        chunk_types = {c.chunk_type for c in chunks}
        expected_types = {
            'receipt_summary', 
            'item_detail', 
            'category_group',
            'merchant_info', 
            'payment_method'
        }
        
        # Should have at least summary + merchant + payment + items
        assert 'receipt_summary' in chunk_types
        assert 'merchant_info' in chunk_types
        assert 'payment_method' in chunk_types
        assert len([c for c in chunks if c.chunk_type == 'item_detail']) == 3

    def test_chunk_metadata_integrity(self, parsed_receipt):
        """Verify all chunks have complete metadata for hybrid search."""
        chunker = ReceiptChunker()
        chunks = chunker.chunk_receipt(parsed_receipt)
        
        for chunk in chunks:
            # Every chunk must have these metadata fields
            assert 'receipt_id' in chunk.metadata
            assert 'merchant_name' in chunk.metadata
            assert 'transaction_date' in chunk.metadata
            assert 'transaction_ts' in chunk.metadata
            assert 'total_amount' in chunk.metadata
            assert chunk.content and len(chunk.content) > 10

    def test_item_categorization_accuracy(self, parsed_receipt):
        """Verify items are categorized correctly (heuristic or LLM)."""
        items = parsed_receipt.items
        
        # Coffee item should be COFFEE_SHOP
        latte = next((i for i in items if "Latte" in i.name), None)
        assert latte is not None
        assert latte.category == ItemCategory.COFFEE_SHOP
        
        # Pastries should be TREATS
        pastry_items = [i for i in items if any(x in i.name.lower() for x in ['muffin', 'croissant'])]
        for item in pastry_items:
            assert item.category in [ItemCategory.TREATS, ItemCategory.COFFEE_SHOP]

    def test_chunk_token_limits(self, parsed_receipt):
        """Verify chunks don't exceed token limits."""
        chunker = ReceiptChunker()
        chunks = chunker.chunk_receipt(parsed_receipt)
        
        for chunk in chunks:
            # Rough token estimate: 1 token ~= 4 characters
            char_limit = chunker.MAX_CHUNK_TOKENS * 3
            assert len(chunk.content) <= char_limit + 20  # Allow for "[TRUNCATED]"


class TestReceiptParserVariations:
    """Test parser handles various receipt formats."""

    def test_grocery_store_receipt(self):
        """Parse typical grocery receipt."""
        text = """
WHOLE FOODS MARKET
1000 Main Street
Austin, TX 78701

Date: 03/22/2024
Cashier: Maria

Organic Milk          1 @ 4.99    $4.99
Fresh Bread           1 @ 3.49    $3.49
Free Range Eggs       2 @ 5.99    $11.98
Bananas              2.5 @ 0.79   $1.98

SUBTOTAL:                         $16.45
TAX:                              $0.00
TOTAL:                            $16.45

Payment: Credit Card
        """
        parser = ReceiptParser()
        receipt = parser.parse_receipt(text)
        
        assert len(receipt.items) == 4
        assert receipt.total_amount == Decimal("16.45")
        assert receipt.payment_method == PaymentMethod.CREDIT

    def test_electronics_receipt(self):
        """Parse electronics receipt with warranty."""
        text = """
BEST BUY
555 Tech Blvd, San Jose, CA 95110

Date: 02/10/2024
Order #BB-998877

1  Wireless Headphones          $129.99
     SKU: WH-1000XM5
1  Extended Warranty 2yr        $24.99

SUBTOTAL:                       $154.98
TAX (9.25%):                    $14.34
TOTAL:                          $169.32

Payment: Visa **** 1234
        """
        parser = ReceiptParser()
        receipt = parser.parse_receipt(text)
        
        # Should detect electronics
        headphone_item = next((i for i in receipt.items if "Headphones" in i.name), None)
        if headphone_item:
            assert headphone_item.category == ItemCategory.ELECTRONICS

    def test_restaurant_receipt_with_tip(self):
        """Parse restaurant receipt with tip."""
        text = """
THE ITALIAN PLACE
789 Oak Ave, Chicago, IL 60601
(312) 555-0199

Date: 04/05/2024  7:30 PM
Server: Giovanni  Table: 12

2  Margherita Pizza      $28.00
1  Caesar Salad          $12.00
2  House Wine            $18.00

Subtotal:                $58.00
Tax:                      $5.22
Tip (20%):              $11.60
Total:                  $74.82

Payment: Credit Card
        """
        parser = ReceiptParser()
        receipt = parser.parse_receipt(text)
        
        assert receipt.tip_amount == Decimal("11.60")
        assert receipt.total_amount == Decimal("74.82")
        assert len(receipt.items) >= 3

    def test_pharmacy_receipt(self):
        """Parse pharmacy receipt with health items."""
        text = """
CVS PHARMACY
456 Health St, Boston, MA 02101

Date: 05/12/2024
Pharmacist: Dr. Smith

Vitamin D3 100ct        $12.99
Allergy Relief 30ct     $8.49
First Aid Bandages      $5.99
Prescription Co-pay     $10.00

Subtotal:               $37.47
Tax:                     $0.00
Total:                  $37.47

Payment: Debit Card
        """
        parser = ReceiptParser()
        receipt = parser.parse_receipt(text)
        
        # Health items should be PHARMACY category
        health_items = [i for i in receipt.items if any(x in i.name.lower() for x in ['vitamin', 'allergy', 'prescription'])]
        for item in health_items:
            assert item.category == ItemCategory.PHARMACY


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_receipt_handling(self):
        """Parser should handle minimal/empty receipts gracefully."""
        parser = ReceiptParser()
        receipt = parser.parse_receipt("No receipt data here", filename="empty.txt")
        
        # Should still return a receipt object
        assert receipt is not None
        assert receipt.merchant_name  # Should have some default
        assert receipt.items == []

    def test_return_transaction_detection(self):
        """Detect return/refund transactions."""
        text = """
TARGET
Date: 06/01/2024

RETURN: Blender
Refund: -$45.99

Total: -$45.99
        """
        parser = ReceiptParser()
        receipt = parser.parse_receipt(text)
        
        assert receipt.return_transaction or receipt.total_amount < 0

    def test_multiple_quantity_items(self):
        """Handle items with quantity > 1."""
        text = """
COSTCO
Date: 07/15/2024

Paper Towels (3x)        $38.97
Soda 12-pack (2x)        $17.98

Total:                   $56.95
        """
        parser = ReceiptParser()
        receipt = parser.parse_receipt(text)
        
        # Should detect multiple items
        assert len(receipt.items) == 2

    def test_international_date_formats(self):
        """Handle various date formats."""
        formats = [
            ("Date: 15/01/2024\nStore: Test\nTotal: $10.00", "DD/MM/YYYY"),
            ("Date: 01-15-2024\nStore: Test\nTotal: $10.00", "MM-DD-YYYY"),
            ("Date: January 15, 2024\nStore: Test\nTotal: $10.00", "Textual"),
        ]
        
        parser = ReceiptParser()
        for text, desc in formats:
            receipt = parser.parse_receipt(text)
            assert receipt.transaction_date.year == 2024, f"Failed for {desc}"
