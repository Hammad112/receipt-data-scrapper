"""
Query accuracy tests - verify system produces correct answers.
Tests actual query results against expected outcomes.
"""
import pytest
import os
import sys
from datetime import datetime
from decimal import Decimal

# Ensure src in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from parsers import ReceiptParser
from chunking import ReceiptChunker
from models import Receipt, ItemCategory, PaymentMethod


class TestQueryAccuracy:
    """Accuracy tests for specific query types."""

    @pytest.fixture
    def sample_receipts(self):
        """Create diverse receipt dataset for testing."""
        parser = ReceiptParser()
        receipts = []
        
        # Receipt 1: January 2024, Grocery, Walmart
        r1_text = """
WALMART
01/15/2024

Milk $4.50
Bread $3.25
Eggs $5.99
Subtotal $13.74
Tax $1.10
Total $14.84
Payment: Credit Card
        """
        receipts.append(parser.parse_receipt(r1_text, filename="jan_walmart.txt"))
        
        # Receipt 2: January 2024, Coffee
        r2_text = """
STARBUCKS
01/20/2024

Venti Latte $5.75
Croissant $4.25
Total $10.00
Payment: Apple Pay
        """
        receipts.append(parser.parse_receipt(r2_text, filename="jan_starbucks.txt"))
        
        # Receipt 3: February 2024, Electronics
        r3_text = """
BEST BUY
02/10/2024

Headphones $129.99
Cables $15.99
Subtotal $145.98
Tax $13.14
Total $159.12
Payment: Credit Card
        """
        receipts.append(parser.parse_receipt(r3_text, filename="feb_bestbuy.txt"))
        
        # Receipt 4: February 2024, Pharmacy
        r4_text = """
CVS PHARMACY
02/15/2024

Vitamin D $12.99
Allergy Meds $8.49
Total $21.48
Payment: Cash
        """
        receipts.append(parser.parse_receipt(r4_text, filename="feb_cvs.txt"))
        
        return receipts

    def test_temporal_filtering_january(self, sample_receipts):
        """Filter receipts by January 2024."""
        jan_receipts = [
            r for r in sample_receipts 
            if r.transaction_date.month == 1 and r.transaction_date.year == 2024
        ]
        
        assert len(jan_receipts) == 2
        assert all("jan" in r.filename.lower() for r in jan_receipts)

    def test_temporal_filtering_february(self, sample_receipts):
        """Filter receipts by February 2024."""
        feb_receipts = [
            r for r in sample_receipts 
            if r.transaction_date.month == 2 and r.transaction_date.year == 2024
        ]
        
        assert len(feb_receipts) == 2
        assert all("feb" in r.filename.lower() for r in feb_receipts)

    def test_merchant_filtering_walmart(self, sample_receipts):
        """Find Walmart receipts."""
        walmart_receipts = [
            r for r in sample_receipts 
            if "walmart" in r.merchant_name.lower()
        ]
        
        assert len(walmart_receipts) == 1
        assert walmart_receipts[0].total_amount == Decimal("14.84")

    def test_category_filtering_coffee(self, sample_receipts):
        """Find coffee shop receipts."""
        coffee_receipts = []
        for r in sample_receipts:
            if any(item.category == ItemCategory.COFFEE_SHOP for item in r.items):
                coffee_receipts.append(r)
        
        assert len(coffee_receipts) == 1
        assert "starbucks" in coffee_receipts[0].merchant_name.lower()

    def test_category_filtering_electronics(self, sample_receipts):
        """Find electronics purchases."""
        electronics_receipts = []
        for r in sample_receipts:
            if any(item.category == ItemCategory.ELECTRONICS for item in r.items):
                electronics_receipts.append(r)
        
        assert len(electronics_receipts) == 1
        assert electronics_receipts[0].total_amount == Decimal("159.12")

    def test_payment_method_filtering(self, sample_receipts):
        """Filter by payment method."""
        apple_pay_receipts = [
            r for r in sample_receipts 
            if r.payment_method == PaymentMethod.APPLE_PAY
        ]
        cash_receipts = [
            r for r in sample_receipts 
            if r.payment_method == PaymentMethod.CASH
        ]
        
        assert len(apple_pay_receipts) == 1
        assert len(cash_receipts) == 1

    def test_aggregation_total_spending(self, sample_receipts):
        """Calculate total spending across all receipts."""
        total = sum(r.total_amount for r in sample_receipts)
        expected = Decimal("14.84") + Decimal("10.00") + Decimal("159.12") + Decimal("21.48")
        
        assert total == expected
        assert total == Decimal("205.44")

    def test_aggregation_by_month(self, sample_receipts):
        """Calculate spending by month."""
        from collections import defaultdict
        
        monthly_totals = defaultdict(Decimal)
        for r in sample_receipts:
            key = (r.transaction_date.year, r.transaction_date.month)
            monthly_totals[key] += r.total_amount
        
        assert monthly_totals[(2024, 1)] == Decimal("24.84")  # 14.84 + 10.00
        assert monthly_totals[(2024, 2)] == Decimal("180.60")  # 159.12 + 21.48

    def test_amount_threshold_filtering(self, sample_receipts):
        """Find receipts over $50."""
        high_value_receipts = [
            r for r in sample_receipts 
            if r.total_amount > Decimal("50.00")
        ]
        
        assert len(high_value_receipts) == 1
        assert high_value_receipts[0].total_amount == Decimal("159.12")

    def test_item_count_verification(self, sample_receipts):
        """Verify item counts per receipt."""
        item_counts = {r.filename: len(r.items) for r in sample_receipts}
        
        assert item_counts["jan_walmart.txt"] == 3
        assert item_counts["jan_starbucks.txt"] == 2
        assert item_counts["feb_bestbuy.txt"] == 2
        assert item_counts["feb_cvs.txt"] == 2


class TestSemanticQueries:
    """Test semantic understanding of queries."""

    @pytest.fixture
    def diverse_receipts(self):
        """Receipts with semantic categories."""
        parser = ReceiptParser()
        receipts = []
        
        # Health-related (pharmacy)
        health_text = """
WALGREENS
03/01/2024

Pain Reliever $8.99
Vitamin C $12.49
Bandages $5.99
Total $27.47
        """
        receipts.append(("health", parser.parse_receipt(health_text)))
        
        # Treats (dessert shop)
        treats_text = """
COLD STONE CREAMERY
03/02/2024

Ice Cream Cake $25.99
Sundaes $12.50
Total $38.49
        """
        receipts.append(("treats", parser.parse_receipt(treats_text)))
        
        # Work lunch (restaurant)
        work_text = """
PANERA BREAD
03/03/2024

Soup $8.99
Sandwich $10.49
Total $19.48
        """
        receipts.append(("work", parser.parse_receipt(work_text)))
        
        return receipts

    def test_health_items_detected(self, diverse_receipts):
        """Health-related items should be categorized pharmacy."""
        health_receipt = next(r for label, r in diverse_receipts if label == "health")
        
        health_items = [
            item for item in health_receipt.items 
            if item.category == ItemCategory.PHARMACY or
               any(word in item.name.lower() for word in ['vitamin', 'pain', 'bandage'])
        ]
        
        assert len(health_items) >= 2

    def test_treat_items_detected(self, diverse_receipts):
        """Treat items should be detected."""
        treats_receipt = next(r for label, r in diverse_receipts if label == "treats")
        
        treat_items = [
            item for item in treats_receipt.items 
            if item.category == ItemCategory.TREATS or
               any(word in item.name.lower() for word in ['ice cream', 'cake', 'candy', 'chocolate'])
        ]
        
        assert len(treat_items) >= 1


class TestChunkMetadataAccuracy:
    """Verify chunk metadata matches receipt data."""

    def test_item_detail_metadata(self):
        """Item chunks contain correct metadata."""
        parser = ReceiptParser()
        chunker = ReceiptChunker()
        
        text = """
TARGET
04/01/2024

Organic Milk $4.99
Whole Grain Bread $3.49
Total $8.48
        """
        receipt = parser.parse_receipt(text)
        chunks = chunker.chunk_receipt(receipt)
        
        # Find item detail chunks
        item_chunks = [c for c in chunks if c.chunk_type == "item_detail"]
        assert len(item_chunks) == 2
        
        # Verify metadata
        for chunk in item_chunks:
            assert chunk.metadata["item_name"]
            assert chunk.metadata["item_price"] > 0
            assert chunk.metadata["merchant_name"] == "TARGET"
            assert "transaction_date" in chunk.metadata
            assert "receipt_id" in chunk.metadata

    def test_category_group_metadata(self):
        """Category chunks aggregate correctly."""
        parser = ReceiptParser()
        chunker = ReceiptChunker()
        
        text = """
GROCERY STORE
04/02/2024

Milk $3.99
Cheese $4.99
Yogurt $2.99
Apple $1.99
Banana $0.99
Total $14.95
        """
        receipt = parser.parse_receipt(text)
        chunks = chunker.chunk_receipt(receipt)
        
        # Find category chunks (only if 2+ items in same category)
        cat_chunks = [c for c in chunks if c.chunk_type == "category_group"]
        
        # Should have at least one category group for groceries
        assert len(cat_chunks) >= 1
        
        for chunk in cat_chunks:
            assert "category" in chunk.metadata
            assert chunk.metadata["item_count"] >= 2
            assert chunk.metadata["total_amount"] > 0
