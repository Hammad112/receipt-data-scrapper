"""
Data models for receipt processing and querying.

This module defines the core data structures used throughout the system,
ensuring type safety and validation via Pydantic.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator

# Absolute imports for industrial stability
from src.utils.logging_config import logger


class PaymentMethod(str, Enum):
    """Supported payment methods for categorical filtering."""
    CASH = "cash"
    CREDIT = "credit"
    DEBIT = "debit"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    OTHER = "other"


class ItemCategory(str, Enum):
    """Industry-standard categories for receipt items."""
    GROCERIES = "groceries"
    RESTAURANT = "restaurant"
    FAST_FOOD = "fast_food"
    COFFEE_SHOP = "coffee_shop"
    ELECTRONICS = "electronics"
    PHARMACY = "pharmacy"
    TREATS = "treats"
    OTHER = "other"


class ReceiptItem(BaseModel):
    """
    Represents a single line item extracted from a receipt.
    Includes validation for names and prices.
    """
    name: str
    quantity: Decimal = Field(default=Decimal('1'))
    unit_price: Decimal
    total_price: Decimal
    category: Optional[ItemCategory] = None  # Deprecated in favor of categories
    categories: List[ItemCategory] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    discount: Optional[Decimal] = None
    warranty_info: Optional[str] = None
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Ensures item names meet minimum length requirements."""
        if not v or len(v.strip()) < 2:
            raise ValueError('Item name must be at least 2 characters')
        return v.strip()


class Receipt(BaseModel):
    """
    The primary data structure representing a fully parsed receipt.
    Orchestrates validation of financial totals.
    """
    receipt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: Optional[str] = None
    merchant_name: str
    transaction_date: datetime
    payment_method: PaymentMethod = PaymentMethod.OTHER
    card_network: Optional[str] = None
    card_last4: Optional[str] = None
    items: List[ReceiptItem] = Field(default_factory=list)
    subtotal: Decimal = Field(default=Decimal('0'))
    tax_amount: Decimal = Field(default=Decimal('0'))
    tip_amount: Optional[Decimal] = None
    delivery_fee: Optional[Decimal] = None
    total_amount: Decimal = Field(default=Decimal('0'))
    discounts: Optional[Decimal] = None
    raw_text: str
    
    # Metadata attributes
    merchant_address: Optional[str] = None
    merchant_city: Optional[str] = None
    merchant_state: Optional[str] = None
    merchant_zip: Optional[str] = None
    merchant_phone: Optional[str] = None
    merchant_website: Optional[str] = None
    cashier: Optional[str] = None
    table_number: Optional[str] = None
    order_number: Optional[str] = None
    transaction_id: Optional[str] = None
    store_number: Optional[str] = None
    loyalty_program: Optional[str] = None
    has_warranty: bool = False
    warranty_text: Optional[str] = None
    return_transaction: bool = False
    
    @field_validator('total_amount')
    @classmethod
    def validate_total_consistency(cls, v, info):
        """Cross-references the grand total with its component parts."""
        data = info.data
        if not data:
            return v
            
        subtotal = data.get('subtotal', Decimal('0'))
        tax = data.get('tax_amount', Decimal('0'))
        tip = data.get('tip_amount', Decimal('0')) or Decimal('0')
        fee = data.get('delivery_fee', Decimal('0')) or Decimal('0')
        disc = data.get('discounts', Decimal('0')) or Decimal('0')
        
        calculated = subtotal + tax + tip + fee - disc
        
        # Log a warning if the mismatch is significant (> $1.00)
        if abs(v - calculated) > Decimal('1.00'):
            logger.warning(
                f"Financial mismatch detected in receipt: {data.get('merchant_name')}. "
                f"Found {v}, Expected {calculated}"
            )
        return v
    
    @property
    def item_count(self) -> int:
        """Helper to count the number of line items."""
        return len(self.items)
    
    @property
    def categories(self) -> List[str]:
        """Returns a unique list of all categories present in the receipt items."""
        all_cats = set()
        for item in self.items:
            # Support both old and new fields during migration
            if item.category: 
                all_cats.add(item.category.value)
            for cat in item.categories:
                all_cats.add(cat.value)
        return list(all_cats)

    @property
    def is_return(self) -> bool:
        """Alias for return_transaction."""
        return self.return_transaction


class ReceiptChunk(BaseModel):
    """
    A specific slice of receipt data ready for vector embedding.
    Includes rigorous content validation.
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    receipt_id: str
    chunk_type: str  # receipt_summary, item_detail, etc.
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator('content')
    @classmethod
    def validate_chunk_density(cls, v):
        """Ensures chunk content is rich enough for meaningful embedding."""
        if not v or len(v.strip()) < 10:
            raise ValueError('Chunk content is too sparse for embedding')
        return v.strip()


class QueryResult(BaseModel):
    """
    The final response structure for the RAG system.
    Encapsulates the answer and its supporting evidence.
    """
    answer: str
    receipts: List[Dict[str, Any]] = Field(default_factory=list)
    items: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    query_type: str
    processing_time: float = Field(ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
