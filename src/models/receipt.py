"""
Data models for receipt processing and querying.

This module defines the core data structures used throughout the system:
- Receipt: Complete receipt information
- ReceiptItem: Individual item details
- ReceiptChunk: Chunk for vector embedding
- QueryResult: Query response structure
- PaymentMethod: Payment method enumeration
- ItemCategory: Item category enumeration
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field, field_validator

try:
    from ..utils.logging_config import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class PaymentMethod(str, Enum):
    """Payment method enumeration."""
    CASH = "cash"
    CREDIT = "credit"
    DEBIT = "debit"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    OTHER = "other"


class ItemCategory(str, Enum):
    """Item category enumeration."""
    GROCERIES = "groceries"
    RESTAURANT = "restaurant"
    COFFEE_SHOP = "coffee_shop"
    ELECTRONICS = "electronics"
    PHARMACY = "pharmacy"
    TREATS = "treats"
    OTHER = "other"


class ReceiptItem(BaseModel):
    """Individual receipt item."""
    name: str
    quantity: Decimal = Field(default=Decimal('1'))
    unit_price: Decimal
    total_price: Decimal
    category: Optional[ItemCategory] = None
    discount: Optional[Decimal] = None
    warranty_info: Optional[str] = None
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:
            raise ValueError('Item name must be at least 2 characters')
        return v.strip()


class Receipt(BaseModel):
    """Complete receipt information."""
    receipt_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: Optional[str] = None
    merchant_name: str
    transaction_date: datetime
    payment_method: PaymentMethod
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
    
    # Additional metadata fields
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
    def validate_total_amount(cls, v, info):
        data = info.data
        if data:
            subtotal = data.get('subtotal', Decimal('0'))
            tax_amount = data.get('tax_amount', Decimal('0'))
            tip_amount = data.get('tip_amount', Decimal('0'))
            delivery_fee = data.get('delivery_fee', Decimal('0'))
            discounts = data.get('discounts', Decimal('0'))
            
            calculated_total = subtotal + tax_amount
            if tip_amount:
                calculated_total += tip_amount
            if delivery_fee:
                calculated_total += delivery_fee
            if discounts:
                calculated_total -= discounts
            
            # Allow more tolerance for rounding errors and additional fees
            if abs(v - calculated_total) > Decimal('1.00'):
                logger.warning(f"Total validation mismatch: found {v}, calculated {calculated_total}")
        
        return v
    
    @property
    def item_count(self) -> int:
        """Get the number of items in the receipt."""
        return len(self.items)
    
    @property
    def categories(self) -> List[str]:
        """Get list of categories in the receipt."""
        return list(set(item.category.value for item in self.items if item.category))


class ReceiptChunk(BaseModel):
    """Chunk of receipt data for vector embedding."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    receipt_id: str
    chunk_type: str  # receipt_summary, item_detail, category_group, etc.
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError('Chunk content must be at least 10 characters')
        return v.strip()


class QueryResult(BaseModel):
    """Result of a query operation."""
    answer: str
    receipts: List[Dict[str, Any]] = Field(default_factory=list)
    items: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    query_type: str
    processing_time: float = Field(ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def receipt_count(self) -> int:
        """Get the number of receipts in the result."""
        return len(self.receipts)
    
    @property
    def item_count(self) -> int:
        """Get the number of items in the result."""
        return len(self.items)
