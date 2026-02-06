"""
Receipt chunking strategy for optimal vector embedding and retrieval.

This module implements a hybrid chunking approach that balances:
- Granularity (receipt-level vs. item-level)
- Context preservation
- Query performance
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re

try:
    from ..models import Receipt, ReceiptItem, ReceiptChunk, ItemCategory
except ImportError:
    from models import Receipt, ReceiptItem, ReceiptChunk, ItemCategory


class ReceiptChunker:
    """
    Implements a multi-level chunking strategy for receipt data.
    
    The chunking strategy is designed to optimize for different query types:
    - Receipt-level queries (merchant, date, total)
    - Item-level queries (specific products, categories)
    - Aggregation queries (spending by category, time periods)
    - Semantic queries (health-related purchases, treats)
    
    Each chunk type contains:
    - Text content suitable for embedding
    - Rich metadata for filtering
    - Relationships to other chunks
    """
    
    def __init__(self):
        """Initialize the chunker."""
        self.chunk_types = {
            'receipt_summary': 'High-level receipt overview',
            'item_detail': 'Individual item information',
            'category_group': 'Items grouped by category',
            'temporal_group': 'Time-based groupings',
            'merchant_info': 'Merchant details',
            'payment_method': 'Payment method information',
        }

    def _normalize_merchant_name(self, name: str) -> str:
        name = (name or "").lower()
        name = re.sub(r"[^a-z0-9\s]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name
    
    def chunk_receipt(self, receipt: Receipt) -> List[ReceiptChunk]:
        """
        Chunk a single receipt into multiple searchable chunks.
        
        Args:
            receipt: The receipt to chunk
            
        Returns:
            List of ReceiptChunk objects
        """
        chunks = []
        
        # 1. Receipt Summary Chunk
        summary_chunk = self._create_summary_chunk(receipt)
        chunks.append(summary_chunk)
        
        # 2. Item Detail Chunks
        item_chunks = self._create_item_chunks(receipt)
        chunks.extend(item_chunks)
        
        # 3. Category Group Chunks
        category_chunks = self._create_category_chunks(receipt)
        chunks.extend(category_chunks)
        
        # 4. Merchant Info Chunk
        merchant_chunk = self._create_merchant_chunk(receipt)
        chunks.append(merchant_chunk)
        
        # 5. Payment Method Chunk
        payment_chunk = self._create_payment_chunk(receipt)
        chunks.append(payment_chunk)
        
        return chunks
    
    def _create_summary_chunk(self, receipt: Receipt) -> ReceiptChunk:
        """
        Create a summary chunk containing high-level receipt information.
        
        This chunk is optimized for queries about:
        - Total spending amounts
        - Receipt dates and times
        - Basic merchant information
        - Payment methods
        - Number of items
        """
        content_parts = [
            f"Receipt from {receipt.merchant_name}",
            f"Date: {receipt.transaction_date.strftime('%Y-%m-%d %H:%M')}",
            f"Subtotal: ${receipt.subtotal:.2f}",
            f"Tax: ${receipt.tax_amount:.2f}",
            f"Total amount: ${receipt.total_amount:.2f}",
            f"Payment method: {receipt.payment_method.value}",
            f"Number of items: {len(receipt.items)}",
        ]
        
        if receipt.discounts:
            content_parts.append(f"Discounts: ${receipt.discounts:.2f}")
        
        if receipt.tip_amount:
            content_parts.append(f"Tip: ${receipt.tip_amount:.2f}")
        
        if receipt.delivery_fee:
            content_parts.append(f"Delivery fee: ${receipt.delivery_fee:.2f}")

        if receipt.loyalty_program:
            content_parts.append(f"Loyalty program: {receipt.loyalty_program}")

        if receipt.has_warranty and receipt.warranty_text:
            content_parts.append(f"Warranty info: {receipt.warranty_text}")
        elif receipt.has_warranty:
            content_parts.append("Warranty info present")

        if receipt.return_transaction:
            content_parts.append("Return transaction")
        
        # Add item summary
        if receipt.items:
            item_names = [item.name for item in receipt.items[:5]]  # First 5 items
            if len(receipt.items) > 5:
                item_names.append(f"and {len(receipt.items) - 5} more items")
            content_parts.append(f"Items: {', '.join(item_names)}")
        
        content = ". ".join(content_parts)
        
        metadata = {
            'receipt_id': receipt.receipt_id,
            'merchant_name': receipt.merchant_name,
            'merchant_name_norm': self._normalize_merchant_name(receipt.merchant_name),
            'transaction_date': receipt.transaction_date.isoformat(),
            'transaction_ts': int(receipt.transaction_date.timestamp()),
            'transaction_year': receipt.transaction_date.year,
            'transaction_month': receipt.transaction_date.month,
            'transaction_day': receipt.transaction_date.day,
            'transaction_weekday': receipt.transaction_date.weekday(),
            'payment_method': receipt.payment_method.value,
            'total_amount': float(receipt.total_amount),
            'subtotal': float(receipt.subtotal),
            'tax_amount': float(receipt.tax_amount),
            'item_count': len(receipt.items),
            'has_discounts': receipt.discounts is not None,
            'has_tip': receipt.tip_amount is not None,
            'has_delivery_fee': receipt.delivery_fee is not None,
            'is_return': receipt.return_transaction,
            'has_warranty': receipt.has_warranty,
            'categories': list(set(item.category.value for item in receipt.items if item.category)),
            'filename': receipt.filename,
        }

        if receipt.card_network:
            metadata['card_network'] = receipt.card_network
        if receipt.card_last4:
            metadata['card_last4'] = receipt.card_last4
        if receipt.tip_amount is not None:
            metadata['tip_amount'] = float(receipt.tip_amount)
        if receipt.discounts is not None:
            metadata['discounts'] = float(receipt.discounts)
        if receipt.delivery_fee is not None:
            metadata['delivery_fee'] = float(receipt.delivery_fee)
        if receipt.merchant_address:
            metadata['merchant_address'] = receipt.merchant_address
        if receipt.merchant_city:
            metadata['merchant_city'] = receipt.merchant_city
        if receipt.merchant_state:
            metadata['merchant_state'] = receipt.merchant_state
        if receipt.merchant_zip:
            metadata['merchant_zip'] = receipt.merchant_zip
        if receipt.has_warranty and receipt.warranty_text:
            metadata['warranty_text'] = receipt.warranty_text
        
        return ReceiptChunk(
            chunk_id=str(uuid.uuid4()),
            receipt_id=receipt.receipt_id,
            chunk_type='receipt_summary',
            content=content,
            metadata=metadata,
            created_at=datetime.utcnow()
        )
    
    def _create_item_chunks(self, receipt: Receipt) -> List[ReceiptChunk]:
        """
        Create individual chunks for each line item.
        
        Strategy:
        - Granularity: One chunk per item ensures precise product searches.
        - Context Injection: Each item chunk is enriched with the purchase date, 
          merchant name, and payment method to allow for filtered retrieval 
          (e.g., "Bananas from Target").
        - Optimization: Ideal for querying specific products, price history, 
          and category-based spending.
        """
        chunks = []
        
        for i, item in enumerate(receipt.items):
            content_parts = [
                f"Item: {item.name}",
                f"Price: ${item.total_price:.2f}",
                f"Quantity: {item.quantity}",
                f"Unit price: ${item.unit_price:.2f}",
            ]
            
            if item.category:
                content_parts.append(f"Category: {item.category.value}")
            
            if item.discount:
                content_parts.append(f"Discount: ${item.discount:.2f}")
            
            if item.warranty_info:
                content_parts.append(f"Warranty: {item.warranty_info}")
            
            # Add essential receipt context for semantic search accuracy
            content_parts.extend([
                f"Purchased at: {receipt.merchant_name}",
                f"Date: {receipt.transaction_date.strftime('%Y-%m-%d')}",
                f"Payment method: {receipt.payment_method.value}",
            ])
            
            content = ". ".join(content_parts)
            
            # Rich metadata for Pinecone hybrid filtering
            metadata = {
                'receipt_id': receipt.receipt_id,
                'item_index': i,
                'item_name': item.name,
                'item_category': item.category.value if item.category else 'other',
                'item_price': float(item.total_price),
                'item_unit_price': float(item.unit_price),
                'item_quantity': float(item.quantity),
                'merchant_name': receipt.merchant_name,
                'merchant_name_norm': self._normalize_merchant_name(receipt.merchant_name),
                'transaction_date': receipt.transaction_date.isoformat(),
                'transaction_ts': int(receipt.transaction_date.timestamp()),
                'transaction_year': receipt.transaction_date.year,
                'transaction_month': receipt.transaction_date.month,
                'payment_method': receipt.payment_method.value,
                'has_discount': item.discount is not None,
                'has_warranty': item.warranty_info is not None,
                'filename': receipt.filename,
            }
            if receipt.card_network:
                metadata['card_network'] = receipt.card_network
            if receipt.card_last4:
                metadata['card_last4'] = receipt.card_last4
            
            chunk = ReceiptChunk(
                chunk_id=str(uuid.uuid4()),
                receipt_id=receipt.receipt_id,
                chunk_type='item_detail',
                content=content,
                metadata=metadata,
                created_at=datetime.utcnow()
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_category_chunks(self, receipt: Receipt) -> List[ReceiptChunk]:
        """
        Create chunks grouping items by category.
        
        These chunks are optimized for:
        - Category-based spending analysis
        - Aggregation queries
        - Semantic category searches
        """
        chunks = []
        
        # Group items by category
        category_groups = defaultdict(list)
        for item in receipt.items:
            category = item.category or ItemCategory.OTHER
            category_groups[category].append(item)
        
        for category, items in category_groups.items():
            if len(items) <= 1:
                continue  # Skip single-item categories, already covered in item chunks
            
            # Calculate category totals
            total_amount = sum(item.total_price for item in items)
            total_quantity = sum(item.quantity for item in items)
            
            content_parts = [
                f"Category: {category.value}",
                f"Store: {receipt.merchant_name}",
                f"Date: {receipt.transaction_date.strftime('%Y-%m-%d')}",
                f"Total items: {len(items)}",
                f"Total quantity: {total_quantity}",
                f"Total amount: ${total_amount:.2f}",
                f"Items: {', '.join(item.name for item in items)}",
            ]
            
            content = ". ".join(content_parts)
            
            metadata = {
                'receipt_id': receipt.receipt_id,
                'category': category.value,
                'item_count': len(items),
                'total_amount': float(total_amount),
                'total_quantity': float(total_quantity),
                'merchant_name': receipt.merchant_name,
                'merchant_name_norm': self._normalize_merchant_name(receipt.merchant_name),
                'transaction_date': receipt.transaction_date.isoformat(),
                'transaction_ts': int(receipt.transaction_date.timestamp()),
                'transaction_year': receipt.transaction_date.year,
                'transaction_month': receipt.transaction_date.month,
                'payment_method': receipt.payment_method.value,
                'item_names': [item.name for item in items],
                'average_item_price': float(total_amount / len(items)),
                'filename': receipt.filename,
            }
            if receipt.card_network:
                metadata['card_network'] = receipt.card_network
            if receipt.card_last4:
                metadata['card_last4'] = receipt.card_last4
            
            chunk = ReceiptChunk(
                chunk_id=str(uuid.uuid4()),
                receipt_id=receipt.receipt_id,
                chunk_type='category_group',
                content=content,
                metadata=metadata,
                created_at=datetime.utcnow()
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_merchant_chunk(self, receipt: Receipt) -> ReceiptChunk:
        """
        Create a chunk with merchant-specific information.
        
        This chunk is optimized for:
        - Merchant-based searches
        - Store location queries
        - Chain store identification
        - High-value item tracking
        """
        content_parts = [
            f"Store: {receipt.merchant_name}",
            f"Transaction date: {receipt.transaction_date.strftime('%Y-%m-%d %H:%M')}",
            f"Total spent: ${receipt.total_amount:.2f}",
            f"Items purchased: {len(receipt.items)}",
            f"Payment method: {receipt.payment_method.value}",
        ]
        
        if receipt.merchant_address:
            content_parts.append(f"Address: {receipt.merchant_address}")
        
        if receipt.merchant_phone:
            content_parts.append(f"Phone: {receipt.merchant_phone}")
        
        if receipt.cashier:
            content_parts.append(f"Cashier: {receipt.cashier}")
        
        if receipt.order_number:
            content_parts.append(f"Order number: {receipt.order_number}")
        
        # Add high-value items (over $20)
        high_value_items = [item for item in receipt.items if item.total_price > 20]
        if high_value_items:
            content_parts.append(f"High-value items: {', '.join(f'{item.name} (${item.total_price:.2f})' for item in high_value_items)}")
        
        content = ". ".join(content_parts)
        
        # Build metadata, filtering out None values (Pinecone doesn't accept nulls)
        metadata = {
            'receipt_id': receipt.receipt_id,
            'merchant_name': receipt.merchant_name,
            'merchant_name_norm': self._normalize_merchant_name(receipt.merchant_name),
            'transaction_date': receipt.transaction_date.isoformat(),
            'transaction_ts': int(receipt.transaction_date.timestamp()),
            'transaction_year': receipt.transaction_date.year,
            'transaction_month': receipt.transaction_date.month,
            'total_amount': float(receipt.total_amount),
            'item_count': len(receipt.items),
            'payment_method': receipt.payment_method.value,
            'has_cashier': receipt.cashier is not None,
            'has_order_number': receipt.order_number is not None,
            'high_value_item_count': len(high_value_items),
            'categories': list(set(item.category.value for item in receipt.items if item.category)),
            'filename': receipt.filename,
        }
        if receipt.card_network:
            metadata['card_network'] = receipt.card_network
        if receipt.card_last4:
            metadata['card_last4'] = receipt.card_last4
        if receipt.merchant_city:
            metadata['merchant_city'] = receipt.merchant_city
        if receipt.merchant_state:
            metadata['merchant_state'] = receipt.merchant_state
        
        # Only add optional fields if they have values (Pinecone rejects null)
        if receipt.merchant_address:
            metadata['merchant_address'] = receipt.merchant_address
        if receipt.merchant_phone:
            metadata['merchant_phone'] = receipt.merchant_phone
        if receipt.merchant_website:
            metadata['merchant_website'] = receipt.merchant_website
        if receipt.store_number:
            metadata['store_number'] = receipt.store_number
        
        return ReceiptChunk(
            chunk_id=str(uuid.uuid4()),
            receipt_id=receipt.receipt_id,
            chunk_type='merchant_info',
            content=content,
            metadata=metadata,
            created_at=datetime.utcnow()
        )
    
    def _create_payment_chunk(self, receipt: Receipt) -> ReceiptChunk:
        """
        Create a chunk with payment method information.
        
        This chunk is optimized for:
        - Payment method analysis
        - Cash vs credit tracking
        - Tip analysis
        - Loyalty program tracking
        """
        content_parts = [
            f"Payment method: {receipt.payment_method.value}",
            f"Store: {receipt.merchant_name}",
            f"Date: {receipt.transaction_date.strftime('%Y-%m-%d')}",
            f"Amount: ${receipt.total_amount:.2f}",
        ]
        
        if receipt.tip_amount:
            content_parts.append(f"Tip: ${receipt.tip_amount:.2f}")
        
        if receipt.discounts:
            content_parts.append(f"Discounts: ${receipt.discounts:.2f}")
        
        if receipt.loyalty_program:
            content_parts.append(f"Loyalty program: {receipt.loyalty_program}")
        
        content = ". ".join(content_parts)
        
        metadata = {
            'receipt_id': receipt.receipt_id,
            'payment_method': receipt.payment_method.value,
            'merchant_name': receipt.merchant_name,
            'merchant_name_norm': self._normalize_merchant_name(receipt.merchant_name),
            'transaction_date': receipt.transaction_date.isoformat(),
            'transaction_ts': int(receipt.transaction_date.timestamp()),
            'transaction_year': receipt.transaction_date.year,
            'transaction_month': receipt.transaction_date.month,
            'total_amount': float(receipt.total_amount),
            'has_tip': receipt.tip_amount is not None,
            'has_discounts': receipt.discounts is not None,
            'has_loyalty_program': receipt.loyalty_program is not None,
            'has_delivery_fee': receipt.delivery_fee is not None,
            'is_return': receipt.return_transaction,
            'has_warranty': receipt.has_warranty,
            'filename': receipt.filename,
        }
        if receipt.card_network:
            metadata['card_network'] = receipt.card_network
        if receipt.card_last4:
            metadata['card_last4'] = receipt.card_last4
        if receipt.tip_amount is not None:
            metadata['tip_amount'] = float(receipt.tip_amount)
        if receipt.discounts is not None:
            metadata['discounts'] = float(receipt.discounts)

        if receipt.delivery_fee is not None:
            metadata['delivery_fee'] = float(receipt.delivery_fee)
        
        return ReceiptChunk(
            chunk_id=str(uuid.uuid4()),
            receipt_id=receipt.receipt_id,
            chunk_type='payment_method',
            content=content,
            metadata=metadata,
            created_at=datetime.utcnow()
        )
    
    def get_chunking_stats(self, receipts: List[Receipt]) -> Dict[str, Any]:
        """
        Get statistics about the chunking process.
        
        Args:
            receipts: List of receipts to analyze
            
        Returns:
            Dictionary with chunking statistics
        """
        total_chunks = 0
        chunk_type_counts = defaultdict(int)
        
        for receipt in receipts:
            chunks = self.chunk_receipt(receipt)
            total_chunks += len(chunks)
            for chunk in chunks:
                chunk_type_counts[chunk.chunk_type] += 1
        
        return {
            'total_receipts': len(receipts),
            'total_chunks': total_chunks,
            'average_chunks_per_receipt': total_chunks / len(receipts) if receipts else 0,
            'chunk_type_distribution': dict(chunk_type_counts),
            'chunk_types': list(self.chunk_types.keys()),
        }
