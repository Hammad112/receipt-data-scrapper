"""
Receipt chunking strategy for optimal vector embedding and retrieval.

This module implements a hybrid chunking approach that balances:
- Granularity (receipt-level vs. item-level)
- Context preservation
- Query performance
"""

import uuid
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Absolute imports for industrial stability
from src.models import Receipt, ReceiptItem, ReceiptChunk, ItemCategory
from src.utils.normalization import normalize_merchant_name


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
    
    MAX_CHUNK_TOKENS = 8000  # Safety limit for text-embedding-3-small (8191 limit)

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

    def _truncate_content(self, content: str) -> str:
        """Heuristic truncation to stay within rough token limits."""
        # Rough heuristic: 1 token ~= 4 characters for English
        max_chars = self.MAX_CHUNK_TOKENS * 3 
        if len(content) > max_chars:
            return content[:max_chars] + "... [TRUNCATED]"
        return content

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
    
    def _get_base_metadata(self, receipt: Receipt) -> Dict[str, Any]:
        """
        Centralizes shared metadata fields across all chunk types.
        Ensures consistency and simplifies chunk-specific logic.
        """
        metadata = {
            'receipt_id': receipt.receipt_id,
            'merchant_name': receipt.merchant_name,
            'merchant_name_norm': normalize_merchant_name(receipt.merchant_name),
            'transaction_date': receipt.transaction_date.isoformat(),
            'transaction_ts': int(receipt.transaction_date.timestamp()),
            'transaction_year': receipt.transaction_date.year,
            'transaction_month': receipt.transaction_date.month,
            'transaction_day': receipt.transaction_date.day,
            'transaction_weekday': receipt.transaction_date.weekday(),
            'payment_method': receipt.payment_method.value,
            'total_amount': float(receipt.total_amount),
            'filename': receipt.filename,
        }
        
        # Add optional financial metadata if present (Pinecone rejects nulls)
        if receipt.card_network: metadata['card_network'] = receipt.card_network
        if receipt.card_last4: metadata['card_last4'] = receipt.card_last4
        if receipt.merchant_city: metadata['merchant_city'] = receipt.merchant_city
        if receipt.merchant_state: metadata['merchant_state'] = receipt.merchant_state
        if receipt.merchant_zip: metadata['merchant_zip'] = receipt.merchant_zip
        if receipt.return_transaction: metadata['is_return'] = True
        
        return metadata

    def _create_summary_chunk(self, receipt: Receipt) -> ReceiptChunk:
        """Creates a high-level overview chunk of the receipt."""
        content_parts = [
            f"Receipt from {receipt.merchant_name}",
            f"Date: {receipt.transaction_date.strftime('%Y-%m-%d %H:%M')}",
            f"Subtotal: ${receipt.subtotal:.2f}",
            f"Tax: ${receipt.tax_amount:.2f}",
            f"Total amount: ${receipt.total_amount:.2f}",
            f"Payment method: {receipt.payment_method.value}",
            f"Number of items: {len(receipt.items)}",
        ]
        
        # Metadata construction
        metadata = self._get_base_metadata(receipt)
        metadata.update({
            'chunk_type': 'receipt_summary',
            'subtotal': float(receipt.subtotal),
            'tax_amount': float(receipt.tax_amount),
            'item_count': len(receipt.items),
            'categories': receipt.categories,  # Uses new multi-label property
            'has_tip': receipt.tip_amount is not None,
            'has_discounts': receipt.discounts is not None,
            'has_delivery_fee': receipt.delivery_fee is not None,
            'has_warranty': receipt.has_warranty,
        })
        
        # Append extras
        if receipt.discounts:
            content_parts.append(f"Discounts: ${receipt.discounts:.2f}")
            metadata['discounts'] = float(receipt.discounts)
        if receipt.tip_amount:
            content_parts.append(f"Tip: ${receipt.tip_amount:.2f}")
            metadata['tip_amount'] = float(receipt.tip_amount)
        if receipt.delivery_fee:
            content_parts.append(f"Delivery fee: ${receipt.delivery_fee:.2f}")
            metadata['delivery_fee'] = float(receipt.delivery_fee)
        if receipt.loyalty_program:
            content_parts.append(f"Loyalty program: {receipt.loyalty_program}")
            metadata['loyalty_program'] = receipt.loyalty_program
        if receipt.has_warranty and receipt.warranty_text:
            content_parts.append(f"Warranty info: {receipt.warranty_text}")
            metadata['warranty_text'] = receipt.warranty_text
        elif receipt.has_warranty:
            content_parts.append("Warranty info present")
        if receipt.return_transaction:
            content_parts.append("Return transaction")
        if receipt.merchant_address:
            metadata['merchant_address'] = receipt.merchant_address
        
        if receipt.items:
            item_names = [item.name for item in receipt.items[:5]]
            if len(receipt.items) > 5:
                item_names.append(f"plus {len(receipt.items) - 5} others")
            content_parts.append(f"Top items: {', '.join(item_names)}")
        
        return ReceiptChunk(
            chunk_id=str(uuid.uuid4()),
            receipt_id=receipt.receipt_id,
            chunk_type='receipt_summary',
            content=". ".join(content_parts),
            metadata=metadata,
            created_at=datetime.now(timezone.utc)
        )
    
    def _create_item_chunks(self, receipt: Receipt) -> List[ReceiptChunk]:
        """Creates individual chunks for each line item with rich context."""
        chunks = []
        base_metadata = self._get_base_metadata(receipt)
        
        for i, item in enumerate(receipt.items):
            item_categories = [c.value for c in item.categories] if item.categories else ['other']
            content = (
                f"Item: {item.name}. Price: ${item.total_price:.2f}. "
                f"Qty: {item.quantity}. Categories: {', '.join(item_categories)}. "
                f"Store: {receipt.merchant_name}. Date: {receipt.transaction_date.strftime('%Y-%m-%d')}."
            )
            
            metadata = base_metadata.copy()
            metadata.update({
                'chunk_type': 'item_detail',
                'item_index': i,
                'item_name': item.name,
                'item_category': item.category.value if item.category else 'other', # Primary for backward compat
                'item_categories': item_categories, # New multi-label support
                'item_price': float(item.total_price),
                'item_unit_price': float(item.unit_price),
                'item_quantity': float(item.quantity)
            })
            
            chunks.append(ReceiptChunk(
                receipt_id=receipt.receipt_id,
                chunk_type='item_detail',
                content=content,
                metadata=metadata
            ))
        return chunks
    
    def _create_category_chunks(self, receipt: Receipt) -> List[ReceiptChunk]:
        """Creates chunks grouping multiple items into spending categories."""
        chunks = []
        base_metadata = self._get_base_metadata(receipt)
        
        # Group items by category (Multi-Label: item can appear in multiple groups)
        category_groups = defaultdict(list)
        for item in receipt.items:
            # If item has no categories, group under OTHER
            if not item.categories:
                category_groups[ItemCategory.OTHER].append(item)
            else:
                for cat in item.categories:
                    category_groups[cat].append(item)
        
        for category, items in category_groups.items():
            if len(items) <= 1: continue 
            
            total_amount = sum(item.total_price for item in items)
            content = (
                f"Category: {category.value}. Total: ${total_amount:.2f}. "
                f"Items ({len(items)}): {', '.join(item.name for item in items)}. "
                f"Store: {receipt.merchant_name}."
            )
            
            metadata = base_metadata.copy()
            metadata.update({
                'chunk_type': 'category_group',
                'category': category.value,
                'item_count': len(items),
                'total_amount': float(total_amount),
                'item_names': [item.name for item in items]
            })
            
            chunks.append(ReceiptChunk(
                receipt_id=receipt.receipt_id,
                chunk_type='category_group',
                content=content,
                metadata=metadata
            ))
        return chunks
    
    def _create_merchant_chunk(self, receipt: Receipt) -> ReceiptChunk:
        """Creates a merchant-focused chunk for geographic or store-based queries."""
        content_parts = [
            f"Merchant: {receipt.merchant_name}",
            f"Location: {receipt.merchant_address or 'Unknown Address'}",
            f"Total visits: 1",
            f"Last total: ${receipt.total_amount:.2f}"
        ]
        
        metadata = self._get_base_metadata(receipt)
        metadata.update({
            'chunk_type': 'merchant_info',
            'has_address': receipt.merchant_address is not None,
            'has_phone': receipt.merchant_phone is not None
        })
        
        return ReceiptChunk(
            receipt_id=receipt.receipt_id,
            chunk_type='merchant_info',
            content=". ".join(content_parts),
            metadata=metadata
        )
    
    def _create_payment_chunk(self, receipt: Receipt) -> ReceiptChunk:
        """Creates a payment-focused chunk for financial audit queries."""
        metadata = self._get_base_metadata(receipt)
        metadata.update({
            'chunk_type': 'payment_method',
            'has_tip': receipt.tip_amount is not None,
            'has_discounts': receipt.discounts is not None
        })
        
        content = (
            f"Payment: {receipt.payment_method.value}. Store: {receipt.merchant_name}. "
            f"Total: ${receipt.total_amount:.2f}. Date: {receipt.transaction_date.strftime('%Y-%m-%d')}."
        )
        
        return ReceiptChunk(
            receipt_id=receipt.receipt_id,
            chunk_type='payment_method',
            content=content,
            metadata=metadata
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
