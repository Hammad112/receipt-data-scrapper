"""
Receipt parsing logic for extracting structured data from receipt text.

This module provides the ReceiptParser class which handles the conversion of raw 
receipt text into structured Pydantic models.
"""

import re
import uuid
import logging
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Dict, Any, Tuple
from dateutil import parser as date_parser

try:
    from ..models import Receipt, ReceiptItem, PaymentMethod, ItemCategory
    from ..utils.logging_config import logger
except ImportError:
    from models import Receipt, ReceiptItem, PaymentMethod, ItemCategory
    from utils.logging_config import logger


class ReceiptParser:
    """
    Industrial-grade parser for extracting structured data from raw receipt text.
    
    Uses a combination of regular expressions and semantic rules to extract 
    merchants, dates, items, and totals from unstructured receipt text.
    """
    
    def __init__(self):
        """
        Initializes the ReceiptParser with all necessary regex patterns.
        """
        self.merchant_patterns = [
            r'^(.*?)(?:\s+(?:STORE|SHOP|MARKET|PHARMACY|CAFE|RESTAURANT))?$',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+.*$',
        ]
        
        self.date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            r'([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})',
        ]
        
        self.payment_patterns = {
            PaymentMethod.CASH: [r'\bcash\b', r'paid\s+cash'],
            PaymentMethod.CREDIT: [r'\bcredit\b', r'\bvisa\b', r'\bmastercard\b', r'\bamex\b', r'\bdiscover\b'],
            PaymentMethod.DEBIT: [r'\bdebit\b', r'debit\s+card'],
            PaymentMethod.APPLE_PAY: [r'apple\s+pay', r'\s*pay'],
            PaymentMethod.GOOGLE_PAY: [r'google\s+pay'],
        }
        
        self.price_patterns = [
            r'\$(\d+(?:\.\d{2})?)',
            r'(\d+(?:\.\d{2})?)\s*$',
        ]
        
        self.quantity_patterns = [
            r'(\d+)\s*[xX]',
            r'qty:\s*(\d+)',
            r'(\d+)\s+(?:pcs|items|units)',
        ]
        
        self.item_patterns = [
            r'^([A-Za-z][\w\s\(\)\-\.]+?)\s+\$\s*(\d+\.\d{2})\s*$',
            r'^(\d+)\s*[xX]\s+([A-Za-z][\w\s\(\)\-\.]+?)\s+\$\s*(\d+\.\d{2})\s*$',
            r'^([A-Za-z][\w\s\(\)\-\.]+?)\s*\((\d+)\)\s+\$\s*(\d+\.\d{2})\s*$',
            r'^([A-Za-z][\w\s\(\)\-\.]+?)\s+@\s+\$\s*(\d+\.\d{2})\s*$',
            r'^([A-Za-z][\w\s\(\)\-\.]+?)\s{2,}\$\s*(\d+\.\d{2})\s*$',
        ]
        
        self.category_keywords = {
            ItemCategory.GROCERIES: [
                'milk', 'bread', 'eggs', 'cheese', 'meat', 'vegetables', 'fruits',
                'cereal', 'pasta', 'rice', 'beans', 'yogurt', 'butter', 'oil'
            ],
            ItemCategory.ELECTRONICS: [
                'phone', 'laptop', 'computer', 'tablet', 'headphones', 'cable',
                'charger', 'battery', 'tv', 'camera', 'gaming', 'console',
                'mouse', 'keyboard', 'monitor', 'printer', 'software', 'hardware',
                'memory', 'drive', 'ssd', 'usb', 'case', 'adapter'
            ],
            ItemCategory.COFFEE_SHOP: [
                'coffee', 'latte', 'espresso', 'cappuccino', 'mocha',
                'drip', 'americano', 'tea', 'pastry', 'croissant', 'muffin'
            ],
            ItemCategory.PHARMACY: [
                'medicine', 'prescription', 'vitamin', 'supplement', 'pain',
                'cold', 'flu', 'allergy', 'first aid', 'bandage'
            ],
            ItemCategory.TREATS: [
                'candy', 'chocolate', 'ice cream', 'cake', 'cookie', 'donut',
                'pie', 'dessert', 'sweet', 'candy bar', 'pastry'
            ],
            ItemCategory.RESTAURANT: [
                'burger', 'pizza', 'sandwich', 'salad', 'soup', 'pasta',
                'steak', 'chicken', 'fish', 'appetizer', 'entree'
            ],
        }

    def parse_receipt(self, text: str, filename: Optional[str] = None) -> Receipt:
        """
        Main entry point for parsing a raw receipt string.
        
        Args:
            text: Raw receipt text.
            filename: Optional source filename for traceability.
            
        Returns:
            Receipt: A structured Pydantic model containing all extracted data.
        """
        logger.debug(f"Parsing receipt: {filename if filename else 'UNNAMED'}")
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Core Extraction
        merchant_name = self._extract_merchant_name(lines)
        transaction_date = self._extract_date(lines)
        payment_method = self._extract_payment_method(lines)
        items = self._extract_items(lines)
        subtotal, tax_amount, tip_amount, total_amount, discounts = self._extract_totals(lines)
        metadata = self._extract_metadata(lines)
        
        logger.info(f"Successfully parsed receipt from {merchant_name} on {transaction_date.date()}")
        
        return Receipt(
            receipt_id=str(uuid.uuid4()),
            filename=filename,
            merchant_name=merchant_name,
            transaction_date=transaction_date,
            payment_method=payment_method,
            items=items,
            subtotal=subtotal,
            tax_amount=tax_amount,
            tip_amount=tip_amount,
            total_amount=total_amount,
            discounts=discounts,
            raw_text=text,
            **metadata
        )

    def _extract_merchant_name(self, lines: List[str]) -> str:
        """
        Extracts the merchant name, typically found in the header.
        
        Args:
            lines: List of receipt lines.
            
        Returns:
            str: Detected merchant name.
        """
        for line in lines[:5]:
            for pattern in self.merchant_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match and len(match.group(1)) > 2:
                    return match.group(1).strip()
        
        return lines[0] if lines else "Unknown Merchant"

    def _extract_date(self, lines: List[str]) -> datetime:
        """
        Statically parses dates from receipt lines using robust regex patterns.
        
        Args:
            lines: List of receipt lines.
            
        Returns:
            datetime: Detected transaction date or current time as fallback.
        """
        for line in lines:
            for pattern in self.date_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    try:
                        return date_parser.parse(match)
                    except Exception:
                        continue
        return datetime.utcnow()

    def _extract_payment_method(self, lines: List[str]) -> PaymentMethod:
        """
        Detects payment method (Cash, Credit, Debit, etc.) from receipt text.
        
        Args:
            lines: List of receipt lines.
            
        Returns:
            PaymentMethod: Enum value representing the payment type.
        """
        text_lower = ' '.join(lines).lower()
        for method, patterns in self.payment_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return method
        return PaymentMethod.OTHER

    def _extract_items(self, lines: List[str]) -> List[ReceiptItem]:
        """
        Extracts individual line items from the receipt.
        
        Handles both single-line (Name + Price) and multi-line (Name on one line, 
        Price on the next) formats.
        
        Args:
            lines: List of receipt lines.
            
        Returns:
            List[ReceiptItem]: List of structured items.
        """
        items = []
        last_item_name_candidate = None
        
        for i, line in enumerate(lines):
            if self._is_non_item_line(line):
                continue
            
            # Scenario 1: Full line (Name + Price)
            item = self._parse_item_line(line)
            if item:
                items.append(item)
                last_item_name_candidate = None
                continue
            
            # Scenario 2: Price-only line (connecting to previous line's name)
            price_only_match = re.search(r'^\s*(?:\$\s*)?(\d+\.\d{2})\s*$', line)
            if price_only_match and last_item_name_candidate:
                price_str = price_only_match.group(1)
                item = self._parse_item_line(f"{last_item_name_candidate} ${price_str}")
                if item:
                    items.append(item)
                    last_item_name_candidate = None
                    continue
            
            # Candidate for next line's price (text but no price)
            if len(line) > 2 and not re.search(r'\d+\.\d{2}', line) and not any(kw in line.lower() for kw in ['total', 'subtotal', 'tax']):
                if not re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', line) and not re.search(r'ID:', line):
                    last_item_name_candidate = line
        
        return items

    def _is_non_item_line(self, line: str) -> bool:
        """
        Heuristic filter to exclude headers, footers, and metadata from item search.
        """
        non_item_patterns = [
            r'total', r'subtotal', r'tax', r'tip', r'discount', r'cash',
            r'credit', r'debit', r'change', r'thank you', r'receipt',
            r'---+', r'===+', r'\*\*\*+', r'\.\.\.+',
            r'date:', r'time:', r'cashier:', r'register:', r'order:',
            r'payment', r'server', r'associate', r'phone:', r'address:',
            r'sku:', r'transaction id:', r'approval code:'
        ]
        line_lower = line.lower()
        return any(re.search(pattern, line_lower) for pattern in non_item_patterns)

    def _parse_item_line(self, line: str) -> Optional[ReceiptItem]:
        """
        Detailed regex parsing for a single candidate item line.
        """
        quantity = Decimal('1')
        item_name = ""
        price = Decimal('0')
        price_str = ""
        matched = False
        
        for pattern in self.item_patterns:
            match = re.search(pattern, line)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    item_name, price_str = groups
                    matched = True
                    break
                elif len(groups) == 3:
                    if groups[0].isdigit():
                        qty_str, item_name, price_str = groups
                        try:
                            quantity = Decimal(qty_str)
                        except Exception:
                            quantity = Decimal('1')
                    else:
                        item_name, qty_str, price_str = groups
                        try:
                            quantity = Decimal(qty_str)
                        except Exception:
                            quantity = Decimal('1')
                    matched = True
                    break
        
        if not matched:
            price_match = re.search(r'\$\s*(\d+\.\d{2})', line)
            if price_match:
                price_str = price_match.group(1)
                item_name = line[:price_match.start()].strip()
                matched = True
        
        if not matched:
            return None
        
        # Hygiene
        if item_name:
            item_name = re.sub(r'\s+', ' ', item_name.strip())
            item_name = re.sub(r'\s*\(\d+\)\s*$', '', item_name)
        
        try:
            price = Decimal(price_str.replace('$', '').replace(',', '').strip())
        except Exception:
            return None
            
        unit_price = price / quantity if quantity > 0 else price
        category = self._categorize_item(item_name)
        
        if item_name and price and len(item_name) > 1:
            return ReceiptItem(
                name=item_name,
                quantity=quantity,
                unit_price=unit_price,
                total_price=price,
                category=category
            )
        return None

    def _categorize_item(self, item_name: str) -> Optional[ItemCategory]:
        """
        Rules-based semantic categorization.
        """
        name_lower = item_name.lower()
        for category, keywords in self.category_keywords.items():
            if any(keyword in name_lower for keyword in keywords):
                return category
        return ItemCategory.OTHER

    def _extract_totals(self, lines: List[str]) -> Tuple[Decimal, Decimal, Optional[Decimal], Decimal, Optional[Decimal]]:
        """
        Extracts monetary totals and tax info.
        """
        subtotal = Decimal('0')
        tax_amount = Decimal('0')
        tip_amount = None
        total_amount = Decimal('0')
        discounts = None
        
        for line in lines:
            ll = line.lower()
            if 'subtotal' in ll:
                amount = self._extract_price_from_line(line)
                if amount: subtotal = amount
            elif 'tax' in ll:
                amount = self._extract_price_from_line(line)
                if amount: tax_amount = amount
            elif 'tip' in ll:
                amount = self._extract_price_from_line(line)
                if amount: tip_amount = amount
            elif 'total' in ll:
                amount = self._extract_price_from_line(line)
                if amount: total_amount = amount
            elif 'discount' in ll:
                amount = self._extract_price_from_line(line)
                if amount: discounts = amount
        
        return subtotal, tax_amount, tip_amount, total_amount, discounts

    def _extract_price_from_line(self, line: str) -> Optional[Decimal]:
        """
        Generic price extraction helper.
        """
        for pattern in self.price_patterns:
            matches = re.findall(pattern, line)
            if matches:
                try:
                    return Decimal(matches[-1])
                except Exception:
                    continue
        return None

    def _extract_metadata(self, lines: List[str]) -> Dict[str, Any]:
        """
        Extracts additional contextual info like Phone, Address, and Cashier.
        """
        metadata = {}
        phone_pattern = r'(\(?\d{3}\)?[\-\.\s]?\d{3}[\-\.\s]?\d{4})'
        address_pattern = r'\d+\s+[A-Za-z0-9\s\.\-]+(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|Drive|Dr\.|Lane|Ln\.|Way|Court|Ct\.)'
        
        for i, line in enumerate(lines):
            ls = line.strip()
            ll = ls.lower()
            
            # Phone / Address
            if not metadata.get('merchant_phone'):
                pm = re.search(phone_pattern, ls)
                if pm: metadata['merchant_phone'] = pm.group(1)
            
            if not metadata.get('merchant_address'):
                if re.search(address_pattern, ls, re.IGNORECASE):
                    metadata['merchant_address'] = ls
                    if i + 1 < len(lines):
                        nl = lines[i+1].strip()
                        if re.search(r'[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}', nl):
                            metadata['merchant_address'] += f", {nl}"
            
            # Staff
            if not metadata.get('cashier'):
                if 'cashier:' in ll: metadata['cashier'] = ls.split(':', 1)[1].strip()
                elif 'server:' in ll: metadata['cashier'] = ls.split(':', 1)[1].strip()
                elif 'associate:' in ll: metadata['cashier'] = ls.split(':', 1)[1].strip()
            
            # References
            if not metadata.get('order_number') and 'order #' in ll:
                metadata['order_number'] = ls.split('#', 1)[1].strip()
            if not metadata.get('transaction_id') and 'transaction id:' in ll:
                metadata['transaction_id'] = ls.split(':', 1)[1].strip()
            if not metadata.get('store_number') and 'store #' in ll:
                metadata['store_number'] = ls.split('#', 1)[1].strip()
            if 'warranty' in ll:
                metadata['has_warranty'] = True
                
        return metadata

