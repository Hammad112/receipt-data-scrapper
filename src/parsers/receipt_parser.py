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
    
    Design Philosophy:
    - Robustness: Uses layered regex patterns to handle varied receipt layouts.
    - Context-Awareness: Identifies non-item lines (tax, totals) to avoid polluting item extraction.
    - Semantic Enrichment: Automatically categorizes items based on keyword mapping.
    """
    
    def __init__(self):
        """
        Initializes the ReceiptParser with prioritized regex patterns.
        
        Patterns are organized by feature area (Merchant, Date, Payment, Price, etc.)
        to allow for easy maintenance and extension as new receipt formats are encountered.
        """
        self.merchant_patterns = [
            # Pattern 1: Bold header/top-line merchant name
            r'^(.*?)(?:\s+(?:STORE|SHOP|MARKET|PHARMACY|CAFE|RESTAURANT))?$',
            # Pattern 2: CamelCase or Title Case merchant name
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+.*$',
        ]
        
        self.date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', # MM/DD/YYYY or DD/MM/YYYY
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})', # YYYY-MM-DD
            r'([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})', # Month DD, YYYY
            r'(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})', # DD Month YYYY
        ]
        
        self.payment_patterns = {
            PaymentMethod.CASH: [r'\bcash\b', r'paid\s+cash'],
            PaymentMethod.CREDIT: [r'\bcredit\b', r'\bvisa\b', r'\bmastercard\b', r'\bamex\b', r'\bdiscover\b'],
            PaymentMethod.DEBIT: [r'\bdebit\b', r'debit\s+card'],
            PaymentMethod.APPLE_PAY: [r'apple\s+pay', r'\s*pay'],
            PaymentMethod.GOOGLE_PAY: [r'google\s+pay'],
        }
        
        self.price_patterns = [
            r'\$(\d+(?:\.\d{2})?)', # Standard $XX.XX
            r'(\d+(?:\.\d{2})?)\s*$', # Price at end of line without $
        ]
        
        self.quantity_patterns = [
            r'(\d+)\s*[xX]', # e.g., 2 x
            r'qty:\s*(\d+)', # e.g., Qty: 3
            r'(\d+)\s+(?:pcs|items|units)',
        ]
        
        # High-confidence item detection patterns
        self.item_patterns = [
            r'^([A-Za-z][\w\s\(\)\-\.]+?)\s+\$\s*(\d+\.\d{2})\s*$', # Name $Price
            r'^(\d+)\s*[xX]\s+([A-Za-z][\w\s\(\)\-\.]+?)\s+\$\s*(\d+\.\d{2})\s*$', # Qty x Name $Price
            r'^([A-Za-z][\w\s\(\)\-\.]+?)\s*\((\d+)\)\s+\$\s*(\d+\.\d{2})\s*$', # Name (Qty) $Price
            r'^([A-Za-z][\w\s\(\)\-\.]+?)\s+@\s+\$\s*(\d+\.\d{2})\s*$', # Name @ $Price
            r'^([A-Za-z][\w\s\(\)\-\.]+?)\s{2,}\$\s*(\d+\.\d{2})\s*$', # Name    $Price
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
        
        Execution pipeline:
        1. Cleanup: Strip whitespace and filter empty lines.
        2. Header Analysis: Extract Merchant, Date, Phone, Address.
        3. Transaction Context: Detect payment method and return status.
        4. Line Item Extraction: Iterate through lines to find product entries.
        5. Totals Extraction: Look for Subtotal, Tax, Tips, and final Total.
        6. Hygiene: Post-process names and ensure numeric consistency.
        """
        logger.debug(f"Parsing receipt: {filename if filename else 'UNNAMED'}")
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 1. Header & Context
        merchant_name = self._extract_merchant_name(lines)
        transaction_date = self._extract_date(lines)
        payment_method = self._extract_payment_method(lines)
        
        # 2. Body Analysis (Items)
        items = self._extract_items(lines)
        
        # 3. Footer Analysis (Finances)
        subtotal, tax_amount, tip_amount, delivery_fee, total_amount, discounts = self._extract_totals(lines)
        
        # 4. Contextual Metadata
        metadata = self._extract_metadata(lines)
        metadata['return_transaction'] = self._detect_return_transaction(lines, total_amount)
        
        logger.info(f"Successfully parsed receipt from {merchant_name} on {transaction_date.date()}")
        
        return Receipt(
            receipt_id=str(uuid.uuid4()),
            filename=filename,
            merchant_name=merchant_name,
            transaction_date=transaction_date,
            payment_method=payment_method,
            delivery_fee=delivery_fee,
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
        Extracts the merchant name from the header (first 5 lines).
        
        Heuristic: The first non-empty line that isn't a date or address is 
        usually the merchant. We also look for specific store suffixes.
        """
        for line in lines[:5]:
            for pattern in self.merchant_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match and len(match.group(1)) > 2:
                    return match.group(1).strip()
        
        return lines[0] if lines else "Unknown Merchant"

    def _extract_date(self, lines: List[str]) -> datetime:
        """
        Statically parses dates using robust regex patterns.
        
        Prioritizes common US and ISO formats. Falls back to current time 
        if no date is found to ensure system stability.
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
        Detects payment method by scanning for identifying keywords.
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
        
        Logic:
        - Filters out 'Non-Item' lines (Totals, Headers, Metadata).
        - Handles Single-Line items: 'Milk $4.50'
        - Handles Multi-Line items: Line n='Milk', Line n+1='$4.50'
        - Handles Quantity patterns: '2 x Milk $4.50'
        """
        items = []
        last_item_name_candidate = None
        
        for i, line in enumerate(lines):
            if self._is_non_item_line(line):
                continue
            
            # Scenario 1: Standard combined line
            item = self._parse_item_line(line)
            if item:
                items.append(item)
                last_item_name_candidate = None
                continue
            
            # Scenario 2: Price-only line (common when names wrap)
            price_only_match = re.search(r'^\s*(?:\$\s*)?(\d+\.\d{2})\s*$', line)
            if price_only_match and last_item_name_candidate:
                price_str = price_only_match.group(1)
                item = self._parse_item_line(f"{last_item_name_candidate} ${price_str}")
                if item:
                    items.append(item)
                    last_item_name_candidate = None
                    continue
            
            # Candidate for next line's price
            if len(line) > 2 and not re.search(r'\d+\.\d{2}', line) and not any(kw in line.lower() for kw in ['total', 'subtotal', 'tax']):
                if not re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', line) and not re.search(r'ID:', line):
                    last_item_name_candidate = line
        
        return items

    def _is_non_item_line(self, line: str) -> bool:
        """
        Heuristic filter to exclude functional lines that look like items.
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
        Low-level regex parser for a single candidate item string.
        Extracts Name, Quantity, and Price.
        """
        quantity = Decimal('1')
        item_name = ""
        price = Decimal('0')
        price_str = ""
        matched = False
        
        # Try structured multi-group patterns first (Qty + Name + Price)
        for pattern in self.item_patterns:
            match = re.search(pattern, line)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    item_name, price_str = groups
                    matched = True
                    break
                elif len(groups) == 3:
                    # Detect if first group is Qty or Name
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
        
        # Fallback to simple "ends with price" detection
        if not matched:
            price_match = re.search(r'\$\s*(\d+\.\d{2})', line)
            if price_match:
                price_str = price_match.group(1)
                item_name = line[:price_match.start()].strip()
                matched = True
        
        if not matched:
            return None
        
        # Cleanup name and strings
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
        Semantic categorization using priority keyword mapping.
        """
        name_lower = item_name.lower()
        for category, keywords in self.category_keywords.items():
            if any(keyword in name_lower for keyword in keywords):
                return category
        return ItemCategory.OTHER

    def _extract_totals(self, lines: List[str]) -> Tuple[Decimal, Decimal, Optional[Decimal], Optional[Decimal], Decimal, Optional[Decimal]]:
        """
        Scans footer lines for Subtotal, Tax, Tip, and Grand Total.
        """
        subtotal = Decimal('0')
        tax_amount = Decimal('0')
        tip_amount = None
        delivery_fee = None
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
            elif 'delivery fee' in ll or (re.search(r'\bdelivery\b', ll) and ('fee' in ll or 'charge' in ll)):
                amount = self._extract_price_from_line(line)
                if amount: delivery_fee = amount
            elif 'total' in ll:
                amount = self._extract_price_from_line(line)
                if amount: total_amount = amount
            elif 'discount' in ll:
                amount = self._extract_price_from_line(line)
                if amount: discounts = amount
        
        return subtotal, tax_amount, tip_amount, delivery_fee, total_amount, discounts

    def _detect_return_transaction(self, lines: List[str], total_amount: Decimal) -> bool:
        """
        Identifies if a receipt represents a return based on negative totals 
        or semantic refund keywords.
        """
        text = " ".join(lines).lower()
        if total_amount < 0:
            return True
        if "return policy" in text:
            text = text.replace("return policy", "")
        return bool(re.search(r"\b(refund|refunded|return|returned|credit memo|credit\s+transaction)\b", text))

    def _extract_price_from_line(self, line: str) -> Optional[Decimal]:
        """
        Helper to find a decimal value at the end of a tagged line (e.g., 'Total: $42.00').
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
        Analyzes lines for contextual data:
        - Merchant contact (Phone, Address, State, Zip)
        - Transaction tracking (Order #, Transaction ID)
        - Staff tracking (Cashier, Server)
        - Payment details (Card Network, Last 4)
        """
        metadata = {}
        phone_pattern = r'(\(?\d{3}\)?[\-\.\s]?\d{3}[\-\.\s]?\d{4})'
        address_pattern = r'\d+\s+[A-Za-z0-9\s\.\-]+(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|Drive|Dr\.|Lane|Ln\.|Way|Court|Ct\.)'
        city_state_zip_pattern = r'([A-Za-z\s]+),\s*([A-Z]{2})\s+(\d{5})'
        
        warranty_lines = []
        card_network = None
        card_last4 = None

        for i, line in enumerate(lines):
            ls = line.strip()
            ll = ls.lower()
            
            # --- Contact Info ---
            if not metadata.get('merchant_phone'):
                pm = re.search(phone_pattern, ls)
                if pm: metadata['merchant_phone'] = pm.group(1)
            
            if not metadata.get('merchant_address'):
                if re.search(address_pattern, ls, re.IGNORECASE):
                    metadata['merchant_address'] = ls
                    # Multi-line address check
                    if i + 1 < len(lines):
                        nl = lines[i+1].strip()
                        if re.search(r'[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}', nl):
                            metadata['merchant_address'] += f", {nl}"
                    
                    # Split into components
                    city_state = re.search(r'([A-Za-z\s]+),\s*([A-Z]{2})\s*(\d{5})?', metadata['merchant_address'])
                    if city_state:
                        metadata['merchant_city'] = city_state.group(1).strip()
                        metadata['merchant_state'] = city_state.group(2).strip()
                        if city_state.group(3):
                            metadata['merchant_zip'] = city_state.group(3).strip()
                else:
                    csz = re.search(city_state_zip_pattern, ls)
                    if csz and not metadata.get('merchant_city'):
                        metadata['merchant_city'] = csz.group(1).strip()
                        metadata['merchant_state'] = csz.group(2).strip()
                        metadata['merchant_zip'] = csz.group(3).strip()
            
            # --- Staff ---
            if not metadata.get('cashier'):
                if 'cashier:' in ll: metadata['cashier'] = ls.split(':', 1)[1].strip()
                elif 'server:' in ll: metadata['cashier'] = ls.split(':', 1)[1].strip()
                elif 'associate:' in ll: metadata['cashier'] = ls.split(':', 1)[1].strip()
            
            # --- References ---
            if not metadata.get('order_number') and 'order #' in ll:
                metadata['order_number'] = ls.split('#', 1)[1].strip()
            if not metadata.get('transaction_id') and 'transaction id:' in ll:
                metadata['transaction_id'] = ls.split(':', 1)[1].strip()
            if not metadata.get('store_number') and 'store #' in ll:
                metadata['store_number'] = ls.split('#', 1)[1].strip()
            if 'warranty' in ll:
                warranty_lines.append(ls)

            # --- Financial IDs ---
            if card_network is None or card_last4 is None:
                card_match = re.search(r"\b(visa|mastercard|amex|american express|discover)\b.*?(\*{2,}|\bending\b)\s*(\d{4})\b", ll)
                if card_match:
                    raw_network = card_match.group(1)
                    if raw_network == "american express":
                        raw_network = "amex"
                    card_network = raw_network
                    card_last4 = card_match.group(3)
                else:
                    card_match2 = re.search(r"\b(visa|mastercard|amex|discover)\b.*?\b(\d{4})\b", ll)
                    if card_match2:
                        card_network = card_match2.group(1)
                        card_last4 = card_match2.group(2)

        if warranty_lines:
            metadata['has_warranty'] = True
            metadata['warranty_text'] = " | ".join(dict.fromkeys(warranty_lines))

        if card_network:
            metadata['card_network'] = card_network
        if card_last4:
            metadata['card_last4'] = card_last4
                
        return metadata
