
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from parsers.receipt_parser import ReceiptParser

def test_walmart_parsing():
    parser = ReceiptParser()
    
    file_path = "data/receipt_samples_100/receipt_001_grocery_20231107.txt"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    receipt = parser.parse_receipt(content, filename="receipt_001_grocery_20231107.txt")
    print(f"Extracted Merchant: '{receipt.merchant_name}'")
    
    # Check lines and patterns
    lines = content.splitlines()
    print(f"First line: '{lines[0]}'")
    
    import re
    p1 = r'^(.*?)(?:\s+(?:STORE|SHOP|MARKET|PHARMACY|CAFE|RESTAURANT))?$'
    m1 = re.search(p1, lines[0], re.IGNORECASE)
    if m1:
        print(f"Pattern 1 Group 1: '{m1.group(1)}'")

    p2 = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+.*$'
    m2 = re.search(p2, lines[0])
    if m2:
        print(f"Pattern 2 Group 1: '{m2.group(1)}'")

if __name__ == "__main__":
    test_walmart_parsing()
