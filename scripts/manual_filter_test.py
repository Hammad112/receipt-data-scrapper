
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from vectorstore.vector_manager import VectorManager

def manual_filter_test():
    load_dotenv()
    vm = VectorManager()
    
    # Test 1: Exact match
    print("Testing Filter: {'merchant_name': 'Walmart Supercenter'}")
    res1 = vm.index.query(
        vector=[0.0] * 1536,
        top_k=5,
        include_metadata=True,
        filter={"merchant_name": "Walmart Supercenter"}
    )
    print(f"Matches: {len(res1['matches'])}")
    
    # Test 2: $in operator
    print("\nTesting Filter: {'merchant_name': {'$in': ['Walmart', 'Walmart Supercenter']}}")
    res2 = vm.index.query(
        vector=[0.0] * 1536,
        top_k=5,
        include_metadata=True,
        filter={"merchant_name": {"$in": ["Walmart", "Walmart Supercenter"]}}
    )
    print(f"Matches: {len(res2['matches'])}")

    # Test 3: Filename + Merchant
    filename = "receipt_001_grocery_20231107.txt"
    print(f"\nTesting Filter: {{'filename': '{filename}', 'merchant_name': 'Walmart Supercenter'}}")
    res3 = vm.index.query(
        vector=[0.0] * 1536,
        top_k=5,
        include_metadata=True,
        filter={"filename": filename, "merchant_name": "Walmart Supercenter"}
    )
    print(f"Matches: {len(res3['matches'])}")

if __name__ == "__main__":
    manual_filter_test()
