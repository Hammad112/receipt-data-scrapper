
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from vectorstore.vector_manager import VectorManager

def deep_check():
    load_dotenv()
    vm = VectorManager()
    
    query_text = "Walmart"
    embedding = vm.generate_embedding(query_text)
    
    # Try multiple filters to see what sticks
    filters_to_test = [
        {"merchant_name": "Walmart Supercenter"},
        {"merchant_name": "WALMART SUPERCENTER"},
        {"filename": "receipt_001_grocery_20231107.txt"},
        {"merchant_name": {"$in": ["Walmart", "Walmart Supercenter"]}}
    ]
    
    print(f"Index Stats: {vm.index.describe_index_stats()}")
    
    for f in filters_to_test:
        print(f"\nTesting filter: {f}")
        res = vm.index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True,
            filter=f
        )
        print(f"Matches: {len(res['matches'])}")
        for i, m in enumerate(res['matches']):
            print(f"  Result {i+1}: {m['metadata'].get('merchant_name')} | {m['metadata'].get('filename')}")

if __name__ == "__main__":
    deep_check()
