
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from vectorstore.vector_manager import VectorManager

def check_stats():
    load_dotenv()
    vm = VectorManager()
    stats = vm.index.describe_index_stats()
    print(f"Index Stats: {stats}")
    
    # Query with high top_k and no vector to see sample
    print("\nFetching broad sample...")
    res = vm.index.query(
        vector=[0.1] * 1536, # Use non-zero
        top_k=10,
        include_metadata=True
    )
    print(f"Matches count: {len(res['matches'])}")
    for m in res['matches']:
        print(f" - Filename: {m['metadata'].get('filename')} | Merchant: {m['metadata'].get('merchant_name')}")

if __name__ == "__main__":
    check_stats()
