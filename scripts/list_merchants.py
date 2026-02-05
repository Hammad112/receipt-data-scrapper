
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from vectorstore.vector_manager import VectorManager

def list_merchants():
    load_dotenv()
    vm = VectorManager()
    
    # Pinecone doesn't have a built-in "list all unique metadata"
    # So we'll query for a broad range and extract from results
    print("Fetching sample vectors to identify merchants...")
    
    # Query with No filter to see what's there
    results = vm.index.query(
        vector=[0.0] * 1536, # Dummy vector
        top_k=100,
        include_metadata=True
    )
    
    merchants = set()
    for match in results['matches']:
        m = match['metadata'].get('merchant_name')
        if m:
            merchants.add(m)
    
    print(f"Total Unique Merchants Found in Sample (100): {len(merchants)}")
    for m in sorted(list(merchants)):
        print(f" - '{m}'")

if __name__ == "__main__":
    list_merchants()
