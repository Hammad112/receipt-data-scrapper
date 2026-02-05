
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from vectorstore.vector_manager import VectorManager

def inspect_walmart_vectors():
    load_dotenv()
    vm = VectorManager()
    
    # Query for "Walmart" without filter to find the vectors
    print("Querying for 'Walmart' vectors...")
    results = vm.index.query(
        vector=vm.generate_embedding("Walmart"),
        top_k=5,
        include_metadata=True
    )
    
    print(f"Found {len(results['matches'])} matches.")
    for i, match in enumerate(results['matches']):
        meta = match['metadata']
        print(f"Match {i+1}:")
        print(f"  Score: {match['score']:.4f}")
        print(f"  Merchant: '{meta.get('merchant_name')}'")
        print(f"  Date: '{meta.get('transaction_date')}'")
        print(f"  Chunk Type: '{meta.get('chunk_type')}'")
        print(f"  Filename: '{meta.get('filename')}'")

if __name__ == "__main__":
    inspect_walmart_vectors()
