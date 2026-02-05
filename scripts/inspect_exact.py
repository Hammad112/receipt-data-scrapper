
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from vectorstore.vector_manager import VectorManager

def inspect_exact_metadatas():
    load_dotenv()
    vm = VectorManager()
    
    # Query for dummy vector to get broad results
    results = vm.index.query(
        vector=[0.0] * 1536,
        top_k=200,
        include_metadata=True
    )
    
    merchants_found = set()
    total_matches = len(results['matches'])
    print(f"Total Matches in Sample: {total_matches}")
    
    for match in results['matches']:
        m = match['metadata'].get('merchant_name')
        if m:
            merchants_found.add(m)
    
    print("\nUnique Merchants in Sample:")
    for m in sorted(list(merchants_found)):
        print(f" - '{m}'")
    
    # Check specifically for any merchant containing walmart (case insensitive)
    walmart_matches = [m for m in merchants_found if 'WALMART' in m.upper()]
    print(f"\nWalmart variations found: {walmart_matches}")

if __name__ == "__main__":
    inspect_exact_metadatas()
