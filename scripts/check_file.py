
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from vectorstore.vector_manager import VectorManager

def check_specific_file():
    load_dotenv()
    vm = VectorManager()
    
    filename = "receipt_001_grocery_20231107.txt"
    print(f"Checking for filename: '{filename}'")
    
    results = vm.index.query(
        vector=[0.0] * 1536,
        top_k=10,
        include_metadata=True,
        filter={"filename": filename}
    )
    
    print(f"Found {len(results['matches'])} matches for this file.")
    for m in results['matches']:
        print(f" - Merchant in metadata: '{m['metadata'].get('merchant_name')}'")

if __name__ == "__main__":
    check_specific_file()
