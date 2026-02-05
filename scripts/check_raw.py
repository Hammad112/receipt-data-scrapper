
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from vectorstore.vector_manager import VectorManager

def check_raw_metadata():
    load_dotenv()
    vm = VectorManager()
    
    filename = "receipt_001_grocery_20231107.txt"
    print(f"Checking for filename: '{filename}'")
    
    results = vm.index.query(
        vector=[0.0] * 1536,
        top_k=1, # Just one to check
        include_metadata=True,
        filter={"filename": filename}
    )
    
    if results['matches']:
        meta = results['matches'][0]['metadata']
        print(f"RAW METADATA KEYS: {list(meta.keys())}")
        print(f"merchant_name value: {repr(meta.get('merchant_name'))}")
        print(f"filename value: {repr(meta.get('filename'))}")
    else:
        print("No matches for that filename!")

if __name__ == "__main__":
    check_raw_metadata()
