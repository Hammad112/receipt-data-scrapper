
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from vectorstore.vector_manager import VectorManager
from query.query_engine import QueryEngine

def simulate_walmart_query():
    load_dotenv()
    vm = VectorManager()
    engine = QueryEngine(vm)
    
    query = "Find all Walmart receipts"
    print(f"Simulating query: '{query}'")
    
    # 1. Parse
    params = engine.parser.parse(query)
    print(f"Parsed Params: {params}")
    
    # 2. Build filters
    filters = engine._build_search_filters(params)
    print(f"Built Filters: {filters}")
    
    # 3. Direct Search
    print("\n--- Executing Direct Vector Search ---")
    search_results = vm.hybrid_search(query, filters=filters, top_k=20)
    print(f"Search Results Count: {len(search_results)}")
    
    for i, res in enumerate(search_results):
        meta = res['metadata']
        print(f"  Match {i+1}: {meta.get('merchant_name')} ({meta.get('filename')}) Score: {res['score']:.4f}")

    # 4. Full Process
    print("\n--- Executing Full Engine Process ---")
    result = engine.process_query(query)
    print(f"Engine Answer: {result.answer}")
    print(f"Found {len(result.receipts)} receipts and {len(result.items)} items.")

if __name__ == "__main__":
    simulate_walmart_query()
