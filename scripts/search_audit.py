
import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from vectorstore.vector_manager import VectorManager
from query.query_engine import QueryEngine

def search_flow_audit():
    load_dotenv()
    vm = VectorManager()
    engine = QueryEngine(vm)
    
    query = "Find all Walmart receipts"
    print(f"Query: {query}")
    
    # 1. Parse
    params = engine.parser.parse(query)
    print(f"Parsed Params: {params}")
    
    # 2. Build filters
    filters = engine._build_search_filters(params)
    print(f"Engine Built Filters: {filters}")
    
    # 3. Search
    print("\nExecuting Hybrid Search...")
    results = vm.hybrid_search(query, filters=filters, top_k=20)
    print(f"Raw Matches: {len(results)}")
    
    # 4. Process Results
    processed = engine._process_search_results(results, params)
    print(f"Processed Results: {len(processed['receipts'])} receipts, {len(processed['items'])} items")
    
    if not results:
        # Try without filter to see similarity
        no_filter = vm.hybrid_search(query, filters=None, top_k=5)
        print("\nTOP SAMPLES (NO FILTER):")
        for m in no_filter:
            print(f" - {m['metadata'].get('merchant_name')} | Score: {m['score']:.4f}")

if __name__ == "__main__":
    search_flow_audit()
