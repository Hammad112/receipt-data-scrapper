
import os
import sys
import json
import time

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vectorstore.vector_manager import VectorManager
from src.query.query_engine import QueryEngine
from src.utils.logging_config import setup_logging

def run_verification():
    setup_logging()
    
    print("Initialize System...")
    try:
        vm = VectorManager()
        engine = QueryEngine(vm)
    except Exception as e:
        print(f"FAILED to initialize system: {e}")
        return

    test_queries = [
        # Temporal
        "How much did I spend in January 2024?",
        "What did I buy last week?",
        "Show me all receipts from December",
        
        # Merchant
        "Find all Whole Foods receipts",
        
        # Category (Fixed!)
        "How much have I spent at coffee shops?",
        "What's my total spending at restaurants?",
        
        # Category / Item Specific
        "Show me all electronics purchases", 
        "What pharmacy items did I buy?",
        "List all groceries over $5",
        
        # Feature Flags
        "Find receipts with warranty information",
        
        # Semantic / Abstract
        "Find health-related purchases",
        "Show me treats I bought"
    ]

    print(f"\n{'='*60}")
    print(f"RUNNING FINAL VERIFICATION: {len(test_queries)} Test Cases")
    print(f"{'='*60}\n")
    
    passed = 0
    results_log = []

    for i, query in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] Query: \"{query}\"")
        start = time.time()
        try:
            result = engine.query(query)
            duration = time.time() - start
            
            # Simple validation: Did we get an answer? matches?
            # Note: We can't strictly assert "correctness" without knowing the dataset content perfectly,
            # but we can check if the pipeline executed and found logical results.
            
            status = "NO RESULTS" if "couldn't find" in result.answer else "SUCCESS"
            if len(result.receipts) > 0 or len(result.items) > 0:
                status = "SUCCESS"
            
            print(f"   Status: {status}")
            print(f"   Answer: {result.answer}")
            print(f"   Time:   {duration:.2f}s")
            
            # Filters Debug
            # We can't easily access the private internal filters from here without modifying code again,
            # but the result counts give us a good proxy.
            print(f"   Matches: {len(result.receipts)} receipts, {len(result.items)} items")
            
            log_entry = f"Query: {query}\nStatus: {status}\nAnswer: {result.answer}\nMatches: {len(result.receipts)}r/{len(result.items)}i\n{'-'*60}\n"
            with open("verification_results.log", "a") as f:
                f.write(log_entry)
            
            results_log.append({
                "query": query,
                "status": status,
                "answer": result.answer,
                "counts": f"{len(result.receipts)}r/{len(result.items)}i"
            })
            
            if status == "SUCCESS":
                passed += 1
                
        except Exception as e:
            print(f"   ERROR: {e}")
            results_log.append({"query": query, "status": "ERROR", "error": str(e)})
        
        print("-" * 60)

    print(f"\nSUMMARY: {passed}/{len(test_queries)} queries returned results.")
    
if __name__ == "__main__":
    run_verification()
