
import os
import sys
import json
from datetime import datetime
from decimal import Decimal
import pandas as pd

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
from vectorstore import VectorManager
from query import QueryEngine

# Validation targets based on user request (50 queries)
TEST_QUERIES = [
    # Temporal (10)
    "How much did I spend in January 2024?",
    "What did I buy last week?",
    "Show me all receipts from December",
    "Show me receipts from October 2023",
    "Show me receipts from November 18, 2023",
    "Did I buy any burger in November?",
    "Show me receipts from November 5, 2023",
    "Show me all receipts from December 13th",
    "List all receipts from 2024",
    "How much was spent in November in total?",
    
    # Merchant (10)
    "Find all Whole Foods receipts",
    "List all items bought at Walmart",
    "What did I spend at CVS?",
    "Find all Safeway receipts",
    "What did I buy at Starbucks?",
    "Find all receipts from Philz Coffee",
    "What did I buy at B&H Photo?",
    "What's my total spending at Target?",
    "Find any receipt from a gas station",
    "List all receipts from Rite Aid",
    
    # Category / Semantic (10)
    "How much have I spent at coffee shops?",
    "What's my total spending at restaurants?",
    "Show me all electronics purchases",
    "What pharmacy items did I buy?",
    "Find health-related purchases",
    "Show me treats I bought",
    "List all items related to medicine",
    "Find healthy snacks",
    "How much was spent on groceries in total?",
    "Show me all pharmacy receipts over $20",
    
    # Cost-based & Aggregations (10)
    "List all groceries over $5",
    "Find all items under $2",
    "What are my most expensive items?",
    "List all coffee purchases over $3",
    "How much tax was paid on electronics?",
    "Show me all receipts with a subtotal over $100",
    "How much tax was paid in total?",
    "List all items from Best Buy over $50",
    "What's the most expensive item regardless of store?",
    "How much did I spend on sushi?",
    
    # Features & Edge Cases (10)
    "Find receipts with warranty information",
    "Show me receipts paid with Apple Pay",
    "Find all receipts with a tip",
    "Show me all discounts received in December",
    "Find any return transactions",
    "Find all receipts paid with cash",
    "Find all receipts from San Francisco",
    "Find all visa transactions",
    "Find receipts from Daly City",
    "Find receipts with delivery fees or tips"
]

def run_tests():
    print("🚀 Starting Comprehensive 50-Query Accuracy Tests...")
    
    # Initialize system
    load_dotenv()
    try:
        vm = VectorManager()
        engine = QueryEngine(vm)
        print(" System initialized successfully")
    except Exception as e:
        print(f" Failed to initialize system: {e}")
        return

    results = []
    
    print(f"\n📋 Running {len(TEST_QUERIES)} test queries...\n")
    
    for i, query in enumerate(TEST_QUERIES):
        print(f"🔹 [{i+1}/50] Testing: '{query}'")
        try:
            start_time = datetime.now()
            result = engine.process_query(query)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Collect matched items for verification
            matched_items_summary = []
            for item in result.items[:5]:
                filename_label = f" (from {item.get('filename')})" if item.get('filename') else ""
                matched_items_summary.append(f"{item.get('name')} (${item.get('price')}) [{item.get('merchant')}]{filename_label}")
            
            if len(result.items) > 5:
                matched_items_summary.append(f"...and {len(result.items) - 5} more")
            
            # Collect receipts for filename citation
            receipt_citations = []
            for r in result.receipts[:10]:
                if r.get('filename'):
                    receipt_citations.append(f"{r['merchant_name']} ({r.get('filename')})")
                
            entry = {
                "Query": query,
                "Answer": result.answer,
                "Items Found": len(result.items),
                "Receipts Found": len(result.receipts),
                "Confidence": result.confidence,
                "Processing Time": f"{duration:.2f}s",
                "Sample Matches": "; ".join(matched_items_summary),
                "Citations": ", ".join(receipt_citations)
            }
            results.append(entry)
            print(f"    Success ({len(result.items)} items, {len(result.receipts)} receipts)")
            
        except Exception as e:
            print(f"    Failed: {e}")
            results.append({
                "Query": query,
                "Answer": f"ERROR: {str(e)}",
                "Items Found": 0,
                "Receipts Found": 0,
                "Confidence": 0,
                "Processing Time": "0s",
                "Sample Matches": "N/A",
                "Citations": "N/A"
            })

    # Generate Report
    report_path = os.path.join(os.path.dirname(__file__), 'query_accuracy_report_50.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("#  50-Query Comprehensive Validation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 🎯 Test Summary\n")
        f.write(f"- Total Queries: {len(TEST_QUERIES)}\n")
        f.write(f"- Traceability: Citations included for original .txt files\n")
        f.write(f"- Coverage: Temporal, Merchant, Category, Semantic, Cost, Features\n\n")
        
        f.write("## 📝 Detailed Results\n\n")
        
        for res in results:
            f.write(f"###  Query: \"{res['Query']}\"\n")
            f.write(f"- **Answer**: {res['Answer']}\n")
            f.write(f"- **Receipts/Files**: {res['Citations'] if res['Citations'] else 'N/A'}\n")
            f.write(f"- **Sample Item Matches**: {res['Sample Matches']}\n")
            f.write(f"- **Stats**: {res['Items Found']} items | {res['Receipts Found']} receipts | {res['Processing Time']}\n")
            f.write("---\n")
            
    print(f"\n 50-Query tests complete. Final report generated at: {report_path}")

if __name__ == "__main__":
    run_tests()
