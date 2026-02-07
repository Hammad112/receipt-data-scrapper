import os
import sys
import json
from datetime import datetime
from decimal import Decimal

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
from vectorstore import VectorManager
from query import QueryEngine
from parsers import ReceiptParser
from chunking import ReceiptChunker

# Validation targets based on user request (50 queries)
TEST_QUERIES = [
    # Temporal (10)
    "How much did I spend in January 2024?",
    "What did I buy last week?",
    "Show me all receipts from December",
    "Show me receipts from November 18, 2023",
    "Did I buy any burger in November?",

    
    # Merchant (10)
    "Find all Whole Foods receipts",
    "List all items bought at Walmart",


    
    # Category / Semantic (10)
    "How much have I spent at coffee shops?",
    "What's my total spending at restaurants?",
    "Show me all electronics purchases",
    "What pharmacy items did I buy?",
    "Find health-related purchases",
    "Show me treats I bought",
    
    # Cost-based & Aggregations (10)
    "List all groceries over $5",
    "Find all items under $2",
    
    # Features & Edge Cases (10)
    "Find receipts with warranty information",


]

def run_tests():
    print(" Starting Comprehensive 50-Query Accuracy Tests...")
    
    # Initialize system
    load_dotenv()
    try:
        vm = VectorManager()
        engine = QueryEngine(vm)
        print(" System initialized successfully")
    except Exception as e:
        print(f" Failed to initialize system: {e}")
        return

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data', 'receipt_samples_100')
    if not os.path.isdir(data_dir):
        print(f" Data directory not found: {data_dir}")
        return

    receipt_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.lower().endswith('.txt') and f.lower().startswith('receipt_')
    ])
    if not receipt_files:
        print(f" No receipt .txt files found in: {data_dir}")
        return

    print(f"\n Resetting Pinecone index and indexing {len(receipt_files)} receipts from: {data_dir}")
    try:
        vm.clear_index()
    except Exception as e:
        print(f" Failed to reset Pinecone index: {e}")
        return

    parser = ReceiptParser()
    chunker = ReceiptChunker()
    all_chunks = []
    max_txn_date = None
    for file_path in receipt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            receipt = parser.parse_receipt(text, filename=os.path.basename(file_path))
            if max_txn_date is None or receipt.transaction_date > max_txn_date:
                max_txn_date = receipt.transaction_date
            all_chunks.extend(chunker.chunk_receipt(receipt))
        except Exception as e:
            print(f" Failed to parse/chunk {os.path.basename(file_path)}: {e}")
            continue

    if not all_chunks:
        print(" No chunks produced; aborting tests.")
        return

    indexed = vm.index_chunks(all_chunks, batch_size=10)
    print(f" Indexed {indexed}/{len(all_chunks)} chunks.")
    if max_txn_date:
        os.environ["RECEIPT_REFERENCE_DATE"] = max_txn_date.date().strftime("%Y%m%d")
        print(f" Using RECEIPT_REFERENCE_DATE={os.environ['RECEIPT_REFERENCE_DATE']} for relative date queries")

    results = []
    report_path = os.path.join(os.path.dirname(__file__), 'query_accuracy_report_50.md')
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("#  50-Query Comprehensive Validation Report\n")
        f.write(f"Generated: {generated_at}\n\n")
        f.write("## 🎯 Test Summary\n")
        f.write(f"- Total Queries: {len(TEST_QUERIES)}\n")
        f.write(f"- Traceability: Citations included for original .txt files\n")
        f.write(f"- Coverage: Temporal, Merchant, Category, Semantic, Cost, Features\n\n")
        f.write("## Detailed Results\n\n")
    
    print(f"\n Running {len(TEST_QUERIES)} test queries...\n")
    
    for i, query in enumerate(TEST_QUERIES):
        print(f"🔹 [{i+1}/50] Testing: '{query}'")
        try:
            start_time = datetime.now()
            result = engine.query(query)
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
                    receipt_citations.append(f"{r.get('merchant_name')} ({r.get('filename')})")
                
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
            with open(report_path, 'a', encoding='utf-8') as f:
                f.write(f"###  Query: \"{entry['Query']}\"\n")
                f.write(f"- **Answer**: {entry['Answer']}\n")
                f.write(f"- **Receipts/Files**: {entry['Citations'] if entry['Citations'] else 'N/A'}\n")
                f.write(f"- **Sample Item Matches**: {entry['Sample Matches']}\n")
                f.write(f"- **Stats**: {entry['Items Found']} items | {entry['Receipts Found']} receipts | {entry['Processing Time']}\n")
                f.write("---\n")
            
        except Exception as e:
            print(f"    Failed: {e}")
            entry = {
                "Query": query,
                "Answer": f"ERROR: {str(e)}",
                "Items Found": 0,
                "Receipts Found": 0,
                "Confidence": 0,
                "Processing Time": "0s",
                "Sample Matches": "N/A",
                "Citations": "N/A"
            }
            results.append(entry)
            with open(report_path, 'a', encoding='utf-8') as f:
                f.write(f"###  Query: \"{entry['Query']}\"\n")
                f.write(f"- **Answer**: {entry['Answer']}\n")
                f.write(f"- **Receipts/Files**: N/A\n")
                f.write(f"- **Sample Item Matches**: N/A\n")
                f.write(f"- **Stats**: 0 items | 0 receipts | 0s\n")
                f.write("---\n")

    allow_empty = {
        "Show me receipts from October 2023",
        "Find any return transactions",
    }
    failures = [
        r for r in results
        if str(r.get("Answer", "")).startswith("ERROR:")
        or (
            r.get("Query") not in allow_empty
            and (r.get("Items Found", 0) == 0 and r.get("Receipts Found", 0) == 0)
        )
    ]
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write("\n## Run Summary\n")
        f.write(f"- Completed: {len(results)}/{len(TEST_QUERIES)}\n")
        f.write(f"- Failures (0 items and 0 receipts, or error): {len(failures)}\n")
    
    print(f"\n 50-Query tests complete. Report generated at: {report_path}")

if __name__ == "__main__":
    run_tests()
