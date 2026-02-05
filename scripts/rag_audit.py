
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from vectorstore.vector_manager import VectorManager
from query.query_engine import QueryEngine
from models.receipt import ReceiptChunk
from utils.logging_config import setup_logging, logger

def run_audit():
    setup_logging("rag_audit")
    load_dotenv()
    
    print("\n--- RAG Pipeline Audit Starting ---")
    
    # 1. Initialize Vector Manager
    vm = VectorManager()
    print("Step 1: Vector Manager initialized.")
    
    # 2. Verify Embedding Generation
    test_text = "Organic avocado from Whole Foods for $5.99"
    embedding = vm.generate_embedding(test_text)
    print(f"Step 2: Embedding created. Dimension: {len(embedding)}")
    
    # 3. Verify Vector Search
    print("Step 3: Performing semantic vector search for 'Whole Foods'...")
    search_results = vm.hybrid_search("Whole Foods", top_k=3)
    
    if search_results:
        print(f"Found {len(search_results)} matches in Pinecone.")
        for i, match in enumerate(search_results):
            print(f"  Match {i+1}: {match['metadata'].get('merchant_name')} - Score: {match['score']:.4f}")
    else:
        print("No matches found in vector search.")
        return

    # 4. Verify LLM Integration
    print("\nStep 4: Providing context to LLM for response generation...")
    engine = QueryEngine(vm)
    query = "How much did I spend at Whole Foods?"
    result = engine.process_query(query)
    
    print(f"\nQUERY: {query}")
    print(f"LLM RESPONSE: {result.answer}")
    print(f"CONFIDENCE: {result.confidence:.1%}")
    print(f"PROCESSING TIME: {result.processing_time:.2f}s")
    
    print("\n--- Audit Complete: Full Pipeline Verified ---")

if __name__ == "__main__":
    run_audit()
