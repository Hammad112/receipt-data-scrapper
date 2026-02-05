import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
from vectorstore import VectorManager
from parsers import ReceiptParser
from chunking import ReceiptChunker

def reindex():
    load_dotenv()
    print("🚀 Starting Re-indexing with Filename Metadata...")
    
    vm = VectorManager()
    
    # Rebuild index to clear old data
    vm.rebuild_index()
    
    # Process all receipts
    receipt_dir = Path("data/receipt_samples_100")
    receipt_files = sorted(receipt_dir.glob("receipt_*.txt"))
    
    print(f"📂 Found {len(receipt_files)} receipts to process")
    
    parser = ReceiptParser()
    chunker = ReceiptChunker()
    
    all_chunks = []
    
    for i, file_path in enumerate(receipt_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Pass filename here!
            receipt = parser.parse_receipt(content, filename=file_path.name)
            chunks = chunker.chunk_receipt(receipt)
            all_chunks.extend(chunks)
            
            if (i + 1) % 10 == 0:
                print(f"  - Parsed {i + 1}/{len(receipt_files)} receipts")
        except Exception as e:
            print(f" Error parsing {file_path}: {e}")
            continue
            
    print(f" Parsed {len(receipt_files)} receipts into {len(all_chunks)} chunks")
    
    # Index in batches
    if all_chunks:
        print(f"📤 Uploading {len(all_chunks)} chunks to Pinecone...")
        vm.index_chunks(all_chunks, batch_size=50)
        print(" Re-indexing complete!")

if __name__ == "__main__":
    reindex()
