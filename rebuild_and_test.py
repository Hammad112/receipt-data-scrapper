import os
os.environ['PYTHONPATH'] = r'd:\Personal\University Projects\Scraping Project'

from src.vectorstore.vector_manager import VectorManager
from src.parsers.receipt_parser import ReceiptParser
from src.chunking.receipt_chunker import ReceiptChunker
from src.query.query_engine import QueryEngine
from pathlib import Path

# Initialize
vm = VectorManager()
parser = ReceiptParser()
chunker = ReceiptChunker()

print(f'Using index: {vm.index_name}')

# Clear and rebuild
print('Clearing index...')
vm.clear_index()

# Process receipts
receipt_dir = Path('data/receipt_samples_100')
receipt_files = sorted(receipt_dir.glob('receipt_*.txt'))

all_chunks = []
for f in receipt_files:
    content = f.read_text(encoding='utf-8')
    receipt = parser.parse_receipt(content, filename=f.name)
    chunks = chunker.chunk_receipt(receipt)
    all_chunks.extend(chunks)

print(f'Indexing {len(all_chunks)} chunks...')
vm.index_chunks(all_chunks, batch_size=10)

# Wait for vectors to be available
import time
time.sleep(5)

# Check index stats
stats = vm.get_index_stats()
print(f'Index stats: {stats}')

# Verify index has data
results = vm.index.query(vector=[0.0]*1536, top_k=10, include_metadata=True)
print(f'Verification query: {len(results.matches)} vectors found')

# Test queries
print('\n--- Testing Queries ---')
engine = QueryEngine(vm)
for q in ['How much have I spent at coffee shops?', 'What is my total spending at restaurants?']:
    result = engine.query(q)
    print(f'{q}: {len(result.receipts)} receipts')
