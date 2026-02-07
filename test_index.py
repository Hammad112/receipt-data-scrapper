from src.vectorstore.vector_manager import VectorManager
from src.query.query_engine import QueryEngine

# Check index
vm = VectorManager()
results = vm.index.query(vector=[0.0]*1536, top_k=100, include_metadata=True)
print(f'Total vectors: {len(results.matches)}')

# Check Starbucks chunks
print('\nStarbucks chunks:')
for m in results.matches:
    if 'Starbucks' in str(m.metadata.get('merchant_name', '')):
        print(f"  type={m.metadata.get('chunk_type')}, item_cat={m.metadata.get('item_category')}, cat={m.metadata.get('category')}")

# Test queries
print('\n--- Testing Queries ---')
engine = QueryEngine(vm)
for q in ['How much have I spent at coffee shops?', 'What is my total spending at restaurants?']:
    result = engine.query(q)
    print(f'{q}: {len(result.receipts)} receipts, {len(result.items)} items')
