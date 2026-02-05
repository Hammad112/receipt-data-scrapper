
import io

def read_results():
    with io.open('tests/query_accuracy_report_50.md', mode='r', encoding='utf-8') as f:
        content = f.read()
    
    sections = [
        'Query: "Find all Walmart receipts"',
        'Query: "List all items bought at Walmart"',
        'Query: "What\'s my total spending at Target?"'
    ]
    
    for s in sections:
        start = content.find(s)
        if start != -1:
            print(f"\n--- Section: {s} ---")
            print(content[start:start+600])
        else:
            print(f"\n--- Section NOT FOUND: {s} ---")

if __name__ == "__main__":
    read_results()
