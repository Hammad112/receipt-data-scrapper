
import re

def test_month_detection():
    query = "Find all Walmart receipts"
    query_lower = query.lower()
    
    months = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
        'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
        'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
        'november': 11, 'nov': 11, 'december': 12, 'dec': 12
    }
    
    for name, num in months.items():
        if name in query_lower:
            print(f"MATCH FOUND: '{name}' (Month {num})")
            
    print("\nCheck word boundaries:")
    for name, num in months.items():
        if re.search(r'\b' + name + r'\b', query_lower):
            print(f"WORD MATCH: '{name}'")

if __name__ == "__main__":
    test_month_detection()
