# Multi-View Chunking Strategy

## 1. Formal Explanation
In traditional RAG systems, documents are often split into fixed-size overlapping chunks. However, for structured documents like **receipts**, this naive approach leads to significant context loss. A single receipt contains diverse information: high-level summaries (total spent, date), granular line items (specific products), and transactional metadata (payment method, cashier).

**Multi-View Chunking** is a strategy where the same source document is indexed multiple times using different "views" or "perspectives." This ensures that regardless of the user's query intent, the system can retrieve the most relevant and precise slice of data.

---

## 2. Design Explanation & Justification: The "Granularity vs. Context" Trade-off

### The Problem
- **If chunks are too large (Full Receipt):** The LLM receives too much noise. If a user asks "How much did I spend on milk?", providing a 50-item receipt wastes tokens and might lead to "lost in the middle" retrieval issues.
- **If chunks are too small (Single Line):** The system loses context. A chunk reading "Milk - $4.50" doesn't know *where* or *when* it was bought, making temporal filters (e.g., "in January") or merchant filters (e.g., "at Target") impossible without complicated joins.

### The Solution: 5-Level Hierarchical Topology
We solve this by creating 5 distinct chunk types for every receipt, each enriched with **Global Receipt Metadata**.

| Chunk Type | Perspective | Optimization Goal |
|------------|-------------|-------------------|
| **Receipt Summary** | Holistic | High-level financial tracking and temporal queries. |
| **Item Detail** | Granular | Precise product search and price comparisons. |
| **Category Group** | Grouped | Sector-wise spending analysis (e.g., Groceries vs. Tech). |
| **Merchant Info** | Locality | Store-specific queries and physical location tracking. |
| **Payment Method**| Transactional | Audit trails, payment behavior, and loyalty tracking. |

---

## 3. Trade-offs

### Advantages (Pros)
1.  **Precision Retrieval:** The system fetches *exactly* what is needed. An aggregation query hits Category chunks; a product query hits Item chunks.
2.  **Rich Metadata Filtering:** By injecting global metadata (date, merchant, total) into every chunk, we enable **Hybrid Search**. We can filter by "January 2024" at the metadata level before performing vector similarity.
3.  **No Hallucinations:** Grounding the LLM in highly specific, structured chunks reduces the "reasoning" it has to do over noisy text.

### Costs (Cons)
1.  **Increased Storage:** We store approximately 5-10x more vectors than a naive approach.
2.  **Indexing Latency:** Initial ingestion takes longer as we must process each receipt through multiple parsing passes.
3.  **Cost:** Higher Pinecone/Vector DB costs due to increased vector count (mitigated by using Serverless indices).

---

## 4. Chunk Breakdown

### Receipt Summary
- **Content:** "Receipt from [Merchant] on [Date]. Total: $[Amount]. Items: [List]"
- **Use Case:** "How much did I spend in total last month?"

### Item Detail
- **Content:** "Item: [Name]. Price: $[Price]. Category: [Category]. Purchased at: [Merchant] on [Date]"
- **Use Case:** "Find the receipt where I bought a Laptop."

### Category Group
- **Content:** "Category: [Category]. Store: [Merchant]. Total: $[Subtotal]. Items: [Item List]"
- **Use Case:** "How much did I spend on Groceries at Whole Foods?"

### Merchant Info
- **Content:** "Store: [Merchant]. Address: [Address]. Total Spent: $[Total]. [High-Value Items]"
- **Use Case:** "What's the phone number for the CVS I visited in SF?"

### Payment Method
- **Content:** "Payment: [Method]. Store: [Merchant]. Date: [Date]. Tip: $[Tip]"
- **Use Case:** "List all transactions paid with Apple Pay."
