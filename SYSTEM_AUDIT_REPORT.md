# System Audit Report: Resolving RAG "Red Flags"

**Status**: 🟡 Remediation in Progress
**Date**: February 7, 2026

## 1. Executive Summary
This report identifies and provides solutions for core technical "red flags" in the Receipt Intelligence System. We are transitioning from a rule-based retrieval system to an industrial-grade **Hybrid RAG** architecture with semantic entity matching.

## 2. Red Flag Remediation Table

| Red Flag | Categorization | Current Status | Solution Strategy |
|:---|:---|:---|:---|
| **Vector-Only Search** | ❌ CRITICAL | Remediation | Enforcing mandatory metadata filtering in `QueryEngine` for Merchant/Date/Category. |
| **Merchant Hardcoding** | ❌ CRITICAL | Remediation | implementing contextual LLM-based NER and semantic merchant expansion. |
| **Chunking Logic Exp.** | ⚠️ PARTIAL | Resolved | Explicitly documented in `CHUNKING_STRATEGY.md`. |
| **Temporal Queries** | ⚠️ PARTIAL | Remediation | Expanding normalization to handle relative dates via `RECEIPT_REFERENCE_DATE`. |
| **Aggregation Support** | ❌ CRITICAL | Remediation | Formalizing `_perform_aggregation_audit` into a dedicated Analytics Layer. |

## 3. Detailed Solutions

### 3.1. Beyond Hardcoded Merchants
The system will now use a two-pass approach:
1.  **Contextual Extraction**: Extract "Suspected Merchant" from context (e.g., following "at" or "from").
2.  **Semantic Normalization**: Use LLM or vector matching against known database merchants to resolve "Walmart", "Walmart Supercenter", and "Wal-mart" to the same entity.

### 3.2. Hybrid Search Constraints
Instead of relying on embeddings to "understand" time, the `QueryParser` will calculate absolute timestamp ranges. These are passed as **hard filters** to Pinecone. Vector similarity is then used *only* to rank results within that specific time window.

### 3.3. Deterministic Aggregation
To prevent LLM "hallucinations" of sums, we implement an **Independent Audit Pattern**. The system retrieves raw metadata for all matching receipts and calculates the sum/average/count mathematically before passing the value to the LLM for natural language formatting.

---
## 4. Next Steps
1.  Implement semantic merchant matching.
2.  Refine temporal range calculations.
3.  Deploy independent audit pattern for all aggregation queries.
4.  Run comprehensive 50-query validation.
