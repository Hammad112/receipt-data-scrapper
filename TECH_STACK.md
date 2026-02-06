# Receipt Intelligence: Technology Stack

This document outlines the core technologies and architectural components used in the Receipt Intelligence System.

## 1. Backend & Language
- **Python 3.10+**: Core programming language.
- **SQLAlchemy/Pydantic**: Data modeling and validation (if applicable).
- **Decimal**: Precise financial calculations.

## 2. Artificial Intelligence & RAG
- **OpenAI GPT-3.5/GPT-4**: Used for:
  - Natural Language Query Parsing (`src/query/query_parser.py`)
  - Grounded Answer Generation (`src/query/answer_generator.py`)
  - Semantic Verification & Auditing.
- **Pinecone (Serverless)**: High-performance vector database for RAG retrieval.
- **OpenAI Text-Embedding-3-Small**: State-of-the-art vector embeddings for semantic search.

## 3. Data Ingestion & Processing
- **OCR/Receipt Parser**: Specialized regex and LLM-based parsing of raw receipt text.
- **Multi-View Chunker**: Custom logic (`src/chunking/receipt_chunker.py`) that generates 5 distinct perspectives for every receipt:
  - Summary
  - Item Detail
  - Merchant Info
  - Category Group
  - Payment Method

## 4. Search & Retrieval
- **Hybrid Search**: Combines semantic vector similarity with deterministic metadata filtering (Merchant, Date, Amount, Category).
- **Deterministic Auditing**: A secondary verification layer that cross-references LLM-generated totals against source metadata for 100% financial fidelity.

## 5. Implementation Standards
- **Modular RAG Architecture**: Decouples parsing, retrieval, and generation for industrial-grade maintenance.
- **ISO 8601 Compliance**: Standardized temporal handling.
- **DRY API Principles**: Shared models and utility layers.
