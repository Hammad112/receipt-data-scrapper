"""
Vector database management using Pinecone for receipt RAG.

This module handles:
- Pinecone index initialization and management.
- Embedding generation using OpenAI's state-of-the-art models.
- Hybrid search execution (combining vector similarity and metadata filters).
"""

import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Absolute imports for industrial stability
from ..utils.logging_config import logger, setup_logging
from ..models import Receipt, ReceiptChunk

try:
    from pinecone import Pinecone, ServerlessSpec
    _PINECONE_SDK = "pinecone"
except Exception:
    Pinecone = None
    ServerlessSpec = None
    import pinecone
    _PINECONE_SDK = "pinecone-client"


class VectorManager:
    """
    Orchestrates vector database operations with Pinecone and OpenAI.
    
    Provides high-level methods for indexing receipt chunks and performing 
    complex hybrid searches with metadata filtering.
    """
    
    def __init__(self):
        """
        Initializes the vector manager, loading environment variables 
        and establishing connections to Pinecone and OpenAI.
        """
        load_dotenv()
        
        # Initialize OpenAI
        self.openai_client = OpenAI()
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
        
        # Pinecone Config
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'receipt-index')

        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required")

        if _PINECONE_SDK == "pinecone":
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.index = self._get_or_create_index_pinecone()
        else:
            self.pc = None
            pinecone_env = os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENV")
            if not pinecone_env:
                raise ValueError("PINECONE_ENVIRONMENT is required for pinecone-client")
            pinecone.init(api_key=self.pinecone_api_key, environment=pinecone_env)
            self.index = self._get_or_create_index_pinecone_client()
        
        logger.info(f"VectorManager initialized with index: {self.index_name}")

    def _get_or_create_index_pinecone(self):
        """
        Retrieves the existing Pinecone index or creates a new one if it doesn't exist.
        
        Returns:
            pinecone.Index: The active index instance.
        """
        try:
            if self.index_name in self.pc.list_indexes().names():
                logger.debug(f"Connecting to existing Pinecone index: {self.index_name}")
                return self.pc.Index(self.index_name)
            else:
                logger.warning(f"Index {self.index_name} not found. Creating new index...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                logger.info(f"Successfully created Pinecone index: {self.index_name}")
                index = self.pc.Index(self.index_name)
                self._wait_for_index_ready(index)
                return index
        except Exception as e:
            logger.error(f"Critical error initializing Pinecone: {e}")
            raise

    def _get_or_create_index_pinecone_client(self):
        try:
            existing = pinecone.list_indexes()
            if self.index_name in existing:
                logger.debug(f"Connecting to existing Pinecone index: {self.index_name}")
                return pinecone.Index(self.index_name)

            logger.warning(f"Index {self.index_name} not found. Creating new index...")
            pod_type = os.getenv("PINECONE_POD_TYPE", "s1.x1")
            pinecone.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                pods=1,
                replicas=1,
                pod_type=pod_type,
            )
            index = pinecone.Index(self.index_name)
            self._wait_for_index_ready(index)
            return index
        except Exception as e:
            logger.error(f"Critical error initializing Pinecone (pinecone-client): {e}")
            raise

    def _wait_for_index_ready(self, index, timeout_seconds: int = 180):
        start = time.time()
        while time.time() - start < timeout_seconds:
            try:
                _ = index.describe_index_stats()
                return
            except Exception:
                time.sleep(2)
        raise TimeoutError(f"Timed out waiting for Pinecone index to be ready: {self.index_name}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates a vector embedding for the given text using OpenAI.
        
        Args:
            text: The text to be embedded.
            
        Returns:
            List[float]: The resulting embedding vector.
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def index_chunks(self, chunks: List[ReceiptChunk], batch_size: int = 50) -> int:
        """
        Indexes a list of receipt chunks in the vector database.
        
        Processes chunks in batches to optimize performance and handle API limits.
        
        Args:
            chunks: List of ReceiptChunk objects to index.
            batch_size: Number of chunks per upsert batch.
            
        Returns:
            int: Number of chunks successfully indexed.
        """
        if not chunks:
            return 0
        
        indexed_count = 0
        logger.info(f"Starting batch indexing: {len(chunks)} chunks, batch size {batch_size}")
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                batch_num = i // batch_size + 1
                logger.info(f"Indexing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
                vectors = []
                embeddings = self.generate_embeddings([chunk.content for chunk in batch])
                for chunk, embedding in zip(batch, embeddings):
                    vectors.append({
                        'id': chunk.chunk_id,
                        'values': embedding,
                        'metadata': {
                            'receipt_id': chunk.receipt_id,
                            'chunk_type': chunk.chunk_type,
                            'content': chunk.content[:1000],
                            **chunk.metadata
                        }
                    })
                
                self.index.upsert(vectors=vectors)
                indexed_count += len(batch)
                logger.debug(f"Indexed batch {i//batch_size + 1}/{len(chunks)//batch_size + 1}")
                
            except Exception as e:
                if "terminated" in str(e).lower():
                    raise
                logger.error(f"Error indexing batch {i//batch_size + 1}: {e}")
                continue
        
        logger.info(f"Indexing complete. Successfully stored {indexed_count}/{len(chunks)} vectors.")
        return indexed_count

    def hybrid_search(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes a hybrid search combining semantic similarity and metadata filters.
        
        Args:
            query: Natural language search string.
            filters: Pinecone-compatible metadata filters.
            top_k: Number of results to retrieve.
            
        Returns:
            List[Dict[str, Any]]: List of matching results with scores and metadata.
        """
        try:
            logger.debug(f"Executing search: query='{query}', filters={filters}")
            query_embedding = self.generate_embedding(query)
            
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filters
            )
            
            results = [{
                'id': m['id'],
                'score': m['score'],
                'metadata': m['metadata']
            } for m in search_results['matches']]
            
            logger.info(f"Search found {len(results)} matches for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Retrieves real-time statistics about the current Pinecone index.
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vector_count': stats['total_vector_count'],
                'dimension': stats.get('dimension', 1536) if isinstance(stats, dict) else 1536,
                'index_fullness': stats.get('index_fullness', 0.0) if isinstance(stats, dict) else 0.0,
                'namespaces': stats.get('namespaces', {}),
            }
        except Exception as e:
            logger.warning(f"Failed to fetch index stats: {e}")
            return {'total_vector_count': 0, 'dimension': 1536}

    def rebuild_index(self):
        """
        Destructive operation: Deletes and recreates the index.
        """
        try:
            logger.warning(f"DELETING INDEX: {self.index_name}")
            if _PINECONE_SDK == "pinecone":
                self.pc.delete_index(self.index_name)
                while self.index_name in self.pc.list_indexes().names():
                    time.sleep(2)
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                self.index = self.pc.Index(self.index_name)
                self._wait_for_index_ready(self.index)
            else:
                pinecone.delete_index(self.index_name)
                while self.index_name in pinecone.list_indexes():
                    time.sleep(2)
                pod_type = os.getenv("PINECONE_POD_TYPE", "s1.x1")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    pods=1,
                    replicas=1,
                    pod_type=pod_type,
                )
                self.index = pinecone.Index(self.index_name)
                self._wait_for_index_ready(self.index)
        except Exception as e:
            logger.error(f"Rebuild failed: {e}")
            raise

    def clear_index(self, timeout_seconds: int = 180):
        try:
            self.index.delete(delete_all=True)
        except Exception as e:
            message = str(e).lower()
            if "namespace not found" in message:
                return
            if "terminated" in message:
                logger.warning(f"Index terminated; rebuilding instead: {e}")
                self.rebuild_index()
                return
            logger.warning(f"Failed to clear index with delete_all: {e}")
            raise

        start = time.time()
        while time.time() - start < timeout_seconds:
            stats = self.get_index_stats()
            if stats.get('total_vector_count', 0) == 0:
                return
            time.sleep(2)
        raise TimeoutError(f"Timed out waiting for index to clear: {self.index_name}")

    def delete_by_receipt_id(self, receipt_id: str) -> bool:
        """
        Deletes all vector data associated with a specific receipt.
        """
        try:
            self.index.delete(filter={'receipt_id': receipt_id})
            logger.info(f"Deleted vectors for receipt_id: {receipt_id}")
            return True
        except Exception as e:
            logger.error(f"Delete failed for receipt_id {receipt_id}: {e}")
            return False

    def get_latest_transaction_date(self) -> Optional[datetime]:
        """
        Get the most recent transaction date from indexed receipts.
        
        Returns:
            datetime of latest receipt, or None if index is empty
        """
        try:
            # Search with no filters to get any recent receipt
            # Use a dummy vector (zeros) since we just want metadata
            dummy_vector = [0.0] * 1536
            
            results = self.index.query(
                vector=dummy_vector,
                top_k=100,
                include_metadata=True,
                filter={'chunk_type': 'receipt_summary'}
            )
            
            max_ts = 0
            for match in results.get('matches', []):
                meta = match.get('metadata', {})
                ts = meta.get('transaction_ts', 0)
                if ts and ts > max_ts:
                    max_ts = ts
            
            if max_ts > 0:
                return datetime.fromtimestamp(max_ts, tz=timezone.utc)
            
            return None
        except Exception as e:
            logger.warning(f"Failed to get latest transaction date: {e}")
            return None
