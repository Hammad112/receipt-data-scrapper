"""
Vector database management using Pinecone for receipt RAG.

This module handles:
- Pinecone index initialization and management.
- Embedding generation using OpenAI's state-of-the-art models.
- Hybrid search execution (combining vector similarity and metadata filters).
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

try:
    from ..models import ReceiptChunk
    from ..utils.logging_config import logger
except ImportError:
    from models import ReceiptChunk
    from utils.logging_config import logger


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
        
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self._get_or_create_index()
        
        logger.info(f"VectorManager initialized with index: {self.index_name}")

    def _get_or_create_index(self):
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
                return self.pc.Index(self.index_name)
        except Exception as e:
            logger.error(f"Critical error initializing Pinecone: {e}")
            raise

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
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                vectors = []
                for chunk in batch:
                    embedding = self.generate_embedding(chunk.content)
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
                'dimension': stats['dimension'],
                'index_fullness': stats['index_fullness'],
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
            self.pc.delete_index(self.index_name)
            
            while self.index_name in [idx.name for idx in self.pc.list_indexes()]:
                logger.debug("Waiting for index deletion...")
                time.sleep(2)
            
            logger.info(f"Recreating index: {self.index_name}")
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
            logger.info("Index successfully rebuilt.")
        except Exception as e:
            logger.error(f"Rebuild failed: {e}")
            raise

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

