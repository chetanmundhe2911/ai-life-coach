"""
app/knowledge/rag_store.py — FAISS Vector Store
=================================================
WHY THIS EXISTS:
  This is the HEART of RAG. It:
  1. Takes text chunks and converts them to vector embeddings
  2. Stores those vectors in FAISS (a fast similarity search library)
  3. Given a query, finds the most semantically similar chunks

CONCEPT — WHAT IS AN EMBEDDING?
  An embedding is a list of ~1500 numbers that represents the MEANING
  of a piece of text. Two sentences with similar meanings will have
  similar number lists (high "cosine similarity").
  
  Example:
  "I love running" → [0.12, -0.45, 0.87, ...]
  "I enjoy jogging" → [0.11, -0.43, 0.85, ...]  ← very similar!
  "The sky is blue"  → [0.92, 0.31, -0.12, ...]  ← very different

CONCEPT — WHAT IS FAISS?
  FAISS (Facebook AI Similarity Search) is a library that:
  - Stores millions of vectors efficiently
  - Finds the K most similar vectors to a query vector
  - Works entirely on disk/RAM (no server needed!)
  
  Think of it as: "given this question's meaning, find the 3 most
  similar document chunks."

CONCEPT — WHY CACHE?
  Calling OpenAI's embedding API costs money and takes time.
  We save the FAISS index to disk so we only re-embed when docs change.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
from openai import OpenAI

from config import settings
from app.knowledge.loader import LoadedDocument


class RAGStore:
    """
    Manages the FAISS vector store for semantic search.
    
    LIFECYCLE:
    1. build(chunks) — embed all chunks and build FAISS index
    2. save() — persist index to disk
    3. load() — restore index from disk (skip re-embedding)
    4. search(query, k) — find top-k similar chunks
    
    USAGE:
        store = RAGStore()
        if not store.load():                    # Try loading cached index
            chunks = DocumentLoader().load_all()
            store.build(chunks)                 # Build fresh index
            store.save()                        # Cache for next time
        
        results = store.search("how to be healthier", k=3)
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # FAISS index — will be populated by build() or load()
        self.index: Optional[faiss.IndexFlatIP] = None
        
        # We store the original chunks alongside the index
        # because FAISS only stores vectors, not the original text
        self.chunks: List[LoadedDocument] = []
        
        # Paths for saving/loading
        self.index_path = settings.faiss_index_path
        self.chunks_path = settings.cache_path / "chunks.pkl"

    # ── Embedding ─────────────────────────────────────────────────

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of text strings to embedding vectors via OpenAI API.
        
        WHY BATCH? Sending all texts in one API call is faster and cheaper
        than sending them one-by-one.
        
        Returns: numpy array of shape (len(texts), embedding_dim)
        """
        print(f"   🔢 Embedding {len(texts)} chunks via OpenAI API...")
        
        # OpenAI can handle up to 2048 texts per request
        # We batch in groups of 100 to be safe
        all_embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=batch
            )
            # Extract the embedding vectors from the response
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            if len(texts) > batch_size:
                print(f"   ... batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Convert to numpy float32 (required by FAISS)
        return np.array(all_embeddings, dtype=np.float32)

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (1, embedding_dim)"""
        response = self.client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=[query]
        )
        vector = np.array([response.data[0].embedding], dtype=np.float32)
        return vector

    # ── Build ─────────────────────────────────────────────────────

    def build(self, chunks: List[LoadedDocument]) -> None:
        """
        Build the FAISS index from document chunks.
        This calls OpenAI's embedding API — costs money but is fast.
        
        FAISS Index Type: IndexFlatIP
          - "Flat" = exact search (no approximation)
          - "IP" = Inner Product (equivalent to cosine similarity
                  when vectors are normalized)
        """
        if not chunks:
            print("⚠️  No chunks to embed. Add documents to /docs first.")
            return

        print(f"🏗️  Building FAISS index from {len(chunks)} chunks...")
        
        # Get all chunk texts
        texts = [chunk.content for chunk in chunks]
        
        # Convert texts → vectors
        embeddings = self._embed_texts(texts)
        
        # Normalize vectors for cosine similarity
        # (FAISS IndexFlatIP computes dot product; cosine sim = dot product of unit vectors)
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        # embeddings.shape[1] = the embedding dimension (e.g., 1536 for text-embedding-3-small)
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Add all vectors to the index
        self.index.add(embeddings)
        
        # Store the original chunks (FAISS only stores vectors, not text)
        self.chunks = chunks
        
        print(f"✅ FAISS index built: {self.index.ntotal} vectors, dim={embedding_dim}")

    # ── Persist ───────────────────────────────────────────────────

    def save(self) -> None:
        """
        Save FAISS index and chunks to disk.
        Next time the app starts, we call load() instead of rebuilding.
        """
        if self.index is None:
            print("⚠️  Nothing to save — index is empty")
            return

        # Save FAISS binary index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save chunks as pickle (Python's native serialization)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        
        print(f"💾 Saved FAISS index → {self.index_path}")

    def load(self) -> bool:
        """
        Load FAISS index from disk.
        Returns True if successful, False if no cache exists.
        
        CALL THIS at startup before build() to avoid re-embedding.
        """
        if not self.index_path.exists() or not self.chunks_path.exists():
            return False

        try:
            self.index = faiss.read_index(str(self.index_path))
            
            with open(self.chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            
            print(f"📦 Loaded FAISS index from cache: {self.index.ntotal} vectors")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}. Will rebuild.")
            return False

    # ── Search ────────────────────────────────────────────────────

    def search(self, query: str, k: Optional[int] = None) -> List[Tuple[LoadedDocument, float]]:
        """
        Find the top-k most semantically similar chunks to the query.
        
        Returns: List of (chunk, similarity_score) tuples
                 sorted by similarity (highest first)
        
        USAGE:
            results = store.search("How can I improve my sleep?", k=3)
            for chunk, score in results:
                print(f"Score: {score:.3f} | {chunk.content[:100]}")
        """
        if self.index is None or self.index.ntotal == 0:
            return []  # No documents loaded

        k = k or settings.TOP_K_RESULTS
        k = min(k, self.index.ntotal)  # Can't return more than we have
        
        # Embed the query
        query_vector = self._embed_query(query)
        faiss.normalize_L2(query_vector)  # Must normalize like we did during build
        
        # Search! Returns distances and indices arrays
        # distances shape: (1, k), indices shape: (1, k)
        distances, indices = self.index.search(query_vector, k)
        
        # Build results list
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            results.append((self.chunks[idx], float(dist)))
        
        return results

    def add_texts(self, texts: List[str], source: str = "runtime") -> None:
        """
        WHY: Add new knowledge at runtime without rebuilding the whole index.
        Useful for adding user-provided info during conversation.
        """
        from app.knowledge.loader import DocumentLoader
        loader = DocumentLoader()
        
        new_chunks = []
        for text in texts:
            chunks = loader.load_text_directly(text, source)
            new_chunks.extend(chunks)
        
        if not new_chunks:
            return
        
        # Embed new chunks
        new_texts = [c.content for c in new_chunks]
        new_embeddings = self._embed_texts(new_texts)
        faiss.normalize_L2(new_embeddings)
        
        # If no index exists yet, create one
        if self.index is None:
            self.index = faiss.IndexFlatIP(new_embeddings.shape[1])
        
        self.index.add(new_embeddings)
        self.chunks.extend(new_chunks)
        
        print(f"➕ Added {len(new_chunks)} chunks to index")

    @property
    def is_ready(self) -> bool:
        """Check if the store has any documents indexed"""
        return self.index is not None and self.index.ntotal > 0
