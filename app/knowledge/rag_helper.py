"""
app/knowledge/rag_helper.py — RAG Search Helper
=================================================
WHY THIS EXISTS:
  rag_store.py handles the low-level vector mechanics.
  rag_helper.py provides a clean, high-level interface that agents use.
  
  This separation means:
  - Agents don't need to know about FAISS internals
  - We can swap FAISS for Pinecone later without touching agent code
  - Easy to test each layer independently

PATTERN USED: Facade Pattern
  We wrap the complex RAGStore behind a simple interface.
"""

from typing import List, Optional
from app.knowledge.rag_store import RAGStore
from app.knowledge.loader import LoadedDocument
from config import settings


class RAGHelper:
    """
    High-level interface for RAG operations.
    
    Agents use this to retrieve relevant context — they don't 
    interact with RAGStore or FAISS directly.
    
    USAGE:
        helper = RAGHelper(store)
        context = helper.get_context("I want to improve my diet")
        # context is a formatted string ready to inject into a prompt
    """

    def __init__(self, store: RAGStore):
        self.store = store

    def get_context(
        self,
        query: str,
        k: Optional[int] = None,
        min_score: float = 0.1,
        domain_filter: Optional[str] = None
    ) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: The user's question or input
            k: Number of chunks to retrieve (default from settings)
            min_score: Minimum similarity score (0-1). 
                       Below this = probably not relevant, skip it.
            domain_filter: Optional filename filter (e.g., "health" 
                          to only use health-related documents)
        
        Returns:
            Formatted string of relevant context, ready for prompt injection.
            Returns empty string if no relevant docs found.
        
        EXAMPLE OUTPUT:
            [Source: my_health_goals.txt]
            I want to run a 5K by December. I have bad knees.
            
            [Source: daily_routine.txt]  
            I wake up at 7am. I skip breakfast often.
        """
        if not self.store.is_ready:
            return ""  # No docs indexed — agent will work without context

        k = k or settings.TOP_K_RESULTS
        results = self.store.search(query, k=k)
        
        # Filter by minimum score
        results = [(chunk, score) for chunk, score in results if score >= min_score]
        
        # Filter by domain if specified
        if domain_filter:
            results = [
                (chunk, score) for chunk, score in results
                if domain_filter.lower() in chunk.source.lower()
            ]
        
        if not results:
            return ""
        
        # Format results into a readable context block
        context_parts = []
        for chunk, score in results:
            filename = chunk.metadata.get("filename", chunk.source)
            context_parts.append(
                f"[Source: {filename} | Relevance: {score:.2f}]\n{chunk.content}"
            )
        
        return "\n\n".join(context_parts)

    def get_raw_results(self, query: str, k: int = 5):
        """
        Returns raw (chunk, score) tuples for advanced use.
        Useful for debugging or displaying sources to the user.
        """
        return self.store.search(query, k=k)

    def has_relevant_context(self, query: str, min_score: float = 0.4) -> bool:
        """
        Quick check: does the store have anything useful for this query?
        Agents use this to decide whether to use RAG at all.
        """
        if not self.store.is_ready:
            return False
        results = self.store.search(query, k=1)
        return bool(results) and results[0][1] >= min_score

    def add_user_knowledge(self, text: str) -> None:
        """
        Add knowledge provided by the user during conversation.
        
        EXAMPLE USE CASE:
        User: "By the way, I'm lactose intolerant"
        → We call add_user_knowledge("User is lactose intolerant")
        → Future health questions will consider this
        """
        self.store.add_texts([text], source="user_provided")
        print(f"💡 Added to knowledge base: {text[:80]}...")
