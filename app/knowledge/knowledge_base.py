"""
app/knowledge/knowledge_base.py — Knowledge Base Orchestrator
=============================================================
WHY THIS EXISTS:
  This is the SINGLE entry point for all knowledge operations.
  It coordinates: DocumentLoader → RAGStore → RAGHelper
  
  Without this file, every agent would need to:
    1. Create a DocumentLoader
    2. Create a RAGStore  
    3. Load or build the index
    4. Create a RAGHelper
    
  With this file, agents just do:
    kb = KnowledgeBase.get_instance()
    context = kb.query("How do I improve my sleep?")

PATTERN: Singleton
  We use a class-level instance so the FAISS index is built ONCE
  and shared across all agents. Building it multiple times would
  be wasteful (API costs) and slow.
"""

from typing import Optional, List
from app.knowledge.loader import DocumentLoader
from app.knowledge.rag_store import RAGStore
from app.knowledge.rag_helper import RAGHelper
from app.knowledge.summarizer import Summarizer
from config import settings


class KnowledgeBase:
    """
    Singleton Knowledge Base that all agents share.
    
    LIFECYCLE:
    1. First call to get_instance() → builds/loads the FAISS index
    2. Subsequent calls → returns the same instance (no rebuilding)
    
    USAGE:
        # In any agent:
        kb = KnowledgeBase.get_instance()
        context = kb.query("tell me about my health goals")
    """
    
    _instance: Optional["KnowledgeBase"] = None

    def __init__(self):
        self.loader = DocumentLoader()
        self.store = RAGStore()
        self.helper = RAGHelper(self.store)
        self.summarizer = Summarizer()
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "KnowledgeBase":
        """
        Returns the shared KnowledgeBase instance.
        Initializes it on first call.
        
        WHY classmethod? So we can call KnowledgeBase.get_instance()
        without needing an existing instance.
        """
        if cls._instance is None:
            cls._instance = KnowledgeBase()
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """
        Build or load the knowledge base.
        Called once when the singleton is first created.
        """
        print("\n📚 Initializing Knowledge Base...")
        
        # Try loading cached FAISS index first (faster, no API cost)
        if self.store.load():
            print("   ✓ Loaded from cache — no re-embedding needed")
        else:
            # No cache — load docs and build fresh index
            print("   ℹ️  No cache found — building fresh index...")
            chunks = self.loader.load_all()
            
            if chunks:
                self.store.build(chunks)
                self.store.save()  # Cache for next time
            else:
                print("   ℹ️  Running without personal documents")
                print("   💡 Add .txt/.pdf/.md files to /docs for personalized coaching")
        
        self._initialized = True
        print(f"   ✓ Knowledge Base ready ({self.store.index.ntotal if self.store.is_ready else 0} vectors)\n")

    def query(
        self,
        question: str,
        domain: Optional[str] = None,
        summarize: bool = False
    ) -> str:
        """
        Main method: get relevant context for a question.
        
        Args:
            question: The user's input or question
            domain: Optional domain filter (e.g., "health", "career")
            summarize: If True, summarize long context (saves tokens)
        
        Returns:
            Formatted context string, or "" if nothing relevant found
        """
        context = self.helper.get_context(
            query=question,
            domain_filter=domain
        )
        
        if summarize and context:
            context = self.summarizer.summarize(context, topic=question)
        
        return context

    def add_knowledge(self, text: str) -> None:
        """Add new knowledge mid-conversation"""
        self.helper.add_user_knowledge(text)

    def rebuild(self) -> None:
        """
        Force a full rebuild of the index.
        Use when you've added new documents to /docs.
        """
        print("🔄 Rebuilding Knowledge Base...")
        chunks = self.loader.load_all()
        self.store.build(chunks)
        self.store.save()
        print("✅ Rebuild complete")

    @property
    def has_documents(self) -> bool:
        """Check if any documents are indexed"""
        return self.store.is_ready
