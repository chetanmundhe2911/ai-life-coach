"""
tests/test_components.py — Unit Tests
=======================================
WHY TESTS MATTER:
  Tests let you change code confidently.
  Without tests, every change risks breaking something invisibly.
  
  We test each component in ISOLATION:
  - Knowledge components (no real API calls)
  - Orchestrator routing (with mock LLM)
  - Session state management

HOW TO RUN:
  pytest tests/ -v
  
  Or with coverage:
  pytest tests/ -v --cov=app

MOCKING:
  We use unittest.mock to avoid real OpenAI API calls in tests.
  This makes tests fast (no network) and free (no API cost).
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Session State Tests ───────────────────────────────────────────

class TestSessionState:
    """Tests for conversation memory management"""
    
    def setup_method(self):
        """Create a fresh SessionState before each test"""
        from app.state.session_state import SessionState
        self.state = SessionState()

    def test_add_user_message(self):
        """Messages are recorded correctly"""
        self.state.add_user_message("Hello")
        assert self.state.message_count == 1
        assert self.state.last_user_message == "Hello"

    def test_add_assistant_message(self):
        """Assistant messages are recorded with agent name"""
        self.state.add_user_message("Hi")
        self.state.add_assistant_message("Hello!", agent="health_wellness")
        assert self.state.message_count == 2

    def test_clear_keeps_profile(self):
        """Clearing conversation preserves user profile"""
        self.state.update_user_profile("name", "Alex")
        self.state.add_user_message("test")
        self.state.clear()
        
        assert self.state.message_count == 0
        assert self.state.user_profile["name"] == "Alex"

    def test_full_reset_clears_everything(self):
        """Full reset removes both messages and profile"""
        self.state.update_user_profile("name", "Alex")
        self.state.add_user_message("test")
        self.state.full_reset()
        
        assert self.state.message_count == 0
        assert len(self.state.user_profile) == 0

    def test_get_messages_for_api_format(self):
        """API messages include system prompt and history"""
        self.state.add_user_message("Hello")
        messages = self.state.get_messages_for_api("You are a coach")
        
        # First message should always be the system prompt
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a coach"
        
        # Second message should be the user message
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_sliding_window(self):
        """History is limited to MAX_CONVERSATION_HISTORY messages"""
        # Add 20 messages (more than the default limit of 10)
        for i in range(20):
            self.state.add_user_message(f"Message {i}")
        
        messages = self.state.get_messages_for_api("system", max_messages=5)
        
        # Should have: 1 system + 5 recent messages = 6 total
        assert len(messages) == 6

    def test_user_profile_text(self):
        """Profile is formatted correctly as text"""
        self.state.update_user_profile("age", "29")
        self.state.update_user_profile("occupation", "engineer")
        
        profile_text = self.state.get_user_profile_text()
        assert "age: 29" in profile_text
        assert "occupation: engineer" in profile_text

    def test_empty_profile_text(self):
        """Empty profile returns empty string"""
        assert self.state.get_user_profile_text() == ""


# ── Document Loader Tests ─────────────────────────────────────────

class TestDocumentLoader:
    """Tests for document loading and chunking"""

    def test_load_text_directly(self, tmp_path):
        """Text can be loaded directly without a file"""
        from app.knowledge.loader import DocumentLoader
        
        loader = DocumentLoader()
        chunks = loader.load_text_directly("This is a test document with some content.", "test")
        
        assert len(chunks) > 0
        assert chunks[0].source == "test"
        assert "test document" in chunks[0].content

    def test_chunk_index(self):
        """Chunks are numbered sequentially"""
        from app.knowledge.loader import DocumentLoader
        
        loader = DocumentLoader()
        # Create text long enough to produce multiple chunks
        long_text = "This is sentence number {i}. " * 100
        chunks = loader.load_text_directly(long_text, "test")
        
        if len(chunks) > 1:
            assert chunks[0].chunk_index == 0
            assert chunks[1].chunk_index == 1

    def test_no_docs_returns_empty(self, tmp_path):
        """Empty directory returns empty list"""
        from app.knowledge.utils import get_supported_files
        from pathlib import Path
        
        files = get_supported_files(Path(tmp_path))
        assert files == []

    def test_supported_file_types(self, tmp_path):
        """Only txt, md, pdf files are returned"""
        from app.knowledge.utils import get_supported_files
        from pathlib import Path
        
        # Create test files
        (tmp_path / "test.txt").write_text("hello")
        (tmp_path / "test.md").write_text("hello")
        (tmp_path / "test.jpg").write_bytes(b"fake image")
        (tmp_path / "test.docx").write_bytes(b"fake docx")
        
        files = get_supported_files(Path(tmp_path))
        filenames = [f.name for f in files]
        
        assert "test.txt" in filenames
        assert "test.md" in filenames
        assert "test.jpg" not in filenames  # Not supported
        assert "test.docx" not in filenames  # Not supported


# ── Utility Tests ─────────────────────────────────────────────────

class TestUtils:
    """Tests for utility functions"""

    def test_file_hash_consistency(self, tmp_path):
        """Same file content produces same hash"""
        from app.knowledge.utils import get_file_hash
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")
        
        hash1 = get_file_hash(str(test_file))
        hash2 = get_file_hash(str(test_file))
        assert hash1 == hash2

    def test_file_hash_changes_with_content(self, tmp_path):
        """Different content produces different hash"""
        from app.knowledge.utils import get_file_hash
        
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content A")
        file2.write_text("Content B")
        
        assert get_file_hash(str(file1)) != get_file_hash(str(file2))

    def test_truncate_text_short(self):
        """Short text is not truncated"""
        from app.knowledge.utils import truncate_text
        
        text = "Short text"
        assert truncate_text(text, max_chars=100) == text

    def test_truncate_text_long(self):
        """Long text is truncated with ellipsis"""
        from app.knowledge.utils import truncate_text
        
        text = "A" * 500
        result = truncate_text(text, max_chars=100)
        assert len(result) <= 103  # 100 + "..."
        assert result.endswith("...")

    def test_token_estimate(self):
        """Token estimation is roughly correct"""
        from app.knowledge.utils import count_tokens_estimate
        
        # "Hello world" is ~3 tokens, our estimate gives len//4 = 2
        # Just check it returns a positive number
        count = count_tokens_estimate("Hello world this is a test sentence")
        assert count > 0


# ── Orchestrator Domain Classification Tests ──────────────────────

class TestOrchestratorKeywords:
    """Test the keyword fallback classifier without API calls"""

    def setup_method(self):
        """Set up orchestrator with mocked dependencies"""
        from app.agents.orchestrator_agent import OrchestratorAgent, DOMAIN_KEYWORDS
        self.keywords = DOMAIN_KEYWORDS

    def test_health_keywords(self):
        """Health-related messages route to health domain"""
        from app.agents.orchestrator_agent import DOMAIN_KEYWORDS
        
        text = "I want to exercise and improve my diet"
        health_score = sum(1 for kw in DOMAIN_KEYWORDS["health_wellness"] if kw in text.lower())
        assert health_score > 0

    def test_career_keywords(self):
        """Career-related messages route to career domain"""
        from app.agents.orchestrator_agent import DOMAIN_KEYWORDS
        
        text = "I'm thinking about a job change and career growth"
        career_score = sum(1 for kw in DOMAIN_KEYWORDS["career_profession"] if kw in text.lower())
        assert career_score > 0

    def test_all_domains_have_keywords(self):
        """Every domain has at least 3 keywords for fallback routing"""
        from app.agents.orchestrator_agent import DOMAIN_KEYWORDS, AGENT_REGISTRY
        
        for domain in AGENT_REGISTRY:
            assert domain in DOMAIN_KEYWORDS, f"{domain} missing from keyword dict"
            assert len(DOMAIN_KEYWORDS[domain]) >= 3, f"{domain} needs more keywords"


# ── RAG Helper Tests ──────────────────────────────────────────────

class TestRAGHelper:
    """Tests for RAG search interface"""

    def test_no_store_returns_empty(self):
        """Empty store returns empty context"""
        from app.knowledge.rag_store import RAGStore
        from app.knowledge.rag_helper import RAGHelper
        
        store = RAGStore.__new__(RAGStore)  # Skip __init__
        store.index = None
        store.chunks = []
        
        helper = RAGHelper(store)
        
        # Should return empty string when no documents
        assert helper.get_context("anything") == ""
        assert helper.has_relevant_context("anything") == False

    def test_format_context(self):
        """Context is formatted with source attribution"""
        from app.knowledge.rag_store import RAGStore
        from app.knowledge.rag_helper import RAGHelper
        from app.knowledge.loader import LoadedDocument
        
        # Mock store
        store = MagicMock(spec=RAGStore)
        store.is_ready = True
        
        mock_chunk = LoadedDocument(
            content="I want to run a 5K",
            source="health_goals.txt",
            chunk_index=0,
            file_hash="abc123",
            metadata={"filename": "health_goals.txt"}
        )
        store.search.return_value = [(mock_chunk, 0.85)]
        
        helper = RAGHelper(store)
        context = helper.get_context("running goals")
        
        assert "health_goals.txt" in context
        assert "5K" in context
        assert "0.85" in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
