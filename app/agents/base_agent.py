"""
app/agents/base_agent.py — Base Agent Class
============================================
WHY THIS EXISTS:
  All domain agents (health, career, etc.) share common behavior:
  - They all call GPT-4
  - They all retrieve RAG context
  - They all use session state
  
  Instead of duplicating this code in every agent, we put it here.
  Each domain agent inherits BaseAgent and only overrides what's different.
  
PATTERN: Template Method Pattern
  BaseAgent defines the STRUCTURE of how an agent works.
  Subclasses fill in domain-specific parts (system_prompt, etc.)
  
  Workflow (defined here, not in subclasses):
  1. get_context(user_input)  ← retrieves RAG context
  2. build_prompt(context)    ← constructs the system prompt
  3. call_llm(messages)       ← calls GPT-4
  4. post_process(response)   ← any cleanup
  
  Step 2 (build_prompt) is overridden by subclasses to inject
  domain-specific instructions.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from openai import OpenAI

from config import settings
from app.knowledge.knowledge_base import KnowledgeBase
from app.state.session_state import SessionState


class BaseAgent(ABC):
    """
    Abstract base class for all domain agents.
    
    SUBCLASS USAGE:
        class HealthAgent(BaseAgent):
            @property
            def domain(self) -> str:
                return "health_wellness"
            
            def get_system_prompt(self, context: str, user_profile: str) -> str:
                return f"You are a health coach. Context: {context}"
    """

    def __init__(self, knowledge_base: KnowledgeBase, session_state: SessionState):
        self.kb = knowledge_base
        self.state = session_state
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # ── Abstract Methods (subclasses MUST implement) ──────────────

    @property
    @abstractmethod
    def domain(self) -> str:
        """
        The domain this agent handles.
        Used for RAG filtering and logging.
        
        Examples: "health_wellness", "career_profession"
        """
        pass

    @abstractmethod
    def get_system_prompt(self, context: str, user_profile: str) -> str:
        """
        Build the system prompt for this agent.
        
        Args:
            context: Retrieved RAG context (from user's documents)
            user_profile: Known facts about the user
        
        Returns:
            The complete system prompt string
        """
        pass

    # ── Core Method (the main entry point) ───────────────────────

    def respond(self, user_input: str) -> str:
        """
        Generate a response to the user's input.
        
        This is called by the orchestrator after routing to this agent.
        
        FLOW:
        1. Retrieve relevant context from knowledge base
        2. Build system prompt with context + user profile
        3. Get conversation history from session state
        4. Call GPT-4
        5. Save response to session state
        6. Return response text
        """
        # Step 1: Retrieve relevant context
        context = self.kb.query(
            question=user_input,
            domain=self.domain
        )

        # Step 2: Build system prompt
        user_profile = self.state.get_user_profile_text()
        system_prompt = self.get_system_prompt(context, user_profile)

        # Step 3: Get full message history for the API call
        messages = self.state.get_messages_for_api(system_prompt)

        # Step 4: Call GPT-4
        response_text = self._call_llm(messages)

        # Step 5: Save to session state (so future turns remember this)
        self.state.add_assistant_message(response_text, agent=self.domain)

        return response_text

    # ── Internal Methods ──────────────────────────────────────────

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Make the actual API call to OpenAI.
        
        WHY SEPARATE METHOD? 
        - Easier to mock in tests
        - Could add retry logic, rate limiting here
        - Could swap for a different LLM without touching agent logic
        """
        try:
            response = self.client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=messages,
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            # Don't crash the whole app on API errors
            print(f"❌ LLM API error in {self.domain}: {e}")
            return (
                "I'm having trouble connecting right now. "
                "Please check your OpenAI API key and try again."
            )

    def _format_context_block(self, context: str) -> str:
        """
        Wrap retrieved context in a clear block for the prompt.
        
        WHY: Makes it obvious to GPT-4 where its own knowledge ends
        and the user's personal information begins.
        """
        if not context:
            return ""
        return f"""
--- PERSONAL CONTEXT FROM USER'S DOCUMENTS ---
{context}
--- END OF PERSONAL CONTEXT ---
"""

    @property
    def name(self) -> str:
        """Human-readable agent name for display"""
        return self.domain.replace("_", " ").title()
