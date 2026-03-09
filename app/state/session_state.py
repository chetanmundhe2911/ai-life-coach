"""
app/state/session_state.py — Conversation Memory
==================================================
WHY THIS EXISTS:
  GPT-4 is stateless — it remembers NOTHING between API calls.
  We need to manually send the conversation history every time.
  
  This module manages that history so the agent can say things like:
  "As you mentioned earlier, you want to lose 10kg..."
  
CONCEPT — MESSAGE ROLES:
  OpenAI's chat API uses three roles:
  - "system"    : Instructions for GPT (never shown to user)
  - "user"      : What the human said
  - "assistant" : What GPT replied
  
  We send the full history like:
  [
    {"role": "system", "content": "You are Aria, a life coach..."},
    {"role": "user", "content": "I want to sleep better"},
    {"role": "assistant", "content": "Let's work on your sleep hygiene..."},
    {"role": "user", "content": "I usually go to bed at 2am"},
  ]
  
  GPT-4 reads all of this and responds in context.

CONCEPT — SLIDING WINDOW:
  We limit history to MAX_CONVERSATION_HISTORY messages.
  Older messages are dropped to avoid exceeding the context window.
  This is called a "sliding window" approach.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from config import settings


@dataclass
class Message:
    """
    Represents a single message in the conversation.
    
    We use a dataclass instead of a plain dict so we can add
    metadata (timestamp, agent) without changing the API call format.
    """
    role: str           # "user", "assistant", or "system"
    content: str        # The message text
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agent: str = "unknown"  # Which agent generated this (for logging)

    def to_api_format(self) -> Dict[str, str]:
        """
        Convert to the format OpenAI's API expects.
        The API only wants role + content, not our metadata.
        """
        return {"role": self.role, "content": self.content}


class SessionState:
    """
    Manages conversation history for a single session.
    
    One SessionState per user session. If you wanted multi-user 
    support, you'd create one SessionState per user ID.
    
    USAGE:
        state = SessionState()
        state.add_user_message("I want to lose weight")
        state.add_assistant_message("Great goal! Tell me more...", agent="health")
        
        # Get messages for API call:
        messages = state.get_messages_for_api(system_prompt="You are...")
    """

    def __init__(self):
        self.messages: List[Message] = []
        self.session_start = datetime.now()
        self.metadata: Dict = {}  # Store arbitrary session data
        
        # User profile built up during conversation
        # Agents update this as they learn about the user
        self.user_profile: Dict[str, str] = {}

    def add_user_message(self, content: str) -> None:
        """Record something the user said"""
        self.messages.append(Message(
            role="user",
            content=content,
            agent="user"
        ))

    def add_assistant_message(self, content: str, agent: str = "unknown") -> None:
        """Record something the assistant said"""
        self.messages.append(Message(
            role="assistant",
            content=content,
            agent=agent
        ))

    def get_messages_for_api(
        self,
        system_prompt: str,
        max_messages: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Build the messages list for an OpenAI API call.
        
        ALWAYS includes:
        1. System prompt (instructions)
        2. Recent conversation history (sliding window)
        
        Args:
            system_prompt: The instructions for GPT
            max_messages: Limit history length (default from settings)
        
        Returns:
            List of dicts in OpenAI format:
            [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]
        """
        max_messages = max_messages or settings.MAX_CONVERSATION_HISTORY
        
        # Start with system prompt
        api_messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (sliding window — most recent messages)
        recent_messages = self.messages[-max_messages:]
        api_messages.extend([msg.to_api_format() for msg in recent_messages])
        
        return api_messages

    def update_user_profile(self, key: str, value: str) -> None:
        """
        Store a fact about the user discovered during conversation.
        
        EXAMPLE:
            state.update_user_profile("sleep_time", "2am")
            state.update_user_profile("goal", "lose 10kg")
        
        These facts can be injected into future prompts.
        """
        self.user_profile[key] = value

    def get_user_profile_text(self) -> str:
        """
        Format user profile as text for prompt injection.
        
        OUTPUT EXAMPLE:
            Known about user:
            - sleep_time: 2am
            - goal: lose 10kg
            - occupation: software engineer
        """
        if not self.user_profile:
            return ""
        
        lines = ["Known about user:"]
        for key, value in self.user_profile.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Start a fresh conversation (keep user profile)"""
        self.messages = []

    def full_reset(self) -> None:
        """Complete reset including user profile"""
        self.messages = []
        self.user_profile = {}

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def last_user_message(self) -> Optional[str]:
        """Get the most recent thing the user said"""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    def get_summary(self) -> str:
        """Brief summary of the session for logging"""
        duration = datetime.now() - self.session_start
        minutes = int(duration.total_seconds() / 60)
        return (
            f"Session: {self.message_count} messages | "
            f"Duration: {minutes}m | "
            f"Profile keys: {list(self.user_profile.keys())}"
        )
