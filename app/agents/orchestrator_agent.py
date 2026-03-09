"""
app/agents/orchestrator_agent.py — The Orchestrator
=====================================================
WHY THIS EXISTS:
  The orchestrator is the BRAIN of the system. When a user sends a message,
  the orchestrator decides which domain agent(s) should respond.
  
  Without an orchestrator, you'd need to manually pick an agent.
  The orchestrator makes this automatic and intelligent.

HOW IT WORKS:
  1. User sends: "I'm stressed about my job and it's affecting my sleep"
  2. Orchestrator analyzes the message
  3. Identifies relevant domains: ["career_profession", "health_wellness"]
  4. Primary domain: "career_profession" (job stress is the root)
  5. Routes to CareerProfessionAgent
  6. The agent's RAG search also considers health context
  
ROUTING APPROACH:
  We use GPT-4 itself to classify the domain.
  Why? It's more accurate than keyword matching and handles nuance well.
  
  Alternative: regex/keyword matching (faster, no API cost, less accurate)
  We've included a keyword fallback in case the LLM call fails.

CONCEPT — WHY NOT SEND TO ALL AGENTS?
  We COULD send every message to all 12 agents and combine responses.
  But that would:
  - Cost 12x more in API calls
  - Take 12x longer
  - Give an incoherent response from multiple "voices"
  
  Better: Route to the 1-2 most relevant agents.
"""

import json
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

from config import settings
from app.knowledge.knowledge_base import KnowledgeBase
from app.state.session_state import SessionState
from app.agents.domain_agents import (
    HealthWellnessAgent,
    CareerProfessionAgent,
    CharacterValuesAgent,
    EducationReadinessAgent,
    FamilyDynamicsAgent,
    HygieneLifestyleAgent,
    LifePhilosophyAgent,
    MedicalLifestyleAgent,
    BehaviourPsychologyAgent,
    PoliticalAlignmentAgent,
    ReligiousValuesAgent,
    SocialPhilosophyAgent,
)


# Registry: maps domain name → agent class
# To add a new domain, just add it here!
AGENT_REGISTRY: Dict = {
    "health_wellness": HealthWellnessAgent,
    "career_profession": CareerProfessionAgent,
    "character_values": CharacterValuesAgent,
    "education_readiness": EducationReadinessAgent,
    "family_dynamics": FamilyDynamicsAgent,
    "hygiene_lifestyle": HygieneLifestyleAgent,
    "life_philosophy": LifePhilosophyAgent,
    "medical_lifestyle": MedicalLifestyleAgent,
    "behaviour_psychology": BehaviourPsychologyAgent,
    "political_alignment": PoliticalAlignmentAgent,
    "religious_values": ReligiousValuesAgent,
    "social_philosophy": SocialPhilosophyAgent,
}

# Keyword fallback for when LLM routing fails
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "health_wellness": ["health", "fitness", "exercise", "diet", "sleep", "weight", "workout", "nutrition", "energy"],
    "career_profession": ["job", "career", "work", "salary", "boss", "promotion", "interview", "business", "profession"],
    "character_values": ["values", "integrity", "honest", "ethics", "character", "identity", "authentic", "principles"],
    "education_readiness": ["learn", "study", "course", "skill", "degree", "education", "read", "improve", "training"],
    "family_dynamics": ["family", "relationship", "partner", "spouse", "children", "parents", "marriage", "friends"],
    "hygiene_lifestyle": ["routine", "habit", "morning", "evening", "clean", "organized", "schedule", "lifestyle"],
    "life_philosophy": ["purpose", "meaning", "life", "happiness", "fulfillment", "why", "legacy", "goal", "vision"],
    "medical_lifestyle": ["medical", "doctor", "medication", "condition", "chronic", "diagnosis", "treatment", "therapy"],
    "behaviour_psychology": ["anxiety", "stress", "mindset", "habit", "procrastinate", "motivation", "emotion", "feeling"],
    "political_alignment": ["politics", "vote", "government", "civic", "policy", "community", "social justice"],
    "religious_values": ["religion", "faith", "spiritual", "god", "prayer", "church", "belief", "soul", "transcend"],
    "social_philosophy": ["society", "community", "belonging", "contribution", "impact", "justice", "connection"],
}


class OrchestratorAgent:
    """
    Routes user messages to the appropriate domain agent.
    
    USAGE:
        orchestrator = OrchestratorAgent(kb, state)
        response = orchestrator.route_and_respond("I can't sleep due to work stress")
    """

    def __init__(self, knowledge_base: KnowledgeBase, session_state: SessionState):
        self.kb = knowledge_base
        self.state = session_state
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Lazy-load agents (create only when first needed)
        # This saves memory if some domains are never used
        self._agents: Dict[str, object] = {}
        
        # Store agent name in session metadata
        self.state.metadata["agent_name"] = settings.AGENT_NAME

    def route_and_respond(self, user_input: str) -> Tuple[str, str]:
        """
        Main method: classify input → route → respond.
        
        Returns:
            Tuple of (response_text, domain_used)
            domain_used is for logging/display
        """
        # Classify which domain this message belongs to
        domain = self._classify_domain(user_input)
        
        print(f"   🧭 Routing to: {domain.replace('_', ' ').title()} Agent")
        
        # Get or create the appropriate agent
        agent = self._get_agent(domain)
        
        # Let the agent generate the response
        response = agent.respond(user_input)
        
        return response, domain

    def _classify_domain(self, user_input: str) -> str:
        """
        Use GPT-4 to classify which domain best fits the user's message.
        Falls back to keyword matching if LLM call fails.
        
        WHY LLM CLASSIFICATION?
        It handles ambiguous cases better than keywords.
        "I feel lost" → life_philosophy (keyword: none obvious)
        "I'm burned out" → behaviour_psychology or career_profession
        """
        try:
            return self._llm_classify(user_input)
        except Exception as e:
            print(f"   ⚠️  LLM routing failed ({e}), using keyword fallback")
            return self._keyword_classify(user_input)

    def _llm_classify(self, user_input: str) -> str:
        """
        Ask GPT-4 to classify the domain.
        We ask for JSON output so we can parse it reliably.
        """
        domains_list = "\n".join([f"- {d}" for d in AGENT_REGISTRY.keys()])
        
        classification_prompt = f"""You are a routing system for a personal life coaching AI.
Classify the user's message into exactly ONE of these domains:

{domains_list}

User message: "{user_input}"

Rules:
- Choose the SINGLE most relevant domain
- If truly ambiguous, prefer "life_philosophy" or "behaviour_psychology"
- Respond with ONLY a JSON object like: {{"domain": "health_wellness"}}
- No explanation, just the JSON"""

        response = self.client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=[{"role": "user", "content": classification_prompt}],
            max_tokens=50,
            temperature=0  # Zero temperature = deterministic classification
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        # Handle cases where GPT wraps it in ```json ... ```
        content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        domain = result.get("domain", "life_philosophy")
        
        # Validate the domain exists
        if domain not in AGENT_REGISTRY:
            print(f"   ⚠️  Unknown domain '{domain}', defaulting to life_philosophy")
            return "life_philosophy"
        
        return domain

    def _keyword_classify(self, user_input: str) -> str:
        """
        Simple keyword-based fallback classifier.
        Counts keyword matches per domain and picks the winner.
        """
        user_lower = user_input.lower()
        scores: Dict[str, int] = {}
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in user_lower)
            scores[domain] = score
        
        # Return domain with highest keyword match count
        best_domain = max(scores, key=scores.get)
        
        # If no keywords matched at all, default to life_philosophy
        if scores[best_domain] == 0:
            return "life_philosophy"
        
        return best_domain

    def _get_agent(self, domain: str):
        """
        Lazy-load and cache agent instances.
        
        WHY LAZY LOADING?
        Creating all 12 agents at startup wastes memory if most
        domains are never used in a session. We create them on demand.
        """
        if domain not in self._agents:
            agent_class = AGENT_REGISTRY[domain]
            self._agents[domain] = agent_class(self.kb, self.state)
        
        return self._agents[domain]

    def get_active_domains(self) -> List[str]:
        """Returns list of domains used in this session (for analytics)"""
        return list(self._agents.keys())
