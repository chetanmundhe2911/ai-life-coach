# 🧠 AI Life Coach Agent — RAG + Multi-Domain Persona System

A production-ready, learning-friendly AI agent that acts as your **personal life coach**.
It understands YOU across 12 life domains (health, career, values, relationships, etc.)
and gives personalized advice using RAG (Retrieval-Augmented Generation).

---

## 🏗️ Architecture Overview

```
User Input
    ↓
[Orchestrator] ← decides which domain(s) are relevant
    ↓
[Domain Agents] ← e.g., health_wellness, career_profession
    ↓
[RAG Pipeline] ← searches your personal knowledge base
    ↓
[GPT-4] ← generates personalized response
    ↓
[State Manager] ← remembers conversation history
```

---

## 📁 Project Structure

```
AI_AGENT/
├── app/
│   ├── agents/               # Domain-specific AI agents
│   │   ├── base_agent.py     # Base class all agents inherit from
│   │   ├── health_agent.py
│   │   ├── career_agent.py
│   │   └── ...
│   ├── knowledge/            # RAG pipeline
│   │   ├── loader.py         # Load & chunk documents
│   │   ├── rag_store.py      # FAISS vector store
│   │   ├── rag_helper.py     # Search & retrieve
│   │   ├── knowledge_base.py # High-level KB interface
│   │   ├── summarizer.py     # Summarize retrieved docs
│   │   └── utils.py          # Helpers
│   ├── prompts/              # Prompt templates per domain
│   │   ├── orchestrator/
│   │   ├── health_wellness/
│   │   ├── career_profession/
│   │   └── ...
│   ├── state/                # Conversation memory
│   │   └── session_state.py
│   └── __init__.py
├── docs/                     # Your personal knowledge docs go here
├── tests/                    # Unit tests
├── .rag_cache/               # Auto-generated FAISS index cache
├── main.py                   # Entry point
├── config.py                 # All configuration
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone <your-repo>
cd AI_AGENT
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 3. Add Your Personal Documents (Optional)
```bash
# Drop any .txt, .pdf, or .md files about yourself into /docs
# e.g., my_health_goals.txt, career_plan.md
```

### 4. Run the Agent
```bash
python main.py
```

---

## 🧩 Key Concepts Explained

### What is RAG?
RAG = Retrieval-Augmented Generation.
Instead of relying only on GPT-4's training data, we:
1. Store YOUR personal documents as vector embeddings (FAISS)
2. When you ask a question, we RETRIEVE relevant chunks
3. We AUGMENT the GPT-4 prompt with those chunks
4. GPT-4 GENERATES a personalized answer

### What is an Orchestrator?
The orchestrator reads your message and decides:
"This question is about health AND career" → routes to both agents.

### What is Session State?
We store your conversation history so the agent remembers
what you said earlier in the conversation.

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `MODEL_NAME` | GPT model (default: gpt-4) |
| `EMBEDDING_MODEL` | Embedding model (default: text-embedding-3-small) |
| `MAX_TOKENS` | Max response tokens (default: 1000) |
| `TEMPERATURE` | Creativity 0-1 (default: 0.7) |
| `TOP_K_RESULTS` | RAG results to retrieve (default: 3) |

---

## 📚 Learning Path

1. Start with `config.py` — understand all settings
2. Read `app/knowledge/loader.py` — how docs become vectors
3. Read `app/knowledge/rag_store.py` — how FAISS stores/searches
4. Read `app/agents/base_agent.py` — the agent pattern
5. Read `app/agents/orchestrator_agent.py` — the routing logic
6. Finally `main.py` — how it all connects

---

## 🧪 Running Tests
```bash
pytest tests/ -v
```
