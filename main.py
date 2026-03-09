"""
main.py — Application Entry Point
===================================
WHY THIS FILE:
  This is WHERE IT ALL STARTS. Run this to launch your life coach.
  
  It connects all the pieces:
  config → knowledge base → session state → orchestrator → UI loop
  
  Think of it like the main() function of a C program —
  everything flows through here.

HOW TO RUN:
  python main.py

WHAT HAPPENS:
  1. Validate config (check API key, etc.)
  2. Initialize Knowledge Base (load/build FAISS index)
  3. Create Session State (fresh conversation memory)
  4. Create Orchestrator (the routing brain)
  5. Run the chat loop (until user types 'quit')
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

from config import settings, validate_settings
from app.knowledge.knowledge_base import KnowledgeBase
from app.state.session_state import SessionState
from app.agents.orchestrator_agent import OrchestratorAgent


# Rich console for beautiful terminal output
console = Console()


def print_banner():
    """Print a welcome banner"""
    banner = Text()
    banner.append(f"\n  🧠 {settings.AGENT_NAME} — Your AI Life Coach\n", style="bold cyan")
    banner.append("  Powered by GPT-4 + RAG (FAISS)\n", style="dim")
    banner.append("  Type 'quit' to exit | 'reset' to clear history\n", style="dim")
    banner.append("  Type 'rebuild' to reload documents | 'profile' to see your profile\n", style="dim")
    
    console.print(Panel(banner, border_style="cyan"))


def print_response(response: str, domain: str):
    """Print the agent's response with domain label"""
    domain_label = domain.replace("_", " ").title()
    console.print(f"\n[dim cyan]● {settings.AGENT_NAME}[/dim cyan] [dim]({domain_label})[/dim]")
    console.print(f"[white]{response}[/white]")
    console.print()


def handle_special_commands(
    user_input: str,
    state: SessionState,
    kb: KnowledgeBase
) -> bool:
    """
    Handle special commands that aren't life coaching questions.
    Returns True if a command was handled (skip normal processing).
    """
    cmd = user_input.strip().lower()

    if cmd in ("quit", "exit", "bye"):
        console.print(f"\n[cyan]✨ {settings.AGENT_NAME}:[/cyan] Take care! See you next time. 👋")
        console.print(f"[dim]{state.get_summary()}[/dim]\n")
        sys.exit(0)

    elif cmd == "reset":
        state.clear()
        console.print("[dim]✓ Conversation cleared. Your profile is preserved.[/dim]\n")
        return True

    elif cmd == "full_reset":
        state.full_reset()
        console.print("[dim]✓ Full reset complete. Profile and history cleared.[/dim]\n")
        return True

    elif cmd == "profile":
        profile = state.get_user_profile_text()
        if profile:
            console.print(f"\n[cyan]Your Profile:[/cyan]\n{profile}\n")
        else:
            console.print("[dim]No profile data yet. Chat more to build your profile![/dim]\n")
        return True

    elif cmd == "rebuild":
        console.print("[dim]Rebuilding knowledge base from /docs...[/dim]")
        kb.rebuild()
        return True

    elif cmd == "help":
        console.print("""
[cyan]Available Commands:[/cyan]
  quit / exit  — End the session
  reset        — Clear conversation history (keep profile)
  full_reset   — Clear everything
  profile      — View your accumulated profile
  rebuild      — Reload documents from /docs folder
  help         — Show this help
        """)
        return True

    return False


def extract_profile_hints(user_input: str, state: SessionState):
    """
    Simple pattern matching to extract profile hints from user messages.
    
    WHY: The more we know about the user, the more personalized the coaching.
    
    This is a basic version — a production system would use the LLM to extract
    structured profile data more accurately.
    """
    lower = user_input.lower()
    
    # Very simple heuristics — good enough for learning project
    if "i'm " in lower or "i am " in lower:
        if "years old" in lower or "year old" in lower:
            # "I'm 28 years old" → store age
            import re
            match = re.search(r"i(?:'m| am) (\d+)", lower)
            if match:
                state.update_user_profile("age", match.group(1))
    
    if "my name is" in lower or "i'm called" in lower:
        import re
        match = re.search(r"(?:my name is|i'm called) ([a-zA-Z]+)", lower)
        if match:
            state.update_user_profile("name", match.group(1).title())
    
    if "i work as" in lower or "i'm a " in lower or "i am a " in lower:
        import re
        match = re.search(r"(?:i work as a?|i(?:'m| am) a?) ([a-zA-Z ]+?)(?:\.|,|$)", lower)
        if match:
            occupation = match.group(1).strip()
            if len(occupation) < 40:  # Sanity check
                state.update_user_profile("occupation", occupation)


def main():
    """
    Main application entry point.
    Initializes all components and runs the chat loop.
    """
    # ── Step 1: Validate Configuration ────────────────────────────
    try:
        validate_settings()
    except ValueError:
        sys.exit(1)

    # ── Step 2: Print Welcome Banner ──────────────────────────────
    print_banner()

    # ── Step 3: Initialize Knowledge Base ─────────────────────────
    # This either loads the cached FAISS index or builds it fresh
    # from documents in /docs
    kb = KnowledgeBase.get_instance()

    # ── Step 4: Create Session State ──────────────────────────────
    # Fresh conversation memory for this session
    state = SessionState()

    # ── Step 5: Create Orchestrator ───────────────────────────────
    orchestrator = OrchestratorAgent(kb, state)

    # ── Step 6: Opening Message ───────────────────────────────────
    opening = (
        f"Hello! I'm {settings.AGENT_NAME}, your personal life coach. "
        f"I'm here to support you across all areas of life — health, career, "
        f"relationships, personal growth, and more. "
        f"What's on your mind today?"
    )
    console.print(f"\n[cyan]● {settings.AGENT_NAME}:[/cyan] {opening}\n")
    state.add_assistant_message(opening, agent="orchestrator")

    # ── Step 7: Main Chat Loop ────────────────────────────────────
    while True:
        try:
            # Get user input
            user_input = console.input("[bold green]You:[/bold green] ").strip()
            
            # Skip empty input
            if not user_input:
                continue

            # Add to session state BEFORE checking commands
            # (so command history is preserved for context)
            if not handle_special_commands(user_input, state, kb):
                # Record user message in history
                state.add_user_message(user_input)
                
                # Extract any profile hints from the message
                extract_profile_hints(user_input, state)
                
                # Route to appropriate agent and get response
                console.print("[dim]Thinking...[/dim]", end="\r")
                response, domain = orchestrator.route_and_respond(user_input)
                
                # Display response
                print_response(response, domain)

        except KeyboardInterrupt:
            console.print(f"\n\n[cyan]{settings.AGENT_NAME}:[/cyan] Goodbye! 👋\n")
            break
        except EOFError:
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            console.print("[dim]Try again or type 'quit' to exit.[/dim]\n")


if __name__ == "__main__":
    main()
