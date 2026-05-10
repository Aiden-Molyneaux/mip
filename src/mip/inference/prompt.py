"""Assemble the message list sent to Ollama: system + RAG context + user."""
from pathlib import Path


def build_messages(
    system_prompt: str,
    rag_context: str | None,
    user_messages: list[dict],
) -> list[dict]:
    full_system = system_prompt
    if rag_context:
        full_system += (
            "\n\n# Retrieved project context\n\n"
            "The following snippets were pulled from the project repository because they may be "
            "relevant. Prefer reusing existing utilities and conventions you see here over "
            "inventing new ones. If a snippet contradicts your assumptions, trust the snippet.\n\n"
            f"{rag_context}"
        )
    return [{"role": "system", "content": full_system}, *user_messages]


def load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")
