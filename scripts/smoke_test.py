"""End-to-end smoke test: query → retrieve → assemble prompt → completion.

Usage:
    uv run python scripts/smoke_test.py --collection inthink \
        --query "how do I add a new server module?"
"""
import argparse
import asyncio
from pathlib import Path

from mip.config import settings
from mip.inference.ollama_client import OllamaClient
from mip.inference.prompt import build_messages, load_system_prompt
from mip.rag.retrieve import format_context, retrieve

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_inthink.md"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    ollama = OllamaClient()
    try:
        chunks = await retrieve(ollama, args.collection, args.query, top_k=args.top_k)
        print(f"\n[retrieve] {len(chunks)} chunks selected")
        for c in chunks:
            tag = c.get("section") or c.get("symbol") or f"L{c.get('line_start', '?')}"
            print(f"  - {c.get('path')} {tag}  (score={c.get('score', 0):.3f})")

        ctx = format_context(chunks)
        system_prompt = load_system_prompt(PROMPT_PATH)
        messages = build_messages(
            system_prompt, ctx, [{"role": "user", "content": args.query}]
        )

        resp = await ollama.chat(model=settings().mip_primary_model, messages=messages)
        print("\n[completion]")
        print(resp.get("message", {}).get("content", ""))
    finally:
        await ollama.aclose()


if __name__ == "__main__":
    asyncio.run(main())
