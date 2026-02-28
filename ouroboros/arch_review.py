"""
Ouroboros — Daily Architecture Review Rotation.

Tracks which module was last reviewed and schedules the next review automatically.
The user asked for incremental daily reviews: one block per day, rotating through
all modules in a cycle.

Blocks (6 total, one per day):
  0 → agent.py + loop.py        (orchestration & execution engine)
  1 → context.py + llm.py       (context building & LLM client)
  2 → tools/ directory           (tool plugin system)
  3 → memory.py + consciousness.py (memory & background loop)
  4 → supervisor/ directory      (telegram, state, events, queue, workers)
  5 → review.py + arch_review.py + metrics (code quality & review system)
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

log = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Block definitions — one per review cycle day
# -------------------------------------------------------------------------

REVIEW_BLOCKS = [
    {
        "index": 0,
        "name": "Orchestration & Execution",
        "files": ["ouroboros/agent.py", "ouroboros/loop.py"],
        "focus": (
            "Review the main orchestrator and LLM tool loop. "
            "Check: task entry/exit flow, parallel tool execution logic, "
            "budget guard, self-check at rounds 50/100/150, context compaction trigger. "
            "Look for: dead code, overly complex branching, missing edge cases."
        ),
    },
    {
        "index": 1,
        "name": "Context & LLM Client",
        "files": ["ouroboros/context.py", "ouroboros/llm.py"],
        "focus": (
            "Review context assembly and LLM API client. "
            "Check: 3-block prompt caching strategy, dynamic section ordering, "
            "model fallback chain, reasoning effort normalization, token counting. "
            "Look for: cache invalidation issues, prompt bloat, unused sections."
        ),
    },
    {
        "index": 2,
        "name": "Tools System",
        "files": ["ouroboros/tools/"],
        "focus": (
            "Review the tool plugin architecture. "
            "Check: auto-discovery via get_tools(), tool schema quality, "
            "parallel vs sequential routing, timeout handling, tool result formatting. "
            "Look for: missing tools, duplicate functionality, schema inaccuracies."
        ),
    },
    {
        "index": 3,
        "name": "Memory & Consciousness",
        "files": ["ouroboros/memory.py", "ouroboros/consciousness.py", "ouroboros/arch_review.py"],
        "focus": (
            "Review memory management and background consciousness loop. "
            "Check: scratchpad/identity/user_context read-write, chat log rotation, "
            "consciousness wakeup interval logic, tool whitelist, budget tracking in BG mode. "
            "Look for: memory leaks, stale data, oversized context."
        ),
    },
    {
        "index": 4,
        "name": "Supervisor Layer",
        "files": ["supervisor/"],
        "focus": (
            "Review the supervisor: state persistence, Telegram polling, "
            "task queue, worker process management, event dispatch. "
            "Check: atomic writes, lock handling, single-consumer routing invariant, "
            "budget tracking accuracy. Look for: race conditions, lost events, state corruption."
        ),
    },
    {
        "index": 5,
        "name": "Prompts & System Prompt",
        "files": ["prompts/SYSTEM.md", "prompts/CONSCIOUSNESS.md", "BIBLE.md"],
        "focus": (
            "Review the system prompt, consciousness prompt, and Bible. "
            "Check: are the instructions still accurate? Is anything outdated? "
            "Are there contradictions between SYSTEM.md and BIBLE.md? "
            "Is the consciousness prompt giving good guidance? "
            "Look for: stale references, missing capabilities, unclear instructions."
        ),
    },
]

NUM_BLOCKS = len(REVIEW_BLOCKS)
REVIEW_INTERVAL_HOURS = 24


def get_block(index: int) -> dict:
    """Return the block definition for the given index (wraps around)."""
    return REVIEW_BLOCKS[index % NUM_BLOCKS]


def is_review_due(last_at: str, interval_hours: float = REVIEW_INTERVAL_HOURS) -> bool:
    """Return True if enough time has passed since the last review."""
    if not last_at:
        return True
    try:
        last_dt = datetime.datetime.fromisoformat(last_at)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        elapsed = (now - last_dt).total_seconds() / 3600.0
        return elapsed >= interval_hours
    except Exception:
        log.debug("Failed to parse arch_review_last_at: %r", last_at, exc_info=True)
        return True


def build_review_task_description(block: dict) -> str:
    """Build the schedule_task description for a given block."""
    files_str = ", ".join(f"`{f}`" for f in block["files"])
    return (
        f"Daily Architecture Review — Block {block['index']}: {block['name']}\n\n"
        f"Files to review: {files_str}\n\n"
        f"Focus: {block['focus']}\n\n"
        "Instructions:\n"
        "1. Read the files listed above.\n"
        "2. Evaluate: complexity, clarity, adherence to Bible (section 8: minimalism), "
        "correctness, missing features.\n"
        "3. Form concrete observations. If improvements are warranted, propose them "
        "with short explanations (1-2 sentences each). Keep it simple — no rewrites "
        "unless there's a clear problem.\n"
        "4. Send a proactive message to the owner summarizing findings + proposals "
        "(use send_owner_message). Keep it concise (≤300 words).\n"
        "5. Update scratchpad with a brief note about what was reviewed and key findings.\n\n"
        "IMPORTANT: This is a review, not a rewrite. Propose only changes with clear value. "
        "No approval needed to review; approval IS needed before implementing changes."
    )


def advance_index(current_index: int) -> int:
    """Return the next block index (wraps around after NUM_BLOCKS)."""
    return (current_index + 1) % NUM_BLOCKS
