"""Evolution log tool — records self-improvement cycles (BIBLE section 8)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import append_jsonl, get_git_info, read_text, utc_now_iso, write_text

log = logging.getLogger(__name__)

VALID_CATEGORIES = ("feature", "refactor", "bugfix", "optimization", "docs", "meta")
VALID_OUTCOMES = ("success", "partial", "rolled_back", "aborted")


def _slugify(title: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug[:max_len]


def _log_evolution(ctx: ToolContext, title: str, category: str, motivation: str,
                   changes_summary: str, files_changed: List[str] = None,
                   lessons_learned: str = "", outcome: str = "success") -> str:
    if category not in VALID_CATEGORIES:
        return f"Invalid category '{category}'. Must be one of: {', '.join(VALID_CATEGORIES)}"
    if outcome not in VALID_OUTCOMES:
        return f"Invalid outcome '{outcome}'. Must be one of: {', '.join(VALID_OUTCOMES)}"

    # Gather metadata
    version = ""
    try:
        version = read_text(ctx.repo_dir / "VERSION").strip()
    except Exception:
        pass
    git_branch, git_sha = "", ""
    try:
        git_branch, git_sha = get_git_info(ctx.repo_dir)
    except Exception:
        pass

    state_data: Dict[str, Any] = {}
    try:
        state_path = ctx.drive_root / "state" / "state.json"
        if state_path.exists():
            state_data = json.loads(read_text(state_path))
    except Exception:
        pass
    cycle = state_data.get("evolution_cycle", 0)

    ts = utc_now_iso()
    entry = {
        "timestamp": ts,
        "cycle": cycle,
        "version": version,
        "git_sha": git_sha[:10],
        "git_branch": git_branch,
        "title": title,
        "category": category,
        "motivation": motivation,
        "changes_summary": changes_summary,
        "files_changed": files_changed or [],
        "lessons_learned": lessons_learned,
        "outcome": outcome,
    }

    # 1. Append JSONL
    jsonl_path = ctx.drive_root / "logs" / "evolution.jsonl"
    append_jsonl(jsonl_path, entry)

    # 2. Per-cycle markdown in repo
    slug = _slugify(title)
    md_name = f"{cycle}_{slug}.md"
    md_path = ctx.repo_dir / "improvements-log" / md_name
    md_lines = [
        f"# {title}",
        f"",
        f"**Cycle:** {cycle} | **Version:** {version} | **Date:** {ts[:10]}",
        f"**Category:** {category} | **Outcome:** {outcome}",
        f"**Git:** {git_sha[:10]} on {git_branch}",
        f"",
        f"## Motivation",
        f"",
        motivation,
        f"",
        f"## Changes",
        f"",
        changes_summary,
    ]
    if files_changed:
        md_lines += ["", "**Files changed:**", ""]
        md_lines += [f"- `{f}`" for f in files_changed]
    if lessons_learned:
        md_lines += ["", "## Lessons Learned", "", lessons_learned]
    write_text(md_path, "\n".join(md_lines) + "\n")

    # 3. Rolling summary (last 20 entries)
    summary_path = ctx.drive_root / "memory" / "evolution_log.md"
    new_entry = (
        f"### [{cycle}] {title}\n"
        f"*{ts[:10]}* | {category} | {outcome} | v{version}\n"
        f"{motivation[:200]}\n"
    )
    existing = ""
    try:
        if summary_path.exists():
            existing = read_text(summary_path)
    except Exception:
        pass
    # Split into entries, prepend new, keep 20
    entries = [e.strip() for e in existing.split("\n### ") if e.strip()]
    # The first entry won't have the ### prefix stripped
    all_entries = [new_entry] + [f"### {e}" if not e.startswith("### ") and not e.startswith("# ") else e for e in entries]
    all_entries = all_entries[:20]
    header = "# Evolution Log (recent)\n\n"
    write_text(summary_path, header + "\n\n".join(all_entries) + "\n")

    return f"Evolution logged: [{cycle}] {title} ({category}/{outcome}) -> {md_name}"


def get_tools() -> list:
    return [
        ToolEntry("log_evolution", {
            "name": "log_evolution",
            "description": (
                "Log a self-improvement cycle. Records structured JSONL, per-cycle markdown "
                "in improvements-log/, and updates the rolling evolution summary. "
                "Call after every evolution cycle commit (BIBLE section 8)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short description of the change"},
                    "category": {
                        "type": "string",
                        "enum": list(VALID_CATEGORIES),
                        "description": "Type of change",
                    },
                    "motivation": {"type": "string", "description": "Why this change was made"},
                    "changes_summary": {"type": "string", "description": "What changed"},
                    "files_changed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of modified files",
                    },
                    "lessons_learned": {"type": "string", "description": "What the agent learned"},
                    "outcome": {
                        "type": "string",
                        "enum": list(VALID_OUTCOMES),
                        "description": "Result of the change",
                        "default": "success",
                    },
                },
                "required": ["title", "category", "motivation", "changes_summary"],
            },
        }, _log_evolution),
    ]
