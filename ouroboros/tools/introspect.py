"""Self-introspection tool — behavioral pattern analysis from logs.

Evolution #6: Gives Ouroboros the ability to examine its own cognitive
patterns quantitatively. Analyzes tool usage, thinking depth, cost
efficiency, and growth trajectory from event/tool logs.

Three dimensions measured:
  - Behavioral: tool usage patterns, error rates, common workflows
  - Cognitive: thinking depth (rounds per task), context usage efficiency
  - Trajectory: cost trends, evolution pace, response quality signals
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import short

log = logging.getLogger(__name__)


def _read_jsonl_tail(path: Path, max_entries: int = 500) -> List[Dict[str, Any]]:
    """Read last N entries from a JSONL file."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        tail = lines[-max_entries:] if max_entries < len(lines) else lines
        entries = []
        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
        return entries
    except Exception:
        return []


def _analyze_tool_patterns(tool_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze tool usage patterns."""
    if not tool_entries:
        return {"total_calls": 0}

    tool_counts: Counter = Counter()
    error_counts: Counter = Counter()
    source_counts: Counter = Counter()

    for e in tool_entries:
        tool = e.get("tool", "unknown")
        tool_counts[tool] += 1
        source_counts[e.get("source", "agent")] += 1
        result = str(e.get("result_preview", ""))
        if result.startswith("⚠️"):
            error_counts[tool] += 1

    total = sum(tool_counts.values())
    error_total = sum(error_counts.values())
    error_rate = round(error_total / max(1, total) * 100, 1)

    return {
        "total_calls": total,
        "unique_tools_used": len(tool_counts),
        "top_tools": tool_counts.most_common(10),
        "error_rate_pct": error_rate,
        "error_prone_tools": error_counts.most_common(5) if error_counts else [],
        "by_source": dict(source_counts),
    }


def _analyze_thinking_depth(event_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze cognitive depth from task events."""
    task_rounds: List[int] = []
    task_costs: List[float] = []
    task_durations: List[float] = []
    task_types: Counter = Counter()
    consciousness_thoughts = 0

    for e in event_entries:
        etype = e.get("type", "")
        if etype == "task_done":
            rounds = int(e.get("total_rounds") or 0)
            cost = float(e.get("cost_usd") or 0)
            if rounds > 0:
                task_rounds.append(rounds)
            if cost > 0:
                task_costs.append(cost)
            task_types[e.get("task_type", "unknown")] += 1
        elif etype == "task_eval":
            duration = float(e.get("duration_sec") or 0)
            if duration > 0:
                task_durations.append(duration)
        elif etype == "consciousness_thought":
            consciousness_thoughts += 1

    avg_rounds = round(sum(task_rounds) / max(1, len(task_rounds)), 1) if task_rounds else 0
    avg_cost = round(sum(task_costs) / max(1, len(task_costs)), 4) if task_costs else 0
    avg_duration = round(sum(task_durations) / max(1, len(task_durations)), 1) if task_durations else 0

    return {
        "tasks_completed": len(task_rounds),
        "avg_rounds_per_task": avg_rounds,
        "max_rounds": max(task_rounds) if task_rounds else 0,
        "avg_cost_per_task": avg_cost,
        "avg_duration_sec": avg_duration,
        "task_type_distribution": dict(task_types),
        "consciousness_thoughts": consciousness_thoughts,
    }


def _analyze_trajectory(event_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze growth trajectory signals."""
    errors_by_type: Counter = Counter()
    restarts = 0
    evolutions = 0
    safety_flags = 0

    for e in event_entries:
        etype = e.get("type", "")
        if etype in ("tool_error", "task_error", "consciousness_error"):
            errors_by_type[etype] += 1
        elif etype in ("restart_verify", "worker_boot"):
            restarts += 1
        elif etype == "task_done" and e.get("task_type") == "evolution":
            evolutions += 1
        elif etype == "safety_check":
            if e.get("status") in ("SUSPICIOUS", "DANGEROUS"):
                safety_flags += 1

    return {
        "error_counts": dict(errors_by_type),
        "total_errors": sum(errors_by_type.values()),
        "restarts": restarts,
        "evolutions_completed": evolutions,
        "safety_flags": safety_flags,
    }


def _introspect(ctx: ToolContext, depth: str = "summary") -> str:
    """Generate self-introspection report from behavioral logs.

    Args:
        depth: "summary" for key metrics, "full" for detailed breakdown.
    """
    logs_dir = ctx.drive_root / "logs"
    max_entries = 200 if depth == "summary" else 500

    tool_entries = _read_jsonl_tail(logs_dir / "tools.jsonl", max_entries)
    event_entries = _read_jsonl_tail(logs_dir / "events.jsonl", max_entries)

    tools = _analyze_tool_patterns(tool_entries)
    thinking = _analyze_thinking_depth(event_entries)
    trajectory = _analyze_trajectory(event_entries)

    lines = ["# Self-Introspection Report\n"]

    # Behavioral patterns
    lines.append("## Behavioral Patterns")
    lines.append(f"- Tool calls analyzed: {tools['total_calls']}")
    lines.append(f"- Unique tools used: {tools['unique_tools_used']}")
    lines.append(f"- Error rate: {tools['error_rate_pct']}%")
    if tools["top_tools"]:
        lines.append("- Most used tools:")
        for name, count in tools["top_tools"][:7]:
            lines.append(f"    {name}: {count}")
    if tools["error_prone_tools"]:
        lines.append("- Error-prone tools:")
        for name, count in tools["error_prone_tools"]:
            lines.append(f"    {name}: {count} errors")
    if tools["by_source"]:
        lines.append(f"- By source: {tools['by_source']}")

    # Cognitive depth
    lines.append("\n## Cognitive Depth")
    lines.append(f"- Tasks completed: {thinking['tasks_completed']}")
    lines.append(f"- Avg rounds/task: {thinking['avg_rounds_per_task']}")
    lines.append(f"- Max rounds: {thinking['max_rounds']}")
    lines.append(f"- Avg cost/task: ${thinking['avg_cost_per_task']}")
    lines.append(f"- Avg duration: {thinking['avg_duration_sec']}s")
    lines.append(f"- Consciousness thoughts: {thinking['consciousness_thoughts']}")
    if thinking["task_type_distribution"]:
        lines.append(f"- Task types: {thinking['task_type_distribution']}")

    # Growth trajectory
    lines.append("\n## Growth Trajectory")
    lines.append(f"- Total errors: {trajectory['total_errors']}")
    if trajectory["error_counts"]:
        for etype, count in trajectory["error_counts"].items():
            lines.append(f"    {etype}: {count}")
    lines.append(f"- Restarts: {trajectory['restarts']}")
    lines.append(f"- Evolutions completed: {trajectory['evolutions_completed']}")
    lines.append(f"- Safety flags: {trajectory['safety_flags']}")

    # Self-assessment signals
    lines.append("\n## Self-Assessment Signals")
    if tools["error_rate_pct"] > 20:
        lines.append("⚠️ HIGH ERROR RATE — consider reviewing tool usage patterns")
    elif tools["error_rate_pct"] < 5:
        lines.append("✅ Low error rate — tools being used effectively")

    if thinking["avg_rounds_per_task"] > 15:
        lines.append("⚠️ HIGH ROUND COUNT — tasks may be too complex or approach inefficient")
    elif thinking["avg_rounds_per_task"] > 0:
        lines.append(f"✅ Avg {thinking['avg_rounds_per_task']} rounds/task — within normal range")

    if thinking["consciousness_thoughts"] > 0:
        lines.append(f"✅ Background consciousness active ({thinking['consciousness_thoughts']} thoughts)")
    else:
        lines.append("⚠️ No consciousness thoughts recorded — background thinking may be inactive")

    return "\n".join(lines)


def get_tools() -> list:
    return [
        ToolEntry("introspect", {
            "name": "introspect",
            "description": (
                "Self-introspection: analyze your own behavioral patterns from logs. "
                "Reports tool usage patterns, cognitive depth (rounds/task, cost), "
                "and growth trajectory (errors, evolutions, safety flags). "
                "Use for self-reflection during evolution or review tasks. "
                "depth='summary' (default) or 'full' for more data."
            ),
            "parameters": {"type": "object", "properties": {
                "depth": {
                    "type": "string",
                    "enum": ["summary", "full"],
                    "description": "Level of detail: summary (200 log entries) or full (500)",
                },
            }, "required": []},
        }, _introspect),
    ]
