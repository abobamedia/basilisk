"""Ouroboros — Smart Task Router.

Classifies incoming tasks and selects the optimal model/provider/effort
based on task type, user context, and budget constraints.

Architecture:
  - Direct user chat → MAIN slot (smart model, e.g. Codex GPT-5.3)
  - Evolution/review  → CODE slot (code-focused smart model)
  - Background tasks   → LIGHT slot (free NVIDIA)
  - Low budget         → FALLBACK slot (free NVIDIA)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    """Result of task routing."""
    model: str
    provider: str
    effort: str
    reason: str  # why this route was chosen (for logging)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def route_task(
    task_type: str = "",
    is_direct_chat: bool = False,
    budget_remaining: float = 999.0,
    depth: int = 0,
) -> RouteDecision:
    """
    Classify a task and select the optimal model/provider/effort.

    Parameters
    ----------
    task_type : str
        Task type string: "task", "evolution", "review", "background", etc.
    is_direct_chat : bool
        True if this is a direct user message (not a scheduled background task).
    budget_remaining : float
        Remaining budget in USD.
    depth : int
        Task depth (0 = top-level, 1+ = sub-task).

    Returns
    -------
    RouteDecision with model, provider, effort, and reason.
    """
    # Read slot configs from env
    main_model = _env("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
    main_provider = _env("PROVIDER_MAIN", "openrouter")
    code_model = _env("OUROBOROS_MODEL_CODE", main_model)
    code_provider = _env("PROVIDER_CODE", main_provider)
    light_model = _env("OUROBOROS_MODEL_LIGHT", main_model)
    light_provider = _env("PROVIDER_LIGHT", main_provider)
    fallback_model = _env("OUROBOROS_MODEL_FALLBACK", main_model)
    fallback_provider = _env("PROVIDER_FALLBACK", main_provider)

    # --- Priority 1: Budget critical → free model ---
    if budget_remaining < 0.10:
        decision = RouteDecision(
            model=fallback_model,
            provider=fallback_provider,
            effort="low",
            reason=f"budget critical (${budget_remaining:.2f} remaining)",
        )
        log.info("Router: %s → %s/%s [%s]", task_type or "task", decision.provider, decision.model, decision.reason)
        return decision

    # --- Priority 2: Direct user chat → smart model ---
    if is_direct_chat:
        decision = RouteDecision(
            model=main_model,
            provider=main_provider,
            effort="medium",
            reason="direct user chat",
        )
        log.info("Router: %s → %s/%s [%s]", task_type or "chat", decision.provider, decision.model, decision.reason)
        return decision

    # --- Priority 3: Evolution / review → code model ---
    task_lower = (task_type or "").lower()
    if task_lower in ("evolution", "review", "code_review"):
        decision = RouteDecision(
            model=code_model,
            provider=code_provider,
            effort="high",
            reason=f"task type: {task_lower}",
        )
        log.info("Router: %s → %s/%s [%s]", task_type, decision.provider, decision.model, decision.reason)
        return decision

    # --- Priority 4: Deep sub-tasks → use lighter effort ---
    if depth >= 2:
        decision = RouteDecision(
            model=main_model,
            provider=main_provider,
            effort="low",
            reason=f"deep sub-task (depth={depth})",
        )
        log.info("Router: %s → %s/%s [%s]", task_type or "subtask", decision.provider, decision.model, decision.reason)
        return decision

    # --- Priority 5: Budget warning → prefer main but lower effort ---
    if budget_remaining < 1.0:
        decision = RouteDecision(
            model=main_model,
            provider=main_provider,
            effort="low",
            reason=f"budget warning (${budget_remaining:.2f} remaining)",
        )
        log.info("Router: %s → %s/%s [%s]", task_type or "task", decision.provider, decision.model, decision.reason)
        return decision

    # --- Default: main smart model ---
    decision = RouteDecision(
        model=main_model,
        provider=main_provider,
        effort="medium",
        reason="default routing",
    )
    log.info("Router: %s → %s/%s [%s]", task_type or "task", decision.provider, decision.model, decision.reason)
    return decision
