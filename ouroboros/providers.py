"""
Ouroboros — LLM Provider Registry.

Defines provider descriptors and helpers for multi-provider routing.
Each provider describes an OpenAI-compatible API endpoint with its quirks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LLMProvider:
    """Descriptor for an OpenAI-compatible LLM API provider."""

    name: str                           # "openrouter", "nvidia", "openai", "local"
    display_name: str                   # Human-readable name
    base_url: str                       # API endpoint (may contain {port} for local)
    api_key_setting: str                # Key name in settings.json (e.g. "NVIDIA_API_KEY")
    supports_reasoning: bool            # extra_body.reasoning (OpenRouter-specific)
    supports_cache_control: bool        # cache_control on messages/tools (OpenRouter/Anthropic)
    supports_provider_routing: bool     # extra_body.provider.order (OpenRouter-specific)
    requires_content_flattening: bool   # Flatten multipart content to plain strings
    tool_choice_values: List[str]       # Supported tool_choice values
    supports_generation_cost_api: bool  # /generation?id= cost endpoint (OpenRouter)
    default_headers: Dict[str, str] = field(default_factory=dict)
    pricing: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)


PROVIDERS: Dict[str, LLMProvider] = {
    "openrouter": LLMProvider(
        name="openrouter",
        display_name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_setting="OPENROUTER_API_KEY",
        supports_reasoning=True,
        supports_cache_control=True,
        supports_provider_routing=True,
        requires_content_flattening=False,
        tool_choice_values=["auto", "none", "required"],
        supports_generation_cost_api=True,
        default_headers={
            "HTTP-Referer": "https://ouroboros.local/",
            "X-Title": "Ouroboros",
        },
        pricing={},  # fetched dynamically from OpenRouter API
    ),
    "nvidia": LLMProvider(
        name="nvidia",
        display_name="NVIDIA NIM",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_setting="NVIDIA_API_KEY",
        supports_reasoning=False,
        supports_cache_control=False,
        supports_provider_routing=False,
        requires_content_flattening=True,
        tool_choice_values=["auto", "none"],
        supports_generation_cost_api=False,
        default_headers={},
        pricing={
            # NVIDIA NIM free tier — all models are free via build.nvidia.com
            "meta/llama-3.3-70b-instruct": (0.0, 0.0, 0.0),
            "meta/llama-3.1-405b-instruct": (0.0, 0.0, 0.0),
            "meta/llama-3.1-70b-instruct": (0.0, 0.0, 0.0),
            "meta/llama-3.1-8b-instruct": (0.0, 0.0, 0.0),
            "nvidia/llama-3.1-nemotron-ultra-253b-v1": (0.0, 0.0, 0.0),
            "nvidia/llama-3.1-nemotron-70b-instruct": (0.0, 0.0, 0.0),
            "nvidia/llama-3.3-nemotron-super-49b-v1": (0.0, 0.0, 0.0),
            "nvidia/mistral-nemo-minitron-8b-8k-instruct": (0.0, 0.0, 0.0),
            "qwen/qwen2.5-coder-32b-instruct": (0.0, 0.0, 0.0),
            "qwen/qwen3-235b-a22b": (0.0, 0.0, 0.0),
            "qwen/qwen3.5-397b-a17b": (0.0, 0.0, 0.0),
            "mistralai/mistral-large-2-instruct": (0.0, 0.0, 0.0),
            "deepseek-ai/deepseek-r1-distill-qwen-32b": (0.0, 0.0, 0.0),
            "mistralai/devstral-2-123b-instruct-2512": (0.0, 0.0, 0.0),
            "nvidia/llama-3.3-nemotron-super-49b-v1.5": (0.0, 0.0, 0.0),
        },
    ),
    "openai": LLMProvider(
        name="openai",
        display_name="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_setting="OPENAI_API_KEY",
        supports_reasoning=False,
        supports_cache_control=False,
        supports_provider_routing=False,
        requires_content_flattening=False,
        tool_choice_values=["auto", "none", "required"],
        supports_generation_cost_api=False,
        default_headers={},
        pricing={
            "gpt-4o": (2.50, 1.25, 10.0),
            "gpt-4-turbo": (10.0, 5.0, 30.0),
            "gpt-4.1": (2.0, 0.50, 8.0),
            "o1": (15.0, 7.5, 60.0),
            "o3": (2.0, 0.50, 8.0),
            "o4-mini": (1.10, 0.275, 4.40),
            "gpt-5.2": (1.75, 0.175, 14.0),
        },
    ),
    "codex": LLMProvider(
        name="codex",
        display_name="Codex (OpenClaw)",
        base_url="http://127.0.0.1:18789/v1",
        api_key_setting="OPENCLAW_API_TOKEN",
        supports_reasoning=False,
        supports_cache_control=False,
        supports_provider_routing=False,
        requires_content_flattening=False,
        tool_choice_values=["auto", "none", "required"],
        supports_generation_cost_api=False,
        default_headers={},
        pricing={
            # ChatGPT subscription — flat monthly cost, no per-token billing
            "openai-codex/gpt-5.3-codex": (0.0, 0.0, 0.0),
            "openai-codex/gpt-5.2-codex": (0.0, 0.0, 0.0),
            "openai-codex/gpt-4o": (0.0, 0.0, 0.0),
            "openai-codex/o3": (0.0, 0.0, 0.0),
            "openai-codex/o4-mini": (0.0, 0.0, 0.0),
        },
    ),
    "bonsai": LLMProvider(
        name="bonsai",
        display_name="Bonsai (Claude via CLI proxy)",
        base_url="https://go.trybons.ai",
        api_key_setting="BONSAI_API_KEY",
        supports_reasoning=False,
        supports_cache_control=False,
        supports_provider_routing=False,
        requires_content_flattening=False,
        tool_choice_values=["auto", "none", "required"],
        supports_generation_cost_api=False,
        default_headers={},
        pricing={
            # Bonsai is free (subscription-based, no per-token billing)
            "bonsai": (0.0, 0.0, 0.0),
        },
    ),
    "local": LLMProvider(
        name="local",
        display_name="Local Model",
        base_url="http://127.0.0.1:{port}/v1",
        api_key_setting="",
        supports_reasoning=False,
        supports_cache_control=False,
        supports_provider_routing=False,
        requires_content_flattening=True,
        tool_choice_values=["auto", "none"],
        supports_generation_cost_api=False,
        default_headers={},
        pricing={},  # free
    ),
}


def get_provider(name: str) -> LLMProvider:
    """Look up a provider by name. Returns openrouter as fallback."""
    return PROVIDERS.get(name, PROVIDERS["openrouter"])


def resolve_api_key(provider: LLMProvider) -> str:
    """Read the API key for a provider from environment/settings."""
    if not provider.api_key_setting:
        return "local"
    # Check provider-specific key first
    key = os.environ.get(provider.api_key_setting, "")
    if key:
        return key
    # Codex (OpenClaw) gateway doesn't require a real key
    if provider.name == "codex":
        return "codex-local"
    # Fallback to generic override
    return os.environ.get("OUROBOROS_LLM_API_KEY", "")
