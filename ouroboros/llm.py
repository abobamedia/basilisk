"""Ouroboros — LLM client.

The only module that communicates with LLM APIs.
Supports multiple providers: OpenRouter, NVIDIA NIM, OpenAI, Local.
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import copy
import json as _json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3-flash-preview"


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch current pricing from OpenRouter API.

    Returns dict of {model_id: (input_per_1m, cached_per_1m, output_per_1m)}.
    Returns empty dict on failure.
    """
    import logging
    log = logging.getLogger("ouroboros.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        # Prefixes we care about
        prefixes = ("anthropic/", "openai/", "google/", "meta-llama/", "x-ai/", "qwen/")

        pricing_dict = {}
        for model in models:
            model_id = model.get("id", "")
            if not model_id.startswith(prefixes):
                continue

            pricing = model.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue

            # OpenRouter pricing is in dollars per token (raw values)
            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None

            # Convert to per-million tokens
            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            if raw_cached is not None:
                cached_price = round(raw_cached * 1_000_000, 4)
            else:
                cached_price = round(prompt_price * 0.1, 4)  # fallback: 10% of prompt

            # Sanity check: skip obviously wrong prices
            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            pricing_dict[model_id] = (prompt_price, cached_price, completion_price)

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


class LLMClient:
    """LLM API wrapper. Routes calls to multiple providers: OpenRouter, NVIDIA, OpenAI, Local."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_local: bool = False,
        local_port: int = 8766,
    ):
        # Legacy env vars for backward compat
        self._api_key = api_key or os.environ.get("OUROBOROS_LLM_API_KEY", "")
        self._base_url = base_url or os.environ.get("OUROBOROS_LLM_BASE_URL", "https://openrouter.ai/api/v1")
        self._use_local = use_local
        self._local_port = local_port
        # Per-provider client cache: {provider_name: OpenAI}
        self._clients: Dict[str, Any] = {}
        # Legacy single-client refs (populated lazily for backward compat)
        self._client = None
        self._local_client = None

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def _get_client_for_provider(self, provider_name: str):
        """Get or create an OpenAI-compatible client for the given provider."""
        if provider_name in self._clients:
            return self._clients[provider_name]

        from openai import OpenAI
        from ouroboros.providers import get_provider, resolve_api_key

        provider = get_provider(provider_name)

        if provider_name == "local":
            port = int(os.environ.get("LOCAL_MODEL_PORT", str(self._local_port)))
            base_url = f"http://127.0.0.1:{port}/v1"
            client = OpenAI(base_url=base_url, api_key="local")
        else:
            api_key = resolve_api_key(provider)
            # Fall back to legacy key if provider-specific key is empty
            if not api_key:
                api_key = self._api_key
            client = OpenAI(
                base_url=provider.base_url,
                api_key=api_key,
                default_headers=provider.default_headers or {},
            )

        self._clients[provider_name] = client
        return client

    def _get_client(self):
        """Legacy: get client for the default provider (backward compat)."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                default_headers={
                    "HTTP-Referer": "https://ouroboros.local/",
                    "X-Title": "Ouroboros",
                },
            )
        return self._client

    def _get_local_client(self):
        """Legacy: get local model client (backward compat)."""
        return self._get_client_for_provider("local")

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_cache_control(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strip cache_control from message content blocks."""
        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block.pop("cache_control", None)
        return cleaned

    @staticmethod
    def _flatten_content(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten multipart content blocks to plain strings."""
        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = "\n\n".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
        return cleaned

    def _local_max_tokens(self, max_tokens: int) -> int:
        """Cap max_tokens for local model context window."""
        local_max = min(max_tokens, 2048)
        try:
            from ouroboros.local_model import get_manager
            ctx_len = get_manager().get_context_length()
            if ctx_len > 0:
                local_max = min(max_tokens, max(256, ctx_len // 4))
        except Exception:
            pass
        return local_max

    # ------------------------------------------------------------------
    # Cost helpers
    # ------------------------------------------------------------------

    def _fetch_generation_cost(self, generation_id: str, provider_name: str = "openrouter") -> Optional[float]:
        """Fetch cost from OpenRouter Generation API as fallback."""
        from ouroboros.providers import get_provider, resolve_api_key
        provider = get_provider(provider_name)
        api_key = resolve_api_key(provider) or self._api_key
        base_url = provider.base_url

        try:
            import requests
            url = f"{base_url.rstrip('/')}/generation?id={generation_id}"
            headers = {"Authorization": f"Bearer {api_key}"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            # Generation might not be ready yet — retry once after short delay
            time.sleep(0.5)
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost", exc_info=True)
        return None

    # ------------------------------------------------------------------
    # Text-based tool call rescue
    # ------------------------------------------------------------------

    _TOOLCALL_TAG_RE = re.compile(
        r"TOOLCALL>\s*(\[.*?\])\s*</TOOLCALL>", re.DOTALL
    )
    _TOOLCALL_JSON_RE = re.compile(
        r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
        re.DOTALL,
    )

    def _rescue_text_tool_calls(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls embedded as text in the model response.

        Some models (NVIDIA NIM free tier) occasionally emit tool calls as
        ``TOOLCALL>[...]</TOOLCALL>`` or similar textual patterns instead of
        using the proper JSON ``tool_calls`` field.  This method detects those
        patterns and converts them into the standard OpenAI tool_calls format
        so the agent loop can execute them normally.

        Returns a list of tool-call dicts or *None* if nothing was found.
        """
        if "TOOLCALL" not in content and '"name"' not in content:
            return None

        calls: List[Dict[str, Any]] = []

        # Pattern 1: TOOLCALL>[{...}]</TOOLCALL>
        tag_match = self._TOOLCALL_TAG_RE.search(content)
        if tag_match:
            try:
                items = _json.loads(tag_match.group(1))
                if isinstance(items, list):
                    for item in items:
                        name = item.get("name", "")
                        args = item.get("arguments") or item.get("parameters") or {}
                        if name:
                            calls.append({
                                "id": f"rescued-{uuid.uuid4().hex[:12]}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": _json.dumps(args) if isinstance(args, dict) else str(args),
                                },
                            })
            except (_json.JSONDecodeError, TypeError):
                pass

        # Pattern 2: raw JSON objects with "name" and "arguments"
        if not calls:
            for m in self._TOOLCALL_JSON_RE.finditer(content):
                name = m.group(1)
                try:
                    args = _json.loads(m.group(2))
                    calls.append({
                        "id": f"rescued-{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": _json.dumps(args),
                        },
                    })
                except _json.JSONDecodeError:
                    pass

        return calls if calls else None

    def _resolve_cost(self, provider_name: str, usage: Dict[str, Any], resp_dict: Dict[str, Any]) -> None:
        """Resolve cost from provider-specific sources. Mutates usage dict in-place."""
        from ouroboros.providers import get_provider
        provider = get_provider(provider_name)

        # Extract cached tokens from nested formats
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])

        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (prompt_details_for_write.get("cache_write_tokens")
                              or prompt_details_for_write.get("cache_creation_tokens")
                              or prompt_details_for_write.get("cache_creation_input_tokens"))
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)

        # Provider-specific cost resolution
        if provider_name == "local":
            usage["cost"] = 0.0
        elif provider.supports_generation_cost_api and not usage.get("cost"):
            gen_id = resp_dict.get("id") or ""
            if gen_id:
                cost = self._fetch_generation_cost(gen_id, provider_name)
                if cost is not None:
                    usage["cost"] = cost

    # ------------------------------------------------------------------
    # Unified provider chat
    # ------------------------------------------------------------------

    def _chat_provider(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        reasoning_effort: str,
        max_tokens: int,
        tool_choice: str,
        provider_name: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Unified chat method that adapts to provider quirks via descriptor flags."""
        from ouroboros.providers import get_provider
        provider = get_provider(provider_name)
        client = self._get_client_for_provider(provider_name)

        # Adapt messages to provider capabilities
        if provider.requires_content_flattening:
            clean_messages = self._flatten_content(messages)
        elif not provider.supports_cache_control:
            clean_messages = self._strip_cache_control(messages)
        else:
            clean_messages = messages

        # Build kwargs
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": clean_messages,
            "max_tokens": max_tokens,
        }

        # Local model: override model name & cap tokens
        if provider_name == "local":
            kwargs["model"] = "local-model"
            kwargs["max_tokens"] = self._local_max_tokens(max_tokens)

        # Codex (OpenClaw) gateway: map to openclaw agent model
        if provider_name == "codex":
            kwargs["model"] = "openclaw:main"

        # OpenRouter-specific: reasoning + provider routing
        if provider.supports_reasoning:
            effort = normalize_reasoning_effort(reasoning_effort)
            extra_body: Dict[str, Any] = {
                "reasoning": {"effort": effort, "exclude": True},
            }
            if provider.supports_provider_routing and model.startswith("anthropic/"):
                extra_body["provider"] = {
                    "order": ["Anthropic"],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                }
            kwargs["extra_body"] = extra_body

        # Tools handling
        if tools:
            if provider.supports_cache_control:
                # Add cache_control to last tool (OpenRouter)
                tools_copy = [t for t in tools]
                if tools_copy:
                    last_tool = {**tools_copy[-1]}
                    last_tool["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                    tools_copy[-1] = last_tool
                kwargs["tools"] = tools_copy
            else:
                # Strip cache_control from tools
                kwargs["tools"] = [
                    {k: v for k, v in t.items() if k != "cache_control"}
                    for t in tools
                ]
            # Clamp tool_choice to supported values
            if tool_choice in provider.tool_choice_values:
                kwargs["tool_choice"] = tool_choice
            else:
                kwargs["tool_choice"] = "auto"

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # Rescue text-based tool calls: some models (especially on NVIDIA
        # NIM free tier) occasionally emit tool calls as text instead of
        # using the proper function-calling JSON.  Detect common patterns
        # (<TOOLCALL>, ```tool_call, etc.) and convert them.
        if not msg.get("tool_calls") and msg.get("content"):
            rescued = self._rescue_text_tool_calls(msg["content"])
            if rescued:
                msg["tool_calls"] = rescued
                msg["content"] = None
                log.info("Rescued %d text-based tool call(s)", len(rescued))

        # Resolve cost
        self._resolve_cost(provider_name, usage, resp_dict)

        return msg, usage

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
        use_local: bool = False,
        provider_name: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call. Returns: (response_message_dict, usage_dict with cost).

        Args:
            provider_name: Explicit provider ("openrouter", "nvidia", "openai", "local").
                           If None, infers from use_local flag or env vars.
        """
        # Determine provider
        if provider_name:
            return self._chat_provider(messages, model, tools, reasoning_effort,
                                       max_tokens, tool_choice, provider_name)

        if use_local or os.environ.get("OUROBOROS_USE_LOCAL", "").lower() == "true":
            return self._chat_provider(messages, model, tools, reasoning_effort,
                                       max_tokens, tool_choice, "local")

        # Infer provider from base_url for backward compat
        inferred = self._infer_provider_from_env()
        return self._chat_provider(messages, model, tools, reasoning_effort,
                                   max_tokens, tool_choice, inferred)

    def _infer_provider_from_env(self) -> str:
        """Infer provider from OUROBOROS_LLM_BASE_URL for backward compat."""
        base_url = os.environ.get("OUROBOROS_LLM_BASE_URL", "")
        if "nvidia" in base_url:
            return "nvidia"
        if "api.openai.com" in base_url:
            return "openai"
        return "openrouter"

    def _chat_local(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
        tool_choice: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Legacy wrapper: route to local provider."""
        return self._chat_provider(messages, "local-model", tools, "medium",
                                   max_tokens, tool_choice, "local")

    def _chat_openrouter(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        reasoning_effort: str,
        max_tokens: int,
        tool_choice: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Legacy wrapper: route to openrouter provider."""
        return self._chat_provider(messages, model, tools, reasoning_effort,
                                   max_tokens, tool_choice, "openrouter")

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "anthropic/claude-sonnet-4.6",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
        provider_name: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Send a vision query to an LLM. Lightweight — no tools, no loop."""
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            provider_name=provider_name,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        return os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models

    def get_pricing_info(self) -> Dict[str, Any]:
        """Fetch current pricing info from the configured provider."""
        from ouroboros.providers import PROVIDERS
        result = {}
        for p in PROVIDERS.values():
            result.update(p.pricing)
        # Also fetch dynamic pricing from OpenRouter
        live = fetch_openrouter_pricing()
        if live:
            result.update(live)
        return result