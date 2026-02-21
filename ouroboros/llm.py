"""
Ouroboros — LLM client.
Supports:
- OpenRouter (default)
- NVIDIA Build / NIM serverless (OpenAI-compatible)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3-pro-preview"


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
    if "cost" in usage and usage.get("cost") is not None:
        total["cost"] = float(total.get("cost") or 0.0) + float(usage.get("cost") or 0.0)


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self._base_url = (base_url or os.environ.get("OUROBOROS_LLM_BASE_URL") or "https://openrouter.ai/api/v1").strip()
        self._is_openrouter = "openrouter.ai" in self._base_url

        self._api_key = (
            api_key
            or os.environ.get("OUROBOROS_LLM_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or ""
        ).strip()

        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            headers = {}
            if self._is_openrouter:
                headers = {
                    "HTTP-Referer": "https://github.com/jkee/ouroboros",
                    "X-Title": "Ouroboros",
                }

            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                default_headers=headers,
            )
        return self._client

    def _fetch_generation_cost(self, generation_id: str) -> Optional[float]:
        """OpenRouter-only fallback cost endpoint."""
        if not self._is_openrouter:
            return None
        try:
            import requests
            url = f"{self._base_url.rstrip('/')}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost", exc_info=True)
        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        client = self._get_client()

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        # OpenRouter-only extensions
        if self._is_openrouter:
            effort = normalize_reasoning_effort(reasoning_effort)
            extra_body: Dict[str, Any] = {"reasoning": {"effort": effort, "exclude": True}}
            if model.startswith("anthropic/"):
                extra_body["provider"] = {"order": ["Anthropic"], "allow_fallbacks": False, "require_parameters": True}
            kwargs["extra_body"] = extra_body

        # Tools: keep OpenAI-compatible shape
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()

        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # cached_tokens / cache_write_tokens if provider returns them
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])

        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (
                    prompt_details_for_write.get("cache_write_tokens")
                    or prompt_details_for_write.get("cache_creation_tokens")
                    or prompt_details_for_write.get("cache_creation_input_tokens")
                )
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)

        # cost: OpenRouter can provide; NVIDIA usually won't. Put 0.0.
        if not usage.get("cost"):
            if self._is_openrouter:
                gen_id = resp_dict.get("id") or ""
                if gen_id:
                    cost = self._fetch_generation_cost(gen_id)
                    if cost is not None:
                        usage["cost"] = cost
            else:
                usage["cost"] = 0.0

        return msg, usage

    def default_model(self) -> str:
        return os.environ.get("OUROBOROS_MODEL", "qwen/qwen3.5-397b-a17b")

    def available_models(self) -> List[str]:
        main = os.environ.get("OUROBOROS_MODEL", self.default_model())
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models
