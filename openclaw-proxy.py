#!/usr/bin/env python3
"""
openclaw-proxy.py — OpenAI-compatible REST proxy for OpenClaw/Codex.

Runs as `openclaw` user on port 18780.
Wraps `openclaw agent -m "message"` CLI calls.
Exposes /v1/chat/completions in OpenAI format with tool calling support.

Tool calling is simulated via prompt injection: tool schemas are appended
to the prompt and the model responds with JSON when it wants to call a tool.

Docker containers can reach this via host.docker.internal:18780
(requires extra_hosts: host-gateway in docker-compose.yml).
"""

import http.server
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from urllib.parse import urlparse

# ── Config ──────────────────────────────────────────────────────────────────
PORT = 18780
HOST = "0.0.0.0"
OPENCLAW_TIMEOUT = 180  # seconds
LOG_FILE = os.path.expanduser("~/openclaw-proxy.log")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("openclaw-proxy")


# ── Prompt Building ───────────────────────────────────────────────────────────

def _flatten_content(content):
    """Flatten multipart content (e.g. vision messages) to string."""
    if isinstance(content, list):
        return " ".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return content or ""


def build_prompt(messages, tools=None):
    """Combine chat messages into a single prompt string.

    If tools is provided, prepends a strong function-calling instruction
    so the model outputs JSON tool calls instead of acting on its own.
    """
    parts = []

    # When tools are provided, put the instruction FIRST so it dominates
    # the model's context before any conversation history.
    if tools:
        tools_json = json.dumps(tools, ensure_ascii=False, indent=2)
        example_id = "call_" + uuid.uuid4().hex[:8]
        header = (
            "=== FUNCTION CALLING MODE — READ THIS FIRST ===\n"
            "You are acting as a FUNCTION ROUTER. Your ONLY job is to decide\n"
            "which function to call and output the call as JSON.\n"
            "\n"
            "MANDATORY RULES:\n"
            "1. DO NOT perform any action yourself (no reading files, no running commands).\n"
            "2. DO NOT explain what you are doing.\n"
            "3. DO NOT use your own built-in tools.\n"
            "4. Output ONLY this JSON and nothing else:\n"
            '   {"tool_calls": [{"id": "' + example_id + '", "type": "function", '
            '"function": {"name": "FUNCTION_NAME", "arguments": "{\\"key\\": \\"value\\"}"}}]}\n'
            "   Where 'arguments' is a JSON-encoded STRING.\n"
            "5. If NO function is needed (pure text answer), respond normally.\n"
            "\n"
            "Available functions:\n"
            f"{tools_json}\n"
            "=== END OF FUNCTION CALLING INSTRUCTIONS ==="
        )
        parts.append(header)

    for m in messages:
        role = m.get("role", "user")
        content = _flatten_content(m.get("content", ""))

        if role == "system":
            parts.append(f"<system>{content}</system>")

        elif role == "assistant":
            tool_calls = m.get("tool_calls")
            if tool_calls:
                tc_json = json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)
                parts.append(f"<assistant_tool_calls>{tc_json}</assistant_tool_calls>")
            else:
                parts.append(f"<assistant>{content}</assistant>")

        elif role == "tool":
            call_id = m.get("tool_call_id", "")
            name = m.get("name", "tool")
            parts.append(
                f"<tool_result name=\"{name}\" call_id=\"{call_id}\">{content}</tool_result>"
            )

        elif role == "user":
            parts.append(content)

    return "\n\n".join(parts)


# ── Tool Call Parsing ─────────────────────────────────────────────────────────

def _normalize_tool_calls(raw_calls):
    """Normalize tool calls list to proper OpenAI format.

    Handles cases where 'arguments' was returned as a dict instead of string.
    """
    result = []
    for tc in raw_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        name = fn.get("name", "")
        if not name:
            continue

        arguments = fn.get("arguments", "{}")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        elif not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)

        result.append({
            "id": tc.get("id") or ("call_" + uuid.uuid4().hex[:8]),
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        })
    return result


def parse_tool_calls(text):
    """Try to parse text as tool calls JSON.

    Returns (text_content_or_None, tool_calls_or_None).
    If tool_calls is not None, text_content will be None (model chose to call a tool).
    """
    stripped = text.strip()

    # Case 1: response is a bare JSON object
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            raw = data.get("tool_calls")
            if raw and isinstance(raw, list):
                normalized = _normalize_tool_calls(raw)
                if normalized:
                    return None, normalized
        except json.JSONDecodeError:
            pass

    # Case 2: JSON embedded somewhere in text (after explanation etc.)
    match = re.search(r'\{\s*"tool_calls"\s*:', stripped, re.DOTALL)
    if match:
        start = match.start()
        depth = 0
        end = start
        for i, ch in enumerate(stripped[start:]):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = start + i
                    break
        try:
            data = json.loads(stripped[start : end + 1])
            raw = data.get("tool_calls")
            if raw and isinstance(raw, list):
                normalized = _normalize_tool_calls(raw)
                if normalized:
                    return None, normalized
        except (json.JSONDecodeError, ValueError):
            pass

    # Case 3: <TOOLCALL>[{"name": "...", "arguments": {...}}]</TOOLCALL>
    # This is the format llm.py's rescue parser knows, added as fallback
    tag_match = re.search(r'TOOLCALL>\s*(\[.*?\])\s*</TOOLCALL>', stripped, re.DOTALL)
    if tag_match:
        try:
            items = json.loads(tag_match.group(1))
            if isinstance(items, list):
                calls = []
                for item in items:
                    name = item.get("name", "")
                    args = item.get("arguments") or item.get("parameters") or {}
                    if name:
                        calls.append({
                            "id": "call_" + uuid.uuid4().hex[:8],
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                            },
                        })
                if calls:
                    return None, calls
        except (json.JSONDecodeError, TypeError):
            pass

    # Case 4: simple {"name": "...", "arguments": {...}} object (llm.py rescue pattern)
    simple_match = re.search(
        r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
        stripped, re.DOTALL
    )
    if simple_match:
        try:
            name = simple_match.group(1)
            args = json.loads(simple_match.group(2))
            return None, [{
                "id": "call_" + uuid.uuid4().hex[:8],
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }]
        except (json.JSONDecodeError, ValueError):
            pass

    return text, None


# ── OpenClaw CLI call ────────────────────────────────────────────────────────

def call_openclaw(prompt):
    """Run `openclaw agent -m <prompt>` and return text response."""
    env = {**os.environ, "HOME": os.path.expanduser("~")}

    try:
        session_id = uuid.uuid4().hex
        result = subprocess.run(
            ["openclaw", "agent", "--agent", "main", "--session-id", session_id, "-m", prompt],
            capture_output=True,
            text=True,
            timeout=OPENCLAW_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        log.error("openclaw agent timed out after %ds", OPENCLAW_TIMEOUT)
        return "[Превышено время ожидания ответа от Codex]"
    except FileNotFoundError:
        log.error("openclaw CLI not found in PATH=%s", env.get("PATH", ""))
        return "[Ошибка: openclaw CLI не найден. Проверьте PATH.]"
    except Exception as e:
        log.error("call_openclaw error: %s", e)
        return f"[Ошибка OpenClaw: {e}]"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()

    if result.returncode != 0 and not stdout:
        log.warning("openclaw rc=%d stderr=%s", result.returncode, stderr[:300])
        return f"[Ошибка OpenClaw (rc={result.returncode}): {stderr[:200] if stderr else 'нет вывода'}]"

    if not stdout:
        log.warning("openclaw returned empty stdout (rc=%d)", result.returncode)
        return "[Нет ответа от Codex]"

    # Try to parse as JSON (some openclaw versions return JSON)
    try:
        data = json.loads(stdout)
        text = (
            data.get("result")
            or data.get("response")
            or data.get("content")
            or data.get("message")
            or data.get("output")
        )
        if text:
            return str(text)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        pass

    return stdout


# ── HTTP Handler ─────────────────────────────────────────────────────────────

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    """Simple OpenAI-compatible HTTP handler."""

    def log_message(self, fmt, *args):  # silence default logging
        log.debug("%s - " + fmt, self.address_string(), *args)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _send_json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    # ── Routes ───────────────────────────────────────────────────────────────

    def do_OPTIONS(self):
        """CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            self._send_json(200, {"status": "ok", "proxy": "openclaw", "port": PORT})

        elif path in ("/v1/models", "/models"):
            self._send_json(200, {
                "object": "list",
                "data": [
                    {"id": "openai-codex/gpt-5.3-codex", "object": "model",
                     "created": 1700000000, "owned_by": "openai"},
                    {"id": "openai-codex/gpt-5.2-codex", "object": "model",
                     "created": 1700000000, "owned_by": "openai"},
                    {"id": "openai-codex/o3", "object": "model",
                     "created": 1700000000, "owned_by": "openai"},
                    {"id": "openai-codex/o4-mini", "object": "model",
                     "created": 1700000000, "owned_by": "openai"},
                ],
            })
        else:
            self._send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):
        path = urlparse(self.path).path

        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": {"message": "endpoint not found"}})
            return

        try:
            raw = self._read_body()
            body = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, ValueError) as e:
            self._send_json(400, {"error": {
                "message": f"invalid JSON: {e}",
                "type": "invalid_request_error",
            }})
            return

        messages = body.get("messages", [])
        model = body.get("model", "openai-codex/gpt-5.3-codex")
        tools = body.get("tools") or None  # treat empty list as None

        if not messages:
            self._send_json(400, {"error": {
                "message": "messages array is required",
                "type": "invalid_request_error",
            }})
            return

        log.info(
            "POST /v1/chat/completions model=%s messages=%d tools=%d",
            model, len(messages), len(tools) if tools else 0,
        )

        prompt = build_prompt(messages, tools)
        t0 = time.time()
        response_text = call_openclaw(prompt)
        elapsed = time.time() - t0

        log.info("openclaw done in %.1fs: %d chars", elapsed, len(response_text))

        # Determine if response contains tool calls
        finish_reason = "stop"
        message = {"role": "assistant"}

        if tools:
            text_content, tool_calls = parse_tool_calls(response_text)
            if tool_calls:
                log.info("Detected %d tool call(s): %s",
                         len(tool_calls),
                         [tc["function"]["name"] for tc in tool_calls])
                message["content"] = None
                message["tool_calls"] = tool_calls
                finish_reason = "tool_calls"
            else:
                message["content"] = text_content or response_text
        else:
            message["content"] = response_text

        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(response_text) // 4)
        self._send_json(200, {
            "id": "chatcmpl-" + uuid.uuid4().hex[:16],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })


# ── Threaded server ───────────────────────────────────────────────────────────

class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    """Handle each request in a separate thread (thread-per-request)."""
    daemon_threads = True  # Don't block exit on active threads


def main():
    log.info("=" * 60)
    log.info("OpenClaw Proxy starting on %s:%d (tool calling enabled)", HOST, PORT)
    log.info("openclaw path: %s", subprocess.run(
        ["which", "openclaw"], capture_output=True, text=True
    ).stdout.strip() or "NOT FOUND")
    try:
        import pwd
        username = pwd.getpwuid(os.getuid()).pw_name
    except Exception:
        username = os.environ.get("USER", str(os.getuid()))
    log.info("Running as user: %s", username)
    log.info("=" * 60)

    server = ThreadedHTTPServer((HOST, PORT), ProxyHandler)
    log.info("OpenClaw Proxy ready — listening on port %d", PORT)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
