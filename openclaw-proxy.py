#!/usr/bin/env python3
"""
openclaw-proxy.py — OpenAI-compatible REST proxy for OpenClaw/Codex.

Runs as `openclaw` user on port 18780.
Wraps `openclaw agent -m "message"` CLI calls.
Exposes /v1/chat/completions in OpenAI format.

Docker containers can reach this via host.docker.internal:18780
(requires extra_hosts: host-gateway in docker-compose.yml).
"""

import http.server
import json
import logging
import os
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


# ── OpenClaw CLI call ────────────────────────────────────────────────────────

def build_prompt(messages: list) -> str:
    """Combine chat messages into a single prompt string."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        # Flatten multipart content (e.g. vision messages)
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if role == "system":
            parts.append(f"<system>{content}</system>")
        elif role == "assistant":
            parts.append(f"<assistant>{content}</assistant>")
        elif role == "user":
            parts.append(content)
    return "\n\n".join(parts)


def call_openclaw(prompt: str) -> str:
    """Run `openclaw agent -m <prompt>` and return text response."""
    env = {**os.environ, "HOME": os.path.expanduser("~")}

    try:
        result = subprocess.run(
            ["openclaw", "agent", "-m", prompt],
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
        # Common fields
        text = (
            data.get("result")
            or data.get("response")
            or data.get("content")
            or data.get("message")
            or data.get("output")
        )
        if text:
            return str(text)
        # If JSON but no known field, return pretty-printed
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

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
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
                    {
                        "id": "openai-codex/gpt-5.3-codex",
                        "object": "model",
                        "created": 1700000000,
                        "owned_by": "openai",
                    },
                    {
                        "id": "openai-codex/gpt-5.2-codex",
                        "object": "model",
                        "created": 1700000000,
                        "owned_by": "openai",
                    },
                    {
                        "id": "openai-codex/o3",
                        "object": "model",
                        "created": 1700000000,
                        "owned_by": "openai",
                    },
                    {
                        "id": "openai-codex/o4-mini",
                        "object": "model",
                        "created": 1700000000,
                        "owned_by": "openai",
                    },
                ],
            })
        else:
            self._send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):
        path = urlparse(self.path).path

        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._send_json(404, {"error": {"message": "endpoint not found"}})
            return

        # Parse request body
        try:
            raw = self._read_body()
            body = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, ValueError) as e:
            self._send_json(400, {"error": {"message": f"invalid JSON: {e}", "type": "invalid_request_error"}})
            return

        messages = body.get("messages", [])
        model = body.get("model", "openai-codex/gpt-5.3-codex")

        if not messages:
            self._send_json(400, {"error": {"message": "messages array is required", "type": "invalid_request_error"}})
            return

        log.info("POST /v1/chat/completions model=%s messages=%d", model, len(messages))

        # Build prompt and call openclaw
        prompt = build_prompt(messages)
        t0 = time.time()
        response_text = call_openclaw(prompt)
        elapsed = time.time() - t0

        log.info("openclaw done in %.1fs: %d chars", elapsed, len(response_text))

        # Return OpenAI-compatible response
        self._send_json(200, {
            "id": "chatcmpl-" + uuid.uuid4().hex[:16],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": max(1, len(prompt) // 4),
                "completion_tokens": max(1, len(response_text) // 4),
                "total_tokens": max(1, (len(prompt) + len(response_text)) // 4),
            },
        })


# ── Threaded server ───────────────────────────────────────────────────────────

class ThreadedHTTPServer(http.server.ThreadingHTTPServer):
    """Handle each request in a separate thread (thread-per-request)."""
    daemon_threads = True  # Don't block exit on active threads


def main():
    log.info("=" * 60)
    log.info("OpenClaw Proxy starting on %s:%d", HOST, PORT)
    log.info("openclaw path: %s", subprocess.run(
        ["which", "openclaw"], capture_output=True, text=True
    ).stdout.strip() or "NOT FOUND")
    log.info("Running as user: %s", os.environ.get("USER", os.getlogin()))
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
