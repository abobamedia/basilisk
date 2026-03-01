"""
Supervisor — Telegram Bot API client + TelegramChatBridge.

TelegramChatBridge implements the same interface as LocalChatBridge
so the supervisor loop works identically with either backend.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TelegramClient — low-level Telegram Bot API wrapper
# ---------------------------------------------------------------------------

class TelegramClient:
    def __init__(self, token: str):
        self.base = f"https://api.telegram.org/bot{token}"
        self._token = token

    def get_updates(self, offset: int, timeout: int = 10) -> List[Dict[str, Any]]:
        last_err = "unknown"
        for attempt in range(3):
            try:
                r = requests.get(
                    f"{self.base}/getUpdates",
                    params={"offset": offset, "timeout": timeout,
                            "allowed_updates": ["message", "edited_message"]},
                    timeout=timeout + 5,
                )
                r.raise_for_status()
                data = r.json()
                if data.get("ok") is not True:
                    raise RuntimeError(f"Telegram getUpdates failed: {data}")
                return data.get("result") or []
            except Exception as e:
                last_err = repr(e)
                if attempt < 2:
                    time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"Telegram getUpdates failed after retries: {last_err}")

    def send_message(self, chat_id: int, text: str, parse_mode: str = "") -> Tuple[bool, str]:
        last_err = "unknown"
        for attempt in range(3):
            try:
                payload: Dict[str, Any] = {"chat_id": chat_id, "text": text,
                                           "disable_web_page_preview": True}
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                r = requests.post(f"{self.base}/sendMessage", data=payload, timeout=30)
                r.raise_for_status()
                data = r.json()
                if data.get("ok") is True:
                    return True, "ok"
                last_err = f"telegram_api_error: {data}"
            except Exception as e:
                last_err = repr(e)
            if attempt < 2:
                time.sleep(0.8 * (attempt + 1))
        return False, last_err

    def send_chat_action(self, chat_id: int, action: str = "typing") -> bool:
        try:
            r = requests.post(
                f"{self.base}/sendChatAction",
                data={"chat_id": chat_id, "action": action},
                timeout=5,
            )
            return r.status_code == 200
        except Exception:
            log.debug("Failed to send chat action to chat_id=%d", chat_id, exc_info=True)
            return False

    def send_photo(self, chat_id: int, photo_bytes: bytes,
                   caption: str = "") -> Tuple[bool, str]:
        last_err = "unknown"
        for attempt in range(3):
            try:
                files = {"photo": ("screenshot.png", photo_bytes, "image/png")}
                data: Dict[str, Any] = {"chat_id": chat_id}
                if caption:
                    data["caption"] = caption[:1024]
                r = requests.post(
                    f"{self.base}/sendPhoto",
                    data=data, files=files, timeout=30,
                )
                r.raise_for_status()
                resp = r.json()
                if resp.get("ok") is True:
                    return True, "ok"
                last_err = f"telegram_api_error: {resp}"
            except Exception as e:
                last_err = repr(e)
            if attempt < 2:
                time.sleep(0.8 * (attempt + 1))
        return False, last_err

    def download_file_base64(self, file_id: str, max_bytes: int = 10_000_000) -> Tuple[Optional[str], str]:
        try:
            r = requests.get(f"{self.base}/getFile", params={"file_id": file_id}, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                return None, ""
            file_path = data["result"].get("file_path", "")
            file_size = int(data["result"].get("file_size") or 0)
            if file_size > max_bytes:
                return None, ""
            download_url = f"https://api.telegram.org/file/bot{self._token}/{file_path}"
            r2 = requests.get(download_url, timeout=30)
            r2.raise_for_status()
            import base64
            b64 = base64.b64encode(r2.content).decode("ascii")
            ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
            mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                        "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp"}
            return b64, mime_map.get(ext, "image/jpeg")
        except Exception:
            log.warning("Failed to download file_id=%s from Telegram", file_id, exc_info=True)
            return None, ""


# ---------------------------------------------------------------------------
# TelegramChatBridge — same interface as LocalChatBridge
# ---------------------------------------------------------------------------

class TelegramChatBridge:
    """Wraps TelegramClient with the same interface as LocalChatBridge.

    The supervisor loop and workers call get_updates/send_message/etc.
    without knowing whether they talk to WebSocket UI or Telegram.
    """

    def __init__(self, token: str):
        self._tg = TelegramClient(token)
        self._broadcast_fn = None  # unused for Telegram, kept for compat

    def get_updates(self, offset: int, timeout: int = 10) -> List[Dict[str, Any]]:
        return self._tg.get_updates(offset=offset, timeout=timeout)

    def send_message(self, chat_id: int, text: str, parse_mode: str = "") -> Tuple[bool, str]:
        text = _sanitize_text(text)
        if not text.strip():
            return True, "empty"
        if parse_mode == "markdown":
            return _send_markdown_telegram(self._tg, chat_id, text)
        return self._tg.send_message(chat_id, text, parse_mode=parse_mode)

    def send_chat_action(self, chat_id: int, action: str = "typing") -> bool:
        return self._tg.send_chat_action(chat_id, action)

    def send_photo(self, chat_id: int, photo_bytes: bytes,
                   caption: str = "") -> Tuple[bool, str]:
        return self._tg.send_photo(chat_id, photo_bytes, caption)

    def download_file_base64(self, file_id: str, max_bytes: int = 10_000_000) -> Tuple[Optional[str], str]:
        return self._tg.download_file_base64(file_id, max_bytes)

    # Stubs for LocalChatBridge-only methods (UI hooks)
    def push_log(self, event: dict):
        pass  # no web UI to stream to

    def ui_poll_logs(self) -> list:
        return []

    def ui_send(self, text: str):
        pass  # Telegram messages come via get_updates, not UI

    def ui_receive(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        return None


# ---------------------------------------------------------------------------
# Telegram formatting helpers
# ---------------------------------------------------------------------------

def _sanitize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        c for c in text
        if (ord(c) >= 32 or c in ("\n", "\t")) and not (0xD800 <= ord(c) <= 0xDFFF)
    )


def _tg_utf16_len(text: str) -> int:
    if not text:
        return 0
    return sum(2 if ord(c) > 0xFFFF else 1 for c in text)


def _markdown_to_telegram_html(md: str) -> str:
    import html as _html
    md = md or ""
    fence_re = re.compile(r"```[^\n]*\n([\s\S]*?)```", re.MULTILINE)
    fenced: list = []

    def _save_fence(m):
        code = m.group(1).rstrip("\n")
        placeholder = f"\x00FENCE{len(fenced)}\x00"
        fenced.append(f"<pre>{_html.escape(code, quote=False)}</pre>")
        return placeholder

    text = fence_re.sub(_save_fence, md)
    inline_re = re.compile(r"`([^`\n]+)`")
    inlines: list = []

    def _save_inline(m):
        placeholder = f"\x00CODE{len(inlines)}\x00"
        inlines.append(f"<code>{_html.escape(m.group(1), quote=False)}</code>")
        return placeholder

    text = inline_re.sub(_save_inline, text)
    text = _html.escape(text, quote=False)
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*\*([^*\n]+?)\*\*\*", r"<b><i>\1</i></b>", text)
    text = re.sub(r"\*\*([^*\n]+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"~~([^~\n]+?)~~", r"<s>\1</s>", text)
    text = re.sub(r"(?<![*\w])\*([^*\n]+?)\*(?![*\w])", r"<i>\1</i>", text)
    text = re.sub(r"\b_([^_\n]+?)_\b", r"<i>\1</i>", text)
    text = re.sub(r"^[\*\-]\s+", "\u2022 ", text, flags=re.MULTILINE)
    for i, code in enumerate(inlines):
        text = text.replace(f"\x00CODE{i}\x00", code)
    for i, block in enumerate(fenced):
        text = text.replace(f"\x00FENCE{i}\x00", block)
    return text


def _chunk_markdown(md: str, max_chars: int = 3500) -> List[str]:
    md = md or ""
    max_chars = max(256, min(4096, int(max_chars)))
    lines = md.splitlines(keepends=True)
    chunks: List[str] = []
    cur = ""
    in_fence = False
    fence_open = "```\n"
    fence_close = "```\n"

    def _flush():
        nonlocal cur
        if cur and cur.strip():
            chunks.append(cur)
        cur = ""

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            if in_fence:
                fence_open = line if line.endswith("\n") else (line + "\n")
        reserve = _tg_utf16_len(fence_close) if in_fence else 0
        if _tg_utf16_len(cur) + _tg_utf16_len(line) > max_chars - reserve:
            if in_fence and cur:
                cur += fence_close
            _flush()
            cur = fence_open if in_fence else ""
        cur += line
    if in_fence:
        cur += fence_close
    _flush()
    return chunks or [md]


def _send_markdown_telegram(tg: TelegramClient, chat_id: int, text: str) -> Tuple[bool, str]:
    chunks = _chunk_markdown(text, max_chars=3200)
    chunks = [c for c in chunks if isinstance(c, str) and c.strip()]
    if not chunks:
        return False, "empty_chunks"
    last_err = "ok"
    for md_part in chunks:
        html_text = _markdown_to_telegram_html(md_part)
        ok, err = tg.send_message(chat_id, _sanitize_text(html_text), parse_mode="HTML")
        if not ok:
            # Fallback to plain text
            plain = re.sub(r"```[^\n]*\n([\s\S]*?)```", r"\1", md_part)
            plain = re.sub(r"`([^`]+)`", r"\1", plain)
            plain = re.sub(r"\*\*(.+?)\*\*", r"\1", plain)
            plain = re.sub(r"\*(.+?)\*", r"\1", plain)
            plain = re.sub(r"~~(.+?)~~", r"\1", plain)
            if not plain.strip():
                return False, err
            ok2, err2 = tg.send_message(chat_id, _sanitize_text(plain))
            if not ok2:
                return False, err2
        last_err = err
    return True, last_err
