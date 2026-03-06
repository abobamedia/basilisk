"""
Ouroboros Agent Server — Self-editable entry point.

This file lives in REPO_DIR and can be modified by the agent.
It runs as a subprocess of the launcher, serving the web UI and
coordinating the supervisor/worker system.

Starlette + uvicorn on localhost:{PORT}.
"""

import asyncio
import json
import logging
import os
import pathlib
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse, FileResponse
from starlette.routing import Route, Mount, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

import uvicorn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR = pathlib.Path(os.environ.get("OUROBOROS_REPO_DIR", pathlib.Path(__file__).parent))
DATA_DIR = pathlib.Path(os.environ.get("OUROBOROS_DATA_DIR",
    pathlib.Path.home() / "Ouroboros" / "data"))
PORT = int(os.environ.get("OUROBOROS_SERVER_PORT", "8765"))

sys.path.insert(0, str(REPO_DIR))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_log_dir = DATA_DIR / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
from logging.handlers import RotatingFileHandler
_file_handler = RotatingFileHandler(
    _log_dir / "server.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, handlers=[_file_handler, logging.StreamHandler()])
log = logging.getLogger("server")

# ---------------------------------------------------------------------------
# Restart signal
# ---------------------------------------------------------------------------
RESTART_EXIT_CODE = 42
PANIC_EXIT_CODE = 99
_restart_requested = threading.Event()

# ---------------------------------------------------------------------------
# WebSocket connections manager
# ---------------------------------------------------------------------------
_ws_clients: List[WebSocket] = []
_ws_lock = threading.Lock()


async def broadcast_ws(msg: dict) -> None:
    """Send a message to all connected WebSocket clients."""
    data = json.dumps(msg, ensure_ascii=False, default=str)
    with _ws_lock:
        clients = list(_ws_clients)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    if dead:
        with _ws_lock:
            for ws in dead:
                try:
                    _ws_clients.remove(ws)
                except ValueError:
                    pass


def broadcast_ws_sync(msg: dict) -> None:
    """Thread-safe sync wrapper for broadcasting.

    Uses the saved _event_loop reference (set in startup_event) rather than
    asyncio.get_event_loop(), which is unreliable from non-main threads
    in Python 3.10+.
    """
    loop = _event_loop
    if loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(broadcast_ws(msg), loop)
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Settings (single source of truth: ouroboros.config)
# ---------------------------------------------------------------------------
from ouroboros.config import (
    SETTINGS_DEFAULTS as _SETTINGS_DEFAULTS,
    load_settings, save_settings, apply_settings_to_env as _apply_settings_to_env,
)


# ---------------------------------------------------------------------------
# Supervisor integration
# ---------------------------------------------------------------------------
_supervisor_ready = threading.Event()
_supervisor_error: Optional[str] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _init_chat_bridge(settings: dict):
    """Initialize chat bridge (Telegram or local WebSocket) based on settings."""
    _tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip() or settings.get("TELEGRAM_BOT_TOKEN", "").strip()
    if _tg_token:
        from supervisor.telegram import TelegramChatBridge
        _mini_url = os.environ.get("MINI_APP_URL", "").strip() or settings.get("MINI_APP_URL", "").strip()
        bridge = TelegramChatBridge(token=_tg_token, mini_app_url=_mini_url)
        log.info("Using Telegram bridge (bot token set). Mini App URL: %s", _mini_url or "(not set)")
    else:
        from supervisor.message_bus import LocalChatBridge
        bridge = LocalChatBridge()
        bridge._broadcast_fn = broadcast_ws_sync
        log.info("Using local WebSocket bridge (no Telegram token).")
    return bridge


def _run_supervisor(settings: dict) -> None:
    """Initialize and run the supervisor loop. Called in a background thread."""
    global _supervisor_error

    _apply_settings_to_env(settings)

    try:
        bridge = _init_chat_bridge(settings)

        from supervisor.message_bus import init as bus_init
        from ouroboros.utils import set_log_sink
        set_log_sink(bridge.push_log)

        bus_init(
            drive_root=DATA_DIR,
            total_budget_limit=float(settings.get("TOTAL_BUDGET", 10.0)),
            budget_report_every=10,
            chat_bridge=bridge,
        )

        from supervisor.state import init as state_init, init_state, load_state, save_state
        from supervisor.state import append_jsonl, update_budget_from_usage, rotate_chat_log_if_needed
        state_init(DATA_DIR, float(settings.get("TOTAL_BUDGET", 10.0)))
        init_state()

        from supervisor.git_ops import init as git_ops_init, ensure_repo_present, safe_restart
        git_ops_init(
            repo_dir=REPO_DIR, drive_root=DATA_DIR, remote_url="",
            branch_dev="ouroboros", branch_stable="ouroboros-stable",
        )
        ensure_repo_present()
        _gh_user = settings.get("GITHUB_USER", "")
        _gh_repo = settings.get("GITHUB_REPO", "")
        _gh_token = settings.get("GITHUB_TOKEN", "")
        # Build repo slug: "owner/repo" format
        if "/" in _gh_repo:
            _repo_slug = _gh_repo  # already full slug
        elif _gh_user and _gh_repo:
            _repo_slug = f"{_gh_user}/{_gh_repo}"
        else:
            _repo_slug = _gh_repo
        if _repo_slug and _gh_token:
            from supervisor.git_ops import configure_remote
            configure_remote(_repo_slug, _gh_token)
        ok, msg = safe_restart(reason="bootstrap", unsynced_policy="rescue_and_reset")
        if not ok:
            log.error("Supervisor bootstrap failed: %s", msg)

        from supervisor.queue import (
            enqueue_task, enforce_task_timeouts, enqueue_evolution_task_if_needed,
            persist_queue_snapshot, restore_pending_from_snapshot,
            cancel_task_by_id, queue_review_task, sort_pending,
        )
        from supervisor.workers import (
            init as workers_init, get_event_q, WORKERS, PENDING, RUNNING,
            spawn_workers, kill_workers, assign_tasks, ensure_workers_healthy,
            handle_chat_direct, _get_chat_agent, auto_resume_after_restart,
        )

        max_workers = int(settings.get("OUROBOROS_MAX_WORKERS", 5))
        soft_timeout = int(settings.get("OUROBOROS_SOFT_TIMEOUT_SEC", 600))
        hard_timeout = int(settings.get("OUROBOROS_HARD_TIMEOUT_SEC", 1800))

        workers_init(
            repo_dir=REPO_DIR, drive_root=DATA_DIR, max_workers=max_workers,
            soft_timeout=soft_timeout, hard_timeout=hard_timeout,
            total_budget_limit=float(settings.get("TOTAL_BUDGET", 10.0)),
            branch_dev="ouroboros", branch_stable="ouroboros-stable",
        )

        from supervisor.events import dispatch_event
        from supervisor.message_bus import send_with_budget
        from ouroboros.consciousness import BackgroundConsciousness
        import types
        import queue as _queue_mod

        kill_workers()
        spawn_workers(max_workers)
        restored_pending = restore_pending_from_snapshot()
        persist_queue_snapshot(reason="startup")

        if restored_pending > 0:
            st_boot = load_state()
            if st_boot.get("owner_chat_id"):
                send_with_budget(int(st_boot["owner_chat_id"]),
                    f"♻️ Restored pending queue from snapshot: {restored_pending} tasks.")

        auto_resume_after_restart()

        def _get_owner_chat_id() -> Optional[int]:
            try:
                st = load_state()
                cid = st.get("owner_chat_id")
                return int(cid) if cid else None
            except Exception:
                return None

        _consciousness = BackgroundConsciousness(
            drive_root=DATA_DIR, repo_dir=REPO_DIR,
            event_queue=get_event_q(), owner_chat_id_fn=_get_owner_chat_id,
        )

        _bg_st = load_state()
        if _bg_st.get("bg_consciousness_enabled"):
            _consciousness.start()
            log.info("Background consciousness auto-restored from saved state.")

        _event_ctx = types.SimpleNamespace(
            DRIVE_ROOT=DATA_DIR, REPO_DIR=REPO_DIR,
            BRANCH_DEV="ouroboros", BRANCH_STABLE="ouroboros-stable",
            bridge=bridge, WORKERS=WORKERS, PENDING=PENDING, RUNNING=RUNNING,
            MAX_WORKERS=max_workers,
            send_with_budget=send_with_budget, load_state=load_state, save_state=save_state,
            update_budget_from_usage=update_budget_from_usage, append_jsonl=append_jsonl,
            enqueue_task=enqueue_task, cancel_task_by_id=cancel_task_by_id,
            queue_review_task=queue_review_task, persist_queue_snapshot=persist_queue_snapshot,
            safe_restart=safe_restart, kill_workers=kill_workers, spawn_workers=spawn_workers,
            sort_pending=sort_pending, consciousness=_consciousness,
            request_restart=_request_restart_exit,
        )
    except Exception as exc:
        _supervisor_error = f"Supervisor init failed: {exc}"
        log.critical("Supervisor initialization failed", exc_info=True)
        _supervisor_ready.set()
        return

    _supervisor_ready.set()
    log.info("Supervisor ready.")

    # Main supervisor loop
    offset = 0
    crash_count = 0
    while not _restart_requested.is_set():
        try:
            rotate_chat_log_if_needed(DATA_DIR)
            ensure_workers_healthy()

            event_q = get_event_q()
            while True:
                try:
                    evt = event_q.get_nowait()
                except _queue_mod.Empty:
                    break
                if evt.get("type") == "restart_request":
                    _handle_restart_in_supervisor(evt, _event_ctx)
                    continue
                dispatch_event(evt, _event_ctx)

            enforce_task_timeouts()
            enqueue_evolution_task_if_needed()
            assign_tasks()
            persist_queue_snapshot(reason="main_loop")

            # Process messages from bridge (WebSocket or Telegram)
            updates = bridge.get_updates(offset=offset, timeout=1)
            for upd in updates:
                offset = int(upd["update_id"]) + 1
                msg = upd.get("message") or {}
                if not msg:
                    continue

                # Extract real chat_id/user_id from Telegram update,
                # fall back to 1 for local WebSocket bridge
                chat_id = int((msg.get("chat") or {}).get("id") or 1)
                user_id = int((msg.get("from") or {}).get("id") or 1)
                text = str(msg.get("text") or "")
                now_iso = datetime.now(timezone.utc).isoformat()

                st = load_state()
                # Always update owner identity from real message
                # (critical for Telegram: chat_id changes between sessions/restarts)
                if st.get("owner_id") is None or (chat_id != 1 and st.get("owner_chat_id") != chat_id):
                    st["owner_id"] = user_id
                    st["owner_chat_id"] = chat_id

                from supervisor.message_bus import log_chat
                log_chat("in", chat_id, user_id, text)
                st["last_owner_message_at"] = now_iso
                save_state(st)

                if not text:
                    continue

                lowered = text.strip().lower()
                if lowered.startswith("/panic"):
                    send_with_budget(chat_id, "🛑 PANIC: killing everything. App will close.")
                    _execute_panic_stop(_consciousness, kill_workers)
                elif lowered.startswith("/restart"):
                    send_with_budget(chat_id, "♻️ Restarting (soft).")
                    ok, restart_msg = safe_restart(reason="owner_restart", unsynced_policy="rescue_and_reset")
                    if not ok:
                        send_with_budget(chat_id, f"⚠️ Restart cancelled: {restart_msg}")
                        continue
                    kill_workers()
                    _request_restart_exit()
                elif lowered.startswith("/review"):
                    queue_review_task(reason="owner:/review", force=True)
                elif lowered.startswith("/evolve"):
                    parts = lowered.split()
                    action = parts[1] if len(parts) > 1 else "on"
                    turn_on = action not in ("off", "stop", "0")
                    st2 = load_state()
                    st2["evolution_mode_enabled"] = bool(turn_on)
                    if turn_on:
                        st2["evolution_consecutive_failures"] = 0
                    save_state(st2)
                    if not turn_on:
                        PENDING[:] = [t for t in PENDING if str(t.get("type")) != "evolution"]
                        sort_pending()
                        persist_queue_snapshot(reason="evolve_off")
                    state_str = "ON" if turn_on else "OFF"
                    send_with_budget(chat_id, f"🧬 Evolution: {state_str}")
                elif lowered.startswith("/bg"):
                    parts = lowered.split()
                    action = parts[1] if len(parts) > 1 else "status"
                    if action in ("start", "on", "1"):
                        result = _consciousness.start()
                        _bg_s = load_state(); _bg_s["bg_consciousness_enabled"] = True; save_state(_bg_s)
                        send_with_budget(chat_id, f"🧠 {result}")
                    elif action in ("stop", "off", "0"):
                        result = _consciousness.stop()
                        _bg_s = load_state(); _bg_s["bg_consciousness_enabled"] = False; save_state(_bg_s)
                        send_with_budget(chat_id, f"🧠 {result}")
                    else:
                        bg_status = "running" if _consciousness.is_running() else "stopped"
                        next_wake = _consciousness.get_next_wakeup()
                        if next_wake:
                            send_with_budget(chat_id, f"🧠 Background consciousness: {bg_status}. Next wakeup: {next_wake}.")
                        else:
                            send_with_budget(chat_id, f"🧠 Background consciousness: {bg_status}.")
                elif lowered.startswith("/status"):
                    from supervisor.workers import get_status_report
                    report = get_status_report()
                    send_with_budget(chat_id, report)
                else:
                    handle_chat_direct(chat_id, user_id, text)

        except KeyboardInterrupt:
            log.info("Supervisor interrupted by user.")
            break
        except Exception as exc:
            crash_count += 1
            log.error("Supervisor loop crashed (count=%d): %s", crash_count, exc, exc_info=True)
            if crash_count >= 5:
                log.critical("Supervisor crashed 5 times. Exiting.")
                break
            time.sleep(2)

    log.info("Supervisor loop exiting.")
    kill_workers()
    _consciousness.stop()


def _execute_panic_stop(consciousness, kill_workers_fn) -> None:
    """Execute emergency stop: kill all workers, stop consciousness, exit with PANIC code."""
    try:
        consciousness.stop()
    except Exception as e:
        log.error("Failed to stop consciousness during panic: %s", e)
    try:
        kill_workers_fn()
    except Exception as e:
        log.error("Failed to kill workers during panic: %s", e)
    log.critical("PANIC STOP executed. Exiting with code %d.", PANIC_EXIT_CODE)
    os._exit(PANIC_EXIT_CODE)


def _request_restart_exit() -> None:
    """Signal the server to exit with RESTART_EXIT_CODE."""
    _restart_requested.set()


def _handle_restart_in_supervisor(evt: dict, ctx) -> None:
    """Handle restart_request event from worker."""
    reason = evt.get("reason", "agent_request")
    log.info("Restart requested by agent: %s", reason)
    from supervisor.git_ops import safe_restart
    ok, msg = safe_restart(reason=reason, unsynced_policy="rescue_and_reset")
    if not ok:
        owner_cid = ctx.load_state().get("owner_chat_id")
        if owner_cid:
            ctx.send_with_budget(int(owner_cid), f"⚠️ Restart cancelled: {msg}")
        return
    ctx.kill_workers()
    _request_restart_exit()


# ---------------------------------------------------------------------------
# HTTP Routes
# ---------------------------------------------------------------------------

async def index(request: Request) -> HTMLResponse:
    """Serve the main UI."""
    html_path = REPO_DIR / "web" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>UI not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


async def settings_page(request: Request) -> HTMLResponse:
    """Serve the settings page."""
    html_path = REPO_DIR / "web" / "settings.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Settings page not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


async def api_health(request: Request) -> JSONResponse:
    """GET /api/health — supervisor readiness check."""
    if _supervisor_error:
        return JSONResponse({"ok": False, "error": _supervisor_error}, status_code=500)
    if not _supervisor_ready.is_set():
        return JSONResponse({"ok": False, "error": "Supervisor not ready"}, status_code=503)
    return JSONResponse({"ok": True})


async def api_send_message(request: Request) -> JSONResponse:
    """POST /api/send_message — send a message to the owner."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)
    
    text = body.get("text", "")
    if not text:
        return JSONResponse({"ok": False, "error": "Missing 'text' field"}, status_code=400)
    
    # Validate chat_id if provided
    chat_id_raw = body.get("chat_id")
    if chat_id_raw is not None:
        try:
            chat_id = int(chat_id_raw)
        except (ValueError, TypeError):
            return JSONResponse(
                {"ok": False, "error": f"Invalid chat_id: must be integer, got {type(chat_id_raw).__name__}"},
                status_code=400
            )
    else:
        from supervisor.state import load_state
        st = load_state()
        chat_id = int(st.get("owner_chat_id") or 1)

    from supervisor.message_bus import get_bridge
    get_bridge().ui_send(text)

    return JSONResponse({"ok": True})


async def api_get_settings(request: Request) -> JSONResponse:
    """GET /api/settings — return current settings."""
    settings = load_settings()
    # Mask sensitive fields
    masked = dict(settings)
    for key in ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", 
                "GITHUB_TOKEN", "TELEGRAM_BOT_TOKEN", "BONSAI_API_KEY"]:
        if key in masked and masked[key]:
            masked[key] = "***" + masked[key][-4:] if len(masked[key]) > 4 else "***"
    return JSONResponse({"ok": True, "settings": masked})


async def api_save_settings(request: Request) -> JSONResponse:
    """POST /api/settings — save settings."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON"}, status_code=400)
    
    new_settings = body.get("settings", {})
    if not isinstance(new_settings, dict):
        return JSONResponse({"ok": False, "error": "'settings' must be a dict"}, status_code=400)
    
    # Merge with existing settings
    current = load_settings()
    for key, value in new_settings.items():
        # Skip masked values (unchanged secrets)
        if isinstance(value, str) and value.startswith("***"):
            continue
        current[key] = value
    
    save_settings(current)
    _apply_settings_to_env(current)
    
    return JSONResponse({"ok": True})


async def api_get_state(request: Request) -> JSONResponse:
    """GET /api/state — return current state."""
    from supervisor.state import load_state
    st = load_state()
    return JSONResponse({"ok": True, "state": st})


async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket /ws — bidirectional chat."""
    await websocket.accept()
    with _ws_lock:
        _ws_clients.append(websocket)
    log.info("WebSocket client connected. Total: %d", len(_ws_clients))
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue
            
            if msg.get("type") == "chat":
                text = msg.get("text", "")
                if text:
                    from supervisor.message_bus import get_bridge
                    get_bridge().ui_send(text)
    except WebSocketDisconnect:
        pass
    finally:
        with _ws_lock:
            try:
                _ws_clients.remove(websocket)
            except ValueError:
                pass
        log.info("WebSocket client disconnected. Total: %d", len(_ws_clients))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

routes = [
    Route("/", index),
    Route("/settings", settings_page),
    Route("/api/health", api_health),
    Route("/api/send_message", api_send_message, methods=["POST"]),
    Route("/api/settings", api_get_settings),
    Route("/api/settings", api_save_settings, methods=["POST"]),
    Route("/api/state", api_get_state),
    WebSocketRoute("/ws", websocket_endpoint),
    Mount("/static", StaticFiles(directory=str(REPO_DIR / "web")), name="static"),
]

app = Starlette(debug=False, routes=routes)


@app.on_event("startup")
async def startup_event() -> None:
    """Start supervisor in background thread."""
    global _event_loop
    _event_loop = asyncio.get_event_loop()
    
    settings = load_settings()
    supervisor_thread = threading.Thread(target=_run_supervisor, args=(settings,), daemon=True)
    supervisor_thread.start()
    
    # Wait for supervisor to be ready (with timeout)
    ready = _supervisor_ready.wait(timeout=30)
    if not ready:
        log.critical("Supervisor did not become ready within 30 seconds.")
        os._exit(1)
    if _supervisor_error:
        log.critical("Supervisor failed to start: %s", _supervisor_error)
        os._exit(1)
    
    log.info("Server startup complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the server."""
    log.info("Starting Ouroboros server on http://127.0.0.1:%d", PORT)
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="info")
    server = uvicorn.Server(config)
    
    try:
        server.run()
    except KeyboardInterrupt:
        log.info("Server interrupted by user.")
    finally:
        if _restart_requested.is_set():
            log.info("Exiting with restart code %d", RESTART_EXIT_CODE)
            sys.exit(RESTART_EXIT_CODE)
        else:
            log.info("Server shutdown complete.")
            sys.exit(0)


if __name__ == "__main__":
    main()
