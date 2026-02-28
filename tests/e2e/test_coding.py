"""E2E: Agent can edit code and commit changes."""
import importlib
import sys
import types

import pytest

from tests.e2e.harness import E2EHarness, git_diff_from_initial


def _load_modified_send_owner_message(repo_dir):
    """Load _send_owner_message from the agent's modified control.py."""
    control_path = repo_dir / "ouroboros" / "tools" / "control.py"
    source = control_path.read_text()

    # Load as an isolated module so it doesn't pollute the real one
    spec = importlib.util.spec_from_file_location("_e2e_control", str(control_path))
    mod = importlib.util.module_from_spec(spec)

    # Provide ouroboros.tools.registry and ouroboros.utils so the module can import
    mod.__dict__["__builtins__"] = __builtins__
    sys.modules["_e2e_control"] = mod
    spec.loader.exec_module(mod)
    del sys.modules["_e2e_control"]

    return getattr(mod, "_send_owner_message")


@pytest.mark.timeout(600)
def test_agent_edits_code_and_commits(e2e_harness: E2EHarness):
    """Agent modifies send_owner_message to append ' :)' and commits."""
    result = e2e_harness.run(
        task_text=(
            "Edit the send_owner_message tool so that every message text "
            "has ' :)' appended to it before sending. Then commit your changes."
        ),
        max_rounds=30,
    )

    # Agent committed something
    assert "repo_commit_push" in result.tools_called, (
        f"Expected repo_commit_push in tools called, got: {result.tools_called}"
    )

    # New commits exist beyond the initial one
    log_lines = result.git_log.strip().splitlines()
    assert len(log_lines) > 1, (
        f"Expected commits beyond initial, got git log:\n{result.git_log}"
    )

    # The diff contains the smiley face change
    diff = git_diff_from_initial(result.repo_dir)
    assert ":)" in diff, (
        f"Expected ':)' in git diff, got:\n{diff[:500]}"
    )

    # Verify the modified code actually works: call the function and check output
    send_fn = _load_modified_send_owner_message(result.repo_dir)

    # Build a minimal mock context
    ctx = types.SimpleNamespace(
        current_chat_id=123,
        pending_events=[],
        drive_logs=lambda: result.drive_root / "logs",
    )
    send_fn(ctx, text="hello")

    # Find the message the function queued
    sent = [e for e in ctx.pending_events if e.get("type") == "send_message"]
    assert sent, "Modified send_owner_message didn't queue any event"
    sent_text = sent[0]["text"]
    assert sent_text.endswith(":)"), (
        f"Expected message to end with ':)', got: {sent_text!r}"
    )

    # Cost guard
    assert result.cost_usd < 5.0, f"Test cost ${result.cost_usd:.2f} exceeded $5.00 limit"
