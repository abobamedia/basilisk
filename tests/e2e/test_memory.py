"""E2E: Agent can persist and recall memory across tasks."""
import pytest

from tests.e2e.harness import E2EHarness


CANARY = "PINEAPPLE_THUNDER_42"


@pytest.mark.timeout(600)
def test_scratchpad_persistence(e2e_harness: E2EHarness):
    """Agent writes to scratchpad, then a second task reads it back."""

    # Task 1: write a unique canary string to the scratchpad
    r1 = e2e_harness.run(
        task_text=(
            f"Write exactly this text to your scratchpad: '{CANARY}'. "
            "Use the update_scratchpad tool. Nothing else."
        ),
        max_rounds=5,
    )

    assert "update_scratchpad" in r1.tools_called, (
        f"Expected update_scratchpad in tools called, got: {r1.tools_called}"
    )

    # Verify scratchpad file was actually written
    scratchpad = (e2e_harness.drive_root / "memory" / "scratchpad.md").read_text()
    assert CANARY in scratchpad, (
        f"Canary not found in scratchpad file: {scratchpad[:200]}"
    )

    # Task 2: ask the agent what's in its scratchpad — it should see
    # the canary in its context (scratchpad is loaded automatically)
    r2 = e2e_harness.run(
        task_text="What is currently in your scratchpad? Tell me the exact contents.",
        max_rounds=5,
    )

    assert CANARY in r2.final_response, (
        f"Expected canary '{CANARY}' in response: {r2.final_response[:300]}"
    )

    # Cost guard
    total_cost = r1.cost_usd + r2.cost_usd
    assert total_cost < 2.0, f"Test cost ${total_cost:.2f} exceeded $2.00 limit"
