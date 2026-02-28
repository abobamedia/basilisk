"""E2E: Agent can boot and answer a simple question."""
import pytest

from tests.e2e.harness import E2EHarness


@pytest.mark.timeout(300)
def test_agent_answers_question(e2e_harness: E2EHarness):
    """System boots, agent reads VERSION file and reports it."""
    result = e2e_harness.run(
        task_text="What version are you running? Read the VERSION file and tell me.",
        max_rounds=10,
    )

    # Agent produced a non-trivial response
    assert result.final_response, "Agent returned empty response"
    assert len(result.final_response) > 10, f"Response too short: {result.final_response!r}"

    # Agent used a file-reading tool
    assert "repo_read" in result.tools_called, (
        f"Expected repo_read in tools called, got: {result.tools_called}"
    )

    # VERSION string appears in the response
    version_file = (result.repo_dir / "VERSION").read_text().strip()
    assert version_file in result.final_response, (
        f"Expected version {version_file!r} in response: {result.final_response[:200]}"
    )

    # Cost guard
    assert result.cost_usd < 1.0, f"Test cost ${result.cost_usd:.2f} exceeded $1.00 limit"
    assert result.rounds <= 10, f"Took {result.rounds} rounds (max 10)"
