"""E2E test configuration — fixtures and skip logic."""
from __future__ import annotations

import os

import pytest

from tests.e2e.harness import E2EHarness


def pytest_collection_modifyitems(config, items):
    """Skip all E2E tests if OPENROUTER_API_KEY is not set."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    skip = pytest.mark.skip(reason="OPENROUTER_API_KEY not set — skipping E2E tests")
    for item in items:
        if "e2e" in str(item.fspath):
            item.add_marker(skip)


@pytest.fixture
def e2e_harness(tmp_path):
    """Create an E2EHarness with a fresh temp directory."""
    harness = E2EHarness(work_dir=tmp_path)
    yield harness
