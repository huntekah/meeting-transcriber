"""Shared pytest fixtures for llm_intelligence tests."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """Temporary directory with valid skill files for testing."""
    skill_md = """---
name: test-skill
description: A skill for testing purposes.
display: 🧪 Test Skill
---
Please analyze: {TRANSCRIPT}

Return markdown.
"""
    (tmp_path / "test-skill.md").write_text(skill_md)
    return tmp_path


@pytest.fixture
def skills_dir_no_transcript(tmp_path: Path) -> Path:
    """Skill file without {TRANSCRIPT} placeholder (warns but loads)."""
    skill_md = """---
name: no-placeholder
description: Skill without transcript placeholder.
display: ⚠ No Placeholder
---
Static prompt with no substitution.
"""
    (tmp_path / "no-placeholder.md").write_text(skill_md)
    return tmp_path
