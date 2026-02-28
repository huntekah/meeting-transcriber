"""Unit tests for skills_loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from llm_intelligence.skills_loader import load_skills


def test_load_skills_parses_frontmatter(skills_dir: Path) -> None:
    skills = load_skills(skills_dir)
    assert "test-skill" in skills
    skill = skills["test-skill"]
    assert skill.name == "test-skill"
    assert skill.description == "A skill for testing purposes."
    assert skill.display == "🧪 Test Skill"


def test_load_skills_captures_prompt_template(skills_dir: Path) -> None:
    skills = load_skills(skills_dir)
    assert "{TRANSCRIPT}" in skills["test-skill"].prompt_template


def test_load_skills_empty_dir(tmp_path: Path) -> None:
    skills = load_skills(tmp_path)
    assert skills == {}


def test_load_skills_nonexistent_dir() -> None:
    skills = load_skills("/nonexistent/path/to/skills")
    assert skills == {}


def test_load_skills_missing_required_field(tmp_path: Path) -> None:
    bad_skill = """---
name: incomplete-skill
---
Prompt here.
"""
    (tmp_path / "bad.md").write_text(bad_skill)
    skills = load_skills(tmp_path)
    # File with missing fields is skipped, not raised
    assert "incomplete-skill" not in skills


def test_load_skills_invalid_frontmatter(tmp_path: Path) -> None:
    (tmp_path / "invalid.md").write_text("No frontmatter at all, just content.")
    skills = load_skills(tmp_path)
    assert skills == {}


def test_load_skills_no_transcript_placeholder_loads(skills_dir_no_transcript: Path) -> None:
    skills = load_skills(skills_dir_no_transcript)
    assert "no-placeholder" in skills
