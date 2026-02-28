"""Skill file loader — parses .md files with YAML-style frontmatter.

Skill file format:
    ---
    name: skill-slug
    description: One line description.
    display: 🎭 Display Label
    ---
    Everything below is the prompt template.
    Use {TRANSCRIPT} as the placeholder for the actual transcript.
"""
from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from llm_intelligence.schemas import SkillMeta

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)
_FIELD_RE = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)
_REQUIRED_FIELDS = {"name", "description", "display"}


def _parse_skill_file(path: Path) -> SkillMeta:
    """Parse a single skill .md file and return a SkillMeta."""
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        raise ValueError(f"Skill file {path.name!r} missing valid frontmatter (expected --- ... --- block)")

    frontmatter_block, prompt_template = match.group(1), match.group(2).strip()
    fields = {m.group(1): m.group(2).strip() for m in _FIELD_RE.finditer(frontmatter_block)}

    missing = _REQUIRED_FIELDS - fields.keys()
    if missing:
        raise ValueError(f"Skill file {path.name!r} missing required fields: {missing}")

    if "{TRANSCRIPT}" not in prompt_template:
        logger.warning("Skill {!r} prompt template has no {{TRANSCRIPT}} placeholder", fields["name"])

    return SkillMeta(
        name=fields["name"],
        description=fields["description"],
        display=fields["display"],
        prompt_template=prompt_template,
    )


def load_skills(skills_dir: str | Path) -> dict[str, SkillMeta]:
    """
    Load all .md skill files from *skills_dir*.

    Returns a dict mapping skill name (slug) → SkillMeta.
    Logs and skips files that fail to parse.
    """
    skills_path = Path(skills_dir)
    if not skills_path.is_dir():
        logger.warning("Skills directory {!r} does not exist — no skills loaded", str(skills_path))
        return {}

    result: dict[str, SkillMeta] = {}
    for md_file in sorted(skills_path.glob("*.md")):
        try:
            skill = _parse_skill_file(md_file)
            result[skill.name] = skill
            logger.debug("Loaded skill {!r} from {}", skill.name, md_file.name)
        except Exception as exc:
            logger.error("Failed to load skill from {!r}: {}", md_file.name, exc)

    logger.info("Loaded {} skill(s) from {!r}", len(result), str(skills_path))
    return result
