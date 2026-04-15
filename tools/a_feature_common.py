"""A 기능 분리 Tool 공통 유틸"""

from __future__ import annotations

from tools import document_ai_tool as core


def _resolve_input(
    content: str = "",
    note_title: str = "",
    source_name: str = "",
    source_type: str = "",
) -> tuple[str, str, str]:
    text = (content or "").strip()
    note_title = (note_title or "").strip()

    if text:
        resolved_source = source_name.strip() or "manual_input"
        resolved_type = source_type.strip() or "text"
        return text, resolved_source, resolved_type

    if note_title:
        filename = f"{note_title.replace(' ', '_')}.md"
        path = core.NOTES_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"노트를 찾지 못했습니다: {path}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        resolved_source = source_name.strip() or filename
        resolved_type = source_type.strip() or "saved_note"
        return text, resolved_source, resolved_type

    raise ValueError("content 또는 note_title 중 하나는 반드시 제공해야 합니다.")


def run_a_analysis(
    content: str = "",
    note_title: str = "",
    source_name: str = "",
    source_type: str = "",
) -> dict:
    text, resolved_source, resolved_type = _resolve_input(
        content=content,
        note_title=note_title,
        source_name=source_name,
        source_type=source_type,
    )
    return core._analyze_text(text, source_name=resolved_source, source_type=resolved_type)
