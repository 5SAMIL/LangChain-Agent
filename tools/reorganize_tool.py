"""기존 노트를 규칙에 따라 다시 분류하는 Tool"""
import os
from datetime import datetime

from langchain_core.tools import tool

from tools.note_ops_utils import (
    build_category_target_path,
    deduplicated_note_path,
    determine_category,
    ensure_inside_notes_dir,
    note_relative_path,
    parse_note_file,
    resolve_note_reference_path,
    write_note_file,
)


def _reorganize_note_file(filepath: str, source_type: str = "") -> tuple[str, str, str]:
    ensure_inside_notes_dir(filepath)
    title, metadata, body = parse_note_file(filepath)
    current_relative_path = note_relative_path(filepath)
    effective_source_type = source_type or metadata.get("문서 유형", "note")
    target_category = determine_category(title, body, effective_source_type)
    target_relative_path = build_category_target_path(
        current_relative_path,
        target_category,
        os.path.basename(filepath),
    )
    target_path = deduplicated_note_path(target_relative_path)
    ensure_inside_notes_dir(target_path)

    metadata["문서 유형"] = effective_source_type
    metadata["분류 폴더"] = target_category
    metadata.setdefault("저장 시각", datetime.now().strftime("%Y-%m-%d %H:%M"))

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    write_note_file(target_path, title, metadata, body)
    if os.path.realpath(target_path) != os.path.realpath(filepath):
        os.remove(filepath)

    return current_relative_path, note_relative_path(target_path), target_category


def _reorganize_note_file_preview(filepath: str, source_type: str = "") -> tuple[str, str, str]:
    ensure_inside_notes_dir(filepath)
    title, metadata, body = parse_note_file(filepath)
    current_relative_path = note_relative_path(filepath)
    effective_source_type = source_type or metadata.get("문서 유형", "note")
    target_category = determine_category(title, body, effective_source_type)
    target_relative_path = build_category_target_path(
        current_relative_path,
        target_category,
        os.path.basename(filepath),
    )
    return current_relative_path, target_relative_path, target_category


@tool
def preview_reorganized_note(current_path: str, source_type: str = "") -> str:
    """기존 노트를 다시 분류했을 때 예상되는 카테고리와 경로를 보여준다. 제목 또는 경로를 받을 수 있다."""
    try:
        source_path = resolve_note_reference_path(current_path)
    except Exception as e:
        return str(e)

    before, after, category = _reorganize_note_file_preview(source_path, source_type)
    return (
        f"예상 재분류 결과:\n"
        f"- 현재: data/notes/{before}\n"
        f"- 분류: {category}\n"
        f"- 이동 예정: data/notes/{after}"
    )


@tool
def reorganize_existing_note(current_path: str, source_type: str = "") -> str:
    """기존 노트를 다시 분류해 알맞은 카테고리 폴더로 이동한다. 제목 또는 경로를 받을 수 있다."""
    try:
        source_path = resolve_note_reference_path(current_path)
    except Exception as e:
        return str(e)

    before, after, category = _reorganize_note_file(source_path, source_type)
    return (
        f"재분류 완료:\n"
        f"- 이전: data/notes/{before}\n"
        f"- 이후: data/notes/{after}\n"
        f"- 적용된 분류: {category}"
    )
