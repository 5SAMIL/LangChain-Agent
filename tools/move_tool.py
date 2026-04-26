"""저장된 노트를 다른 카테고리로 이동하는 Tool"""
import os
from datetime import datetime

from langchain_core.tools import tool

from tools.note_ops_utils import (
    build_category_target_path,
    deduplicated_note_path,
    ensure_inside_notes_dir,
    note_relative_path,
    parse_note_file,
    resolve_note_reference_path,
    write_note_file,
)


@tool
def move_note_to_category(
    current_path: str,
    new_category: str,
    target_month: str = "",
) -> str:
    """기존 노트를 다른 카테고리 폴더로 이동한다. current_path 는 제목 또는 경로를 받을 수 있다."""
    try:
        source_path = resolve_note_reference_path(current_path)
    except Exception as e:
        return str(e)

    title, metadata, body = parse_note_file(source_path)
    source_relative_path = note_relative_path(source_path)
    filename = os.path.basename(source_path)

    if target_month:
        destination_relative_path = os.path.join(new_category, target_month, filename)
    else:
        destination_relative_path = build_category_target_path(
            source_relative_path,
            new_category,
            filename,
        )

    destination_path = deduplicated_note_path(destination_relative_path)
    ensure_inside_notes_dir(destination_path)

    metadata["분류 폴더"] = new_category
    metadata.setdefault("저장 시각", datetime.now().strftime("%Y-%m-%d %H:%M"))

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    write_note_file(destination_path, title, metadata, body)
    if os.path.realpath(destination_path) != os.path.realpath(source_path):
        os.remove(source_path)

    return (
        f"노트 이동 완료:\n"
        f"- 이전: data/notes/{source_relative_path}\n"
        f"- 이후: data/notes/{note_relative_path(destination_path)}"
    )
