"""저장된 노트 파일 이름과 제목을 변경하는 Tool"""
import os

from langchain_core.tools import tool

from tools.note_ops_utils import (
    deduplicated_note_path,
    ensure_inside_notes_dir,
    ensure_markdown_filename,
    note_relative_path,
    parse_note_file,
    resolve_note_reference_path,
    write_note_file,
)


@tool
def rename_note_file(
    current_path: str,
    new_name: str,
    update_title: bool = True,
) -> str:
    """기존 노트의 파일명을 바꾸고 필요하면 문서 제목도 함께 변경한다. 제목 또는 경로를 받을 수 있다."""
    try:
        source_path = resolve_note_reference_path(current_path)
    except Exception as e:
        return str(e)

    title, metadata, body = parse_note_file(source_path)
    new_filename = ensure_markdown_filename(new_name)
    destination_relative_path = os.path.join(
        os.path.dirname(note_relative_path(source_path)),
        new_filename,
    )
    destination_path = deduplicated_note_path(destination_relative_path)
    ensure_inside_notes_dir(destination_path)

    next_title = new_name if update_title else title
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    write_note_file(destination_path, next_title, metadata, body)

    if os.path.realpath(destination_path) != os.path.realpath(source_path):
        os.remove(source_path)

    return (
        f"노트 이름 변경 완료:\n"
        f"- 이전: data/notes/{note_relative_path(source_path)}\n"
        f"- 이후: data/notes/{note_relative_path(destination_path)}"
    )
