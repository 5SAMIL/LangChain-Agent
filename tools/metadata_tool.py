"""노트 메타데이터를 수정하는 Tool"""
import os

from langchain_core.tools import tool

from tools.note_ops_utils import (
    note_relative_path,
    parse_note_file,
    resolve_note_reference_path,
    write_note_file,
)


@tool
def update_note_metadata(
    current_path: str,
    category: str = "",
    source_name: str = "",
    source_type: str = "",
    tags: str = "",
    saved_at: str = "",
) -> str:
    """노트 상단 메타데이터를 수정한다. current_path 는 제목 또는 경로를 받을 수 있고, 태그는 쉼표로 구분해 입력한다."""
    try:
        target_path = resolve_note_reference_path(current_path)
    except Exception as e:
        return str(e)

    title, metadata, body = parse_note_file(target_path)

    if category:
        metadata["분류 폴더"] = category
    if source_name:
        metadata["출처"] = source_name
    if source_type:
        metadata["문서 유형"] = source_type
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        metadata["태그"] = ", ".join(tag_list)
    if saved_at:
        metadata["저장 시각"] = saved_at

    write_note_file(target_path, title, metadata, body)
    return f"메타데이터 수정 완료: data/notes/{note_relative_path(target_path)}"
