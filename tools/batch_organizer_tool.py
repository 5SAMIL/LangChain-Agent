"""폴더 단위로 노트를 일괄 재정리하는 Tool"""
import os

from langchain_core.tools import tool

from tools.note_ops_utils import ensure_inside_notes_dir, note_absolute_path
from tools.reorganize_tool import _reorganize_note_file, _reorganize_note_file_preview


@tool
def batch_reorganize_notes(
    source_folder: str = "",
    source_type: str = "",
    dry_run: bool = True,
    limit: int = 50,
) -> str:
    """특정 폴더 아래의 마크다운 노트를 한 번에 재분류한다."""
    base_path = note_absolute_path(source_folder) if source_folder else note_absolute_path("")
    ensure_inside_notes_dir(base_path)

    if not os.path.isdir(base_path):
        folder_label = f"data/notes/{source_folder}" if source_folder else "data/notes"
        return f"폴더를 찾을 수 없습니다: {folder_label}"

    markdown_files = []
    for root, _, files in os.walk(base_path):
        for name in sorted(files):
            if name.endswith(".md"):
                markdown_files.append(os.path.join(root, name))

    if not markdown_files:
        return "재정리할 마크다운 노트가 없습니다."

    if limit > 0:
        markdown_files = markdown_files[:limit]

    results = []
    for filepath in markdown_files:
        if dry_run:
            before, after, category = _reorganize_note_file_preview(filepath, source_type)
        else:
            before, after, category = _reorganize_note_file(filepath, source_type)
        results.append((before, after, category))

    header = "배치 재정리 미리보기" if dry_run else "배치 재정리 완료"
    lines = [header]
    for before, after, category in results:
        lines.append(f"- {category}: data/notes/{before} -> data/notes/{after}")

    if len(markdown_files) == limit and limit > 0:
        lines.append(f"- 처리 제한: 처음 {limit}개 파일만 확인했습니다.")

    return "\n".join(lines)
