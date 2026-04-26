"""노트 저장/읽기/목록 조회 Tool"""
import os
from datetime import datetime
from langchain_core.tools import tool
from tools.note_ops_utils import note_relative_path, note_root_dir, resolve_note_reference_path


@tool
def save_note(title: str, content: str) -> str:
    """마크다운 파일로 노트를 저장한다. title은 파일명, content는 노트 내용."""
    NOTES_DIR = note_root_dir()
    os.makedirs(NOTES_DIR, exist_ok=True)
    filename = f"{title.replace(' ', '_')}.md"
    filepath = os.path.join(NOTES_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"*저장 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        f.write(content)

    return f"노트 저장 완료: {filename}"


@tool
def read_note(title: str) -> str:
    """저장된 노트를 제목으로 읽는다."""
    try:
        filepath = resolve_note_reference_path(title)
    except Exception as e:
        return str(e)

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


@tool
def list_notes() -> str:
    """저장된 모든 노트 목록을 반환한다."""
    NOTES_DIR = note_root_dir()
    if not os.path.exists(NOTES_DIR):
        return "저장된 노트가 없습니다."

    files = []
    for root, _, filenames in os.walk(NOTES_DIR):
        for name in filenames:
            if name.endswith(".md"):
                files.append(note_relative_path(os.path.join(root, name)).replace(".md", ""))

    if not files:
        return "저장된 노트가 없습니다."

    files = sorted(path.replace("_", " ") for path in files)
    return "저장된 노트 목록:\n" + "\n".join(f"- {f}" for f in files)
