"""노트 저장/읽기/목록 조회 Tool"""
import os
from datetime import datetime
from langchain_core.tools import tool

NOTES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "notes")


@tool
def save_note(title: str, content: str) -> str:
    """마크다운 파일로 노트를 저장한다. title은 파일명, content는 노트 내용."""
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
    filename = f"{title.replace(' ', '_')}.md"
    filepath = os.path.join(NOTES_DIR, filename)

    if not os.path.exists(filepath):
        return f"노트를 찾을 수 없습니다: {title}"

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


@tool
def list_notes() -> str:
    """저장된 모든 노트 목록을 반환한다."""
    if not os.path.exists(NOTES_DIR):
        return "저장된 노트가 없습니다."

    files = [f.replace(".md", "").replace("_", " ") for f in os.listdir(NOTES_DIR) if f.endswith(".md")]
    if not files:
        return "저장된 노트가 없습니다."

    return "저장된 노트 목록:\n" + "\n".join(f"- {f}" for f in files)
