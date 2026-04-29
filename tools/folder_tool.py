"""정리된 노트 폴더의 이름을 변경하는 Tool"""
import os
import re
import unicodedata

from langchain_core.tools import tool

DEFAULT_NOTES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "notes"
)


def _notes_dir() -> str:
    return os.getenv("PKM_NOTES_DIR", DEFAULT_NOTES_DIR)


def _sanitize_folder_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = normalized.replace(" ", "_")
    normalized = re.sub(r'[\\/:\*\?"<>\|]+', "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("._")
    return normalized or "untitled"


def _resolve_relative_folder(relative_folder: str) -> str:
    normalized = (relative_folder or "").strip().strip("'\"").replace("\\", "/")
    notes_dir = os.path.realpath(_notes_dir())

    if os.path.isabs(normalized):
        real_value = os.path.realpath(normalized)
        try:
            if os.path.commonpath([notes_dir, real_value]) == notes_dir:
                normalized = os.path.relpath(real_value, notes_dir).replace("\\", "/")
        except ValueError:
            pass

    if normalized.startswith("data/notes/"):
        normalized = normalized[len("data/notes/"):]

    sanitized_parts = [
        _sanitize_folder_name(part)
        for part in normalized.split("/")
        if part.strip()
    ]
    return os.path.join(*sanitized_parts) if sanitized_parts else ""


def _assert_inside_notes_dir(path: str, notes_dir: str) -> None:
    real_notes_dir = os.path.realpath(notes_dir)
    real_path = os.path.realpath(path)
    if os.path.commonpath([real_notes_dir, real_path]) != real_notes_dir:
        raise ValueError("data/notes 바깥 경로로 이동할 수 없습니다.")


@tool
def rename_organized_folder(current_folder: str, new_folder_name: str) -> str:
    """data/notes 아래의 기존 폴더 이름을 새 이름으로 변경한다."""
    notes_dir = _notes_dir()
    current_relative = _resolve_relative_folder(current_folder)
    if not current_relative:
        return "변경할 폴더 경로를 입력해주세요."

    current_path = os.path.join(notes_dir, current_relative)
    _assert_inside_notes_dir(current_path, notes_dir)

    if not os.path.isdir(current_path):
        return f"폴더를 찾을 수 없습니다: data/notes/{current_relative}"

    new_folder = _sanitize_folder_name(new_folder_name)
    parent_dir = os.path.dirname(current_path)
    target_path = os.path.join(parent_dir, new_folder)
    _assert_inside_notes_dir(target_path, notes_dir)

    if os.path.exists(target_path):
        target_relative = os.path.relpath(target_path, notes_dir)
        return f"같은 이름의 폴더가 이미 있습니다: data/notes/{target_relative}"

    os.rename(current_path, target_path)

    before_relative = os.path.relpath(current_path, notes_dir)
    after_relative = os.path.relpath(target_path, notes_dir)
    return (
        f"폴더 이름 변경 완료:\n"
        f"- 이전: data/notes/{before_relative}\n"
        f"- 이후: data/notes/{after_relative}"
    )
