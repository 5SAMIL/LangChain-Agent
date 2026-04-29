"""문서를 분류 규칙에 따라 정리하고 저장하는 Tool"""
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from tools.file_tool import _read_content, _resolve_path

DEFAULT_NOTES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "notes"
)

FOLDER_RULES = {
    "meetings": [
        "회의", "미팅", "meeting", "agenda", "standup", "회의록",
        "minutes", "action item", "weekly", "sync",
    ],
    "study": [
        "공부", "학습", "강의", "lecture", "course", "정리", "복습", "책",
        "교안", "수업", "tutorial", "실습", "연구", "논문", "과제",
        "알고리즘", "머신러닝", "딥러닝", "인공지능", "langchain", "python",
    ],
    "projects": [
        "프로젝트", "project", "개발", "구현", "배포", "버그", "task",
        "issue", "기능", "roadmap", "요구사항", "requirement", "report",
    ],
    "ideas": ["아이디어", "idea", "brainstorm", "생각", "기획", "컨셉", "제안"],
    "todos": ["할 일", "todo", "to-do", "체크리스트", "action item", "checklist"],
    "references": ["문서", "자료", "reference", "가이드", "매뉴얼", "문법", "manual", "spec", "faq"],
    "journal": ["회고", "일기", "journal", "retrospective", "다이어리", "일상", "오늘의", "개인"],
}
CATEGORY_PRIORITY = ["meetings", "study", "projects", "ideas", "todos", "references", "journal", "inbox"]
SOURCE_TYPE_FALLBACKS = {
    "pdf": "references",
    "document": "references",
    "docs": "references",
    "paper": "references",
    "journal": "journal",
    "meeting": "meetings",
}
_RECENT_SAVED_NOTES: dict[str, str] = {}
_PENDING_ORGANIZATION: dict[str, str] = {}
SUGGESTION_CATEGORY_MAP = {
    "project": "projects",
    "reference": "references",
    "personal": "journal",
    "finance": "references",
    "legal": "references",
    "media": "references",
}


def _notes_dir() -> str:
    return os.getenv("PKM_NOTES_DIR", DEFAULT_NOTES_DIR)


def _sanitize_path_component(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = normalized.replace(" ", "_")
    normalized = re.sub(r'[\\/:\*\?"<>\|]+', "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("._")
    return normalized or "untitled"


def _note_lookup_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = normalized.replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", normalized)


def recent_saved_note_path(note_reference: str) -> str:
    """현재 프로세스에서 같은 제목으로 마지막 저장된 노트 경로를 반환한다."""
    reference = (note_reference or "").strip().strip("'\"")
    if not reference or "/" in reference or reference.endswith(".md"):
        return ""
    return _RECENT_SAVED_NOTES.get(_note_lookup_key(reference), "")


def _store_pending_organization(
    *,
    content: str = "",
    file_path: str = "",
    note_title: str = "",
    source_type: str = "",
    source_name: str = "",
    suggested_category: str = "",
) -> None:
    _PENDING_ORGANIZATION.clear()
    _PENDING_ORGANIZATION.update(
        {
            "content": content,
            "file_path": file_path,
            "note_title": note_title,
            "source_type": source_type,
            "source_name": source_name,
            "suggested_category": suggested_category,
        }
    )


def _classify_document(title: str, content: str, source_type: str) -> str:
    title_text = (title or "").lower()
    body_text = (content or "").lower()
    source_text = (source_type or "").lower()

    scores = {category: 0.0 for category in CATEGORY_PRIORITY}
    for category, keywords in FOLDER_RULES.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in title_text:
                scores[category] += 3.0
            if keyword_lower in body_text:
                scores[category] += 1.0
            if keyword_lower in source_text:
                scores[category] += 1.5

    fallback_category = SOURCE_TYPE_FALLBACKS.get(source_text, "")
    if fallback_category:
        scores[fallback_category] += 0.75

    best_category = max(CATEGORY_PRIORITY, key=lambda category: (scores[category], -CATEGORY_PRIORITY.index(category)))
    if scores[best_category] <= 0:
        return fallback_category or "inbox"
    return best_category


def _normalize_suggested_category(category: str) -> str:
    normalized = (category or "").strip().lower()
    normalized = SUGGESTION_CATEGORY_MAP.get(normalized, normalized)
    if normalized in CATEGORY_PRIORITY:
        return normalized
    return ""


def _extract_suggested_category(suggestion: str) -> str:
    match = re.search(r"^category:\s*(.+)$", suggestion or "", re.MULTILINE)
    return _normalize_suggested_category(match.group(1)) if match else ""


def _target_relative_path(
    title: str,
    content: str,
    source_type: str,
    now=None,  # type: Optional[datetime]
    category_override: str = "",
) -> tuple[str, str]:
    current_time = now or datetime.now()
    category = _normalize_suggested_category(category_override) or _classify_document(title, content, source_type)
    month_folder = current_time.strftime("%Y-%m")
    filename = f"{_sanitize_path_component(title)}.md"
    relative_path = os.path.join(category, month_folder, filename)
    return category, relative_path


def _deduplicated_path(base_dir: str, relative_path: str) -> str:
    absolute_path = os.path.join(base_dir, relative_path)
    if not os.path.exists(absolute_path):
        return absolute_path

    stem, ext = os.path.splitext(absolute_path)
    counter = 2
    while True:
        candidate = f"{stem}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _save_organized_note(
    title: str,
    content: str,
    source_type: str = "note",
    source_name: str = "manual",
    category_override: str = "",
) -> str:
    notes_dir = _notes_dir()
    category, relative_path = _target_relative_path(title, content, source_type, category_override=category_override)
    filepath = _deduplicated_path(notes_dir, relative_path)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    relative_saved_path = os.path.relpath(filepath, notes_dir)
    saved_note_path = f"data/notes/{relative_saved_path}"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"*저장 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
        f.write(f"*문서 유형: {source_type}*\n")
        f.write(f"*출처: {source_name}*\n")
        f.write(f"*분류 폴더: {category}*\n\n")
        f.write(content)

    _RECENT_SAVED_NOTES[_note_lookup_key(title)] = saved_note_path
    return f"자동 정리 완료: {saved_note_path}"


@tool
def preview_organized_path(title: str, content: str, source_type: str = "note") -> str:
    """문서를 어떤 폴더에 저장할지 미리 분류해서 예상 경로를 반환한다."""
    category, relative_path = _target_relative_path(title, content, source_type)
    return f"예상 분류: {category}\n예상 경로: data/notes/{relative_path}"


@tool
def organize_and_save_note(
    title: str,
    content: str,
    source_type: str = "note",
    source_name: str = "manual",
) -> str:
    """문서를 규칙에 따라 자동 분류하고 필요한 폴더를 생성해 마크다운으로 저장한다."""
    return _save_organized_note(title, content, source_type, source_name)


@tool
def preview_file_organized_path(
    file_path: str,
    note_title: str = "",
    source_type: str = "",
) -> str:
    """파일을 읽은 뒤 자동 정리했을 때의 예상 저장 경로를 미리 보여준다."""
    resolved_path = _resolve_path(file_path)
    if not os.path.exists(resolved_path):
        return f"파일을 찾을 수 없습니다: {resolved_path}"

    try:
        content = _read_content(resolved_path)
    except Exception as e:
        return f"파일 읽기 실패: {e}"

    title = note_title.strip() or os.path.splitext(os.path.basename(resolved_path))[0]
    effective_source_type = source_type.strip() or os.path.splitext(resolved_path)[1].lstrip(".") or "document"
    return preview_organized_path.invoke(
        {"title": title, "content": content, "source_type": effective_source_type}
    )


@tool
def organize_file_and_save_note(
    file_path: str,
    note_title: str = "",
    source_type: str = "",
) -> str:
    """파일을 읽어 내용을 자동 분류한 뒤 data/notes/<카테고리>/<YYYY-MM>/ 아래에 저장한다."""
    resolved_path = _resolve_path(file_path)
    if not os.path.exists(resolved_path):
        return f"파일을 찾을 수 없습니다: {resolved_path}"

    try:
        content = _read_content(resolved_path)
    except Exception as e:
        return f"파일 읽기 실패: {e}"

    title = note_title.strip() or os.path.splitext(os.path.basename(resolved_path))[0]
    effective_source_type = source_type.strip() or os.path.splitext(resolved_path)[1].lstrip(".") or "document"
    return _save_organized_note(title, content, effective_source_type, os.path.basename(resolved_path))


@tool
def suggest_organization_for_approval(
    content: str = "",
    file_path: str = "",
    note_title: str = "",
    source_name: str = "",
    source_type: str = "",
) -> str:
    """정리 결과를 제안하고, 사용자가 승인하면 실제 정리를 실행할 수 있도록 대기 상태로 저장한다."""
    content = (content or "").strip()
    file_path = (file_path or "").strip()
    note_title = (note_title or "").strip()
    source_name = (source_name or "").strip()
    source_type = (source_type or "").strip()

    suggestion_content = content
    effective_source_name = source_name
    effective_source_type = source_type

    if file_path:
        resolved_path = _resolve_path(file_path)
        if not os.path.exists(resolved_path):
            return f"파일을 찾을 수 없습니다: {resolved_path}"
        try:
            suggestion_content = _read_content(resolved_path)
        except Exception as e:
            return f"파일 읽기 실패: {e}"
        effective_source_name = source_name or os.path.basename(resolved_path)
        effective_source_type = source_type or Path(resolved_path).suffix.lstrip(".") or "document"
        file_path = resolved_path

    if not suggestion_content and not note_title:
        return "정리 결과를 제안할 내용, 파일 경로, 또는 저장된 노트 제목을 입력해주세요."

    from tools.a_organization_tool import a_suggest_organization

    suggestion = a_suggest_organization.invoke(
        {
            "content": suggestion_content,
            "note_title": note_title,
            "source_name": effective_source_name,
            "source_type": effective_source_type,
        }
    )
    suggested_category = _extract_suggested_category(suggestion)

    _store_pending_organization(
        content=content,
        file_path=file_path,
        note_title=note_title,
        source_type=effective_source_type,
        source_name=effective_source_name,
        suggested_category=suggested_category,
    )
    return f"{suggestion}\n\n이대로 정리를 시작할까요?"


@tool
def approve_organization() -> str:
    """직전에 제안된 정리 결과를 사용자가 승인했을 때 실제 자동 정리 저장을 실행한다."""
    if not _PENDING_ORGANIZATION:
        return "승인할 정리 제안이 없습니다. 먼저 정리 결과 제안을 요청해주세요."

    pending = dict(_PENDING_ORGANIZATION)
    _PENDING_ORGANIZATION.clear()

    if pending.get("file_path"):
        resolved_path = _resolve_path(pending["file_path"])
        try:
            content = _read_content(resolved_path)
        except Exception as e:
            return f"파일 읽기 실패: {e}"
        title = pending.get("note_title") or os.path.splitext(os.path.basename(resolved_path))[0]
        source_type = pending.get("source_type") or os.path.splitext(resolved_path)[1].lstrip(".") or "document"
        return _save_organized_note(
            title,
            content,
            source_type,
            os.path.basename(resolved_path),
            pending.get("suggested_category", ""),
        )

    title = pending.get("note_title") or pending.get("source_name") or "정리된 노트"
    content = pending.get("content", "")
    if not content:
        return "저장할 원문 내용이 없습니다. 파일 경로나 본문으로 다시 정리 결과 제안을 요청해주세요."

    return _save_organized_note(
        title,
        content,
        pending.get("source_type") or "note",
        pending.get("source_name") or "manual",
        pending.get("suggested_category", ""),
    )
