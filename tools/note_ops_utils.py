"""노트 이동/이름변경/메타데이터 편집용 유틸리티"""
import os
import re
import unicodedata
from datetime import datetime

from tools.organizer_tool import (
    _classify_document,
    _deduplicated_path,
    _notes_dir,
    _sanitize_path_component,
)

METADATA_ORDER = ["저장 시각", "문서 유형", "출처", "분류 폴더", "태그"]
MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")
NOTES_PREFIX_PATTERN = re.compile(r"^(?:\.?/)?data/notes/")


def resolve_relative_path(relative_path: str) -> str:
    relative_path = normalize_notes_reference(relative_path)
    parts = [
        _sanitize_path_component(part)
        for part in relative_path.replace("\\", "/").split("/")
        if part.strip()
    ]
    return os.path.join(*parts) if parts else ""


def note_root_dir() -> str:
    return _notes_dir()


def normalize_notes_reference(reference: str) -> str:
    value = (reference or "").strip().strip("'\"")
    if not value:
        return ""

    normalized = value.replace("\\", "/")
    notes_dir = os.path.realpath(note_root_dir())

    if os.path.isabs(value):
        real_value = os.path.realpath(value)
        try:
            if os.path.commonpath([notes_dir, real_value]) == notes_dir:
                normalized = os.path.relpath(real_value, notes_dir).replace("\\", "/")
            else:
                return value
        except ValueError:
            return value

    normalized = NOTES_PREFIX_PATTERN.sub("", normalized)
    return normalized.lstrip("/")


def ensure_inside_notes_dir(path: str) -> None:
    notes_dir = os.path.realpath(note_root_dir())
    target = os.path.realpath(path)
    if os.path.commonpath([notes_dir, target]) != notes_dir:
        raise ValueError("data/notes 바깥 경로는 사용할 수 없습니다.")


def note_absolute_path(relative_path: str) -> str:
    raw_value = (relative_path or "").strip().strip("'\"")
    if not raw_value:
        return note_root_dir()

    if os.path.isabs(raw_value):
        return raw_value

    resolved = resolve_relative_path(raw_value)
    return os.path.join(note_root_dir(), resolved)


def note_relative_path(absolute_path: str) -> str:
    return os.path.relpath(absolute_path, note_root_dir())


def ensure_markdown_filename(filename: str) -> str:
    base_name = filename[:-3] if filename.lower().endswith(".md") else filename
    return f"{_sanitize_path_component(base_name)}.md"


def extract_month_folder(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/")
    for part in reversed(normalized.split("/")):
        if MONTH_PATTERN.match(part):
            return part
    return datetime.now().strftime("%Y-%m")


def parse_note_file(filepath: str) -> tuple[str, dict[str, str], str]:
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    lines = raw_text.splitlines()
    title = os.path.splitext(os.path.basename(filepath))[0].replace("_", " ")
    metadata: dict[str, str] = {}
    index = 0

    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip() or title
        index = 1

    while index < len(lines) and lines[index].strip() == "":
        index += 1

    while index < len(lines):
        stripped = lines[index].strip()
        match = re.match(r"^\*(.+?):\s*(.*?)\*$", stripped)
        if not match:
            break
        metadata[match.group(1).strip()] = match.group(2).strip()
        index += 1

    while index < len(lines) and lines[index].strip() == "":
        index += 1

    body = "\n".join(lines[index:]).strip()
    return title, metadata, body


def write_note_file(filepath: str, title: str, metadata: dict[str, str], body: str) -> None:
    ordered_keys = [key for key in METADATA_ORDER if metadata.get(key)]
    extra_keys = sorted(key for key in metadata if key not in METADATA_ORDER and metadata.get(key))

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for key in ordered_keys + extra_keys:
            f.write(f"*{key}: {metadata[key]}*\n")
        if ordered_keys or extra_keys:
            f.write("\n")
        if body:
            f.write(body)


def determine_category(title: str, content: str, source_type: str) -> str:
    return _classify_document(title, content, source_type)


def deduplicated_note_path(relative_path: str) -> str:
    return _deduplicated_path(note_root_dir(), relative_path)


def build_category_target_path(current_relative_path: str, category: str, filename: str) -> str:
    month_folder = extract_month_folder(current_relative_path)
    return os.path.join(_sanitize_path_component(category), month_folder, ensure_markdown_filename(filename))


def _title_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = normalized.replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", normalized)


def resolve_note_reference_path(note_reference: str) -> str:
    raw_reference = (note_reference or "").strip().strip("'\"")
    if not raw_reference:
        raise FileNotFoundError("노트 경로 또는 제목을 입력해주세요.")

    direct_path = note_absolute_path(raw_reference)
    ensure_inside_notes_dir(direct_path)
    if os.path.isfile(direct_path):
        return direct_path

    normalized_reference = normalize_notes_reference(raw_reference)
    dirname = os.path.dirname(normalized_reference)
    filename = os.path.basename(normalized_reference)
    if filename:
        with_md = note_absolute_path(
            os.path.join(dirname, ensure_markdown_filename(filename))
        )
        ensure_inside_notes_dir(with_md)
        if os.path.isfile(with_md):
            return with_md

    target_filename = ensure_markdown_filename(os.path.basename(raw_reference))
    target_key = _title_key(os.path.splitext(os.path.basename(raw_reference))[0])
    matches: list[tuple[int, int, str, str]] = []

    for root, _, files in os.walk(note_root_dir()):
        for name in sorted(files):
            if not name.endswith(".md"):
                continue

            absolute_path = os.path.join(root, name)
            relative_path = note_relative_path(absolute_path).replace("\\", "/")
            score: int | None = None

            if relative_path == normalized_reference:
                score = 0
            elif name == target_filename:
                score = 1
            else:
                title, _, _ = parse_note_file(absolute_path)
                if _title_key(title) == target_key or _title_key(os.path.splitext(name)[0]) == target_key:
                    score = 2

            if score is not None:
                matches.append((score, relative_path.count("/"), relative_path, absolute_path))

    if not matches:
        raise FileNotFoundError(f"노트를 찾을 수 없습니다: {raw_reference}")

    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    best_score = matches[0][0]
    best_matches = [match for match in matches if match[0] == best_score]

    if len(best_matches) > 1:
        candidates = "\n".join(f"- data/notes/{match[2]}" for match in best_matches[:5])
        raise ValueError(
            "같은 제목의 노트가 여러 개 있습니다. 경로를 더 구체적으로 입력해주세요:\n"
            f"{candidates}"
        )

    return best_matches[0][3]
