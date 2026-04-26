"""A 역할 — 태그(Tag) Tool (독립 구현)

문서 내용을 보고 어울리는 태그를 생성한다.

알고리즘:
1) 텍스트를 정규화한다.
2) data/taxonomy.json 의 tag_rules 를 읽는다.
   예) "urgent": ["긴급", "마감", "asap"]
3) 각 룰마다, 룰 단어가 본문에 몇 개 포함됐는지 세서 confidence 를 만든다.
   confidence = min(1.0, matched/len(rule) + 0.3)
4) 룰에 걸리지 않은 태그는, 자주 등장한 단어 상위 6개로 보조 태그를 만든다.
5) confidence 내림차순으로 정렬해서 반환.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

from langchain_core.tools import tool
from tools.note_ops_utils import note_relative_path, resolve_note_reference_path


# ----- 설정값 ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = PROJECT_ROOT / "data" / "taxonomy.json"

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9가-힣][A-Za-z0-9가-힣_+.-]{1,}")
DATE_TOKEN_PATTERN = re.compile(r"^\d{2,4}[-/.]\d{1,2}([\-/.]\d{1,2})?$")

STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "have", "will",
    "into", "over", "under", "are", "was", "were", "been", "being", "there",
    "here", "such", "than", "then", "per", "via", "each", "using", "used",
    "use", "in", "to", "of", "on", "at", "by", "as", "an", "a", "or", "is",
    "be", "we", "you", "your", "our", "their", "it", "its",
    "이번", "다음", "해당", "관련", "통해", "위해", "및", "등", "또는",
    "그리고", "하지만", "또한", "에서", "으로", "하는", "하기", "대한",
}

DEFAULT_TAXONOMY = {"tag_rules": {}, "categories": [], "stopwords": []}


# ----- 내부 유틸 ------------------------------------------------------------

def _normalize(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln).strip()


def _is_noise(token: str) -> bool:
    if not token:
        return True
    if DATE_TOKEN_PATTERN.match(token):
        return True
    if re.fullmatch(r"[0-9]+([./:-][0-9]+)+", token):
        return True
    if token.isascii() and token.isalpha() and len(token) <= 2 and token not in {"ai", "ui", "ux"}:
        return True
    return False


def _tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN_PATTERN.findall(text):
        t = raw.strip("._-:;,'\"()[]{}!?").lower()
        if t and t not in STOPWORDS and not t.isdigit() and len(t) >= 2 and not _is_noise(t):
            tokens.append(t)
    return tokens


def _top_keywords(text: str, top_k: int = 6) -> list[tuple[str, float]]:
    counts = Counter(_tokenize(text))
    if not counts:
        return []
    max_count = max(counts.values())
    return [(term, freq / max_count) for term, freq in counts.most_common(top_k)]


def _load_taxonomy() -> dict:
    if not TAXONOMY_PATH.exists():
        return DEFAULT_TAXONOMY
    try:
        return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_TAXONOMY


def _build_tags(text: str, taxonomy: dict) -> list[dict]:
    text_lower = text.lower()
    tag_rules = taxonomy.get("tag_rules", {}) or {}

    tags: dict[str, float] = {}

    # 1) 규칙 기반 태그
    for tag_name, rule_terms in tag_rules.items():
        matched = [t for t in rule_terms if t.lower() in text_lower]
        if matched:
            confidence = min(1.0, len(matched) / max(1, len(rule_terms)) + 0.3)
            tags[tag_name] = round(confidence, 4)

    # 2) 빈도 기반 보조 태그 (규칙에 없는 것만)
    for term, score in _top_keywords(text, top_k=6):
        if term not in tags:
            tags[term] = round(max(0.35, score), 4)

    return [
        {"name": name, "confidence": conf}
        for name, conf in sorted(tags.items(), key=lambda x: x[1], reverse=True)
    ]


def _resolve_input(content: str, note_title: str, source_name: str, source_type: str):
    content = (content or "").strip()
    note_title = (note_title or "").strip()

    if content:
        return _normalize(content), (source_name.strip() or "manual_input"), (source_type.strip() or "text")

    if note_title:
        path = Path(resolve_note_reference_path(note_title))
        raw = path.read_text(encoding="utf-8", errors="ignore")
        resolved_source = source_name.strip() or os.path.basename(note_relative_path(str(path)))
        return _normalize(raw), resolved_source, (source_type.strip() or "saved_note")

    raise ValueError("content 또는 note_title 중 하나는 반드시 제공해야 합니다.")


# ----- Tool -----------------------------------------------------------------

@tool
def a_generate_tags(
    content: str = "",
    note_title: str = "",
    source_name: str = "",
    source_type: str = "",
    top_k: int = 10,
) -> str:
    """A 역할: 문서 태그를 생성한다. content 또는 note_title 을 입력받는다."""
    try:
        text, resolved_source, resolved_type = _resolve_input(
            content, note_title, source_name, source_type
        )
    except Exception as e:
        return f"A 태그 생성 실패: {e}"

    taxonomy = _load_taxonomy()
    top_k = max(1, min(top_k, 30))
    tags = _build_tags(text, taxonomy)[:top_k]

    lines = [
        "A 태그 결과",
        f"source: {resolved_source} ({resolved_type})",
        f"top_k: {top_k}",
    ]
    if not tags:
        lines.append("- 태그 없음")
        return "\n".join(lines)

    for item in tags:
        lines.append(f"- {item['name']} (confidence={item['confidence']})")
    return "\n".join(lines)
