"""A 역할 — 키워드(Keyword) Tool (독립 구현)

입력 텍스트에서 빈도수 기반 핵심 단어(top_k 개)를 추출한다.

알고리즘:
1) 텍스트 정규화.
2) 한/영/숫자 단어만 뽑기 (TOKEN_PATTERN).
3) 불용어(STOPWORDS)/짧은 단어/날짜 같은 노이즈 제거.
4) Counter 로 빈도 집계 → 상위 top_k 개 반환.
5) score 는 "가장 많이 나온 단어 대비 비율" 로 정규화.
"""

from __future__ import annotations

import re
import os
from collections import Counter
from pathlib import Path

from langchain_core.tools import tool
from tools.note_ops_utils import note_relative_path, resolve_note_reference_path


# ----- 설정값 ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


# ----- 내부 유틸 ------------------------------------------------------------

def _normalize(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    # 파일 리더가 이미지 블록 위치에 삽입하는 플레이스홀더 제거
    text = re.sub(r"\(이미지\)", "", text)
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


def _extract_keywords(text: str, top_k: int) -> list[dict]:
    counts = Counter(_tokenize(text))
    if not counts:
        return []
    max_count = max(counts.values())
    return [
        {"term": term, "freq": freq, "score": round(freq / max_count, 4)}
        for term, freq in counts.most_common(top_k)
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
def a_extract_keywords(
    content: str = "",
    note_title: str = "",
    source_name: str = "",
    source_type: str = "",
    top_k: int = 10,
) -> str:
    """A 역할: 문서 키워드를 추출한다. content 또는 note_title 을 입력받는다."""
    try:
        text, resolved_source, resolved_type = _resolve_input(
            content, note_title, source_name, source_type
        )
    except Exception as e:
        return f"A 키워드 추출 실패: {e}"

    top_k = max(1, min(top_k, 30))
    keywords = _extract_keywords(text, top_k)

    lines = [
        "A 키워드 결과",
        f"source: {resolved_source} ({resolved_type})",
        f"top_k: {top_k}",
    ]
    if not keywords:
        lines.append("- 키워드 없음")
        return "\n".join(lines)

    for item in keywords:
        lines.append(f"- {item['term']} (freq={item['freq']}, score={item['score']})")
    return "\n".join(lines)
