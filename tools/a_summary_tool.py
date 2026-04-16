"""A 역할 — 요약(Summary) Tool (독립 구현)

입력 텍스트를 LLM(GPT-4o-mini)에게 보내 짧은 요약(short)과 상세 요약(detailed)을
생성한다. LLM 호출 실패 시 빈도 기반 발췌 요약으로 자동 폴백한다.

LLM 요약 알고리즘:
1) 텍스트를 정규화한다.
2) GPT-4o-mini 에게 short/detailed 요약을 JSON 형식으로 요청한다.
3) 응답에서 short / detailed 를 파싱해 반환한다.

폴백(발췌 요약) 알고리즘:
1) 빈도수 기반 키워드 가중치를 만든다.
2) 문장 점수 = 문장 내 키워드 가중치 합산.
3) 상위 2문장 → short, 상위 5문장 → detailed.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ----- 설정값 ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = PROJECT_ROOT / "data" / "notes"

LLM_MODEL = "gpt-4o-mini"
LLM_MAX_INPUT_CHARS = 12_000  # 토큰 절약을 위해 긴 문서는 앞부분만 전달

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


def _keyword_weights(text: str, top_k: int = 12) -> dict[str, float]:
    counts = Counter(_tokenize(text))
    if not counts:
        return {}
    max_count = max(counts.values())
    return {term: freq / max_count for term, freq in counts.most_common(top_k)}


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?。])\s+|\n+", text)
    return [c.strip() for c in chunks if len(c.strip()) >= 12]


def _extractive_summary(text: str) -> dict[str, str]:
    """LLM 없이 빈도 기반으로 문장을 발췌하는 폴백 요약."""
    if not text:
        return {
            "short": "문서 내용이 비어 있어 요약할 수 없습니다.",
            "detailed": "텍스트가 없어 상세 요약을 생성하지 못했습니다.",
        }
    sentences = _split_sentences(text)
    if not sentences:
        return {"short": text[:250], "detailed": text[:500]}

    weights = _keyword_weights(text)
    scored = []
    for idx, sentence in enumerate(sentences):
        score = sum(weights.get(tok, 0.0) for tok in _tokenize(sentence))
        scored.append((idx, sentence, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    top_short = sorted(scored[:2], key=lambda x: x[0])
    top_detailed = sorted(scored[:5], key=lambda x: x[0])

    return {
        "short": " ".join(s for _, s, _ in top_short)[:400],
        "detailed": " ".join(s for _, s, _ in top_detailed)[:1200],
    }


def _llm_summary(text: str) -> dict[str, str]:
    """GPT-4o-mini 를 사용해 문서를 이해하고 요약한다."""
    truncated = text[:LLM_MAX_INPUT_CHARS]

    prompt = (
        "다음 문서를 한국어로 요약해줘.\n"
        "반드시 아래 JSON 형식으로만 응답해. 다른 말은 하지 마.\n\n"
        "{\n"
        '  "short": "2~3문장의 핵심 요약",\n'
        '  "detailed": "5~8문장의 상세 요약. 주요 개념과 흐름을 포함할 것."\n'
        "}\n\n"
        f"[문서]\n{truncated}"
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    response = llm.invoke(prompt)
    raw = response.content.strip()

    # JSON 블록 파싱 (```json ... ``` 감싸여 있을 수도 있음)
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        raise ValueError("LLM 응답에서 JSON을 찾을 수 없음")

    parsed = json.loads(json_match.group())
    return {
        "short": str(parsed.get("short", "")).strip(),
        "detailed": str(parsed.get("detailed", "")).strip(),
    }


def _make_summary(text: str) -> tuple[dict[str, str], str]:
    """LLM 요약을 시도하고, 실패 시 발췌 요약으로 폴백한다.
    반환: (summary_dict, method)  — method 는 'llm' 또는 'extractive'
    """
    if not text:
        return _extractive_summary(text), "extractive"

    if not os.getenv("OPENAI_API_KEY"):
        return _extractive_summary(text), "extractive"

    try:
        return _llm_summary(text), "llm"
    except Exception:
        return _extractive_summary(text), "extractive"


def _resolve_input(content: str, note_title: str, source_name: str, source_type: str):
    content = (content or "").strip()
    note_title = (note_title or "").strip()

    if content:
        return _normalize(content), (source_name.strip() or "manual_input"), (source_type.strip() or "text")

    if note_title:
        filename = f"{note_title.replace(' ', '_')}.md"
        path = NOTES_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"노트를 찾지 못했습니다: {path}")
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return _normalize(raw), (source_name.strip() or filename), (source_type.strip() or "saved_note")

    raise ValueError("content 또는 note_title 중 하나는 반드시 제공해야 합니다.")


# ----- Tool -----------------------------------------------------------------

@tool
def a_generate_summary(
    content: str = "",
    note_title: str = "",
    source_name: str = "",
    source_type: str = "",
) -> str:
    """A 역할: 문서 요약을 생성한다. content 또는 note_title 을 입력받는다."""
    try:
        text, resolved_source, resolved_type = _resolve_input(
            content, note_title, source_name, source_type
        )
    except Exception as e:
        return f"A 요약 실패: {e}"

    summary, method = _make_summary(text)

    return (
        "A 요약 결과\n"
        f"source: {resolved_source} ({resolved_type})\n"
        f"text_length: {len(text)}\n"
        f"method: {method}\n"
        f"short: {summary['short']}\n"
        f"detailed: {summary['detailed']}"
    )
