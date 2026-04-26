"""A 역할 — 자동 분류(Classification) Tool (독립 구현)

문서가 어느 카테고리에 속하는지 자동으로 판별한다.
예) project / study / finance / legal / personal / reference / media / inbox

알고리즘:
1) 텍스트 정규화.
2) data/taxonomy.json 의 categories 를 로드.
   각 카테고리에는 name/subcategories/keywords/default_folder 가 있다.
3) 본문에 카테고리 keyword 가 얼마나 포함됐는지 점수화.
4) 최고점 카테고리 선택. 점수 0이면 default_category(=inbox) 로 폴백.
5) 신뢰도(confidence) 는 '최고점 포화도 + 2위와의 격차' 로 계산.
6) subcategory 는 본문에 먼저 등장하는 것을 선택 (없으면 첫 번째).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from langchain_core.tools import tool
from tools.note_ops_utils import note_relative_path, resolve_note_reference_path


# ----- 설정값 ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = PROJECT_ROOT / "data" / "taxonomy.json"

DEFAULT_TAXONOMY = {
    "default_category": "inbox",
    "categories": [
        {
            "name": "inbox",
            "subcategories": ["uncategorized"],
            "keywords": [],
            "default_folder": "data/sorted/inbox",
        }
    ],
}


# ----- 내부 유틸 ------------------------------------------------------------

def _normalize(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln).strip()


def _load_taxonomy() -> dict:
    if not TAXONOMY_PATH.exists():
        return DEFAULT_TAXONOMY
    try:
        return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_TAXONOMY


def _find_best_subcategory(text: str, category: dict) -> str:
    subs = category.get("subcategories", []) or []
    if not subs:
        return "uncategorized"
    text_lower = text.lower()
    for sub in subs:
        if sub.lower() in text_lower:
            return sub
    return subs[0]


def _classify(text: str, taxonomy: dict) -> dict:
    categories = taxonomy.get("categories", []) or []
    default_category = taxonomy.get("default_category", "inbox")
    text_lower = text.lower()

    # 각 카테고리의 점수와 매칭 근거를 모은다
    scored: list[tuple[dict, float, list[str], int]] = []
    for cat in categories:
        score = 0.0
        evidence: list[str] = []
        matched_count = 0
        for kw in cat.get("keywords", []):
            if kw.lower() in text_lower:
                score += 0.4
                evidence.append(kw)
                matched_count += 1
        scored.append((cat, score, evidence, matched_count))

    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored or scored[0][1] <= 0.0:
        # 아무 키워드도 안 걸림 → 기본 카테고리로 폴백
        selected = next(
            (c for c in categories if c.get("name") == default_category),
            {
                "name": default_category,
                "subcategories": ["uncategorized"],
                "default_folder": "data/sorted/inbox",
            },
        )
        return {
            "category": selected.get("name", default_category),
            "subcategory": _find_best_subcategory(text, selected),
            "confidence": 0.2,
            "reason": ["키워드 근거 부족"],
            "default_folder": selected.get("default_folder", "data/sorted/inbox"),
        }

    top_cat, top_score, top_evidence, top_hits = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0

    # 신뢰도: 점수 포화도(45%) + 2위와의 격차(55%)
    saturation = top_score / (top_score + 1.5)
    dominance = max(0.0, (top_score - second_score) / (top_score + 1e-9))
    confidence = 0.45 * saturation + 0.55 * dominance
    if top_hits >= 2:
        confidence = max(confidence, 0.65)
    confidence = max(0.25, min(0.99, confidence))

    return {
        "category": top_cat.get("name", default_category),
        "subcategory": _find_best_subcategory(text, top_cat),
        "confidence": round(confidence, 4),
        "reason": top_evidence[:6],
        "default_folder": top_cat.get("default_folder", "data/sorted/inbox"),
    }


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
def a_classify_document(
    content: str = "",
    note_title: str = "",
    source_name: str = "",
    source_type: str = "",
) -> str:
    """A 역할: 문서를 자동 분류한다. content 또는 note_title 을 입력받는다."""
    try:
        text, resolved_source, resolved_type = _resolve_input(
            content, note_title, source_name, source_type
        )
    except Exception as e:
        return f"A 자동 분류 실패: {e}"

    taxonomy = _load_taxonomy()
    cls = _classify(text, taxonomy)
    reason_text = ", ".join(cls["reason"]) if cls["reason"] else "근거 없음"

    return (
        "A 자동 분류 결과\n"
        f"source: {resolved_source} ({resolved_type})\n"
        f"category: {cls['category']}\n"
        f"subcategory: {cls['subcategory']}\n"
        f"confidence: {cls['confidence']}\n"
        f"reason: {reason_text}\n"
        f"default_folder: {cls['default_folder']}"
    )
