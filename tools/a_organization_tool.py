"""A 역할 — 정리 결과 제안(Organization Suggestion) Tool (독립 구현)

분류 결과를 바탕으로 "어디에, 어떤 이름으로 저장하면 좋은지" 제안한다.
실제 파일 이동/리네임은 하지 않는다. (그건 C팀의 책임)

알고리즘:
1) 텍스트 정규화.
2) 내부에서 자동 분류를 수행 → 카테고리/폴더를 얻는다.
3) 추천 파일명 = YYYYMMDD_<category>_<slug(source_name)>.md
4) 추천 폴더 = 카테고리의 default_folder (절대 경로로 변환)
5) 실행 여부(execution) 는 항상 "not_executed" — 제안만 한다.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool


# ----- 설정값 ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = PROJECT_ROOT / "data" / "notes"
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


def _classify(text: str, taxonomy: dict) -> dict:
    """정리 제안을 위해 내부에서 자동 분류를 수행한다.
    a_classification_tool 과 같은 로직이지만, 파일 독립성을 위해 복제했다."""
    categories = taxonomy.get("categories", []) or []
    default_category = taxonomy.get("default_category", "inbox")
    text_lower = text.lower()

    scored = []
    for cat in categories:
        score = 0.0
        matched = 0
        for kw in cat.get("keywords", []):
            if kw.lower() in text_lower:
                score += 0.4
                matched += 1
        scored.append((cat, score, matched))

    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored or scored[0][1] <= 0.0:
        selected = next(
            (c for c in categories if c.get("name") == default_category),
            {"name": default_category, "default_folder": "data/sorted/inbox"},
        )
        return {
            "category": selected.get("name", default_category),
            "default_folder": selected.get("default_folder", "data/sorted/inbox"),
            "confidence": 0.2,
        }

    top_cat, top_score, top_hits = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    saturation = top_score / (top_score + 1.5)
    dominance = max(0.0, (top_score - second_score) / (top_score + 1e-9))
    confidence = 0.45 * saturation + 0.55 * dominance
    if top_hits >= 2:
        confidence = max(confidence, 0.65)
    confidence = max(0.25, min(0.99, confidence))

    return {
        "category": top_cat.get("name", default_category),
        "default_folder": top_cat.get("default_folder", "data/sorted/inbox"),
        "confidence": round(confidence, 4),
    }


def _build_basename(source_name: str) -> str:
    """파일명을 YYYYMMDD_category_ 뒤에 붙을 형태로 변환한다.
    ' - ' 구분자가 있으면: 앞부분(공백제거) + (뒷부분) 형태로 만든다.
    예) '강의교안 01 - 인공지능 개요' → '강의교안01(인공지능 개요)'
    없으면: 공백만 제거해서 반환한다.
    """
    stem = Path(source_name).stem or source_name or "document"
    # ' - ' 구분자로 분리
    if " - " in stem:
        parts = stem.split(" - ", 1)
        prefix = parts[0].replace(" ", "")          # 공백 제거
        suffix = parts[1].strip()                    # 공백 유지
        return f"{prefix}({suffix})"
    else:
        return stem.replace(" ", "")


def _build_suggestion(classification: dict, source_name: str) -> dict:
    date_str = datetime.now().strftime("%Y%m%d")
    category = classification.get("category", "inbox")
    base_name = _build_basename(source_name)

    recommended_filename = f"{date_str}_{category}_{base_name}.md"
    recommended_folder = classification.get("default_folder", "data/sorted/inbox")
    confidence = round(max(0.4, float(classification.get("confidence", 0.4))), 4)

    return {
        "recommended_folder": str((PROJECT_ROOT / recommended_folder).resolve()),
        "recommended_filename": recommended_filename,
        "reason": f"분류 결과({category}) 기반 정리 제안",
        "confidence": confidence,
        "next_action_owner": "C팀(정리 실행/승인)",
        "execution": "not_executed",
    }


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
def a_suggest_organization(
    content: str = "",
    note_title: str = "",
    source_name: str = "",
    source_type: str = "",
) -> str:
    """A 역할: 문서 정리 결과를 제안한다. 실제 파일 이동은 하지 않는다."""
    try:
        text, resolved_source, resolved_type = _resolve_input(
            content, note_title, source_name, source_type
        )
    except Exception as e:
        return f"A 정리 제안 실패: {e}"

    taxonomy = _load_taxonomy()
    classification = _classify(text, taxonomy)
    suggestion = _build_suggestion(classification, resolved_source)

    return (
        "A 정리 결과 제안\n"
        f"source: {resolved_source} ({resolved_type})\n"
        f"category: {classification['category']}\n"
        f"recommended_folder: {suggestion['recommended_folder']}\n"
        f"recommended_filename: {suggestion['recommended_filename']}\n"
        f"reason: {suggestion['reason']}\n"
        f"confidence: {suggestion['confidence']}\n"
        f"next_action_owner: {suggestion['next_action_owner']}\n"
        f"execution: {suggestion['execution']}"
    )
