"""A 역할 전용: 문서 내용 분석/요약/키워드/태그/자동 분류/정리 제안 Tool"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
INDEX_PATH = ANALYSIS_DIR / "index.json"
TAXONOMY_PATH = DATA_DIR / "taxonomy.json"
NOTES_DIR = DATA_DIR / "notes"

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9가-힣][A-Za-z0-9가-힣_+.-]{1,}")
DATE_TOKEN_PATTERN = re.compile(r"^\d{2,4}[-/.]\d{1,2}([\-/.]\d{1,2})?$")

# 너무 자주 나오는 기능어/불용어(한영 혼합)
DEFAULT_EXTRA_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "have",
    "will",
    "into",
    "over",
    "under",
    "are",
    "was",
    "were",
    "been",
    "being",
    "there",
    "here",
    "such",
    "than",
    "then",
    "per",
    "via",
    "each",
    "using",
    "used",
    "use",
    "in",
    "to",
    "of",
    "on",
    "at",
    "by",
    "as",
    "an",
    "a",
    "or",
    "is",
    "be",
    "we",
    "you",
    "your",
    "our",
    "their",
    "it",
    "its",
    "이번",
    "다음",
    "해당",
    "관련",
    "통해",
    "위해",
    "및",
    "등",
    "또는",
}

DEFAULT_TAXONOMY = {
    "version": "1.0",
    "default_category": "inbox",
    "categories": [
        {
            "name": "inbox",
            "subcategories": ["uncategorized"],
            "keywords": [],
            "default_folder": "data/sorted/inbox",
        }
    ],
    "tag_rules": {},
    "stopwords": [],
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        _write_json(path, default)
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        _write_json(path, default)
        return default


def _ensure_storage() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    _read_json(INDEX_PATH, {"version": "1.0", "analyses": [], "updated_at": _now()})


def _load_index() -> dict[str, Any]:
    return _read_json(INDEX_PATH, {"version": "1.0", "analyses": [], "updated_at": _now()})


def _save_index(index: dict[str, Any]) -> None:
    index["updated_at"] = _now()
    _write_json(INDEX_PATH, index)


def _load_taxonomy() -> dict[str, Any]:
    if not TAXONOMY_PATH.exists():
        _write_json(TAXONOMY_PATH, DEFAULT_TAXONOMY)
    return _read_json(TAXONOMY_PATH, DEFAULT_TAXONOMY)


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _clean_token(token: str) -> str:
    return token.strip("._-:;,'\"()[]{}!?")


def _is_noise_token(token: str) -> bool:
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
    tokens: list[str] = []
    for raw in TOKEN_PATTERN.findall(text):
        cleaned = _clean_token(raw.lower())
        if cleaned:
            tokens.append(cleaned)
    return tokens


def _extract_keywords(text: str, stopwords: set[str], top_k: int = 12) -> list[dict[str, Any]]:
    merged_stopwords = set(stopwords) | DEFAULT_EXTRA_STOPWORDS
    tokens = [
        t
        for t in _tokenize(text)
        if t not in merged_stopwords and not t.isdigit() and len(t) >= 2 and not _is_noise_token(t)
    ]
    if not tokens:
        return []

    counts = Counter(tokens)
    max_count = max(counts.values())
    return [
        {"term": term, "freq": freq, "score": round(freq / max_count, 4)}
        for term, freq in counts.most_common(top_k)
    ]


def _sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?。])\s+|\n+", text)
    return [c.strip() for c in chunks if len(c.strip()) >= 12]


def _build_summary(text: str, keywords: list[dict[str, Any]]) -> dict[str, str]:
    if not text:
        return {
            "short": "문서 내용이 비어 있어 요약할 수 없습니다.",
            "detailed": "텍스트가 없어 상세 요약을 생성하지 못했습니다.",
        }

    sentences = _sentences(text)
    if not sentences:
        return {"short": text[:250], "detailed": text[:500]}

    weights = {item["term"]: float(item["score"]) for item in keywords}
    scored = []
    for i, sentence in enumerate(sentences):
        sentence_score = sum(weights.get(token, 0.0) for token in _tokenize(sentence))
        scored.append((i, sentence, sentence_score))

    scored.sort(key=lambda x: x[2], reverse=True)
    top_short = sorted(scored[:2], key=lambda x: x[0])
    top_detailed = sorted(scored[:5], key=lambda x: x[0])

    return {
        "short": " ".join(s for _, s, _ in top_short)[:400],
        "detailed": " ".join(s for _, s, _ in top_detailed)[:1200],
    }


def _build_tags(text: str, keywords: list[dict[str, Any]], taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    text_lower = text.lower()
    rules: dict[str, list[str]] = taxonomy.get("tag_rules", {})
    tags: dict[str, float] = {}

    for tag_name, rule_terms in rules.items():
        matched = [term for term in rule_terms if term.lower() in text_lower]
        if matched:
            confidence = min(1.0, len(matched) / max(1, len(rule_terms)) + 0.3)
            tags[tag_name] = round(confidence, 4)

    for kw in keywords[:6]:
        if kw["term"] not in tags:
            tags[kw["term"]] = max(0.35, float(kw["score"]))

    return [
        {"name": name, "confidence": round(score, 4)}
        for name, score in sorted(tags.items(), key=lambda x: x[1], reverse=True)
    ]


def _find_best_subcategory(text: str, category: dict[str, Any]) -> str:
    subcategories = category.get("subcategories", [])
    if not subcategories:
        return "uncategorized"

    text_lower = text.lower()
    for sub in subcategories:
        if sub.lower() in text_lower:
            return sub
    return subcategories[0]


def _classify(
    text: str,
    keywords: list[dict[str, Any]],
    tags: list[dict[str, Any]],
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    categories = taxonomy.get("categories", [])
    default_category = taxonomy.get("default_category", "inbox")

    keyword_scores = {item["term"]: float(item["score"]) for item in keywords}
    tag_names = {item["name"] for item in tags}
    text_lower = text.lower()

    scored: list[tuple[dict[str, Any], float, list[str], int, bool]] = []
    for category in categories:
        score = 0.0
        evidence: list[str] = []
        matched_keyword_count = 0
        tag_hit = False

        for kw in category.get("keywords", []):
            kw_lower = kw.lower()
            if kw_lower in text_lower:
                score += max(0.25, keyword_scores.get(kw_lower, 0.4))
                evidence.append(kw)
                matched_keyword_count += 1

        if category.get("name") in tag_names:
            score += 0.7
            evidence.append(f"tag:{category['name']}")
            tag_hit = True

        scored.append((category, score, evidence, matched_keyword_count, tag_hit))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_category, top_score, top_evidence, top_keyword_hits, top_tag_hit = (
        scored[0] if scored else ({"name": default_category}, 0.0, [], 0, False)
    )

    if top_score <= 0.0:
        selected = next(
            (c for c in categories if c.get("name") == default_category),
            {"name": default_category, "subcategories": ["uncategorized"], "default_folder": "data/sorted/inbox"},
        )
        confidence = 0.2
        evidence = ["키워드/태그 근거 부족"]
    else:
        selected = top_category
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        dominance = max(0.0, (top_score - second_score) / (top_score + 1e-9))
        saturation = top_score / (top_score + 1.5)
        confidence = 0.45 * saturation + 0.55 * dominance
        if top_keyword_hits >= 2:
            confidence = max(confidence, 0.65)
        if top_tag_hit:
            confidence = max(confidence, 0.6)
        confidence = max(0.25, min(0.99, confidence))
        evidence = top_evidence

    return {
        "category": selected.get("name", default_category),
        "subcategory": _find_best_subcategory(text, selected),
        "confidence": round(confidence, 4),
        "reason": evidence[:6],
        "default_folder": selected.get("default_folder", "data/sorted/inbox"),
    }


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9가-힣_\-]", "", value)
    return value[:80] if value else "document"


def _build_organization_suggestion(classification: dict[str, Any], source_name: str) -> dict[str, Any]:
    date_str = datetime.now().strftime("%Y%m%d")
    category = classification.get("category", "inbox")
    base_name = _slugify(Path(source_name).stem or source_name or "document")

    suggested_filename = f"{date_str}_{category}_{base_name}.md"
    suggested_folder = classification.get("default_folder", "data/sorted/inbox")

    return {
        "recommended_folder": str((PROJECT_ROOT / suggested_folder).resolve()),
        "recommended_filename": suggested_filename,
        "reason": f"분류 결과({category}) 기반 정리 제안",
        "confidence": round(max(0.4, float(classification.get("confidence", 0.4))), 4),
        "next_action_owner": "C팀(정리 실행/승인)",
        "execution": "not_executed",
    }


def _analyze_text(content: str, source_name: str, source_type: str = "text") -> dict[str, Any]:
    _ensure_storage()
    taxonomy = _load_taxonomy()

    normalized = _normalize_text(content)
    stopwords = {s.lower() for s in taxonomy.get("stopwords", [])}

    keywords = _extract_keywords(normalized, stopwords)
    summary = _build_summary(normalized, keywords)
    tags = _build_tags(normalized, keywords, taxonomy)
    classification = _classify(normalized, keywords, tags, taxonomy)
    organization = _build_organization_suggestion(classification, source_name)

    analysis = {
        "analysis_id": str(uuid.uuid4()),
        "source_name": source_name,
        "source_type": source_type,
        "content_hash": hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest(),
        "text_length": len(normalized),
        "summary": summary,
        "keywords": keywords,
        "tags": tags,
        "classification": classification,
        "organization_suggestion": organization,
        "analyzed_at": _now(),
    }

    index = _load_index()
    index.setdefault("analyses", []).append(analysis)
    _save_index(index)
    return analysis


def _format_analysis(analysis: dict[str, Any]) -> str:
    keywords = ", ".join(item["term"] for item in analysis.get("keywords", [])[:8]) or "없음"
    tags = ", ".join(item["name"] for item in analysis.get("tags", [])[:8]) or "없음"

    cls = analysis.get("classification", {})
    org = analysis.get("organization_suggestion", {})

    return (
        "A 역할 분석 완료\n"
        f"analysis_id: {analysis.get('analysis_id')}\n"
        f"source: {analysis.get('source_name')} ({analysis.get('source_type')})\n"
        f"텍스트 길이: {analysis.get('text_length')}\n"
        f"요약(짧게): {analysis.get('summary', {}).get('short', '')}\n"
        f"키워드: {keywords}\n"
        f"태그: {tags}\n"
        f"자동 분류: {cls.get('category')}/{cls.get('subcategory')} "
        f"(신뢰도 {round(float(cls.get('confidence', 0.0))*100, 1)}%)\n"
        "정리 결과 제안(실행 안 함):\n"
        f"- 추천 폴더: {org.get('recommended_folder')}\n"
        f"- 추천 파일명: {org.get('recommended_filename')}\n"
        f"- 근거: {org.get('reason')}\n"
        f"- 다음 실행 담당: {org.get('next_action_owner')}"
    )


def analyze_content_ai(content: str, source_name: str = "manual_input", source_type: str = "text") -> str:
    """A 역할 전용 분석: 텍스트 내용을 받아 요약/키워드/태그/자동 분류/정리 제안을 생성한다."""
    if not content or not content.strip():
        return "분석할 content가 비어 있습니다."

    try:
        analysis = _analyze_text(content, source_name=source_name, source_type=source_type)
        return _format_analysis(analysis)
    except Exception as e:
        return f"A 역할 분석 실패: {e}"


def analyze_saved_note(title: str) -> str:
    """저장된 노트 내용을 읽어 A 역할 분석을 수행한다. 파일 스캔/이동은 하지 않는다."""
    filename = f"{title.replace(' ', '_')}.md"
    path = NOTES_DIR / filename
    if not path.exists():
        return f"노트를 찾지 못했습니다: {path}"

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        analysis = _analyze_text(content, source_name=filename, source_type="saved_note")
        return _format_analysis(analysis)
    except Exception as e:
        return f"노트 분석 실패: {e}"


def get_latest_analyses(limit: int = 5) -> str:
    """최근 A 분석 결과를 조회한다."""
    _ensure_storage()
    index = _load_index()
    analyses = index.get("analyses", [])

    if not analyses:
        return "저장된 분석 결과가 없습니다."

    recent = analyses[-max(1, min(limit, 20)):][::-1]
    lines = [f"최근 분석 {len(recent)}건"]
    for item in recent:
        cls = item.get("classification", {})
        lines.append(
            f"- {item.get('analysis_id')} | {item.get('source_name')} | "
            f"{cls.get('category')}/{cls.get('subcategory')} "
            f"({round(float(cls.get('confidence', 0.0))*100, 1)}%)"
        )

    return "\n".join(lines)
