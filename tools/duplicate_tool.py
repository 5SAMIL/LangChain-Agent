"""노트 제목/경로 기반 문서 중복 탐지 Tool

흐름:
1. 노트 제목 또는 경로로 파일을 찾아 전체 내용을 읽는다.
2. 전체 텍스트를 OpenAI 임베딩으로 벡터화한다.
3. vectorDB의 기존 문서들과 코사인 유사도를 비교한다.
4. 유사도 상위 3개 결과를 반환하고 data/duplicate_cache.json에 저장한다.
5. Agent가 결과를 자연어로 설명한다 (Tool은 계산만 담당).
"""
import json
import os
import re
from collections import Counter
from datetime import datetime

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from tools.note_ops_utils import resolve_note_reference_path, note_relative_path
from tools.file_tool import _read_content, _resolve_path

VECTORDB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vectordb")
NOTES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "notes")
READABLE_EXTS = {".md", ".txt", ".pdf", ".pptx", ".ppt", ".docx", ".doc"}
CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "duplicate_cache.json")

HIGH_THRESHOLD = 0.85
LOW_THRESHOLD = 0.70
TOP_K = 3

TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z]{2,}")
STOPWORDS = {
    "이번", "다음", "해당", "관련", "통해", "위해", "및", "등", "또는",
    "그리고", "하지만", "또한", "에서", "으로", "하는", "하기", "대한",
    "the", "and", "for", "with", "this", "that", "from", "have", "will",
}


def _get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=VECTORDB_DIR, embedding_function=embeddings)


def _top_keywords(text: str, k: int = 5) -> list:
    tokens = [t.lower() for t in TOKEN_PATTERN.findall(text) if t.lower() not in STOPWORDS]
    return [term for term, _ in Counter(tokens).most_common(k)]


def _load_cache() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _run_similarity_search(source_path: str, content: str) -> list:
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search_with_score(content, k=TOP_K + 2)

    source_filename = os.path.basename(source_path)
    entries = []
    for doc, score in results:
        similarity = round(1 - score, 4)
        doc_source = doc.metadata.get("source", "unknown")

        if source_filename in doc_source or source_path in doc_source:
            continue
        if similarity < LOW_THRESHOLD:
            continue

        judgment = "중복 가능성 높음" if similarity >= HIGH_THRESHOLD else "유사 문서"
        common_kw = _top_keywords(doc.page_content)

        entries.append({
            "rank": len(entries) + 1,
            "source": doc_source,
            "similarity": round(similarity * 100, 1),
            "judgment": judgment,
            "keywords": common_kw,
        })

        if len(entries) == TOP_K:
            break

    return entries


def _resolve_file(reference: str):
    """노트 제목/경로 또는 일반 파일 경로를 받아 (filepath, source_label, content)를 반환한다."""
    # 1) data/notes 안의 .md 노트로 시도
    try:
        filepath = resolve_note_reference_path(reference)
        source_label = "data/notes/" + note_relative_path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return filepath, source_label, content
    except Exception:
        pass

    # 2) 일반 파일 경로로 시도 (PDF, PPTX, DOCX, TXT 등)
    resolved = _resolve_path(reference)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {reference}")

    ext = os.path.splitext(resolved)[1].lower()
    if ext not in READABLE_EXTS:
        raise ValueError(f"지원하지 않는 형식입니다: {ext} (지원: {', '.join(sorted(READABLE_EXTS))})")

    content = _read_content(resolved).strip()
    source_label = os.path.relpath(resolved)
    return resolved, source_label, content


@tool
def check_note_duplicates(note_reference: str) -> str:
    """노트 제목, 노트 경로, 또는 PDF/PPTX/DOCX/TXT 파일 경로를 받아 vectorDB의 기존 문서들과 전체 내용 기반 유사도를 비교하고 상위 3개를 반환한다. Agent는 이 결과를 자연어로 설명한다."""
    try:
        filepath, rel_path, content = _resolve_file(note_reference)
    except Exception as e:
        return f"파일 처리 실패: {e}"

    if not content:
        return "문서 내용이 비어 있습니다."

    try:
        entries = _run_similarity_search(rel_path, content)
    except Exception as e:
        return f"유사도 계산 실패 (vectorDB가 비어 있을 수 있습니다): {e}"

    cache = _load_cache()
    cache[rel_path] = {
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "char_count": len(content),
        "results": entries,
    }
    _save_cache(cache)

    filename = os.path.basename(filepath)

    if not entries:
        return (
            f"중복 탐지 결과 | 기준 문서: {filename} ({len(content)}자)\n"
            f"유사 문서 없음 (기준 임계값: 유사={int(LOW_THRESHOLD*100)}% / 중복={int(HIGH_THRESHOLD*100)}%)"
        )

    lines = [f"중복 탐지 결과 | 기준 문서: {filename} ({len(content)}자)"]
    for e in entries:
        kw_str = ", ".join(e["keywords"]) if e["keywords"] else "-"
        lines.append(
            f"\n{e['rank']}위 | 유사도: {e['similarity']}% | {e['judgment']}\n"
            f"   경로: {e['source']}\n"
            f"   공통 키워드: {kw_str}"
        )

    return "\n".join(lines)
