"""연결고리 탐색 및 중복 탐지 Tool"""
import os
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

VECTORDB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vectordb")
NOTES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "notes")
DUPLICATE_THRESHOLD = 0.95  # 유사도 이 이상이면 중복으로 판단


def _l2_to_cosine(l2_distance: float) -> float:
    """L2 거리 → 코사인 유사도 변환 (OpenAI 단위 벡터 기준: cosine_sim = 1 - L2²/2)"""
    return max(0.0, 1 - (l2_distance ** 2) / 2)


def _get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=VECTORDB_DIR, embedding_function=embeddings)


@tool
def find_connections(topic: str, k: int = 5) -> str:
    """특정 주제와 연관된 지식들을 벡터DB에서 찾아 연결고리를 탐색한다."""
    try:
        vectorstore = _get_vectorstore()
        results = vectorstore.similarity_search_with_score(topic, k=k)

        if not results:
            return "연결된 지식을 찾지 못했습니다."

        output = [f"'{topic}'과 연결된 지식:"]
        for i, (doc, score) in enumerate(results, 1):
            similarity = round(_l2_to_cosine(score) * 100, 1)
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:100].replace("\n", " ")
            output.append(f"[{i}] 유사도 {similarity}% (출처: {source})\n    {preview}...")

        return "\n".join(output)
    except Exception as e:
        return f"연결고리 탐색 실패: {e}"


@tool
def detect_duplicates(content: str) -> str:
    """입력한 내용과 중복되는 지식이 벡터DB에 있는지 탐지한다."""
    try:
        vectorstore = _get_vectorstore()
        results = vectorstore.similarity_search_with_score(content, k=3)

        if not results:
            return "중복 없음: 유사한 지식이 없습니다."

        duplicates = []
        similar = []

        for doc, score in results:
            cosine_sim = _l2_to_cosine(score)
            similarity = round(cosine_sim * 100, 1)
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:80].replace("\n", " ")

            if cosine_sim >= DUPLICATE_THRESHOLD:
                duplicates.append(f"  - [{similarity}%] (출처: {source}) {preview}...")
            elif cosine_sim >= 0.7:
                similar.append(f"  - [{similarity}%] (출처: {source}) {preview}...")

        output = []
        if duplicates:
            output.append(f"중복 감지 ({len(duplicates)}건):")
            output.extend(duplicates)
        if similar:
            output.append(f"유사 내용 ({len(similar)}건):")
            output.extend(similar)
        if not output:
            return "중복 없음: 유사한 지식이 없습니다."

        return "\n".join(output)
    except Exception as e:
        return f"중복 탐지 실패: {e}"
