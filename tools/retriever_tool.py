"""벡터DB(Chroma) 기반 지식 저장/검색 Tool"""
import os
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

VECTORDB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vectordb")


def _get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=VECTORDB_DIR, embedding_function=embeddings)


@tool
def add_to_knowledge(content: str, source: str = "manual") -> str:
    """텍스트를 벡터DB에 저장해서 나중에 검색할 수 있게 한다. source는 출처 메타데이터."""
    vectorstore = _get_vectorstore()
    doc = Document(page_content=content, metadata={"source": source})
    vectorstore.add_documents([doc])
    return f"지식 저장 완료 (출처: {source})"


@tool
def search_knowledge(query: str, k: int = 3) -> str:
    """저장된 지식에서 쿼리와 관련된 내용을 검색한다. k는 반환할 결과 수."""
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)

    if not results:
        return "관련 지식을 찾지 못했습니다."

    output = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        output.append(f"[{i}] (출처: {source})\n{doc.page_content}")

    return "\n\n".join(output)
