"""data/notes 전체 또는 단일 노트를 vectorDB에 인덱싱하는 Tool"""
import os
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from tools.note_ops_utils import resolve_note_reference_path

VECTORDB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vectordb")
NOTES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "notes")


def _get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=VECTORDB_DIR, embedding_function=embeddings)


@tool
def index_all_notes() -> str:
    """data/notes 아래 모든 마크다운 파일을 읽어 vectorDB에 일괄 등록한다. vectorDB가 비어 있거나 재구축이 필요할 때 실행한다."""
    if not os.path.isdir(NOTES_DIR):
        return "data/notes 폴더가 없습니다."

    docs = []
    errors = []
    for root, _, files in os.walk(NOTES_DIR):
        for name in sorted(files):
            if not name.endswith(".md"):
                continue
            filepath = os.path.join(root, name)
            source = "data/notes/" + os.path.relpath(filepath, NOTES_DIR)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    docs.append(Document(page_content=content, metadata={"source": source}))
            except Exception as e:
                errors.append(f"{name}: {e}")

    if not docs:
        return "인덱싱할 노트가 없습니다."

    vectorstore = _get_vectorstore()
    vectorstore.add_documents(docs)

    result = f"인덱싱 완료: {len(docs)}개 노트를 vectorDB에 추가했습니다."
    if errors:
        result += f"\n오류 {len(errors)}건: " + ", ".join(errors[:3])
    return result


@tool
def index_note(note_reference: str) -> str:
    """노트 제목 또는 경로로 지정한 단일 노트를 읽어 vectorDB에 등록한다. 파일 저장 직후 자동 인덱싱에 사용한다."""
    try:
        filepath = resolve_note_reference_path(note_reference)
    except Exception as e:
        return f"노트를 찾을 수 없습니다: {e}"

    source = "data/notes/" + os.path.relpath(filepath, NOTES_DIR)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        return f"파일 읽기 실패: {e}"

    if not content:
        return "문서 내용이 비어 있습니다."

    vectorstore = _get_vectorstore()
    vectorstore.add_documents([Document(page_content=content, metadata={"source": source})])
    return f"인덱싱 완료: {source} ({len(content)}자)"
