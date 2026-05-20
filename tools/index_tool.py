"""data/notes 전체 또는 단일 노트를 vectorDB에 인덱싱하는 Tool"""
import os
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from tools.note_ops_utils import resolve_note_reference_path
from tools.file_tool import _read_content, _resolve_path, SUPPORTED_EXTENSIONS

VECTORDB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vectordb")
NOTES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "notes")


def _get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=VECTORDB_DIR, embedding_function=embeddings)


def _dedup_and_add(vectorstore, docs: list):
    """같은 source 기존 문서 삭제 후 재등록 — 중복 누적 방지"""
    sources = [d.metadata.get("source") for d in docs if d.metadata.get("source")]
    if sources:
        existing = vectorstore.get(where={"source": {"$in": sources}})
        if existing["ids"]:
            vectorstore.delete(ids=existing["ids"])
    vectorstore.add_documents(docs)


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
    _dedup_and_add(vectorstore, docs)

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
    _dedup_and_add(vectorstore, [Document(page_content=content, metadata={"source": source})])
    return f"인덱싱 완료: {source} ({len(content)}자)"


@tool
def index_file(file_path: str) -> str:
    """단일 파일(pdf, ppt, pptx, doc, docx, txt, md, 이미지 등)을 읽어 vectorDB에 인덱싱한다.
    파일 경로를 직접 입력받아 텍스트를 추출한 뒤 등록한다."""
    resolved = _resolve_path(file_path)
    if not os.path.isfile(resolved):
        return f"파일을 찾을 수 없습니다: {file_path}"

    ext = os.path.splitext(resolved)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return f"지원하지 않는 파일 형식입니다: {ext}"

    try:
        content = _read_content(resolved)
    except Exception as e:
        return f"파일 읽기 실패: {os.path.basename(resolved)} — {e}"

    if not content or "추출할 수 없습니다" in content or content.startswith("지원하지 않는"):
        return f"텍스트를 추출할 수 없습니다: {resolved}"

    vectorstore = _get_vectorstore()
    _dedup_and_add(vectorstore, [Document(page_content=content, metadata={"source": resolved})])
    return f"인덱싱 완료: {resolved} ({len(content)}자)"


@tool
def index_folder(folder_path: str) -> str:
    """폴더 내 모든 지원 파일(pdf, ppt, pptx, doc, docx, txt, md 등)을 재귀적으로 읽어 vectorDB에 일괄 인덱싱한다.
    ~$ 로 시작하는 Office 임시 파일은 자동으로 제외된다."""
    if not os.path.isdir(folder_path):
        return f"폴더를 찾을 수 없습니다: {folder_path}"

    entries = []
    for root, _, files in os.walk(folder_path):
        for name in sorted(files):
            if name.startswith("~$"):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                entries.append(os.path.join(root, name))

    if not entries:
        return f"지원되는 파일이 없습니다: {folder_path}"

    docs = []
    errors = []
    for filepath in entries:
        try:
            content = _read_content(filepath)
            if content and "추출할 수 없습니다" not in content and not content.startswith("지원하지 않는"):
                docs.append(Document(page_content=content, metadata={"source": filepath}))
            else:
                errors.append(f"{os.path.basename(filepath)}: 텍스트 추출 실패")
        except Exception as e:
            errors.append(f"{os.path.basename(filepath)}: {e}")

    if not docs:
        result = "인덱싱할 내용이 없습니다."
        if errors:
            result += "\n오류:\n" + "\n".join(errors[:5])
        return result

    vectorstore = _get_vectorstore()
    _dedup_and_add(vectorstore, docs)

    result = f"인덱싱 완료: {len(docs)}개 파일을 vectorDB에 추가했습니다."
    if errors:
        result += f"\n오류 {len(errors)}건:\n" + "\n".join(errors[:5])
    return result
