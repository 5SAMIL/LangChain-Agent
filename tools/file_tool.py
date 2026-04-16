"""파일 업로드 및 스캔, 읽기, 이동, 복사 Tool"""
import os
import shutil
import logging
import warnings
from datetime import datetime
from langchain_core.tools import tool

# pdfplumber/pdfminer 내부 경고 억제
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="pdfplumber")
warnings.filterwarnings("ignore", module="pdfminer")

# EasyOCR reader 싱글톤 (모델 로딩은 최초 1회만)
_easyocr_reader = None

def _get_ocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(['ko', 'en'], gpu=False)
    return _easyocr_reader

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
NOTES_DIR = os.path.join(PROJECT_ROOT, "data", "notes")

# 파일명만 입력했을 때 탐색할 기본 경로들
SEARCH_DIRS = [
    PROJECT_ROOT,
    NOTES_DIR,
    os.path.join(PROJECT_ROOT, "data"),
    os.path.expanduser("~/Downloads"),
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~"),
]


def _resolve_path(file_path: str) -> str:
    """파일 경로 해석. 절대/상대 경로가 존재하면 그대로, 파일명만이면 탐색 후 반환.
    파일명이 깨진 경우 같은 디렉토리에서 동일 확장자 파일을 탐색."""
    # 앞뒤 따옴표 및 공백 제거
    file_path = file_path.strip().strip("'\"")

    if os.path.exists(file_path):
        return file_path

    ext = os.path.splitext(file_path)[1].lower()
    parent_dir = os.path.dirname(file_path)

    # 파일명이 깨졌을 때: 같은 디렉토리에서 동일 확장자 파일 탐색
    if parent_dir and os.path.isdir(parent_dir) and ext:
        matches = [
            os.path.join(parent_dir, f)
            for f in os.listdir(parent_dir)
            if os.path.splitext(f)[1].lower() == ext
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # 가장 최근 수정된 파일 반환
            return max(matches, key=os.path.getmtime)

    # 파일명만 입력된 경우 — 기본 경로들에서 탐색
    filename = os.path.basename(file_path)
    for search_dir in SEARCH_DIRS:
        candidate = os.path.join(search_dir, filename)
        if os.path.exists(candidate):
            return candidate

    return file_path  # 못 찾으면 원본 반환 (이후 에러 메시지 처리)
MAX_PREVIEW_CHARS = 2000

SUPPORTED_EXTENSIONS = {
    # 문서
    ".pdf", ".txt", ".md",
    ".ppt", ".pptx",
    ".doc", ".docx",
    # 이미지
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp",
}


def _fix_pdf_text(text: str) -> str:
    """PDF 추출 텍스트 후처리 — 분리된 문장부호를 앞 줄에 붙임"""
    import re
    lines = text.split("\n")
    result = []
    punct_only = re.compile(r'^[\s→\?\.\!\,\;\:\"\'\(\)\[\]\-]+$')

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue
        # 문장부호/화살표만 있는 줄은 앞 줄에 붙이기
        if punct_only.match(stripped) and result:
            prev = result[-1].rstrip()
            result[-1] = prev + " " + stripped
        else:
            result.append(line)

    return "\n".join(result)


def _filter_pdf_noise(text: str) -> str:
    """PDF에서 세로 텍스트/워터마크로 인해 한 글자씩 줄바꿈된 노이즈 줄을 제거한다.
    3줄 이상 연속으로 한 글자(대문자 알파벳)만 있는 구간을 통째로 제거한다."""
    import re
    lines = text.split("\n")
    result = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # 한 글자 대문자 알파벳 줄(끝에 숫자 포함 가능)이 연속으로 이어지는지 확인
        if re.fullmatch(r"[A-Z]\s*\d*", stripped) and len(stripped) <= 4:
            run_start = i
            while i < len(lines) and re.fullmatch(r"[A-Z]\s*\d*", lines[i].strip()) and len(lines[i].strip()) <= 4:
                i += 1
            run_len = i - run_start
            # 3줄 이상 연속이면 노이즈로 판단하고 제거
            if run_len >= 3:
                continue
            # 3줄 미만이면 정상 텍스트로 유지
            result.extend(lines[run_start:i])
        else:
            result.append(lines[i])
            i += 1
    return "\n".join(result)


def _read_content(file_path: str) -> str:
    """파일 확장자에 따라 텍스트 추출"""
    file_path = _resolve_path(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
        text = "\n".join(pages_text).strip()
        return text or "PDF에서 텍스트를 추출할 수 없습니다."

    if ext in {".ppt", ".pptx"}:
        from pptx import Presentation
        prs = Presentation(file_path)
        sections = []
        for i, slide in enumerate(prs.slides, 1):
            lines = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text and text != str(i):  # 슬라이드 번호 단독 줄 제거
                            lines.append(text)
            if lines:
                sections.append(f"## [슬라이드 {i}]\n" + "\n".join(lines))
        return "\n\n".join(sections) or "PPT에서 텍스트를 추출할 수 없습니다."

    if ext in {".doc", ".docx"}:
        from docx import Document
        doc = Document(file_path)
        lines = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return "\n".join(lines) or "DOCX에서 텍스트를 추출할 수 없습니다."

    if ext in {".txt", ".md"}:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
        from PIL import Image
        import numpy as np

        img = Image.open(file_path).convert("RGB")

        # 해상도가 낮으면 2배 업스케일
        w, h = img.size
        if w < 1000 or h < 1000:
            img = img.resize((w * 2, h * 2), Image.LANCZOS)

        img_array = np.array(img)

        reader = _get_ocr_reader()
        # adjust_contrast: EasyOCR 내부 대비 보정, paragraph=False로 인식 누락 방지
        results = reader.readtext(img_array, detail=0, paragraph=False, adjust_contrast=0.5)
        text = "\n".join(results).strip()
        return text or "이미지에서 텍스트를 추출할 수 없습니다."

    return f"지원하지 않는 파일 형식입니다: {ext}"


def _sort_entries(entries: list, sort_by: str) -> list:
    """정렬 기준에 따라 (name, full_path) 리스트 정렬"""
    if sort_by == "name":
        return sorted(entries, key=lambda x: x[0].lower())
    if sort_by == "created":
        return sorted(entries, key=lambda x: os.path.getctime(x[1]))
    if sort_by == "modified":
        return sorted(entries, key=lambda x: os.path.getmtime(x[1]))
    return sorted(entries, key=lambda x: x[0].lower())


@tool
def list_directory(directory: str, sort_by: str = "name", folders_only: bool = False) -> str:
    """폴더 안의 파일과 하위 폴더 목록을 반환한다. 재귀 탐색 없이 해당 폴더 바로 아래만 조회한다.
    sort_by: 'name'(가나다순), 'created'(생성일순), 'modified'(수정일순)
    folders_only: True면 하위 폴더 목록만 반환"""
    if not os.path.exists(directory):
        return f"경로를 찾을 수 없습니다: {directory}"
    if not os.path.isdir(directory):
        return f"폴더가 아닙니다: {directory}"

    folder_entries = []
    file_entries = []

    for name in os.listdir(directory):
        full_path = os.path.join(directory, name)
        if os.path.isdir(full_path):
            folder_entries.append((name, full_path))
        elif not folders_only:
            file_entries.append((name, full_path))

    folder_entries = _sort_entries(folder_entries, sort_by)
    file_entries = _sort_entries(file_entries, sort_by)

    def fmt_time(path):
        if sort_by == "created":
            return datetime.fromtimestamp(os.path.getctime(path)).strftime("%Y-%m-%d %H:%M")
        if sort_by == "modified":
            return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        return ""

    folders_out = []
    for name, full_path in folder_entries:
        t = fmt_time(full_path)
        folders_out.append(f"  📁 {name}/" + (f"  ({t})" if t else ""))

    files_out = []
    for name, full_path in file_entries:
        ext = os.path.splitext(name)[1].lower()
        size_kb = os.path.getsize(full_path) // 1024
        t = fmt_time(full_path)
        tag = f"[{ext}]" if ext in SUPPORTED_EXTENSIONS else "📄"
        files_out.append(f"  {tag} {name} ({size_kb}KB)" + (f"  ({t})" if t else ""))

    sort_label = {"name": "가나다순", "created": "생성일순", "modified": "수정일순"}.get(sort_by, sort_by)
    label = "폴더 목록" if folders_only else "목록"
    output = [f"{directory} {label} ({sort_label}):"]
    if folders_out:
        output.append(f"\n폴더 ({len(folders_out)}개):")
        output.extend(folders_out)
    if files_out:
        output.append(f"\n파일 ({len(files_out)}개):")
        output.extend(files_out)
    if not folders_out and not files_out:
        return "빈 폴더입니다." if not folders_only else "하위 폴더가 없습니다."

    return "\n".join(output)


@tool
def count_files(directory: str) -> str:
    """폴더 내 파일 수, 폴더 수, 전체 항목 수를 반환한다."""
    if not os.path.exists(directory):
        return f"경로를 찾을 수 없습니다: {directory}"
    if not os.path.isdir(directory):
        return f"폴더가 아닙니다: {directory}"

    items = os.listdir(directory)
    folders = [i for i in items if os.path.isdir(os.path.join(directory, i))]
    files = [i for i in items if os.path.isfile(os.path.join(directory, i))]

    return (
        f"{directory} 항목 수:\n"
        f"  폴더: {len(folders)}개\n"
        f"  파일: {len(files)}개\n"
        f"  전체: {len(items)}개"
    )


@tool
def scan_files(directory: str, sort_by: str = "name") -> str:
    """디렉토리를 재귀적으로 스캔해서 지원되는 파일 목록을 반환한다. (pdf, ppt, pptx, doc, docx, txt, md, png, jpg, jpeg 등)
    sort_by: 'name'(가나다순), 'created'(생성일순), 'modified'(수정일순)"""
    if not os.path.exists(directory):
        return f"경로를 찾을 수 없습니다: {directory}"

    entries = []
    for root, _, files in os.walk(directory):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, filename)
                entries.append((filename, full_path))

    entries = _sort_entries(entries, sort_by)

    if not entries:
        return f"지원되는 파일이 없습니다: {directory}"

    found = []
    for filename, full_path in entries:
        ext = os.path.splitext(filename)[1].lower()
        size_kb = os.path.getsize(full_path) // 1024
        if sort_by == "created":
            t = datetime.fromtimestamp(os.path.getctime(full_path)).strftime("%Y-%m-%d %H:%M")
            found.append(f"- [{ext}] {full_path} ({size_kb}KB)  ({t})")
        elif sort_by == "modified":
            t = datetime.fromtimestamp(os.path.getmtime(full_path)).strftime("%Y-%m-%d %H:%M")
            found.append(f"- [{ext}] {full_path} ({size_kb}KB)  ({t})")
        else:
            found.append(f"- [{ext}] {full_path} ({size_kb}KB)")

    sort_label = {"name": "가나다순", "created": "생성일순", "modified": "수정일순"}.get(sort_by, sort_by)
    return f"파일 {len(found)}개 발견 ({sort_label}):\n" + "\n".join(found)


@tool
def find_file(keyword: str, extension: str = "") -> str:
    """파일명을 정확히 모를 때 키워드로 파일을 검색한다.
    keyword: 파일명에 포함된 단어 (예: '오픽', 'opic', 'CartoonGAN')
    extension: 확장자 필터 (예: 'pdf', 'pptx') — 생략 시 전체 검색"""
    keyword_lower = keyword.lower().strip().strip("'\"")
    ext_filter = extension.lower().strip().strip(".") if extension else ""

    matches = []
    for search_dir in SEARCH_DIRS:
        if not os.path.isdir(search_dir):
            continue
        for root, _, files in os.walk(search_dir):
            for fname in files:
                fname_lower = fname.lower()
                if keyword_lower not in fname_lower:
                    continue
                if ext_filter and not fname_lower.endswith(f".{ext_filter}"):
                    continue
                full_path = os.path.join(root, fname)
                size_kb = os.path.getsize(full_path) // 1024
                matches.append(f"{full_path} ({size_kb}KB)")

    if not matches:
        return f"'{keyword}' 키워드로 파일을 찾을 수 없습니다."
    return f"검색 결과 {len(matches)}개:\n" + "\n".join(matches)


def _resolve_dest(destination_dir: str) -> tuple:
    """목적지 경로 결정. 이미 존재하면 그대로 사용, 없으면 생성. (경로, 신규생성여부) 반환"""
    dest = destination_dir.strip() if destination_dir.strip() else NOTES_DIR
    already_exists = os.path.exists(dest)
    os.makedirs(dest, exist_ok=True)
    return dest, already_exists


@tool
def duplicate_note(source_title: str, new_title: str) -> str:
    """data/notes 폴더 내 노트를 새 이름으로 복제한다.
    source_title: 원본 노트 제목 (파일명에서 .md 제외), new_title: 새 노트 제목"""
    source_filename = f"{source_title.replace(' ', '_')}.md"
    source_path = os.path.join(NOTES_DIR, source_filename)

    if not os.path.exists(source_path):
        return f"노트를 찾을 수 없습니다: {source_filename}"

    new_filename = f"{new_title.replace(' ', '_')}.md"
    new_path = os.path.join(NOTES_DIR, new_filename)

    shutil.copy2(source_path, new_path)
    return f"복제 완료: {source_filename} → {new_filename}"


@tool
def move_file(source_path: str, destination_dir: str = "") -> str:
    """파일을 source_path에서 destination_dir 폴더로 이동한다. 원본은 삭제된다.
    destination_dir 미지정 시 기본값: LangChain-Agent/data/notes
    이미 존재하는 폴더면 해당 폴더로 이동, 없으면 새로 생성."""
    if not os.path.exists(source_path):
        return f"파일을 찾을 수 없습니다: {source_path}"
    if not os.path.isfile(source_path):
        return f"파일이 아닙니다: {source_path}"

    dest, already_exists = _resolve_dest(destination_dir)
    filename = os.path.basename(source_path)
    dest_path = os.path.join(dest, filename)

    shutil.move(source_path, dest_path)
    status = "기존 폴더" if already_exists else "새 폴더 생성"
    return f"이동 완료 ({status}): {source_path} → {dest_path}"


@tool
def copy_file(source_path: str, destination_dir: str = "") -> str:
    """파일을 source_path에서 destination_dir 폴더로 복사한다. 원본은 유지된다.
    destination_dir 미지정 시 기본값: LangChain-Agent/data/notes
    이미 존재하는 폴더면 해당 폴더로 복사, 없으면 새로 생성."""
    if not os.path.exists(source_path):
        return f"파일을 찾을 수 없습니다: {source_path}"
    if not os.path.isfile(source_path):
        return f"파일이 아닙니다: {source_path}"

    dest, already_exists = _resolve_dest(destination_dir)
    filename = os.path.basename(source_path)
    dest_path = os.path.join(dest, filename)

    shutil.copy2(source_path, dest_path)
    status = "기존 폴더" if already_exists else "새 폴더 생성"
    return f"복사 완료 ({status}): {source_path} → {dest_path}"



def _split_docx_pages(file_path: str) -> list:
    """DOCX를 실제 페이지 단위로 분할.
    우선순위: lastRenderedPageBreak(Word 렌더 기준) → 명시적 페이지 나누기 → 50단락 단위"""
    from docx import Document
    from docx.oxml.ns import qn
    from lxml import etree

    doc = Document(file_path)
    pages = []
    current = []

    for para in doc.paragraphs:
        # 단락 시작 전 page_break_before 속성
        if para.paragraph_format.page_break_before and current:
            pages.append("\n".join(current))
            current = []

        para_text_parts = []

        for run in para.runs:
            run_el = run._element
            for child in run_el:
                tag = etree.QName(child.tag).localname if child.tag != etree.Comment else ""
                # lastRenderedPageBreak — Word가 렌더링 시 기록한 실제 페이지 나누기
                if tag == "lastRenderedPageBreak":
                    text_so_far = run.text[:list(run_el).index(child)] if para_text_parts else ""
                    if text_so_far.strip():
                        para_text_parts.append(text_so_far.strip())
                    if para_text_parts or current:
                        current.extend(para_text_parts)
                        para_text_parts = []
                        pages.append("\n".join(current))
                        current = []
                # 명시적 페이지 나누기 (w:br w:type="page")
                elif tag == "br" and child.get(qn("w:type")) == "page":
                    current.extend(para_text_parts)
                    para_text_parts = []
                    if current:
                        pages.append("\n".join(current))
                        current = []

        text = para.text.strip()
        if text:
            current.append(text)

    if current:
        pages.append("\n".join(current))

    # 페이지 나누기를 전혀 감지 못한 경우 50단락 단위로 분할
    if len(pages) <= 1:
        all_paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        pages = [
            "\n".join(all_paras[i:i + 50])
            for i in range(0, len(all_paras), 50)
        ]

    return pages or ["(내용 없음)"]


def _split_txt_pages(file_path: str, lines_per_page: int = 50) -> list:
    """TXT/MD를 줄 단위로 분할 (기본 50줄)."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [l.rstrip() for l in f.readlines()]

    pages = []
    for i in range(0, max(len(lines), 1), lines_per_page):
        chunk = "\n".join(lines[i:i + lines_per_page]).strip()
        if chunk:
            pages.append(chunk)

    return pages or ["(내용 없음)"]


@tool
def read_file_full(file_path: str, page: int = 1) -> str:
    """파일을 실제 페이지 단위로 읽는다.
    - PDF: 실제 PDF 페이지
    - PPTX/PPT: 실제 슬라이드
    - DOCX/DOC: 명시적 페이지 나누기 기준 (없으면 50단락 단위)
    - TXT/MD: 50줄 단위
    지원 형식: pdf, ppt, pptx, doc, docx, txt, md
    page: 읽을 페이지 번호 (1부터 시작)"""
    file_path = _resolve_path(file_path)
    if not os.path.exists(file_path):
        return f"파일을 찾을 수 없습니다: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()

    # PDF — 실제 페이지
    if ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                if page < 1 or page > total_pages:
                    return f"페이지 범위 초과. 이 PDF는 총 {total_pages}페이지입니다. 1~{total_pages} 사이로 입력하세요."
                text = pdf.pages[page - 1].extract_text() or ""
                text = _filter_pdf_noise(text)
                return f"[PDF {page}/{total_pages} 페이지]\n\n{text.strip()}"
        except Exception as e:
            return f"PDF 읽기 실패: {e}"

    # PPTX/PPT — 실제 슬라이드
    if ext in {".pptx", ".ppt"}:
        try:
            from pptx import Presentation
            prs = Presentation(file_path)
            total_slides = len(prs.slides)
            if page < 1 or page > total_slides:
                return f"슬라이드 범위 초과. 이 파일은 총 {total_slides}슬라이드입니다. 1~{total_slides} 사이로 입력하세요."
            slide = prs.slides[page - 1]
            lines = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text and text != str(page):  # 슬라이드 번호 단독 줄 제거
                            lines.append(text)
            content = "\n".join(lines) or "(이미지 전용 슬라이드 — 텍스트 없음)"
            return f"[PPTX {page}/{total_slides} 슬라이드]\n\n{content}"
        except Exception as e:
            return f"PPTX 읽기 실패: {e}"

    # DOCX/DOC — 명시적 페이지 나누기 or 50단락 단위
    if ext in {".docx", ".doc"}:
        try:
            pages = _split_docx_pages(file_path)
            total_pages = len(pages)
            if page < 1 or page > total_pages:
                return f"페이지 범위 초과. 이 파일은 총 {total_pages}페이지입니다. 1~{total_pages} 사이로 입력하세요."
            return f"[DOCX {page}/{total_pages} 페이지]\n\n{pages[page - 1]}"
        except Exception as e:
            return f"DOCX 읽기 실패: {e}"

    # TXT/MD — 50줄 단위
    if ext in {".txt", ".md"}:
        try:
            pages = _split_txt_pages(file_path)
            total_pages = len(pages)
            if page < 1 or page > total_pages:
                return f"페이지 범위 초과. 이 파일은 총 {total_pages}페이지입니다. 1~{total_pages} 사이로 입력하세요."
            return f"[{ext.lstrip('.').upper()} {page}/{total_pages} 페이지]\n\n{pages[page - 1]}"
        except Exception as e:
            return f"파일 읽기 실패: {e}"

    return f"페이지 단위 읽기를 지원하지 않는 형식입니다: {ext}"


@tool
def file_to_note(file_path: str, note_title: str) -> str:
    """파일을 읽어서 바로 노트로 저장한다. 파일 읽기와 노트 저장을 한 번에 처리한다.
    지원 형식: pdf, ppt, pptx, doc, docx, txt, md, png, jpg, jpeg"""
    file_path = _resolve_path(file_path)
    if not os.path.exists(file_path):
        return f"파일을 찾을 수 없습니다: {file_path}"

    try:
        content = _read_content(file_path)
    except Exception as e:
        return f"파일 읽기 실패: {e}"

    os.makedirs(NOTES_DIR, exist_ok=True)
    filename = f"{note_title.replace(' ', '_')}.md"
    filepath = os.path.join(NOTES_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {note_title}\n\n")
        f.write(content)

    return f"저장 완료: data/notes/{filename} ({len(content)}자)"


@tool
def create_file(file_path: str, content: str = "") -> str:
    """지정한 경로에 파일을 생성한다. 이미 존재하면 덮어쓴다.
    file_path: 생성할 파일 전체 경로, content: 파일 내용 (기본값: 빈 파일)"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"파일 생성 완료: {file_path}"
    except Exception as e:
        return f"파일 생성 실패: {e}"


@tool
def delete_file(file_path: str) -> str:
    """지정한 파일을 삭제한다. 폴더는 삭제할 수 없다."""
    if not os.path.exists(file_path):
        return f"파일을 찾을 수 없습니다: {file_path}"
    if os.path.isdir(file_path):
        return f"폴더는 삭제할 수 없습니다. 파일 경로를 입력하세요: {file_path}"

    os.remove(file_path)
    return f"삭제 완료: {file_path}"


@tool
def delete_folder(folder_path: str) -> str:
    """지정한 폴더와 내부 파일을 모두 삭제한다."""
    if not os.path.exists(folder_path):
        return f"폴더를 찾을 수 없습니다: {folder_path}"
    if not os.path.isdir(folder_path):
        return f"폴더가 아닙니다: {folder_path}"

    shutil.rmtree(folder_path)
    return f"폴더 삭제 완료: {folder_path}"
