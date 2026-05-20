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
    ".pptx",  # .ppt(구버전) 제외 — python-pptx 미지원
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


def _display_width(text: str) -> int:
    """한국어 등 전각 문자는 2, 나머지는 1로 계산한 표시 너비."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def _format_table(rows: list) -> str:
    """2D 리스트(rows)를 텍스트 표 형태로 변환."""
    if not rows:
        return ""
    # 열 수 통일
    col_count = max(len(r) for r in rows)
    rows = [r + [""] * (col_count - len(r)) for r in rows]
    # 열별 최대 너비 계산
    col_widths = [
        max(_display_width(str(rows[r][c])) for r in range(len(rows)))
        for c in range(col_count)
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    lines = [sep]
    for row in rows:
        cells = []
        for c, cell in enumerate(row):
            cell = str(cell)
            pad = col_widths[c] - _display_width(cell)
            cells.append(f" {cell}{' ' * pad} ")
        lines.append("|" + "|".join(cells) + "|")
        lines.append(sep)
    return "\n".join(lines)


def _fitz_page_text(page) -> str:
    """pymupdf dict 모드 기반 텍스트 추출.
    - block→line 구조를 그대로 사용해 글머리 기호가 같은 줄에 자연스럽게 포함
    - 로고/워터마크 블록(밀도 낮은 블록) 제거
    - 표는 find_tables()로 행×열(좌→우, 위→아래) 순서로 추출
    """
    import re

    # 1. 표 먼저 추출 및 영역 기록
    table_entries = []  # (y0, text)
    table_bboxes = []
    try:
        for tbl in page.find_tables():
            table_bboxes.append(tbl.bbox)
            rows = [[str(c or "").strip() for c in row] for row in tbl.extract()]
            table_entries.append((tbl.bbox[1], _format_table(rows)))
    except Exception:
        pass

    def _in_table(x0, y0, x1, y1):
        for bbox in table_bboxes:
            if x0 >= bbox[0] - 2 and y0 >= bbox[1] - 2 \
               and x1 <= bbox[2] + 2 and y1 <= bbox[3] + 2:
                return True
        return False

    # 2. 노이즈 블록 영역 파악 (blocks 모드로 로고/워터마크 bbox 수집)
    noise_bboxes = []
    for b in page.get_text("blocks"):
        if b[6] != 0:
            continue
        text = b[4].strip()
        if not text:
            continue
        area = (b[2] - b[0]) * (b[3] - b[1])
        char_count = len(text.replace("\n", "").replace(" ", ""))
        has_korean = bool(re.search(r"[\uac00-\ud7af]", text))
        has_bullet = bool(re.search(r"[•·▪▸►◦‣⁃]", text))
        if not has_korean and not has_bullet and area > 0 and char_count > 0:
            if (char_count / area) < 0.01 and char_count <= 10:
                noise_bboxes.append((b[0], b[1], b[2], b[3]))

    def _is_noise(x0, y0, x1, y1):
        for bbox in noise_bboxes:
            if x0 >= bbox[0] - 1 and y0 >= bbox[1] - 1 \
               and x1 <= bbox[2] + 1 and y1 <= bbox[3] + 1:
                return True
        return False

    # 3. rawdict 모드로 문자 단위 위치 분석 — 공백 문자 없이 간격으로만 띄어쓰기를
    #    표현하는 PDF도 처리 (문자 간격 > 폰트 크기 * 0.25 이면 공백 삽입)
    line_entries = []  # (y0, x0, text)
    for block in page.get_text("rawdict", sort=True).get("blocks", []):
        # 이미지 블록 → 위치에 "(이미지)" 표시
        if block.get("type") == 1:
            line_entries.append((block["bbox"][1], block["bbox"][0], "(이미지)"))
            continue
        if block.get("type") != 0:
            continue
        bx0, by0, bx1, by1 = block["bbox"]
        if _in_table(bx0, by0, bx1, by1):
            continue
        if _is_noise(bx0, by0, bx1, by1):
            continue
        for line in block.get("lines", []):
            # 모든 span의 문자를 x 순으로 수집
            chars = []
            for span in line.get("spans", []):
                size = span.get("size", 10)
                for ch in span.get("chars", []):
                    c = ch.get("c", "")
                    if not c:
                        continue
                    ox = ch["origin"][0]
                    x1c = ch["bbox"][2]
                    chars.append({"c": c, "x": ox, "x1": x1c, "size": size})
            if not chars:
                continue
            chars.sort(key=lambda c: c["x"])

            # 문자 간격으로 공백 여부 판단
            text = chars[0]["c"]
            for i in range(1, len(chars)):
                gap = chars[i]["x"] - chars[i - 1]["x1"]
                if gap > chars[i - 1]["size"] * 0.25:
                    text += " "
                text += chars[i]["c"]

            text = text.strip()
            if text:
                line_entries.append((line["bbox"][1], line["bbox"][0], text))

    # 4. y0 기준 5pt 이내 줄은 같은 행으로 묶어 x0 순으로 공백 조인
    line_entries.sort(key=lambda x: (x[0], x[1]))
    rows = []
    for entry in line_entries:
        if rows and abs(entry[0] - rows[-1][0][0]) <= 5:
            rows[-1].append(entry)
        else:
            rows.append([entry])

    text_entries = []  # (y0, text)
    for row in rows:
        y0 = row[0][0]
        row_text = " ".join(t for _, _, t in sorted(row, key=lambda x: x[1]))
        text_entries.append((y0, row_text))

    # 5. 표와 텍스트를 y 위치 순으로 합치기
    for y0, ttext in table_entries:
        text_entries.append((y0, ttext))
    text_entries.sort(key=lambda x: x[0])

    return "\n".join(t for _, t in text_entries)


def _read_content(file_path: str) -> str:
    """파일 확장자에 따라 텍스트 추출"""
    file_path = _resolve_path(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        import fitz
        doc = fitz.open(file_path)
        pages_text = [_fitz_page_text(doc[i]) for i in range(doc.page_count)]
        doc.close()
        text = "\n".join(pages_text).strip()
        return text or "PDF에서 텍스트를 추출할 수 없습니다."

    if ext in {".ppt", ".pptx"}:
        if ext == ".ppt":
            return "지원하지 않는 형식입니다: 구버전 .ppt(PowerPoint 97-2003)는 읽을 수 없습니다. .pptx로 변환 후 사용하세요."
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
    우선순위: lastRenderedPageBreak(Word 렌더 기준) → 명시적 페이지 나누기 → 50단락 단위
    단락과 표를 문서 순서대로 처리."""
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from lxml import etree

    doc = Document(file_path)
    pages = []
    current = []

    def _flush():
        if current:
            pages.append("\n".join(current))
            current.clear()

    # body 요소를 순서대로 순회 (단락 + 표 혼재)
    for child in doc.element.body:
        tag = etree.QName(child.tag).localname if child.tag != etree.Comment else ""

        # 표 처리
        if tag == "tbl":
            tbl = Table(child, doc)
            tbl_rows = []
            for row in tbl.rows:
                tbl_rows.append([cell.text.strip() for cell in row.cells])
            current.append(_format_table(tbl_rows))
            continue

        # 단락 처리
        if tag != "p":
            continue
        para = Paragraph(child, doc)

        if para.paragraph_format.page_break_before and current:
            _flush()

        for run in para.runs:
            run_el = run._element
            for rchild in run_el:
                rtag = etree.QName(rchild.tag).localname if rchild.tag != etree.Comment else ""
                if rtag == "lastRenderedPageBreak":
                    _flush()
                elif rtag == "br" and rchild.get(qn("w:type")) == "page":
                    _flush()

        text = para.text.strip()
        if text:
            current.append(text)

    _flush()

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
            import fitz
            doc = fitz.open(file_path)
            total_pages = doc.page_count
            if page < 1 or page > total_pages:
                doc.close()
                return f"페이지 범위 초과. 이 PDF는 총 {total_pages}페이지입니다. 1~{total_pages} 사이로 입력하세요."
            text = _fitz_page_text(doc[page - 1])
            doc.close()
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
            # top 기준으로 정렬해 시각적 위→아래 순서 보장
            sorted_shapes = sorted(slide.shapes, key=lambda s: (s.top or 0, s.left or 0))
            for shape in sorted_shapes:
                # 이미지 shape → 위치에 "(이미지)" 표시
                if shape.shape_type in (13, 3):  # PICTURE, LINKED_PICTURE
                    lines.append("(이미지)")
                    continue
                # 표 shape → _format_table로 텍스트 표 변환
                if shape.shape_type == 19:  # TABLE
                    tbl_rows = []
                    for row in shape.table.rows:
                        tbl_rows.append([cell.text.strip() for cell in row.cells])
                    lines.append(_format_table(tbl_rows))
                    continue
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


@tool
def search_in_files(keyword: str, directory: str = "", file_types: str = "all") -> str:
    """파일 내용에서 키워드를 직접 검색한다. vectorDB 인덱싱 없이 실제 파일을 읽어 텍스트를 탐색한다.
    keyword: 검색할 텍스트 (대소문자 구분 없음)
    directory: 검색할 폴더 경로 (기본값: data/notes 전체)
    file_types: 'text'(md/txt만), 'doc'(pdf/pptx/ppt/docx/doc), 'all'(전체, 기본값)
    """
    TEXT_EXTS = {".md", ".txt"}
    DOC_EXTS = {".pdf", ".pptx", ".ppt", ".docx", ".doc"}

    search_dir = directory.strip() if directory.strip() else NOTES_DIR
    if not os.path.isdir(search_dir):
        return f"폴더를 찾을 수 없습니다: {search_dir}"

    if file_types == "text":
        target_exts = TEXT_EXTS
    elif file_types == "doc":
        target_exts = DOC_EXTS
    else:
        target_exts = TEXT_EXTS | DOC_EXTS

    keyword_lower = keyword.lower()
    matches = []
    errors = []

    for root, _, files in os.walk(search_dir):
        for name in sorted(files):
            if name.startswith("~$"):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in target_exts:
                continue

            filepath = os.path.join(root, name)
            try:
                if ext in TEXT_EXTS:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                else:
                    content = _read_content(filepath)

                if keyword_lower not in content.lower():
                    continue

                lines = content.split("\n")
                matched_lines = []
                total_count = 0
                for i, line in enumerate(lines):
                    if keyword_lower in line.lower():
                        total_count += 1
                        if len(matched_lines) < 3:
                            matched_lines.append(f"  줄 {i + 1}: {line.strip()[:120]}")

                rel_path = os.path.relpath(filepath, PROJECT_ROOT)
                matches.append({"path": rel_path, "lines": matched_lines, "count": total_count})

            except Exception as e:
                errors.append(f"{name}: {e}")

    if not matches:
        result = f"'{keyword}' 검색 결과 없음."
        if errors:
            result += f"\n오류 {len(errors)}건: " + ", ".join(errors[:3])
        return result

    out = [f"'{keyword}' 검색 결과: {len(matches)}개 파일에서 발견"]
    for m in matches:
        out.append(f"\n📄 {m['path']} ({m['count']}개 매칭)")
        out.extend(m["lines"])
        if m["count"] > 3:
            out.append(f"  ... 외 {m['count'] - 3}개 더")

    if errors:
        out.append(f"\n오류 {len(errors)}건: " + ", ".join(errors[:3]))

    return "\n".join(out)
