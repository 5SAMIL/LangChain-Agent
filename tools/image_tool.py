"""이미지 OCR 읽기 Tool"""
import os
from langchain_core.tools import tool

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


@tool
def read_image(file_path: str) -> str:
    """이미지 파일에서 OCR로 텍스트를 추출한다. (png, jpg, jpeg, bmp, tiff, webp)"""
    if not os.path.exists(file_path):
        return f"파일을 찾을 수 없습니다: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return f"지원하지 않는 이미지 형식입니다: {ext}"

    try:
        import pytesseract
        from PIL import Image
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img, lang="kor+eng")
        return text.strip() or "이미지에서 텍스트를 추출할 수 없습니다."
    except pytesseract.TesseractNotFoundError:
        return "Tesseract가 설치되어 있지 않습니다. 'brew install tesseract tesseract-lang' 실행 후 다시 시도하세요."
    except Exception as e:
        return f"이미지 읽기 실패: {e}"
