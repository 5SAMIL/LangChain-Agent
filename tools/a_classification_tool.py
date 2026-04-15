"""A 역할 - 자동 분류 Tool"""

from langchain_core.tools import tool
from tools.a_feature_common import run_a_analysis


@tool
def a_classify_document(
    content: str = "",
    note_title: str = "",
    source_name: str = "",
    source_type: str = "",
) -> str:
    """A 역할: 문서를 자동 분류한다. content 또는 note_title을 입력받는다."""
    try:
        analysis = run_a_analysis(
            content=content,
            note_title=note_title,
            source_name=source_name,
            source_type=source_type,
        )
    except Exception as e:
        return f"A 자동 분류 실패: {e}"

    cls = analysis.get("classification", {})
    reasons = cls.get("reason", []) or []
    reason_text = ", ".join(reasons) if reasons else "근거 없음"

    return (
        "A 자동 분류 결과\n"
        f"analysis_id: {analysis.get('analysis_id')}\n"
        f"source: {analysis.get('source_name')} ({analysis.get('source_type')})\n"
        f"category: {cls.get('category')}\n"
        f"subcategory: {cls.get('subcategory')}\n"
        f"confidence: {cls.get('confidence')}\n"
        f"reason: {reason_text}\n"
        f"default_folder: {cls.get('default_folder')}"
    )
