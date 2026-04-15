"""A 역할 - 정리 결과 제안 Tool"""

from langchain_core.tools import tool
from tools.a_feature_common import run_a_analysis


@tool
def a_suggest_organization(
    content: str = "",
    note_title: str = "",
    source_name: str = "",
    source_type: str = "",
) -> str:
    """A 역할: 문서 정리 결과를 제안한다. content 또는 note_title을 입력받는다."""
    try:
        analysis = run_a_analysis(
            content=content,
            note_title=note_title,
            source_name=source_name,
            source_type=source_type,
        )
    except Exception as e:
        return f"A 정리 제안 실패: {e}"

    suggestion = analysis.get("organization_suggestion", {})
    return (
        "A 정리 결과 제안\n"
        f"analysis_id: {analysis.get('analysis_id')}\n"
        f"source: {analysis.get('source_name')} ({analysis.get('source_type')})\n"
        f"recommended_folder: {suggestion.get('recommended_folder')}\n"
        f"recommended_filename: {suggestion.get('recommended_filename')}\n"
        f"reason: {suggestion.get('reason')}\n"
        f"confidence: {suggestion.get('confidence')}\n"
        f"next_action_owner: {suggestion.get('next_action_owner')}\n"
        f"execution: {suggestion.get('execution')}"
    )
