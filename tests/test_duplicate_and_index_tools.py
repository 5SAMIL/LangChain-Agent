import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from tools.duplicate_tool import (
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    TOP_K,
    _top_keywords,
    check_note_duplicates,
)
from tools.index_tool import index_all_notes


def _write_note(path: str, title: str, body: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{body}")


def _make_mock_vectorstore(results=None):
    mock_vs = MagicMock()
    mock_vs.similarity_search_with_score.return_value = results or []
    mock_vs.add_documents = MagicMock()
    return mock_vs


# ── index_all_notes ──────────────────────────────────────────────────────────

class IndexAllNotesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.notes_dir = os.path.join(self.temp_dir, "notes")
        os.makedirs(self.notes_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_indexes_all_md_files_and_returns_count(self):
        _write_note(os.path.join(self.notes_dir, "study", "2026-04", "note1.md"), "노트1", "머신러닝 기초 내용")
        _write_note(os.path.join(self.notes_dir, "study", "2026-04", "note2.md"), "노트2", "딥러닝 신경망 내용")

        mock_vs = _make_mock_vectorstore()
        with patch("tools.index_tool.NOTES_DIR", self.notes_dir), \
             patch("tools.index_tool._get_vectorstore", return_value=mock_vs):
            result = index_all_notes.invoke({})

        self.assertIn("인덱싱 완료", result)
        self.assertIn("2개", result)
        added = mock_vs.add_documents.call_args[0][0]
        self.assertEqual(len(added), 2)

    def test_skips_non_md_files(self):
        _write_note(os.path.join(self.notes_dir, "note.md"), "노트", "내용")
        with open(os.path.join(self.notes_dir, "image.png"), "wb") as f:
            f.write(b"fake image data")

        mock_vs = _make_mock_vectorstore()
        with patch("tools.index_tool.NOTES_DIR", self.notes_dir), \
             patch("tools.index_tool._get_vectorstore", return_value=mock_vs):
            index_all_notes.invoke({})

        added = mock_vs.add_documents.call_args[0][0]
        self.assertEqual(len(added), 1)

    def test_returns_no_notes_message_when_directory_empty(self):
        mock_vs = _make_mock_vectorstore()
        with patch("tools.index_tool.NOTES_DIR", self.notes_dir), \
             patch("tools.index_tool._get_vectorstore", return_value=mock_vs):
            result = index_all_notes.invoke({})

        self.assertIn("인덱싱할 노트가 없습니다", result)
        mock_vs.add_documents.assert_not_called()

    def test_returns_error_when_notes_dir_does_not_exist(self):
        with patch("tools.index_tool.NOTES_DIR", "/nonexistent/path/notes"):
            result = index_all_notes.invoke({})

        self.assertIn("data/notes 폴더가 없습니다", result)

    def test_document_source_uses_data_notes_prefix(self):
        _write_note(os.path.join(self.notes_dir, "study", "note.md"), "노트", "내용")

        mock_vs = _make_mock_vectorstore()
        with patch("tools.index_tool.NOTES_DIR", self.notes_dir), \
             patch("tools.index_tool._get_vectorstore", return_value=mock_vs):
            index_all_notes.invoke({})

        added = mock_vs.add_documents.call_args[0][0]
        self.assertTrue(added[0].metadata["source"].startswith("data/notes/"))

    def test_indexes_notes_in_nested_subdirectories(self):
        _write_note(os.path.join(self.notes_dir, "a", "b", "c", "deep.md"), "깊은노트", "내용")

        mock_vs = _make_mock_vectorstore()
        with patch("tools.index_tool.NOTES_DIR", self.notes_dir), \
             patch("tools.index_tool._get_vectorstore", return_value=mock_vs):
            result = index_all_notes.invoke({})

        self.assertIn("1개", result)


# ── check_note_duplicates ────────────────────────────────────────────────────

class CheckNoteDuplicatesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["PKM_NOTES_DIR"] = self.temp_dir
        self.cache_path = os.path.join(self.temp_dir, "duplicate_cache.json")

    def tearDown(self):
        os.environ.pop("PKM_NOTES_DIR", None)
        shutil.rmtree(self.temp_dir)

    def _note(self, rel_path, title, body):
        path = os.path.join(self.temp_dir, rel_path)
        _write_note(path, title, body)
        return path

    def test_returns_no_duplicates_when_vectordb_empty(self):
        self._note("study/2026-04/머신러닝.md", "머신러닝", "머신러닝 기초 내용")

        mock_vs = _make_mock_vectorstore([])
        with patch("tools.duplicate_tool._get_vectorstore", return_value=mock_vs), \
             patch("tools.duplicate_tool.CACHE_PATH", self.cache_path):
            result = check_note_duplicates.invoke({"note_reference": "머신러닝"})

        self.assertIn("중복 탐지 결과", result)
        self.assertIn("유사 문서 없음", result)

    def test_classifies_high_similarity_as_duplicate(self):
        self._note("study/2026-04/note_a.md", "노트A", "딥러닝 신경망 학습률 역전파")

        # 유사도 88% = score 0.12
        similar_doc = Document(
            page_content="딥러닝 신경망 학습률 역전파 내용",
            metadata={"source": "data/notes/study/2026-03/note_b.md"},
        )
        mock_vs = _make_mock_vectorstore([(similar_doc, 0.12)])
        with patch("tools.duplicate_tool._get_vectorstore", return_value=mock_vs), \
             patch("tools.duplicate_tool.CACHE_PATH", self.cache_path):
            result = check_note_duplicates.invoke({"note_reference": "노트A"})

        self.assertIn("중복 가능성 높음", result)
        self.assertIn("88.0%", result)

    def test_classifies_medium_similarity_as_similar_document(self):
        self._note("study/2026-04/note_a.md", "노트A", "머신러닝 기초")

        # 유사도 75% = score 0.25
        similar_doc = Document(
            page_content="인공지능과 머신러닝 개요",
            metadata={"source": "data/notes/study/2026-03/note_c.md"},
        )
        mock_vs = _make_mock_vectorstore([(similar_doc, 0.25)])
        with patch("tools.duplicate_tool._get_vectorstore", return_value=mock_vs), \
             patch("tools.duplicate_tool.CACHE_PATH", self.cache_path):
            result = check_note_duplicates.invoke({"note_reference": "노트A"})

        self.assertIn("유사 문서", result)
        self.assertIn("75.0%", result)

    def test_excludes_self_from_results(self):
        self._note("study/2026-04/note_a.md", "노트A", "동일한 내용")

        # vectorDB가 자기 자신을 99% 유사도로 반환
        self_doc = Document(
            page_content="동일한 내용",
            metadata={"source": "data/notes/study/2026-04/note_a.md"},
        )
        mock_vs = _make_mock_vectorstore([(self_doc, 0.01)])
        with patch("tools.duplicate_tool._get_vectorstore", return_value=mock_vs), \
             patch("tools.duplicate_tool.CACHE_PATH", self.cache_path):
            result = check_note_duplicates.invoke({"note_reference": "노트A"})

        self.assertIn("유사 문서 없음", result)

    def test_filters_results_below_low_threshold(self):
        self._note("study/2026-04/note_a.md", "노트A", "내용")

        # 유사도 60% = LOW_THRESHOLD(70%) 미만 → 출력 안 함
        low_doc = Document(
            page_content="전혀 다른 내용",
            metadata={"source": "data/notes/other.md"},
        )
        mock_vs = _make_mock_vectorstore([(low_doc, 0.40)])
        with patch("tools.duplicate_tool._get_vectorstore", return_value=mock_vs), \
             patch("tools.duplicate_tool.CACHE_PATH", self.cache_path):
            result = check_note_duplicates.invoke({"note_reference": "노트A"})

        self.assertIn("유사 문서 없음", result)

    def test_returns_at_most_top_k_results(self):
        self._note("study/2026-04/note_a.md", "노트A", "내용")

        docs = [
            (Document(page_content=f"유사 내용 {i}", metadata={"source": f"data/notes/other_{i}.md"}), 0.10)
            for i in range(TOP_K + 3)
        ]
        mock_vs = _make_mock_vectorstore(docs)
        with patch("tools.duplicate_tool._get_vectorstore", return_value=mock_vs), \
             patch("tools.duplicate_tool.CACHE_PATH", self.cache_path):
            result = check_note_duplicates.invoke({"note_reference": "노트A"})

        count = result.count("위 |")
        self.assertLessEqual(count, TOP_K)

    def test_saves_result_to_cache_file(self):
        self._note("study/2026-04/note_a.md", "노트A", "내용")
        mock_vs = _make_mock_vectorstore([])
        with patch("tools.duplicate_tool._get_vectorstore", return_value=mock_vs), \
             patch("tools.duplicate_tool.CACHE_PATH", self.cache_path):
            check_note_duplicates.invoke({"note_reference": "노트A"})

        self.assertTrue(os.path.exists(self.cache_path))
        with open(self.cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        entry = list(cache.values())[0]
        self.assertIn("checked_at", entry)
        self.assertIn("char_count", entry)
        self.assertIn("results", entry)

    def test_output_includes_path_and_keywords(self):
        self._note("study/2026-04/note_a.md", "노트A", "머신러닝 딥러닝 신경망")

        similar_doc = Document(
            page_content="딥러닝 신경망 머신러닝 학습",
            metadata={"source": "data/notes/study/2026-03/note_b.md"},
        )
        mock_vs = _make_mock_vectorstore([(similar_doc, 0.10)])
        with patch("tools.duplicate_tool._get_vectorstore", return_value=mock_vs), \
             patch("tools.duplicate_tool.CACHE_PATH", self.cache_path):
            result = check_note_duplicates.invoke({"note_reference": "노트A"})

        self.assertIn("경로:", result)
        self.assertIn("공통 키워드:", result)
        self.assertIn("note_b.md", result)

    def test_returns_error_for_nonexistent_note(self):
        mock_vs = _make_mock_vectorstore([])
        with patch("tools.duplicate_tool._get_vectorstore", return_value=mock_vs), \
             patch("tools.duplicate_tool.CACHE_PATH", self.cache_path):
            result = check_note_duplicates.invoke({"note_reference": "존재하지않는노트xyz"})

        self.assertIn("파일 처리 실패", result)

    def test_returns_error_for_empty_note_content(self):
        empty_path = os.path.join(self.temp_dir, "study", "2026-04", "empty.md")
        os.makedirs(os.path.dirname(empty_path), exist_ok=True)
        with open(empty_path, "w", encoding="utf-8") as f:
            f.write("")

        mock_vs = _make_mock_vectorstore([])
        with patch("tools.duplicate_tool._get_vectorstore", return_value=mock_vs), \
             patch("tools.duplicate_tool.CACHE_PATH", self.cache_path):
            result = check_note_duplicates.invoke({"note_reference": "empty"})

        self.assertIn("문서 내용이 비어 있습니다", result)

    def test_returns_error_for_unsupported_file_extension(self):
        # notes 디렉토리 밖에 생성해야 step1이 skip되고 step2의 확장자 검사에 걸림
        outside_dir = tempfile.mkdtemp()
        unsupported = os.path.join(outside_dir, "file.xyz")
        try:
            with open(unsupported, "w") as f:
                f.write("내용")

            mock_vs = _make_mock_vectorstore([])
            with patch("tools.duplicate_tool._get_vectorstore", return_value=mock_vs), \
                 patch("tools.duplicate_tool.CACHE_PATH", self.cache_path):
                result = check_note_duplicates.invoke({"note_reference": unsupported})

            self.assertIn("지원하지 않는 형식", result)
        finally:
            shutil.rmtree(outside_dir)

    def test_accepts_direct_file_path_instead_of_note_title(self):
        path = os.path.join(self.temp_dir, "study", "2026-04", "직접경로.md")
        _write_note(path, "직접경로", "파일 경로로 직접 입력한 내용")

        mock_vs = _make_mock_vectorstore([])
        with patch("tools.duplicate_tool._get_vectorstore", return_value=mock_vs), \
             patch("tools.duplicate_tool.CACHE_PATH", self.cache_path):
            result = check_note_duplicates.invoke({"note_reference": path})

        self.assertIn("중복 탐지 결과", result)
        self.assertIn("직접경로.md", result)


# ── _top_keywords ────────────────────────────────────────────────────────────

class TopKeywordsTests(unittest.TestCase):
    def test_returns_most_frequent_terms_in_order(self):
        text = "머신러닝 머신러닝 딥러닝 신경망 머신러닝 딥러닝"
        result = _top_keywords(text, k=2)
        self.assertEqual(result[0], "머신러닝")
        self.assertEqual(result[1], "딥러닝")

    def test_filters_stopwords(self):
        text = "이번 그리고 머신러닝 딥러닝 해당 및"
        result = _top_keywords(text, k=10)
        self.assertNotIn("이번", result)
        self.assertNotIn("그리고", result)
        self.assertIn("머신러닝", result)

    def test_returns_empty_list_for_stopword_only_text(self):
        text = "이번 그리고 해당 및 또는"
        result = _top_keywords(text, k=5)
        self.assertEqual(result, [])

    def test_respects_k_limit(self):
        text = "가나다 나다라 다라마 라마바 마바사 바사아"
        result = _top_keywords(text, k=3)
        self.assertLessEqual(len(result), 3)


# ── 툴 자동 등록 확인 ─────────────────────────────────────────────────────────

class ToolRegistrationTests(unittest.TestCase):
    def test_new_tools_are_auto_loaded(self):
        from tools import load_all_tools
        tool_names = {t.name for t in load_all_tools()}
        self.assertIn("index_all_notes", tool_names)
        self.assertIn("check_note_duplicates", tool_names)

    def test_constants_match_expected_values(self):
        self.assertEqual(HIGH_THRESHOLD, 0.85)
        self.assertEqual(LOW_THRESHOLD, 0.70)
        self.assertEqual(TOP_K, 3)


if __name__ == "__main__":
    unittest.main()
