import os
import shutil
import tempfile
import unittest

from tools import load_all_tools
from tools.a_classification_tool import a_classify_document
from tools.note_ops_utils import (
    ensure_markdown_filename,
    extract_month_folder,
    parse_note_file,
    resolve_note_reference_path,
    resolve_relative_path,
    write_note_file,
)
from tools.organizer_tool import (
    organize_and_save_note,
    organize_file_and_save_note,
    preview_file_organized_path,
    preview_organized_path,
)


class OrganizerAndUtilsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["PKM_NOTES_DIR"] = self.temp_dir

    def tearDown(self):
        os.environ.pop("PKM_NOTES_DIR", None)
        shutil.rmtree(self.temp_dir)

    def test_load_all_tools_includes_added_tools(self):
        tool_names = sorted(tool.name for tool in load_all_tools())

        expected = {
            "organize_and_save_note",
            "organize_file_and_save_note",
            "preview_organized_path",
            "preview_file_organized_path",
            "rename_organized_folder",
            "move_note_to_category",
            "rename_note_file",
            "preview_reorganized_note",
            "reorganize_existing_note",
            "update_note_metadata",
            "batch_reorganize_notes",
        }
        self.assertTrue(expected.issubset(set(tool_names)))

    def test_preview_organized_path_classifies_document(self):
        result = preview_organized_path.invoke(
            {
                "title": "주간 회의 문서",
                "content": "회의 agenda 와 action item 정리",
                "source_type": "document",
            }
        )

        self.assertIn("예상 분류: meetings", result)
        self.assertIn("data/notes/meetings/", result)

    def test_organize_and_save_note_creates_file_and_deduplicates_name(self):
        first_result = organize_and_save_note.invoke(
            {
                "title": "강의 정리",
                "content": "LangChain 강의와 학습 정리",
                "source_type": "pdf",
                "source_name": "lecture.pdf",
            }
        )
        second_result = organize_and_save_note.invoke(
            {
                "title": "강의 정리",
                "content": "LangChain 강의와 학습 정리",
                "source_type": "pdf",
                "source_name": "lecture.pdf",
            }
        )

        self.assertIn("자동 정리 완료: data/notes/study/", first_result)
        self.assertIn("자동 정리 완료: data/notes/study/", second_result)
        created_files = []
        for root, _, files in os.walk(self.temp_dir):
            for name in files:
                created_files.append(os.path.relpath(os.path.join(root, name), self.temp_dir))

        self.assertEqual(len(created_files), 2)
        self.assertTrue(any(path.endswith("강의_정리.md") for path in created_files))
        self.assertTrue(any(path.endswith("강의_정리_2.md") for path in created_files))

    def test_note_ops_utils_parse_and_write_round_trip(self):
        filepath = os.path.join(self.temp_dir, "projects", "2026-04", "api_정리.md")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        write_note_file(
            filepath,
            "API 정리",
            {
                "저장 시각": "2026-04-15 10:00",
                "문서 유형": "document",
                "출처": "notion",
                "분류 폴더": "projects",
                "태그": "api, backend",
            },
            "본문 내용",
        )

        title, metadata, body = parse_note_file(filepath)
        self.assertEqual(title, "API 정리")
        self.assertEqual(metadata["출처"], "notion")
        self.assertEqual(metadata["태그"], "api, backend")
        self.assertEqual(body, "본문 내용")

    def test_note_ops_utils_helpers_normalize_paths(self):
        self.assertEqual(resolve_relative_path(" meetings/2026-04/주간 회의 "), "meetings/2026-04/주간_회의")
        self.assertEqual(ensure_markdown_filename("새 제목"), "새_제목.md")
        self.assertEqual(extract_month_folder("inbox/2026-04/노트.md"), "2026-04")

    def test_resolve_note_reference_path_supports_title_and_prefixed_paths(self):
        filepath = os.path.join(self.temp_dir, "meetings", "2026-04", "주간_회의.md")
        write_note_file(filepath, "주간 회의", {"분류 폴더": "meetings"}, "회의 안건")

        by_title = resolve_note_reference_path("주간 회의")
        by_prefixed_path = resolve_note_reference_path("data/notes/meetings/2026-04/주간_회의.md")

        self.assertEqual(os.path.realpath(by_title), os.path.realpath(filepath))
        self.assertEqual(os.path.realpath(by_prefixed_path), os.path.realpath(filepath))

    def test_preview_and_organize_file_tools_support_file_input(self):
        source_file = os.path.join(self.temp_dir, "인공지능_강의.txt")
        with open(source_file, "w", encoding="utf-8") as f:
            f.write("인공지능 강의와 학습 정리")

        preview = preview_file_organized_path.invoke(
            {"file_path": source_file, "note_title": "AI 강의 정리", "source_type": "document"}
        )
        result = organize_file_and_save_note.invoke(
            {"file_path": source_file, "note_title": "AI 강의 정리", "source_type": "document"}
        )

        self.assertIn("예상 분류: study", preview)
        self.assertIn("자동 정리 완료: data/notes/study/", result)

    def test_a_tools_can_read_nested_saved_note_by_title(self):
        filepath = os.path.join(self.temp_dir, "study", "2026-04", "강의요약.md")
        write_note_file(filepath, "강의요약", {"분류 폴더": "study"}, "강의 정리와 학습 내용")

        result = a_classify_document.invoke({"note_title": "강의요약"})

        self.assertIn("A 자동 분류 결과", result)
        self.assertIn("category: study", result)


if __name__ == "__main__":
    unittest.main()
