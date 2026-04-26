import os
import shutil
import tempfile
import unittest

from tools.folder_tool import rename_organized_folder


class FolderToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        os.environ["PKM_NOTES_DIR"] = self.temp_dir

    def tearDown(self):
        os.environ.pop("PKM_NOTES_DIR", None)
        shutil.rmtree(self.temp_dir)

    def test_rename_organized_folder_renames_existing_folder(self):
        source_dir = os.path.join(self.temp_dir, "meetings", "2026-04")
        os.makedirs(source_dir, exist_ok=True)
        with open(os.path.join(source_dir, "note.md"), "w", encoding="utf-8") as f:
            f.write("content")

        result = rename_organized_folder.invoke(
            {"current_folder": "meetings/2026-04", "new_folder_name": "2026-04-weekly"}
        )

        self.assertIn("폴더 이름 변경 완료", result)
        self.assertTrue(os.path.isdir(os.path.join(self.temp_dir, "meetings", "2026-04-weekly")))
        self.assertTrue(
            os.path.isfile(os.path.join(self.temp_dir, "meetings", "2026-04-weekly", "note.md"))
        )

    def test_rename_organized_folder_rejects_duplicate_target(self):
        os.makedirs(os.path.join(self.temp_dir, "meetings", "2026-04"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, "meetings", "archive"), exist_ok=True)

        result = rename_organized_folder.invoke(
            {"current_folder": "meetings/2026-04", "new_folder_name": "archive"}
        )

        self.assertIn("같은 이름의 폴더가 이미 있습니다", result)

    def test_rename_organized_folder_rejects_missing_folder(self):
        result = rename_organized_folder.invoke(
            {"current_folder": "ideas/2026-04", "new_folder_name": "renamed"}
        )

        self.assertIn("폴더를 찾을 수 없습니다", result)


if __name__ == "__main__":
    unittest.main()
