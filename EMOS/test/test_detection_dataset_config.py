"""Check the four-task experiment scope without requiring ignored traces."""

import json
from pathlib import Path
import unittest


CONFIG = Path(__file__).resolve().parents[1] / "configs/detection/emos.json"


class DetectionDatasetConfigTest(unittest.TestCase):
    def datasets(self):
        self.assertTrue(CONFIG.is_file(), "EMOS detector dataset config missing")
        return json.loads(CONFIG.read_text(encoding="utf-8"))["datasets"]

    def test_all_four_user_selected_tasks_have_complete_episode_scope(self):
        datasets = self.datasets()
        self.assertEqual(len(datasets), 4)
        counts = {row["task"]: row["expected_episodes"] for row in datasets}
        self.assertEqual(counts, {
            "rearrange": 49,
            "mobility": 99,
            "perception": 103,
            "manipulation": 139,
        })
        self.assertEqual(sum(counts.values()), 390)

    def test_manifest_paths_resolve_from_repository_and_do_not_overlap(self):
        datasets = self.datasets()
        identifiers = [row["dataset"] for row in datasets]
        manifests = [row["manifest"] for row in datasets]
        self.assertEqual(len(set(identifiers)), len(datasets))
        self.assertEqual(len(set(manifests)), len(datasets))
        for row in datasets:
            with self.subTest(dataset=row["dataset"]):
                path = Path(row["manifest"])
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)
                self.assertEqual(path.parts, (
                    "analysis_results", row["dataset"], "manual_review",
                    "manifest.json",
                ))


if __name__ == "__main__":
    unittest.main()
