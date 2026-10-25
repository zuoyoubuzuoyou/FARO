"""Fixed-cohort reporting checks; no simulator or API dependencies."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.episode_stability_report import summarize, write_reports


def manifest_for(rows):
    cohort = [dict(task="Manipulation", scene_id="scene", episode_id=str(i),
                   dataset_file="episodes.json.gz") for i in range(len(rows))]
    trials = []
    for identity, outcomes in zip(cohort, rows):
        for round_number, outcome in enumerate(outcomes, 1):
            trials.append(dict(identity, round=round_number,
                               trial_id=f"{identity['episode_id']}-{round_number}",
                               trial_dir="unused", status="completed",
                               task_success=outcome))
    return dict(settings={"repetitions": 3}, cohort=cohort, trials=trials,
                source_hashes={}, input_hashes={}, status="completed")


class EpisodeStabilityReportTests(unittest.TestCase):
    def test_native_outcomes_and_directional_pairwise_denominators(self):
        summary = summarize(manifest_for([
            [True, True, True], [False, False, False], [True, False, True],
            [False, True, False], [True, False, False],
        ]))
        self.assertEqual([r["success"] for r in summary["rounds"]], [3, 2, 2])
        self.assertEqual([r["failure"] for r in summary["rounds"]], [2, 3, 3])
        self.assertEqual([r["success_rate"] for r in summary["rounds"]], [.6, .4, .4])
        self.assertAlmostEqual(summary["success_rate_range_pp"], 20)
        pair = summary["pairwise"][0]
        self.assertEqual((pair["round_a"], pair["round_b"], pair["common_valid"]), (1, 2, 5))
        self.assertEqual((pair["flips"], pair["success_to_failure"], pair["failure_to_success"]), (3, 2, 1))
        self.assertEqual((pair["flip_rate"], pair["success_to_failure_rate"], pair["failure_to_success_rate"]), (.6, .4, .2))
        self.assertEqual([p["flips"] for p in summary["pairwise"]], [3, 1, 2])
        self.assertEqual(summary["any_change"], {"common_valid": 5, "changed": 3, "rate": .6})
        self.assertEqual([t["outcomes"] for t in summary["task_outcomes"]],
                         [["S", "S", "S"], ["F", "F", "F"], ["S", "F", "S"],
                          ["F", "S", "F"], ["S", "F", "F"]])

    def test_errors_missing_and_pending_never_become_native_failures(self):
        manifest = manifest_for([[False, True, False], [True, False, True]])
        manifest["trials"][1].update(status="api_error", task_success=False)
        manifest["trials"][4].update(status="pending", task_success=None)
        manifest["trials"].pop(5)
        result = summarize(manifest)
        self.assertEqual([r["completed"] for r in result["rounds"]], [2, 0, 1])
        self.assertEqual([r["failure"] for r in result["rounds"]], [1, 0, 1])
        self.assertEqual([r["success_rate"] for r in result["rounds"]], [.5, None, None])
        self.assertEqual(result["rounds"][1]["status_counts"], {"api_error": 1, "pending": 1})
        self.assertEqual(result["rounds"][2]["status_counts"], {"completed": 1, "missing": 1})
        self.assertEqual(result["rounds"][2]["completion_coverage"], .5)
        self.assertIsNone(result["success_rate_range_pp"])
        self.assertEqual(result["pairwise"][0]["common_valid"], 0)
        self.assertIsNone(result["pairwise"][0]["flip_rate"])
        self.assertEqual(result["pairwise"][1]["common_valid"], 1)
        self.assertEqual(result["any_change"], {"common_valid": 0, "changed": 0, "rate": None})
        self.assertEqual(result["task_outcomes"][0]["outcomes"], ["F", "api_error", "F"])
        self.assertEqual(result["task_outcomes"][1]["outcomes"], ["S", "pending", "missing"])

    def test_completed_requires_real_boolean(self):
        for invalid in (None, 0, 1, "false", "True"):
            with self.subTest(value=invalid):
                manifest = manifest_for([[True, True, True]])
                manifest["trials"][0]["task_success"] = invalid
                result = summarize(manifest)
                self.assertEqual(result["rounds"][0]["completed"], 0)
                self.assertIsNone(result["rounds"][0]["success_rate"])
                self.assertEqual(result["task_outcomes"][0]["outcomes"][0], "invalid_completed")

    def test_non_completed_status_cannot_spoof_a_native_outcome(self):
        for status in ("S", "F", "unknown"):
            manifest = manifest_for([[True, True, True]])
            manifest["trials"][0]["status"] = status
            with self.subTest(status=status):
                with self.assertRaises(ValueError):
                    summarize(manifest)

    def test_reject_duplicate_foreign_and_invalid_round_identities(self):
        original = manifest_for([[True, False, True]])
        variants = []
        duplicate_trial = copy.deepcopy(original)
        duplicate_trial["trials"].append(copy.deepcopy(duplicate_trial["trials"][0]))
        variants.append(duplicate_trial)
        duplicate_cohort = copy.deepcopy(original)
        duplicate_cohort["cohort"].append(copy.deepcopy(duplicate_cohort["cohort"][0]))
        variants.append(duplicate_cohort)
        for field in ("task", "scene_id", "episode_id"):
            foreign = copy.deepcopy(original)
            foreign["trials"][0][field] = "foreign"
            variants.append(foreign)
        for invalid in (0, 4, True, "1", 1.0):
            invalid_round = copy.deepcopy(original)
            invalid_round["trials"][0]["round"] = invalid
            variants.append(invalid_round)
        for variant in variants:
            with self.subTest(manifest=variant):
                with self.assertRaises(ValueError):
                    summarize(variant)

    def test_initial_state_agreement_requires_all_round_hashes(self):
        manifest = manifest_for([[True] * 3, [False] * 3, [True] * 3])
        for trial, digest in zip(manifest["trials"], ["a", "a", "a", "b", "b", None, "c", "d", "c"]):
            if digest:
                trial["initial_state_sha256"] = digest
        result = summarize(manifest)
        self.assertEqual([r["initial_state_agreement"] for r in result["task_outcomes"]], [True, None, False])
        self.assertIs(result["initial_state_agreement"], False)
        del manifest["trials"][7]["initial_state_sha256"]
        self.assertIsNone(summarize(manifest)["initial_state_agreement"])
        for trial in manifest["trials"]:
            trial["initial_state_sha256"] = "same"
        self.assertIs(summarize(manifest)["initial_state_agreement"], True)

    def test_writes_json_and_bom_csv_reports_without_mutating_manifest(self):
        manifest = manifest_for([[True, False, True]])
        original = copy.deepcopy(manifest)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reports"
            result = write_reports(manifest, output)
            self.assertEqual(json.loads((output / "summary.json").read_text(encoding="utf-8")), result)
            for filename in ("rounds.csv", "task_outcomes.csv"):
                self.assertTrue((output / filename).read_bytes().startswith(b"\xef\xbb\xbf"))
            with (output / "rounds.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["success"] for r in rows], ["1", "0", "1"])
            with (output / "task_outcomes.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([rows[0][f"round_{i}"] for i in (1, 2, 3)], ["S", "F", "S"])
        self.assertEqual(manifest, original)


if __name__ == "__main__":
    unittest.main()
