"""Independent episode-level detector scoring and human label workflows."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "habitat-mas/habitat_mas/detection/evaluation.py"
)


def load_evaluation():
    assert MODULE_PATH.is_file(), "Episode evaluation component is missing"
    spec = importlib.util.spec_from_file_location("detection_evaluation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(identifier, label, dataset="Mobility", scene="scene", success=False):
    return {
        "dataset": dataset,
        "scene_id": scene,
        "episode_id": str(identifier),
        "runtime_system_error": label,
        "reviewer": "人工核查者" if label in (0, 1) else "",
        "native_success": success,
    }


def prediction(identifier, detected, method="rules", **extra):
    row = {
        "dataset": "Mobility",
        "scene_id": "scene",
        "episode_id": str(identifier),
        "method": method,
        "detected": detected,
        "complete": True,
    }
    row.update(extra)
    return row


class EpisodeEvaluationTest(unittest.TestCase):
    def test_user_example_counts_episodes_and_scores_precision_recall(self):
        # Reversing the FP/FN branches would reverse these hand-computed metrics.
        module = load_evaluation()
        refs = [reference(i, int(i < 25)) for i in range(40)]
        preds = [prediction(i, i < 15 or 25 <= i < 30) for i in range(40)]
        result = module.score_episodes(refs, preds)["methods"]["rules"]
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (15, 5, 10, 10))
        self.assertEqual(result["precision"], 0.75)
        self.assertEqual(result["recall"], 0.6)
        self.assertEqual(result["status"], "complete")

    def test_different_methods_do_not_share_predictions(self):
        module = load_evaluation()
        result = module.score_episodes(
            [reference("p", 1), reference("n", 0)],
            [prediction("p", True), prediction("n", True),
             prediction("p", False, "llm"), prediction("n", False, "llm")],
        )["methods"]
        self.assertEqual(result["rules"]["precision"], 0.5)
        self.assertEqual(result["rules"]["recall"], 1.0)
        self.assertIsNone(result["llm"]["precision"])
        self.assertEqual(result["llm"]["recall"], 0.0)

    def test_unknown_references_never_become_negatives_or_formal_scores(self):
        module = load_evaluation()
        for unknown in (None, "", "unknown"):
            with self.subTest(label=unknown):
                result = module.score_episodes(
                    [reference(1, 1), reference(2, unknown)],
                    [prediction(1, False), prediction(2, True)],
                )["methods"]["rules"]
                self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (0, 0, 1, 0))
                self.assertEqual(result["status"], "awaiting_manual_labels")
                self.assertEqual(result["coverage"]["unknown_labels"], 1)
                self.assertIsNone(result["precision"])
                self.assertIsNone(result["recall"])

    def test_success_episode_may_contain_anomaly_and_failed_only_is_explicit(self):
        module = load_evaluation()
        refs = [reference(1, 1, success=True), reference(2, 0, success=False)]
        preds = [prediction(1, True), prediction(2, False)]
        self.assertEqual(module.score_episodes(refs, preds)["methods"]["rules"]["tp"], 1)
        failed = module.score_episodes(refs, preds, scope="failed_episodes_only")
        self.assertEqual(failed["reference_counts"]["episodes"], 1)
        self.assertEqual(failed["methods"]["rules"]["tn"], 1)
        self.assertEqual(failed["methods"]["rules"]["tp"], 0)

    def test_missing_error_and_uncertain_predictions_cannot_be_false_negatives(self):
        module = load_evaluation()
        refs = [reference(i, 1) for i in range(4)]
        preds = [prediction(0, True),
                 prediction(1, False, complete=False, status="error", error="timeout"),
                 prediction(2, None, status="uncertain")]
        result = module.score_episodes(refs, preds)["methods"]["rules"]
        self.assertEqual((result["tp"], result["fn"]), (1, 0))
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["coverage"]["scored"], 1)
        self.assertEqual(result["coverage"]["incomplete_predictions"], 2)
        self.assertEqual(result["coverage"]["missing_predictions"], 1)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])

    def test_zero_denominators_are_null_instead_of_perfect(self):
        module = load_evaluation()
        result = module.score_episodes([reference(0, 0)], [prediction(0, False)])["methods"]["rules"]
        self.assertEqual(result["tn"], 1)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])

    def test_completed_uncertain_queries_are_negative_and_can_be_missed_errors(self):
        module = load_evaluation()
        result = module.score_episodes(
            [reference(1, 1)],
            [prediction(1, False, status="completed", uncertain_queries=3)],
        )["methods"]["rules"]
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["recall"], 0.0)

    def test_annotation_generation_never_overwrites_human_work(self):
        module = load_evaluation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            original = b"existing human answers"
            path.write_bytes(original)
            with self.assertRaises(FileExistsError):
                module.write_annotation_template(path, [reference(1, "")])
            self.assertEqual(path.read_bytes(), original)

    def test_identity_includes_dataset_and_scene(self):
        module = load_evaluation()
        refs = [reference(1, 1), reference(1, 0, scene="other"), reference(1, 0, dataset="Perception")]
        preds = [prediction(1, True), prediction(1, False, scene_id="other"), prediction(1, True, dataset="Perception")]
        result = module.score_episodes(refs, preds)["methods"]["rules"]
        self.assertEqual((result["tp"], result["fp"], result["tn"]), (1, 1, 1))

    def test_multiple_alarms_within_an_episode_count_once(self):
        module = load_evaluation()
        result = module.score_episodes(
            [reference(1, 1)],
            [prediction(1, True, alarms=[{"step": 10}, {"step": 20}])],
        )["methods"]["rules"]
        self.assertEqual(result["tp"], 1)

    def test_duplicate_rows_and_unknown_prediction_identity_are_rejected(self):
        module = load_evaluation()
        ref = reference(1, 1)
        pred = prediction(1, True)
        for refs, preds in (([ref, ref], [pred]), ([ref], [pred, pred]), ([ref], [prediction(2, False)])):
            with self.subTest(refs=refs, preds=preds):
                with self.assertRaises(ValueError):
                    module.score_episodes(refs, preds)

    def test_no_human_labels_never_produces_a_score(self):
        module = load_evaluation()
        result = module.score_episodes([reference(1, "")], [prediction(1, True)])["methods"]["rules"]
        self.assertEqual(result["status"], "awaiting_manual_labels")
        self.assertEqual(result["coverage"]["scored"], 0)
        self.assertIsNone(result["precision"])

    def test_labels_need_human_reviewer_and_valid_intervals(self):
        module = load_evaluation()
        for patch in ({"reviewer": ""}, {"runtime_system_error": "model_error"},
                      {"onset_step": -1}, {"onset_step": 10, "offset_step": 5}):
            with self.subTest(patch=patch):
                ref = reference(1, 1)
                ref.update(patch)
                with self.assertRaises(ValueError):
                    module.score_episodes([ref], [prediction(1, True)])

    def test_template_has_bom_and_does_not_import_existing_causal_labels(self):
        module = load_evaluation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            episodes = [dict(reference(1, 1), reason="旧模型归因", report_path="episodes/1.md"),
                        reference(2, 0, success=True)]
            module.write_annotation_template(path, episodes)
            self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
            loaded = module.load_annotations(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["runtime_system_error"], None)
            self.assertEqual(loaded[1]["runtime_system_error"], None)
            self.assertEqual(loaded[0]["reviewer"], "")
            self.assertEqual(loaded[0]["reason"], "")
            self.assertEqual(loaded[0]["report_path"], "episodes/1.md")
            self.assertIs(loaded[1]["native_success"], True)

    def test_load_annotations_validates_and_preserves_chinese(self):
        module = load_evaluation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            module.export_csv(path, [dict(reference(1, 1), onset_step="12", offset_step="20", reason="全队无法推进")])
            row = module.load_annotations(path)[0]
            self.assertEqual((row["onset_step"], row["offset_step"]), (12, 20))
            self.assertEqual(row["reason"], "全队无法推进")
            self.assertEqual(row["runtime_system_error"], 1)
            module.export_csv(path, [reference(1, 1), reference(1, 0)])
            with self.assertRaises(ValueError):
                module.load_annotations(path)

    def test_failed_only_requires_explicit_native_outcome(self):
        module = load_evaluation()
        ref = reference(1, 0)
        del ref["native_success"]
        with self.assertRaises(ValueError):
            module.score_episodes([ref], [prediction(1, False)], scope="failed_episodes_only")


if __name__ == "__main__":
    unittest.main()
