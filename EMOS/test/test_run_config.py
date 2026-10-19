"""Run provenance is explicit, redacted, and based on source bytes."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))

from habitat_mas.utils.run_config import build_run_metadata


class RunConfigTests(unittest.TestCase):
    def config(self):
        return {
            "benchmark": "declared-benchmark", "habitat": {
                "seed": 27,
                "dataset": {"type": "RearrangeDataset-v0", "split": "val",
                            "mode": "manipulation", "should_terminate_on_wait": False},
                "simulator": {"seed": 28, "agents_order": ["robot_b", "robot_a"],
                              "agents": {"robot_a": {"articulated_agent_type": "FetchRobot"},
                                         "robot_b": {"articulated_agent_type": "SpotRobot"}}},
                "environment": {"max_episode_steps": 12, "max_episode_seconds": 30},
                "task": {"success_measure": "existing_measure", "end_on_success": False,
                         "constraint_violation_ends_episode": True,
                         "measurements": {"existing_measure": {"type": "ExistingSuccess", "threshold": 0.2}}},
            },
        }

    def test_declared_identity_and_original_success_definition(self):
        metadata = build_run_metadata(self.config(), 7, 12)
        self.assertEqual(metadata["benchmark"], "declared-benchmark")
        self.assertEqual(metadata["task_category"], "manipulation")
        self.assertEqual(metadata["agents_order"], ["robot_b", "robot_a"])
        self.assertEqual([agent["robot_type"] for agent in metadata["agents"]],
                         ["SpotRobot", "FetchRobot"])
        self.assertEqual(metadata["agent_count"], 2)
        self.assertEqual(metadata["success_measure"], "existing_measure")
        self.assertEqual(metadata["success_definition"]["measure_config"],
                         {"type": "ExistingSuccess", "threshold": 0.2})
        self.assertFalse(metadata["termination"]["end_on_success"])
        self.assertTrue(metadata["termination"]["task_conditions"]["constraint_violation_ends_episode"])

    def test_redacts_nested_config_credentials_and_urls_without_reading_environment(self):
        config = self.config()
        config["service"] = {"api_key": "credential-marker", "nested": [
            {"password": "password-marker", "endpoint": "https://user:pass@example.test/api?secret=value"},
        ]}
        with patch.dict("os.environ", {"OPENAI_API_KEY": "environment-marker"}):
            metadata = build_run_metadata(config)
        encoded = json.dumps(metadata)
        for secret in ("credential-marker", "password-marker", "user:pass", "secret=value", "environment-marker"):
            self.assertNotIn(secret, encoded)
        self.assertEqual(metadata["config_snapshot"]["service"]["api_key"], "[REDACTED]")

    def test_omegaconf_snapshot_preserves_unresolved_values(self):
        from omegaconf import OmegaConf
        config = OmegaConf.create({"habitat": {"seed": "${unregistered:never-evaluate}",
                                               "dataset": {"mode": "???"}},
                                   "api_key": "${oc.env:UNSET_SECRET}"})
        metadata = build_run_metadata(config)
        self.assertEqual(metadata["config_snapshot"]["habitat"]["seed"], "${unregistered:never-evaluate}")
        self.assertEqual(metadata["config_snapshot"]["api_key"], "[REDACTED]")
        self.assertIsNone(metadata["seed"])
        self.assertIsNone(metadata["task_category"])
        self.assertIn("seed", metadata["missing_fields"])

    def test_unknown_settings_are_null_with_reasons(self):
        metadata = build_run_metadata(SimpleNamespace(habitat=SimpleNamespace()))
        for field in ("benchmark", "dataset", "task_category", "config_path", "seed", "success_measure"):
            self.assertIsNone(metadata[field])
            self.assertIn(field, metadata["missing_fields"])
        for module, model in metadata["models"].items():
            self.assertIsNone(model["model"])
            self.assertIsNone(model["decoding"])
            self.assertIn(f"models.{module}.model", metadata["missing_fields"])
            self.assertIn(f"models.{module}.decoding", metadata["missing_fields"])

    def test_explicit_model_settings_are_separate_from_runtime_defaults(self):
        config = self.config()
        config["habitat_baselines"] = {"eval": {"models": {
            "execution": {"model": "declared-model", "temperature": 0.4, "max_tokens": 100},
        }}}
        metadata = build_run_metadata(config)
        self.assertEqual(metadata["models"]["execution"]["model"], "declared-model")
        self.assertEqual(metadata["models"]["execution"]["decoding"],
                         {"temperature": 0.4, "max_tokens": 100})
        self.assertIsNone(metadata["models"]["group_discussion"]["model"])

    def test_hydra_reports_local_config_path_and_hashes_selected_sources(self):
        from hydra.core.hydra_config import HydraConfig
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_root = root / "configs"
            (config_root / "benchmark/multi_agent").mkdir(parents=True)
            main = config_root / "main.yaml"
            selected = config_root / "benchmark/multi_agent/declared.yaml"
            main.write_text("defaults: []\n", encoding="utf-8")
            selected.write_text("habitat: {}\n", encoding="utf-8")
            hydra = {"job": {"config_name": "main"}, "runtime": {
                "choices": {"benchmark/multi_agent": "declared"},
                "config_sources": [{"schema": "file", "path": str(config_root)}],
            }}
            with patch.object(HydraConfig, "initialized", return_value=True), \
                    patch.object(HydraConfig, "get", return_value=hydra):
                metadata = build_run_metadata({}, repository_root=root)
            self.assertEqual(metadata["config_path"], str(main))
            self.assertEqual(metadata["benchmark"], "declared")
            self.assertEqual(metadata["version"]["config_files"], {
                "configs/main.yaml": hashlib.sha256(main.read_bytes()).hexdigest(),
                "configs/benchmark/multi_agent/declared.yaml": hashlib.sha256(selected.read_bytes()).hexdigest(),
            })

    def test_actual_source_hash_and_snapshot_hash_are_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "habitat-mas/habitat_mas/utils/models.py"
            source.parent.mkdir(parents=True)
            source.write_text("version = 1\n", encoding="utf-8")
            first = build_run_metadata(self.config(), repository_root=root)
            second = build_run_metadata(self.config(), repository_root=root)
            key = str(source.relative_to(root))
            self.assertEqual(first["version"]["source_files"][key],
                             hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(first["version"], second["version"])
            self.assertIsNone(first["version"]["git_head"])
            self.assertIn("version.git_head", first["missing_fields"])
            source.write_text("version = 2\n", encoding="utf-8")
            changed = build_run_metadata(self.config(), repository_root=root)
            self.assertNotEqual(first["version"]["source_files"][key],
                                changed["version"]["source_files"][key])
            self.assertEqual(first["version"]["config_snapshot_sha256"],
                             changed["version"]["config_snapshot_sha256"])


if __name__ == "__main__":
    unittest.main()
