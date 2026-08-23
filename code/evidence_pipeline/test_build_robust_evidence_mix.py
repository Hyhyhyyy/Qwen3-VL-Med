from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from build_robust_evidence_mix import build


class RobustEvidenceMixTest(unittest.TestCase):
    def test_builds_balanced_train_only_mix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = [
                {"case_index": 0, "images": ["synthetic/a.png"], "prediction": "finding A", "error": None},
                {"case_index": 1, "images": ["synthetic/b.png"], "prediction": "finding B", "error": None},
            ]
            cases = [
                {
                    "case_uid": "synthetic-train-0",
                    "split": "train",
                    "oracle_sample_index": 0,
                    "evidence_sample_indices": [0, 1],
                    "image_paths": ["synthetic/a.png", "synthetic/b.png"],
                },
                {
                    "case_uid": "synthetic-test-0",
                    "split": "test",
                    "oracle_sample_index": 99,
                    "evidence_sample_indices": [99],
                    "image_paths": ["synthetic/test.png"],
                },
            ]
            oracle = [{
                "messages": [
                    {"role": "user", "content": "Evidence 1: reference finding"},
                    {"role": "assistant", "content": "Synthetic diagnosis"},
                ]
            }]
            for name, value in [("predictions.json", predictions), ("cases.json", cases), ("oracle.json", oracle)]:
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            args = argparse.Namespace(
                predictions=root / "predictions.json",
                case_manifest=root / "cases.json",
                oracle_train=root / "oracle.json",
                output_dir=root / "output",
                output_name="mix.json",
                seed=7,
                expected_cases=1,
                train_split="train",
                split_key="split",
                case_id_key="case_uid",
                oracle_index_key="oracle_sample_index",
                evidence_indices_key="evidence_sample_indices",
                image_paths_key="image_paths",
                prediction_index_key="case_index",
                prediction_text_key="prediction",
                prompt_prefix="Evidence from the same case:\n",
                evidence_label="Evidence ",
            )
            manifest = build(args)
            self.assertEqual(manifest["total_records"], 2)
            self.assertEqual(manifest["non_train_records_used"], 0)
            mixed = json.loads((root / "output" / "mix.json").read_text())
            self.assertEqual(len(mixed), 2)
            self.assertTrue(all("Synthetic diagnosis" in row["messages"][1]["content"] for row in mixed))


if __name__ == "__main__":
    unittest.main()
