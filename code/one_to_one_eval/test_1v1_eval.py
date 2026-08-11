from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
METRIC_ROOT = HERE.parent if (HERE.parent / "compute_metrics.py").is_file() else HERE.parent / "metric_eval"


class EvaluationPipelineTest(unittest.TestCase):
    def test_view_metrics_and_leakage_safe_case_builder(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            audit = [
                {"split": "test", "one_to_one_index": 0, "clean_case_index": 10, "source_index": 100, "image": "a.jpg", "case_level_reference_diagnosis_not_used_as_target": "慢性肝炎"},
                {"split": "test", "one_to_one_index": 1, "clean_case_index": 10, "source_index": 100, "image": "b.jpg", "case_level_reference_diagnosis_not_used_as_target": "慢性肝炎"},
                {"split": "test", "one_to_one_index": 2, "clean_case_index": 11, "source_index": 101, "image": "c.jpg", "case_level_reference_diagnosis_not_used_as_target": "脂肪性肝病"},
            ]
            predictions = [
                {"case_index": 0, "images": ["a.jpg"], "reference": "图像分析：汇管区炎症", "prediction": "图像分析：汇管区炎症", "error": None, "hit_max_new_tokens": False},
                {"case_index": 1, "images": ["b.jpg"], "reference": "图像分析：胆管损伤", "prediction": "图像分析：胆管损伤", "error": None, "hit_max_new_tokens": False},
                {"case_index": 2, "images": ["c.jpg"], "reference": "图像分析：肝细胞脂肪变", "prediction": "图像分析：肝细胞脂肪变", "error": None, "hit_max_new_tokens": False},
            ]
            audit_path, pred_path = tmp / "audit.jsonl", tmp / "predictions.jsonl"
            audit_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in audit), encoding="utf-8")
            pred_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in predictions), encoding="utf-8")
            out = tmp / "metrics"
            subprocess.run([
                sys.executable, str(HERE / "compute_1v1_metrics.py"), "--predictions", str(pred_path),
                "--audit", str(audit_path), "--metric-root", str(METRIC_ROOT), "--output-dir", str(out),
                "--bootstrap-samples", "20",
            ], check=True)
            metrics = json.loads((out / "metrics_1v1.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["metadata"]["view_count"], 3)
            self.assertEqual(metrics["metadata"]["case_count"], 2)
            self.assertEqual(metrics["structure"]["position_token_rate"], 0.0)
            self.assertEqual(metrics["structure"]["case_diagnosis_phrase_rate"], 0.0)

            case_path = tmp / "cases.json"
            subprocess.run([
                sys.executable, str(HERE / "build_case_aggregation_inputs.py"), "--predictions", str(pred_path),
                "--audit", str(audit_path), "--output", str(case_path),
            ], check=True)
            cases = json.loads(case_path.read_text(encoding="utf-8"))
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0]["view_indices"], [0, 1])
            user_prompt = cases[0]["messages"][0]["content"]
            assistant_reference = cases[0]["messages"][1]["content"]
            self.assertNotIn("慢性肝炎", user_prompt)
            self.assertEqual(assistant_reference, "病理诊断：慢性肝炎")


if __name__ == "__main__":
    unittest.main()
