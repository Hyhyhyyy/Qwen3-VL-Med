#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from build_joint_dataset import build_split
from render_predictions import parse_object, render, validate


class JointReportTest(unittest.TestCase):
    def test_build_validate_and_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = "synthetic://case-a/image-1"
            cases = [{
                "images": [image],
                "messages": [
                    {"role": "user", "content": "synthetic"},
                    {"role": "assistant", "content": "Synthetic finding.\nDiagnosis:Synthetic class."},
                ],
            }]
            images = [{
                "images": [image],
                "messages": [
                    {"role": "user", "content": "synthetic"},
                    {"role": "assistant", "content": "Synthetic finding."},
                ],
            }]
            case_path, image_path, output_path = root / "cases.json", root / "images.json", root / "joint.json"
            case_path.write_text(json.dumps(cases), encoding="utf-8")
            image_path.write_text(json.dumps(images), encoding="utf-8")
            report = build_split(case_path, image_path, output_path, "Diagnosis:")
            self.assertEqual(report["case_count"], 1)
            target = json.loads(json.loads(output_path.read_text())[0]["messages"][1]["content"])
            self.assertEqual(validate(target, 1), (True, []))
            self.assertEqual(render(target, "Diagnosis:"), "Synthetic finding.\nDiagnosis:Synthetic class.")
            self.assertIsNotNone(parse_object(json.dumps(target)))


if __name__ == "__main__":
    unittest.main()
