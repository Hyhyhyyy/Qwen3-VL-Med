from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from archive_interpretability import archive_case


class InterpretabilityArchiveTest(unittest.TestCase):
    def test_archives_full_arrays_and_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attention = np.full((2, 4, 6), 0.1, dtype=np.float32)
            gradient = np.arange(12, dtype=np.float32).reshape(2, 6) + 1
            images = [
                {
                    "image_index": 0,
                    "path": "synthetic/image-a.png",
                    "sha256": "0" * 64,
                    "original_width": 200,
                    "original_height": 100,
                    "processed_width": 200,
                    "processed_height": 100,
                    "vision_token_start": 0,
                    "vision_token_end_exclusive": 6,
                    "sequence_start": 10,
                    "token_grid_t": 1,
                    "token_grid_h": 2,
                    "token_grid_w": 3,
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                }
            ]
            manifest = archive_case(
                root,
                run_id="synthetic-run",
                case_uid="synthetic-case",
                comparison_group="synthetic",
                target_span_id="diagnosis_full_v1",
                target_text="synthetic target",
                target_token_ids=np.array([11, 12]),
                target_token_texts=["synthetic", "target"],
                attention_heads=attention,
                abs_gradient=gradient,
                images=images,
                decoder_layer=35,
            )
            case_dir = root / "synthetic-run" / "synthetic-case" / "diagnosis_full_v1"
            with np.load(case_dir / "attribution.float16.npz") as arrays:
                self.assertEqual(arrays["attention_heads"].shape, (2, 4, 6))
                self.assertEqual(arrays["grad_x_attention_heads"].shape, (2, 4, 6))
                self.assertEqual(arrays["visual_token_bbox_original_xyxy"].shape, (6, 4))
                self.assertEqual(arrays["attention_heads"].dtype, np.float16)
            self.assertEqual(manifest["decoder_layer"], 35)
            saved = json.loads((case_dir / "manifest.json").read_text())
            self.assertTrue(saved["qc"]["spatial_mapping_complete"])
            self.assertTrue((case_dir / "image_00_grad_x_attention.png").is_file())
            self.assertTrue((case_dir / "target_000_image_00_grad_x_attention.png").is_file())


if __name__ == "__main__":
    unittest.main()
