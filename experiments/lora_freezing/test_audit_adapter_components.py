#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "audit_adapter_components.py"


class AdapterComponentAuditTest(unittest.TestCase):
    def run_audit(self, run_id: str, keys: list[str]) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir)
            tensors = {key: torch.zeros(1) for key in keys}
            save_file(tensors, adapter_dir / "adapter_model.safetensors")
            return subprocess.run(
                [sys.executable, str(SCRIPT), "--run-id", run_id, "--adapter-dir", str(adapter_dir)],
                text=True,
                capture_output=True,
                check=False,
            )

    def test_expected_component_matrix(self) -> None:
        vision = "base_model.model.visual.blocks.0.attn.qkv.lora_A.default.weight"
        projector = "base_model.model.visual.merger.mlp.0.modules_to_save.default.weight"
        language = "base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight"
        cases = {
            "R04": [vision, projector, language],
            "R05": [projector, language],
            "R06": [vision, language],
            "R07": [vision, projector],
        }
        for run_id, keys in cases.items():
            with self.subTest(run_id=run_id):
                result = self.run_audit(run_id, keys)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertTrue(json.loads(result.stdout)["checks_passed"])

    def test_mismatch_fails(self) -> None:
        result = self.run_audit(
            "R05",
            ["base_model.model.visual.blocks.0.attn.qkv.lora_A.default.weight"],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(json.loads(result.stdout)["checks_passed"])


if __name__ == "__main__":
    unittest.main()
