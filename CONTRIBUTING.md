# Contributing

This repository accepts focused improvements to reproducibility, privacy gates,
synthetic examples, evaluation code, and documentation. It does not accept clinical
data, case-level outputs, model weights, adapters, checkpoints, private paths, or
credentials—even when encrypted.

## Before opening a pull request

1. Open or claim an issue for changes that affect protocols or public conclusions.
2. Use only synthetic inputs in tests and examples.
3. Run:

```powershell
./tools/privacy_audit.ps1
python tools/privacy_audit_history.py
python tools/privacy_audit_commit_metadata.py
python -m compileall -q code experiments scripts tools
python code/metric_eval/tests/test_metric_rules.py
python code/one_to_one_eval/test_1v1_eval.py
python code/evidence_pipeline/test_build_robust_evidence_mix.py
python code/joint_report/test_joint_report.py
python code/standard_eval/test_archive_interpretability.py
python experiments/lora_freezing/test_audit_adapter_components.py
./tools/update_release_hashes.ps1
git diff --exit-code -- RELEASE_SHA256.txt
```

Install the lightweight test dependencies with
`python -m pip install numpy pillow safetensors`. GPU training dependencies are not
required for these public gates.

## Evidence rules

- Separate synthetic validation, internal experiments, and external clinical evidence.
- Never describe automatic text metrics as clinical accuracy.
- Include the exact command, input boundary, and known limitation for new results.
- Keep each pull request small enough to review independently.

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
