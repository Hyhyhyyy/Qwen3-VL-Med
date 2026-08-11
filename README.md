# Qwen3-VL-Med

A code-only, privacy-screened release for reproducible Qwen3-VL medical-domain fine-tuning research.

## Privacy boundary

This public repository contains **no clinical images, case-level records, reports, predictions, model weights, adapters, internal hostnames, credentials, or institution storage paths**. The included JSON example is synthetic and does not describe a real person.

Clinical data must remain in an access-controlled institutional environment. Do not commit encrypted clinical archives either: public Git history is permanent and future key compromise would expose the archive.

## Included

- Full-SFT configuration templates for cleaned multi-image training, single-view training, case diagnosis, and evidence-to-diagnosis aggregation.
- Four controlled LoRA freezing configurations:
  - R04: vision LoRA + full multimodal projector + language LoRA.
  - R05: frozen vision tower.
  - R06: frozen multimodal projector.
  - R07: frozen language backbone.
- Dataset cleaning and transformation code.
- Case aggregation and evaluation code.
- Adapter-component auditing and paired bootstrap comparison tools.
- A fail-closed privacy audit for files about to be committed.

## Not included

- Hospital datasets or derived row-level artifacts.
- Images, identifiers, free-text reports, exclusions, audit rows, or prediction files.
- Aggregate results derived from the private cohort.
- Model or optimizer weights.
- Machine-specific deployment scripts and runtime manifests.

## Quick start

1. Install Qwen3-VL, LLaMA-Factory, PyTorch, Transformers and the optional packages in `requirements.txt` in an isolated environment.
2. Keep the private dataset outside this repository.
3. Copy a YAML template and replace `/path/to/...` values locally. Never commit the edited private paths.
4. Register the private dataset with LLaMA-Factory inside the protected environment.
5. Run the privacy gate before every commit:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\privacy_audit.ps1
```

6. Launch a configuration with LLaMA-Factory, for example:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train experiments/lora_freezing/lora_01_r04_all_components.yaml
```

The YAML files are templates; their placeholder paths intentionally do not point to real data.

## Research design

See [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md). Data handling and internal encryption rules are in [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md).

## Clinical-use warning

This repository is research software, not a medical device. Reference-based metrics do not establish clinical safety or diagnostic performance. Independent pathology review and institution-specific governance are required before any clinical use.
