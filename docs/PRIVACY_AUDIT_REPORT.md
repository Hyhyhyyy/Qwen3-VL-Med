# Public release privacy audit

Audit date: 2026-08-11

## Release construction

The release was rebuilt in a separate empty Git repository using an explicit source-code allowlist. The private project tree was never staged. Runtime reports, cohort manifests, row-level metrics, predictions, exclusions, images, weights, adapters and deployment metadata were not copied.

## Automated gates

- Blocked clinical/image/model/archive/key file extensions.
- Maximum committed file size: 5 MiB.
- Scanned for private-key headers, common provider tokens, cloud credentials, direct-identifier patterns, workstation user paths, private mount markers and infrastructure endpoints.
- Parsed every committed JSON file.
- Compiled all Python files.
- Ran metric-rule, view-to-case aggregation and adapter-component unit tests.
- Parsed the four LoRA configurations with LLaMA-Factory and verified the controlled-variable matrix.

## Data classification

The only dataset-shaped artifact is `examples/synthetic_dataset.json`. Every row is explicitly labeled synthetic and contains no real image, person, accession, encounter or free-text clinical record.

## Residual risk and approval

Automated scanning cannot replace institutional privacy review. Anyone adding files must rerun `tools/privacy_audit.ps1` and review the staged diff. Real or encrypted clinical artifacts must never be committed.
