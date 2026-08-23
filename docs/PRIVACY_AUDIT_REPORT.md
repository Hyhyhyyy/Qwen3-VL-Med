# Public release privacy audit

Audit date: 2026-08-14

## Release construction

The release was rebuilt in a separate empty Git repository using an explicit source-code allowlist. The private project tree was never staged. Runtime reports, cohort manifests, row-level metrics, predictions, exclusions, images, weights, adapters and deployment metadata were not copied.

## Automated gates

- Blocked clinical/image/model/archive/key file extensions.
- Maximum committed file size: 5 MiB.
- Scanned for private-key headers, common provider tokens, cloud credentials, direct-identifier patterns, workstation user paths, private mount markers and infrastructure endpoints.
- Parsed every committed JSON file.
- Compiled all Python files.
- Ran the dependency-light metric-rule and view-to-case aggregation unit tests.
- Kept adapter-component tests and LLaMA-Factory configuration parsing as additional checks for an isolated environment with the full PyTorch/training dependency stack; they were not rerun in this lightweight public-release environment.
- Regenerated SHA-256 values for every tracked release file except the self-referential manifest.

## Data classification

The only dataset-shaped artifact is `examples/synthetic_dataset.json`. Every row is explicitly labeled synthetic and contains no real image, person, accession, encounter or free-text clinical record.

## Residual risk and approval

Automated scanning cannot replace institutional privacy review. Anyone adding files must rerun `tools/privacy_audit.ps1` and review the staged diff. Real or encrypted clinical artifacts must never be committed.

The 2026-08-14 documentation update publishes only methodology, directional engineering conclusions and explicit evidence limitations. It does not add private-cohort scores, sample-level outputs, images, reports, weights or infrastructure metadata.
