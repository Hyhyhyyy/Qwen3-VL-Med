# Roadmap

## Current: reproducible research package

- Keep privacy, history, syntax, synthetic tests, and release hashes green.
- Make the fixed synthetic example runnable without a GPU or private dataset.
- Document which conclusions are supported by public artifacts and which are not.

## Next: independent reproduction

- Add a single command that exercises data validation, rendering, and metric checks.
- Publish environment lock guidance for CPU-only validation and GPU training separately.
- Invite two independent reproductions of the public synthetic workflow.

## Later: external evaluation protocol

- Add institution-independent evaluation templates without bundling clinical data.
- Version the metric schema and publish migration notes for breaking changes.
- Accept third-party adapters only as code/configuration contributions; weights remain out
  of scope.

## Stable success criteria

- A new contributor can run all public gates from a clean clone.
- Every reported number links to a protocol, aggregation boundary, and limitation.
- At least two independent users reproduce the synthetic workflow.
- No protected data or weight artifact enters the reachable Git history.
