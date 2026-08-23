#!/usr/bin/env python3
"""Validate and merge rank-specific JSONL prediction shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    records = json.load(open(cfg["test_json"], "r", encoding="utf-8"))
    output_dir = Path(cfg["output_dir"])

    latest: dict[int, dict] = {}
    for rank in range(int(cfg.get("world_size", 4))):
        path = output_dir / f"predictions.part{rank}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing prediction shard: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                try:
                    row = json.loads(line)
                    latest[int(row["case_index"])] = row
                except Exception as exc:
                    raise SystemExit(f"invalid JSON at {path}:{line_no}: {exc}") from exc

    missing = sorted(set(range(len(records))) - set(latest))
    errors = [(idx, latest[idx].get("error")) for idx in sorted(latest) if latest[idx].get("error")]
    empty = [idx for idx in sorted(latest) if not str(latest[idx].get("prediction") or "").strip()]
    if missing or errors or empty:
        raise SystemExit(
            f"prediction set incomplete: missing={missing[:20]} errors={errors[:20]} empty={empty[:20]}"
        )

    final_path = output_dir / "predictions.jsonl"
    with final_path.open("w", encoding="utf-8") as out:
        for idx in range(len(records)):
            out.write(json.dumps(latest[idx], ensure_ascii=False) + "\n")
    print(f"merged {len(records)} predictions into {final_path}")


if __name__ == "__main__":
    main()
