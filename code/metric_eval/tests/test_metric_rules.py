#!/usr/bin/env python3

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compute_metrics", ROOT / "compute_metrics.py")
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def main() -> None:
    reference = (
        "左上图：肝穿组织共见7个中小汇管区，小叶结构可辨\n"
        "右上图：汇管区中度炎性细胞浸润，可见淋巴细胞聚集灶\n"
        "左下图：环绕并攻击小胆管，可见上皮样肉芽肿\n"
        "右下图：未见明显界面炎，未见细胆管反应，未见纤维组织增生\n"
        "病理诊断：原发性胆汁性胆管炎 I期"
    )
    good = reference
    bad = reference.replace("未见明显界面炎", "可见明显界面炎").replace("I期", "III期")

    assert module.canonical_diagnosis(reference) == "原发性胆汁性胆管炎"
    assert module.extract_stage(reference) == 1
    assert module.extract_stage(bad) == 3
    assert module.extract_entities(reference)["interface_hepatitis"]["status"] == "absent"
    assert module.extract_entities(bad)["interface_hepatitis"]["status"] == "present"
    assert module.contradiction_stats(reference, good)[0] == 0
    assert module.contradiction_stats(reference, bad)[0] >= 1
    assert module.set_prf(module.fact_set(reference), module.fact_set(good)) == (1.0, 1.0, 1.0)
    assert module.rouge_l(module.tokenize(reference), module.tokenize(good))[2] == 1.0
    assert module.meteor_exact(module.tokenize(reference), module.tokenize(good)) > 0.99
    print("metric rule tests passed")


if __name__ == "__main__":
    main()
