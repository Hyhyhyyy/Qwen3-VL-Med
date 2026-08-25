# Joint structured report pipeline

This module reproduces the public, data-agnostic part of the R16/R18 target construction:

1. align one case-level report with ordered image-level evidence;
2. build one JSON target containing an overall diagnosis, per-image findings and cross-image evidence;
3. reject duplicate/missing images and train/test image overlap;
4. validate model JSON and deterministically render it into the common report format before text evaluation.

All input and output paths are mandatory CLI arguments. Run it only inside the approved private environment. Never copy the generated JSON, audit manifest, predictions or images into this repository.

The builder checks structural alignment, not clinical truth. Negative evidence and uncertainty must come from approved annotations or an audited deterministic mapping; they must not be invented from absence of a phrase.
