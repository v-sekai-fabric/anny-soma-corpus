---
license: apache-2.0
tags:
  - keypoints
  - motion
  - anny
  - soma
  - kimodo
  - rfd-1173
  - rfd-2203
configs:
  # HERO fills one config_name per rendered subset at publish time. The subset directory
  # naming convention is <subset-index>-<motion-source-summary>-<n-views>view (e.g.
  # subset-01-walks-crouches-getups-8view/*.parquet), and each subset ships as its own
  # commit set per RFD 2203's living-dataset doctrine. Two configs per subset — one
  # `train.parquet` and one `val.parquet` per the split_config field in the shard's
  # manifest — so the HF viewer surfaces the split shape without additional URL routing.
  # A subset shipped without a matching config_name entry renders as nothing per RFD 2196
  # rule 5 (the subdirectory prefix is what makes the viewer render).
  - config_name: subset-01-train
    data_files: "subset-01-*/train.parquet"
  - config_name: subset-01-val
    data_files: "subset-01-*/val.parquet"
---

# anny-soma-corpus

The ANNY-SOMA keypoint corpus. Task #67 render side of the RFD 1173 keypoints stub. Living
dataset: first subset lands, motions can be refined in place, new subsets add on top, each
subset a separate commit set with the prior hash cited.

## Row shape

Hybrid per [RFD 2203](https://github.com/weftspun/request-for-discussion/tree/main/rfd/2203-anny-soma-first-subset-corpus-shape).

| column | shape | notes |
| --- | --- | --- |
| `image` | `struct<bytes, path>` | rendered frame per RFD 2196 |
| `camera` | `float32[4,4]` | extrinsics × intrinsics |
| `anny_posed_vertices` | `float32[19158, 3]` | makehuman topology, indexed by `wholebody133.pth` |
| `keypoints_2d` | `float32[133, 3]` | `(x, y, visible)` baked at the `.pth` hash |
| `soma_pose` | `float32[77, 3]` | axis-angle rotvecs as Kimodo emits |

Per-shard manifest carries `wholebody133.pth` SHA-256, observed SOMA joint count, motion
source, sampler configuration, render seed, and the co-tenancy dump at kick. Training
consumers regress fresh 2D keypoints from `anny_posed_vertices` via
`KeypointsRegressor.load_precomputed(wholebody133.pth)` when they want the latest anchors,
or use the baked `keypoints_2d` column as the immutable comparison baseline. The dataset
viewer shows the baked column.

## Subsets

Each subset directory follows `subset-<index>-<motion-source-summary>-<n-views>view/`, and
contains `train.parquet`, `val.parquet`, and `manifest.json`. The `configs:` block above
routes the HF viewer at each subset's shards; a shard without a matching `config_name`
entry does not surface.

The living-dataset shape means the corpus grows in two ways:

- **Additive subsets** add new motion sources, camera policies, lighting conditions,
  blendshape sweeps, or environmental context. Each is its own commit set.
- **Motion refinements** re-render an earlier motion source with better sampler
  configuration; the refined shards ship as a new commit set with the prior shard's hash
  cited in the manifest as retraction-in-place.

Each subset's manifest records what it was rendered against, so a training run that spans
subsets can cite the exact snapshot per RFD 2203.

## Publish path

`hf upload-large-folder` for per-file resumable commits. `HF_HUB_DISABLE_XET=1
HF_HUB_ENABLE_HF_TRANSFER=1` per RFD 2196 rule 5.

Every subset is verified against `check_corpus_manifest.py --self-test` before publish. The
gate rejects a shard whose `anny_posed_vertices` column is not `(19158, 3)`, whose
`soma_pose` column is not `(77, 3)`, whose manifest is missing any required key, whose
compression is not zstd, whose `projection_check` field does not cover all 133 anchors, or
whose accuracy fields carry a number without naming the quantity (`bone_projection_accuracy_max_mm`
passes; `projection_accuracy_max_mm` fails). Twelve controls total.

## License and synthetic class (per subset)

Apache-2.0. **Synthetic class is stated per subset, not per corpus**, because the two
motion sources this corpus draws from sit on opposite sides of CLAUDE.md's
constructed/generated line:

| motion source | synthetic class | manifest `motion_source` |
| --- | --- | --- |
| ANNY / SOMA pose library (assets we hold, deterministic) | constructed | `soma-library` |
| Kimodo (a diffusion sampler) | **constructed renders over generated poses** | `kimodo` |

Renders are constructed either way — the pixels come from Mitsuba rendering ANNY posed
against `sphere_hammersley_sequence` cameras. But Kimodo is a sampled generative model, so
a Kimodo-driven subset's poses are generated synthetic and the four conditions in CLAUDE.md
bind on those subsets:

1. **Generator, checkpoint, and prompt/conditioning recorded** — the manifest's
   `motion_source` (name, repo, commit) and `sampler_config` (integrator, spp, max_depth,
   render seed; extended for Kimodo subsets to carry the Kimodo checkpoint hash and the
   prompt string) satisfy this.
2. **Stored and manifested separately from constructed and real data** — each subset's own
   commit set + `motion_source` label satisfies this by construction.
3. **Not the sole distribution for a model deployed on real inputs.** *Corpus-level
   invariant:* the corpus must carry at least one constructed-pose subset (`soma-library`
   or equivalent) before any training run consumes it. `check_corpus_manifest.py` asserts
   this at the corpus level, not just per-shard.
4. **Evaluation uses real or constructed data only** — the blinded holdout
   `coco_person_commercial_val2017` per CLAUDE.md; downstream training reads the manifest's
   `synthetic_class` per subset and includes generated subsets in gradient steps only.

License upstreams: ANNY (Apache-2.0, NAVER), Kimodo (Apache-2.0 code + NVIDIA Open Model
weights) — for the Kimodo checkpoint the ID / prompt / conditioning are the per-subset
provenance condition 1 requires — labels regressed from vertex weights
([weftspun/anny-keypoint-anchors](https://github.com/weftspun/anny-keypoint-anchors),
Apache-2.0 OR MIT, part-derived from NAVER's `coco.pth`).

## Citation

See `CITATION.cff` at the repository root. Every upstream this dataset consumed is a
separate `references:` entry with its license recorded honestly — inclusion decision is
"did the dataset use it", not "does its license require a credit".

## Related

RFD 2203 (row shape decision) · RFD 2196 (HF publishing rules) · RFD 1173 (multimodal
pipeline; parent) ·
[weftspun/anny-keypoint-anchors](https://github.com/weftspun/anny-keypoint-anchors)
(source of `wholebody133.pth`) ·
[weftspun/interactor-kimodo-text-to-motion](https://github.com/weftspun/interactor-kimodo-text-to-motion)
(source of SOMA motion output)
