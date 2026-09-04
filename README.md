# anny-soma-corpus

The ANNY-SOMA keypoint corpus. Task #67 render side of the RFD 1173 keypoints stub. Living dataset: first subset lands, motions can be refined in place, new subsets add on top, each subset a separate commit set with the prior hash cited.

## Row shape

Hybrid per [RFD 2203](https://github.com/weftspun/request-for-discussion/tree/main/rfd/2203-anny-soma-first-subset-corpus-shape).

| column | shape | notes |
| --- | --- | --- |
| `image` | `struct<bytes, path>` | rendered frame per RFD 2196 |
| `camera` | `float32[4,4]` | extrinsics × intrinsics |
| `anny_posed_vertices` | `float32[19158, 3]` | makehuman topology, indexed by `wholebody133.pth` |
| `keypoints_2d` | `float32[133, 3]` | `(x, y, visible)` baked at the `.pth` hash |
| `soma_pose` | `float32[77, 3]` | axis-angle rotvecs as Kimodo emits |

Per-shard manifest carries `wholebody133.pth` SHA-256, observed SOMA joint count, motion source, sampler config, render seed.

## Publish path

`hf upload-large-folder` for per-file resumable commits. `HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1` per RFD 2196 rule 5.

## License

Apache-2.0. Renders are constructed from ANNY (Apache-2.0, NAVER) posed by Kimodo SOMA output (Apache-2.0 code, NVIDIA Open Model checkpoints).

## Related

RFD 2203 · RFD 2196 · RFD 1173 · `weftspun/anny-keypoint-anchors` (source of `wholebody133.pth`) · `weftspun/interactor-kimodo-text-to-motion` (source of SOMA output)
