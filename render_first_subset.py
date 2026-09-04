"""Render the first subset of the anny-soma keypoint corpus.

Reads Kimodo `.npz` motion files, poses anny wearing the SOMA rig on
the 19,158-vertex makehuman topology (RFD 2203 topology decision),
regresses 2D keypoints via `KeypointsRegressor.load_precomputed(
wholebody133.pth)` at a `sphere_hammersley_sequence` camera, renders
the image with Mitsuba, and writes one parquet row per (motion, frame,
camera) triple.

Row shape per RFD 2203:
  image                  struct<bytes, path>
  camera                 float32[4, 4]  extrinsics x intrinsics
  anny_posed_vertices    float32[19158, 3]
  keypoints_2d           float32[133, 3]  (x, y, visible)
  soma_pose              float32[77, 3]   rotvecs as Kimodo emits

Per-shard manifest carries: `wholebody133.pth` SHA-256, observed SOMA
joint count, motion source, sampler config, render seed, and the
`nvidia-smi --query-compute-apps` co-tenancy dump at kick.

Publish path per RFD 2196 rule 5: `hf upload-large-folder` with
`HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1`.

Vertex-side verification (per HERD's rule-2 concern on
interactor#2's follow-up):
  1. Compute makehuman -> SOMA_wrap barycentric map once on rest meshes
  2. Per pose interpolate SOMA_wrap posed verts at those barycentrics
  3. Diff against makehuman posed verts (positive number)
  4. Two negative controls: zero mid-hierarchy bone; zero limb bone on
     limb-moving poses. Both on the SAME diff apparatus as the positive.
  5. Gate at anny thresholds (max<15mm, mean<5mm); if positive fails,
     the subset is discarded before publish.

This script is a draft skeleton per HERD's directive. Section by section
below; TODOs mark the parts that need first-subset scope decisions from
RFD 2203 review before they land.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# CPU cap before torch import per VRChat-headroom rule extension for
# CPU-heavy work. GPU work below is gated on VRChat's card detection.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


def detect_vrchat_gpu() -> tuple[int | None, dict]:
    """Return (vrchat_gpu_pci_index_or_none, dump).

    Reads `nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid`
    and matches on VRChat.exe / vrserver / vrcompositor process names.
    The dump goes verbatim into the manifest so a re-run can name what
    was co-tenant at kick.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid", "--format=csv"],
        capture_output=True, text=True, check=False,
    )
    dump = {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    vrchat_uuid = None
    for line in result.stdout.splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        _, proc, uuid = parts[0], parts[1], parts[2]
        if any(needle in proc.lower() for needle in ("vrchat.exe", "vrserver", "vrcompositor")):
            vrchat_uuid = uuid
            break

    # Map uuid -> torch cuda index
    idx = None
    if vrchat_uuid is not None:
        import torch
        for i in range(torch.cuda.device_count()):
            if torch.cuda.get_device_properties(i).uuid.hex.lower() == vrchat_uuid.lower().replace("gpu-", "").replace("-", ""):
                idx = i
                break

    return idx, dump


def free_gpu_index(vrchat_idx: int | None) -> int:
    """Return the CUDA index that VRChat is NOT on. If nothing was
    detected, default to cuda:0."""
    import torch
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("NOT-MEASURED: no CUDA devices visible")
    if vrchat_idx is None:
        return 0
    for i in range(count):
        if i != vrchat_idx:
            return i
    # Only one GPU and VRChat is on it — headroom-cap mode
    return vrchat_idx


def load_anny_model(device: str):
    """RFD 2203 topology decision: rig=soma + makehuman(19158)."""
    import anny
    import torch
    from anny.models.model_data import TopologyConfig
    return anny.Anny(
        rig="soma",
        topology=TopologyConfig(base_mesh="makehuman", remove_unattached_vertices=False),
        pose_parameterization="local-ref",
        phenotypes="all",
    ).to(device=device, dtype=torch.float32)


def hammersley_cameras(n_views: int) -> list:
    """`sphere_hammersley_sequence` per CLAUDE.md convention 6.
    TODO(first-subset review): 8 views per RFD 2203 review convergence.
    """
    # Ref implementation lives in `anny-render-corpus` per prior work;
    # for the draft return placeholder camera dicts.
    return [{"view_index": i, "extrinsics": None, "intrinsics": None} for i in range(n_views)]


def _load_kimodo_npz(path: Path, device: str):
    """Kimodo emits `local_rot_mats (T,J,3,3)` and `root_positions (T,3)`.
    Generated-synthetic class per CLAUDE.md (sampler output)."""
    import numpy as np
    import roma
    import torch
    data = np.load(str(path), allow_pickle=False)
    lrm = torch.from_numpy(np.asarray(data["local_rot_mats"])).to(device=device, dtype=torch.float32)
    rp = torch.from_numpy(np.asarray(data["root_positions"])).to(device=device, dtype=torch.float32)
    return lrm, rp


def _load_soma_library(path: Path, device: str):
    """SOMA/ANNY pose-library file. Constructed-synthetic class per
    CLAUDE.md (deterministic assets, no learned sampler in the loop).
    Same downstream shape as Kimodo: `(T,J,3,3)` local rotations +
    `(T,3)` root translation. Schema shipped by ANCHOR alongside the
    library enumeration.

    TODO(implementation): ANCHOR is enumerating the library; once the
    file schema lands, wire the loader here. Placeholder raises for now.
    """
    raise NotImplementedError(
        "SOMA-library loader not yet wired; awaiting ANCHOR's library "
        "enumeration + file-schema spec. See RFD 2203 followup on the "
        "constructed-pose subset for the schema definition."
    )


def pose_anny_batched(anny_model, pose_path: Path, pose_kind: str, device: str):
    """Load posed rotations from a Kimodo `.npz` or a SOMA-library
    file (kind = 'kimodo' | 'soma-library'), apply anny per frame,
    return posed vertices + bone poses + soma_pose per frame + the
    raw upstream joint count. Rotvec conversion + root prepend copied
    from `interactor-kimodo-text-to-motion/server.py:
    _pose_anny_from_soma`.

    The two pose kinds carry different synthetic-class semantics per
    CLAUDE.md (kimodo=generated, soma-library=constructed); the
    downstream shape is identical from anny's forward onward. The
    manifest records `motion_source.kind` and derives
    `synthetic_class` from it.
    """
    import roma
    import torch
    if pose_kind == "kimodo":
        local_rot_mats, root_positions = _load_kimodo_npz(pose_path, device)
    elif pose_kind == "soma-library":
        local_rot_mats, root_positions = _load_soma_library(pose_path, device)
    else:
        raise RuntimeError(f"unknown pose_kind={pose_kind!r}; expected 'kimodo' or 'soma-library'")
    T, J = int(local_rot_mats.shape[0]), int(local_rot_mats.shape[1])
    if J not in (77, 78):
        raise RuntimeError(f"unexpected SOMA joint count J={J}")
    rotvec = roma.rotmat_to_rotvec(local_rot_mats)
    if J == 77:
        rotvec_ext = torch.cat((torch.zeros((T, 1, 3), device=device, dtype=torch.float32), rotvec), dim=1)
    else:
        rotvec_ext = rotvec
    pose_parameters = roma.Rigid(roma.rotvec_to_rotmat(rotvec_ext), translation=None).to_homogeneous()
    pose_parameters[:, 0, :3, 3] = root_positions
    phenotype = torch.zeros((T, len(anny_model.phenotype_labels)), device=device, dtype=torch.float32)
    out = anny_model(pose_parameters=pose_parameters, phenotype_kwargs=phenotype, local_changes_kwargs={})
    return {
        "vertices": out["vertices"],       # (T, 19158, 3)
        "bone_poses": out["bone_poses"],   # (T, 78, 4, 4)
        "soma_pose": rotvec,               # (T, 77, 3) as Kimodo emitted
        "raw_joint_count": J,              # RAW upstream count (R2): task #76 back-port answers "did this
                                           # subset see 77 or 78 from Kimodo?" — soma_pose.shape[1] is always
                                           # 77 post-conversion regardless of truth, so read the raw shape
                                           # before conversion.
        "T": T,
    }


def regress_keypoints_2d(anny_model, posed_vertices, camera):
    """Read `wholebody133.pth` weights, apply to posed vertices per
    KeypointsRegressor.load_precomputed. Project to 2D via camera.
    Return (133, 3) with (x, y, visible)."""
    # TODO(implementation): wire `anny.keypoints.KeypointsRegressor`
    # against `wholebody133.pth` (path from anny-keypoint-anchors main).
    # Visibility via Z-test against depth from the same camera.
    raise NotImplementedError("regress_keypoints_2d: wire KeypointsRegressor + camera projection")


def render_mitsuba(posed_vertices, faces, camera, image_size: tuple[int, int]):
    """Mitsuba 3 render at (H, W). TODO(first-subset review): 384x384
    per RFD 2203 review convergence."""
    # TODO(implementation): mitsuba pipeline (scene dict from anny mesh
    # + camera, integrator=path, spp modest).
    raise NotImplementedError("render_mitsuba: wire Mitsuba 3 scene + render")


def verify_projection_accuracy(anny_model, posed_vertices, n_poses: int = 4) -> dict:
    """Run the rule-2 apparatus per HERD's critique on interactor#2:
    positive number and negative controls on the SAME apparatus
    (barycentric-interpolated vertex diff against a SOMA_wrap-topology
    reference).

    Two negative controls both on the same diff apparatus:
      (a) zero mid-hierarchy bone (bone 5) — mm-scale sensitivity
      (b) zero limb bone (upperarm.L or hip) on limb-moving poses —
          cm-scale sensitivity

    Gates at anny thresholds (max<15mm, mean<5mm) on the positive
    diff. If positive fails, the caller discards the subset before
    publish per RFD 2203.

    R6: state the detection floor per CLAUDE.md rule 5. With n=4
    poses this cell only sees defects that appear in more than ~75%
    of frames; per-motion drift below that rate is invisible. For a
    fixed motion population, enumerate all poses (or a stratified
    subset covering all motion categories) rather than sample four.
    A production run should scale n_poses with the shard's motion
    coverage; the returned dict names the floor so the manifest can
    record what fraction of drift the check could catch.
    """
    # TODO(implementation): use anny's point_to_mesh_distance_and_face_uvs
    # on rest meshes to build makehuman->SOMA_wrap barycentric map once,
    # interpolate per pose, diff against makehuman posed vertices.
    # Two controls above on the same barycentric-interpolated apparatus.
    # Return dict:
    #   passed: bool
    #   positive_max_mm, positive_mean_mm, positive_p99_mm
    #   negative_a_max_mm (mid-hierarchy bone; expect > positive)
    #   negative_b_max_mm (limb bone on limb-mover; expect >> positive)
    #   detection_floor_pct: 100.0 * 3 / n_poses (rule 5)
    #   n_poses_sampled, thresholds_mm
    raise NotImplementedError("verify_projection_accuracy: wire barycentric-map vertex diff")


def build_projection_check(pth_path: Path, unverified_reasons: dict,
                           bone_tracking_results: dict) -> dict:
    """Rule-3-complete projection_check field for every one of the 133
    anchors. Writer iterates the full label set from wholebody133.json
    and emits a `kind` for each, never omits. ANCHOR's gate requires:
    `kind` in {body_surface, bone_tracking, none} + counts add to 133.
    """
    import json as _json
    import torch
    labels = list(torch.load(pth_path, weights_only=True).keys())
    # Loading pth to enumerate labels; the actual weights aren't needed here.
    # Some anchors may be face_kpt_* placeholders present in wholebody133.json
    # but not in the .pth if ANCHOR hasn't shipped weights for them yet;
    # count both paths.
    entries = {}
    for anchor in labels:
        if anchor in unverified_reasons:
            entries[anchor] = unverified_reasons[anchor]  # {kind, reason, ...}
        elif anchor in bone_tracking_results:
            r = bone_tracking_results[anchor]
            entries[anchor] = {
                "kind": "bone_tracking",
                "reference_bone": r["reference_bone"],
                "rest_offset_mm": r["rest_offset_mm"],
                "variation_across_poses_mm": r["variation_across_poses_mm"],
            }
        else:
            entries[anchor] = {"kind": "body_surface"}
    if len(entries) != 133:
        raise RuntimeError(f"projection_check has {len(entries)} entries, expected 133 (rule 3: named and counted)")
    return entries


def write_manifest(out_dir: Path, pth_path: Path, observed_J: int, motion_source: dict,
                   sampler_config: dict, split_config: dict, seed: int,
                   cotenancy_dump: dict, verify_result: dict,
                   projection_check: dict,
                   keypoints_2d_face_status: str = "v3-axis-corrected"):
    """Per-shard manifest with the fields RFD 2203 requires.

    R5: motion_source is an object with `{repo, commit, categories, npz_count}`
    not a free-form TODO string.
    R4: sampler_config carries the Mitsuba knobs (integrator, spp,
    max_depth) alongside the view/resolution numbers.
    R3: split_config records the train/val split shape + fresh
    per-subset seed.
    R2: observed_J is the RAW upstream joint count (from
    local_rot_mats.shape[1] before rotvec conversion), NOT
    soma_pose.shape[1] which is always 77 post-conversion.
    R6: verify_result carries the per-motion max/mean plus the
    detection-floor statement per CLAUDE.md rule 5.
    """
    with open(pth_path, "rb") as f:
        pth_sha = hashlib.sha256(f.read()).hexdigest()
    # Derive synthetic_class from motion_source.kind per CLAUDE.md's
    # generated vs constructed distinction. Kimodo output comes from a
    # diffusion sampler = generated; SOMA-library poses are
    # deterministic assets we hold = constructed. #65 training must
    # see at least one constructed subset per condition 3 of the
    # generated-synthetic rule (not the sole distribution for a model
    # deployed on real inputs).
    SYNTHETIC_CLASS_BY_KIND = {"kimodo": "generated", "soma-library": "constructed"}
    synthetic_class = SYNTHETIC_CLASS_BY_KIND.get(motion_source.get("kind"))
    if synthetic_class is None:
        raise RuntimeError(f"motion_source.kind={motion_source.get('kind')!r} does not map to a synthetic_class")
    manifest = {
        "wholebody133_pth_sha256": pth_sha,
        "observed_soma_joint_count_raw": observed_J,
        "motion_source": motion_source,
        "synthetic_class": synthetic_class,     # derived, not asked separately; the source of truth is motion_source.kind
        "sampler_config": sampler_config,
        "split_config": split_config,
        "render_seed": seed,
        "cotenancy_at_kick": cotenancy_dump,
        "verify_result": verify_result,
        # Rule-3 completeness: every one of the 133 anchors named + counted.
        # Six fingertips at kind=bone_tracking, 2 hips at kind=body_surface
        # with checked_mass_pct notes, other 125 at kind=body_surface (v3
        # anchors closed face_axis_bug: all 68 face anchors pass mass
        # check under wholebody133.pth SHA 5eb5e244…).
        "projection_check": projection_check,
        # Under v2 anchors this was "v2-axis-bug" (jawline at chin height,
        # eye/mouth X-compressed ~50%; face_anchors.py assumed +Y up but
        # ANNY is +Z up). Anchors-v3 (SHA 5eb5e244…) corrected the axis
        # + closed the interior-mouth-vert leak; face anchors now
        # geographically-correct within 6mm median / 12mm worst-cap.
        # Refinement subsets can carry other values as new labels land.
        "keypoints_2d_face_status": keypoints_2d_face_status,
        # Rule-3 honesty on the sweep envelope's source (constructed
        # subsets) and per-pose validation (all subsets). Both start
        # as literal "hand-picked" / "none" because ANCHOR's ROM
        # envelope + HERO's RFD 0007 gate are tracked deliverables
        # not yet landed. Values flip to the source path / "rfd_0007"
        # when those land in follow-up subsets. Gate script asserts
        # the field is present with an allowlisted value; "none"
        # explicit is legal, missing is not.
        "sweep_ranges": "hand-picked",   # or "<source>" once anatomy ROM envelope lands
        "pose_validation": "none",       # or "rfd_0007" once foot-Y+joint-limit gate lands
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_parquet_shard(out_path: Path, rows: list):
    """One row per (motion, frame, camera) triple. Wide-row per RFD
    2196 (not ETNF). ZStandard compression, row_group_size 100 for
    viewer safety.

    R1: assert every row's anny_posed_vertices has shape (19158, 3)
    before write — a mis-sized column indicates the topology
    setting drifted and would corrupt every downstream training run
    silently. Contract with wholebody133.pth's 19,158 index space.
    """
    for i, row in enumerate(rows):
        v = row["anny_posed_vertices"]
        # Accept numpy or lists, but check the shape either way
        shape = getattr(v, "shape", None) or (len(v), len(v[0]) if len(v) else 0)
        if shape != (19158, 3):
            raise RuntimeError(
                f"row {i}: anny_posed_vertices shape {shape} != (19158, 3); "
                f"topology setting drifted, refusing to publish"
            )
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path, compression="zstd", row_group_size=100)


def split_train_val(rows: list, val_fraction: float, seed: int) -> tuple[list, list]:
    """R3: train/val 90/10 with a fresh per-subset seed. The seed is
    fresh so an additive subset's val split is not deterministic
    against the first subset's — a stable seed across subsets would
    let earlier training frames leak into a new val split.

    Splits at motion-source granularity, not row granularity, so all
    (frame, camera) rows for a given motion clip stay together in one
    split. Row-level shuffling would let a val frame's neighbours
    train.
    """
    import random
    rng = random.Random(seed)
    # Group rows by motion source id (path stem from row's image.path)
    by_motion: dict[str, list] = {}
    for row in rows:
        stem = row["image"]["path"].split("_f")[0]
        by_motion.setdefault(stem, []).append(row)
    motions = list(by_motion.keys())
    rng.shuffle(motions)
    n_val = max(1, int(round(len(motions) * val_fraction)))
    val_motions = set(motions[:n_val])
    train_rows = [r for m in motions if m not in val_motions for r in by_motion[m]]
    val_rows = [r for m in val_motions for r in by_motion[m]]
    return train_rows, val_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose-dir", required=True, help="Directory of pose source files (Kimodo .npz or SOMA-library files)")
    ap.add_argument("--pose-kind", choices=["kimodo", "soma-library"], required=True,
                    help="Pose source class. kimodo = generated synthetic (diffusion sampler); "
                         "soma-library = constructed synthetic (deterministic assets). "
                         "Determines the manifest's synthetic_class per CLAUDE.md.")
    ap.add_argument("--pth", required=True, help="Path to wholebody133.pth")
    ap.add_argument("--out-dir", required=True, help="Output directory for the shard + manifest")
    ap.add_argument("--n-views", type=int, default=8, help="hammersley view count (RFD 2203: 8 first, additive 16 later)")
    ap.add_argument("--image-size", type=int, default=384, help="Render resolution (RFD 2203: 384x384)")
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    # 1. Detect VRChat's GPU and route to the other
    vrchat_idx, dump = detect_vrchat_gpu()
    idx = free_gpu_index(vrchat_idx)
    import torch
    device = f"cuda:{idx}"
    print(f"VRChat on cuda:{vrchat_idx}, rendering on {device}")
    print(f"co-tenancy dump: {dump['stdout'][:300]}")

    # 2. Load anny once
    anny_model = load_anny_model(device)

    # 3. For each motion .npz, pose, verify, render, and collect rows
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = hammersley_cameras(args.n_views)

    rows = []
    observed_J_raw = None
    verify_result_agg = None
    glob_pattern = "*.npz" if args.pose_kind == "kimodo" else "*.pose"  # SOMA-library ext TBD; adjust when ANCHOR ships the schema
    for motion_path in sorted(Path(args.pose_dir).glob(glob_pattern)):
        posed = pose_anny_batched(anny_model, motion_path, args.pose_kind, device)
        if observed_J_raw is None:
            # R2: RAW upstream count from local_rot_mats.shape[1] before
            # rotvec conversion. soma_pose.shape[1] is always 77
            # post-prepend regardless of truth.
            observed_J_raw = int(posed["raw_joint_count"])

        # Verify BEFORE publishing — if the positive vertex check fails,
        # discard the subset per RFD 2203 rather than publish garbage.
        verify_result = verify_projection_accuracy(anny_model, posed["vertices"][:4])
        verify_result_agg = verify_result  # last-wins is fine; per-motion drift shows in per-motion runs
        if not verify_result.get("passed", False):
            print(f"NOT-MEASURED: verify failed for {motion_path.name}: {verify_result}")
            return 2

        for frame in range(posed["T"]):
            for cam in cameras:
                img_bytes = render_mitsuba(posed["vertices"][frame], anny_model.faces, cam, (args.image_size, args.image_size))
                kp2d = regress_keypoints_2d(anny_model, posed["vertices"][frame], cam)
                rows.append({
                    "image": {"bytes": img_bytes, "path": f"{motion_path.stem}_f{frame:04d}_v{cam['view_index']}.png"},
                    "camera": cam["extrinsics"],  # TODO: pack extrinsics + intrinsics
                    "anny_posed_vertices": posed["vertices"][frame].detach().cpu().numpy(),
                    "keypoints_2d": kp2d,
                    "soma_pose": posed["soma_pose"][frame].detach().cpu().numpy(),
                })

    # 4. Split train/val at motion granularity (R3); write both shards + manifest
    train_rows, val_rows = split_train_val(rows, val_fraction=0.10, seed=args.seed)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    write_parquet_shard(train_path, train_rows)
    write_parquet_shard(val_path, val_rows)

    # R4: Mitsuba sampler knobs recorded in manifest so a repro run
    # matches the same integrator / spp / max_depth. TODO: wire these
    # from render_mitsuba's real config.
    sampler_config = {
        "n_views": args.n_views,
        "image_size": args.image_size,
        "mitsuba": {
            "integrator": "path",
            "spp": 64,          # TODO(implementation): finalize spp against render-cost/quality tradeoff
            "max_depth": 4,     # TODO(implementation): 4 keeps GI cost modest for a body render
            "sampler": "independent",
        },
    }

    # R3: split config in manifest so a downstream consumer can pin
    # to the exact split shape + seed.
    split_config = {
        "shape": "90/10 train/val",
        "granularity": "motion",       # not per-row shuffle — clip integrity preserved
        "val_fraction": 0.10,
        "seed": args.seed,             # fresh per subset (not stable across subsets by design)
    }

    # R5: motion_source as a structured object with a `kind` enum
    # that CLAUDE.md's synthetic classes derive from
    # (kimodo=generated, soma-library=constructed). Manifest also
    # records the derived synthetic_class so a consumer never has to
    # rediscover the mapping.
    motion_source = {
        "kind": args.pose_kind,           # enum: 'kimodo' | 'soma-library'
        "repo": "TODO(first-subset review): e.g. weftspun/anny-render-corpus or nv-tlabs/kimodo",
        "commit": "TODO(first-subset review): git rev of the motion set or Kimodo checkpoint",
        "categories": ["walk", "crouch", "getup"],  # per RFD 2203 review convergence
        "file_count": len(list(Path(args.pose_dir).glob(glob_pattern))),
    }

    # Build the rule-3-complete projection_check: 6 fingertips from the
    # bone-tracking check, face landmarks (all 68) as kind=none pending
    # v3, hips as body_surface with checked_mass_pct note, rest as
    # body_surface implicit.
    fingertip_bone_map = {
        "left_middle_finger4":  "LeftHandMiddleEnd",
        "left_ring_finger4":    "LeftHandRingEnd",
        "left_pinky_finger4":   "LeftHandPinkyEnd",
        "right_middle_finger4": "RightHandMiddleEnd",
        "right_ring_finger4":   "RightHandRingEnd",
        "right_pinky_finger4":  "RightHandPinkyEnd",
    }
    # Bone-tracking results measured by interactor#5's
    # verify_projection_vertex.py on the SAME 4 poses; rest_offsets and
    # variations are the values from that run and are reproducible.
    # Real production writes these from the check's return dict; here
    # they're inlined so the manifest structure is complete and
    # rule-3-checked.
    bone_tracking_results = {
        anchor: {"reference_bone": bone, "rest_offset_mm": 6.5, "variation_across_poses_mm": 0.0}
        for anchor, bone in fingertip_bone_map.items()
    }
    # anchors-v3 (wholebody133.pth SHA 5eb5e244…, merged 2026-09-04)
    # closed the face_axis_bug: all 68 face anchors now clear the 1%
    # excluded-mass threshold and take kind=body_surface. Only the 8
    # exceptions remain: 6 fingertips (kind=bone_tracking, above) and
    # 2 hips (kind=body_surface with checked_mass_pct note). Ship
    # against v3 from the start per HERD's directive; no v2-flag
    # subset needed.
    unverified_reasons = {}
    unverified_reasons["left_hip"] = {
        "kind": "body_surface", "checked_mass_pct": 95.5,
        "note": "small overspill onto interior verts; upstream NAVER anchor",
    }
    unverified_reasons["right_hip"] = {
        "kind": "body_surface", "checked_mass_pct": 98.6,
        "note": "small overspill onto interior verts; upstream NAVER anchor",
    }
    projection_check = build_projection_check(
        Path(args.pth), unverified_reasons, bone_tracking_results,
    )

    write_manifest(
        out_dir=out_dir,
        pth_path=Path(args.pth),
        observed_J=observed_J_raw,
        motion_source=motion_source,
        sampler_config=sampler_config,
        split_config=split_config,
        seed=args.seed,
        cotenancy_dump=dump,
        verify_result=verify_result_agg or {"passed": False, "reason": "no motions processed"},
        projection_check=projection_check,
        keypoints_2d_face_status="v3-axis-corrected",
    )
    print(f"wrote train={len(train_rows)} val={len(val_rows)} rows to {out_dir}")
    print(f"manifest at {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
