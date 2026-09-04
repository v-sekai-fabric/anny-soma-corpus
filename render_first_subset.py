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


def pose_anny_batched(anny_model, kimodo_npz_path: Path, device: str):
    """Load Kimodo output, apply anny per frame, return posed vertices
    + bone poses + soma_pose per frame. Rotvec conversion + root
    prepend copied from `interactor-kimodo-text-to-motion/server.py:
    _pose_anny_from_soma`."""
    import numpy as np
    import roma
    import torch
    data = np.load(str(kimodo_npz_path), allow_pickle=False)
    local_rot_mats = torch.from_numpy(np.asarray(data["local_rot_mats"])).to(device=device, dtype=torch.float32)
    root_positions = torch.from_numpy(np.asarray(data["root_positions"])).to(device=device, dtype=torch.float32)
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
    """Run the same rule-2 apparatus interactor#2 landed, adapted so
    the positive number and the negative controls sit on the same
    apparatus (vertex diff via barycentric-interpolated reference).

    Two negative controls:
      (a) zero mid-hierarchy bone (bone 5) — mm-scale sensitivity
      (b) zero limb bone on limb-moving poses — cm-scale sensitivity

    Gates at anny thresholds (max<15mm, mean<5mm). If positive fails,
    the caller discards the subset before publish per RFD 2203.
    """
    # TODO(implementation): use anny's point_to_mesh_distance_and_face_uvs
    # on rest meshes to build makehuman->SOMA_wrap barycentric map once,
    # interpolate per pose, diff. Two controls as above.
    raise NotImplementedError("verify_projection_accuracy: wire barycentric-map vertex diff")


def write_manifest(out_dir: Path, pth_path: Path, observed_J: int, motion_source: str,
                   sampler_config: dict, seed: int, cotenancy_dump: dict):
    """Per-shard manifest with the fields RFD 2203 requires."""
    with open(pth_path, "rb") as f:
        pth_sha = hashlib.sha256(f.read()).hexdigest()
    manifest = {
        "wholebody133_pth_sha256": pth_sha,
        "observed_soma_joint_count": observed_J,
        "motion_source": motion_source,
        "sampler_config": sampler_config,
        "render_seed": seed,
        "cotenancy_at_kick": cotenancy_dump,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_parquet_shard(out_path: Path, rows: list):
    """One row per (motion, frame, camera) triple. Wide-row per RFD
    2196 (not ETNF). ZStandard compression, row_group_size 100 for
    viewer safety."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path, compression="zstd", row_group_size=100)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--motion-dir", required=True, help="Directory of Kimodo .npz motion files")
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
    observed_J = None
    for motion_path in sorted(Path(args.motion_dir).glob("*.npz")):
        posed = pose_anny_batched(anny_model, motion_path, device)
        if observed_J is None:
            observed_J = int(posed["soma_pose"].shape[1])

        # Verify BEFORE publishing — if the positive vertex check fails,
        # discard the subset per RFD 2203 rather than publish garbage.
        verify_result = verify_projection_accuracy(anny_model, posed["vertices"][:4])
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

    # 4. Write shard + manifest
    shard_path = out_dir / "shard_00.parquet"
    write_parquet_shard(shard_path, rows)
    write_manifest(
        out_dir=out_dir,
        pth_path=Path(args.pth),
        observed_J=observed_J,
        motion_source="TODO(first-subset review): walks + crouches/getups per RFD 2203 review convergence",
        sampler_config={"n_views": args.n_views, "image_size": args.image_size},
        seed=args.seed,
        cotenancy_dump=dump,
    )
    print(f"wrote {len(rows)} rows to {shard_path}")
    print(f"manifest at {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
