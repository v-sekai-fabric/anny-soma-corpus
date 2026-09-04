"""Gate the anny-soma-corpus render output against RFD 2203.

Every rule below is stated in RFD 2203 (`rfd/2203-anny-soma-first-subset-corpus-shape/`
in the manuals repo); this script is the machine-checked companion so a document rule
and a gate cannot drift. Self-test carries a negative control per rule, so a gate that
passes on known-broken input surfaces its own defect rather than certifying it.

    python check_corpus_manifest.py --subset <path>              gate one subset
    python check_corpus_manifest.py --self-test                  plant + reject 6 controls
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import sys
import tempfile

MAKEHUMAN_VERTEX_COUNT = 19158
SOMA_JOINT_COUNT = 77

QUANTITY_NOUNS = ("bone", "vertex", "joint", "mesh")
ACCURACY_KEY_MARKERS = ("accuracy", "verified", "spread")
REQUIRED_MANIFEST_KEYS = (
    "wholebody133_pth_sha256",
    "observed_soma_joint_count_raw",
    "motion_source",
    "sampler_config",
    "render_seed",
    "cotenancy_at_kick",
    "projection_check",
)
WHOLEBODY133_ANCHOR_COUNT = 133
PROJECTION_CHECK_KINDS = ("body_surface", "bone_tracking", "none")
MOTION_SOURCE_NAMES = ("soma-library", "kimodo")
CONSTRUCTED_MOTION_SOURCES = frozenset(("soma-library",))
SYNTHETIC_CLASS_FOR = {
    "soma-library": "constructed",
    "kimodo": "constructed-renders-over-generated-poses",
}
REQUIRED_ROW_COLUMNS = (
    "image", "camera", "anny_posed_vertices", "keypoints_2d", "soma_pose",
)


def sha256_of(path: pathlib.Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _walk_keys(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            path = "%s.%s" % (prefix, k) if prefix else k
            yield path
            yield from _walk_keys(v, path)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_keys(v, "%s[%d]" % (prefix, i))


def check_subset(subset: pathlib.Path, anchors_pth: pathlib.Path | None) -> list[str]:
    """Return a list of failure messages; empty list is a pass."""
    bad = []

    manifest_path = subset / "manifest.json"
    if not manifest_path.is_file():
        return ["no manifest.json at %s" % manifest_path]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return ["manifest.json is not valid JSON: %s" % e]

    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            bad.append("manifest is missing required key %r" % key)

    if manifest.get("observed_soma_joint_count_raw") not in (SOMA_JOINT_COUNT, 78):
        bad.append("observed_soma_joint_count_raw is %r, not 77 (or 78 for pre-conversion .npz)"
                   % manifest.get("observed_soma_joint_count_raw"))

    proj = manifest.get("projection_check")
    if proj is None:
        bad.append("manifest is missing projection_check field (RFD 2203 + rule 3: "
                   "every anchor's projection-verification state is named and counted)")
    else:
        entries = proj if isinstance(proj, list) else list(proj.values() if isinstance(proj, dict) else [])
        if len(entries) != WHOLEBODY133_ANCHOR_COUNT:
            bad.append("projection_check covers %d anchors, not %d (rule 3: unchecked "
                       "things are named and counted, never omitted)"
                       % (len(entries), WHOLEBODY133_ANCHOR_COUNT))
        for i, e in enumerate(entries):
            kind = e.get("kind") if isinstance(e, dict) else None
            if kind not in PROJECTION_CHECK_KINDS:
                bad.append("projection_check entry %d has kind %r, not one of %s"
                           % (i, kind, ", ".join(PROJECTION_CHECK_KINDS)))

    for key in _walk_keys(manifest):
        low = key.lower()
        if any(m in low for m in ACCURACY_KEY_MARKERS):
            if not any(n in low for n in QUANTITY_NOUNS):
                bad.append("manifest key %r names an accuracy/verified/spread field but no "
                           "quantity noun (%s)" % (key, ", ".join(QUANTITY_NOUNS)))

    ms = manifest.get("motion_source")
    ms_name = ms.get("name") if isinstance(ms, dict) else ms
    if ms_name not in MOTION_SOURCE_NAMES:
        bad.append("motion_source.name %r not in %s (RFD 2203)"
                   % (ms_name, MOTION_SOURCE_NAMES))
    else:
        sc = manifest.get("synthetic_class")
        expected = SYNTHETIC_CLASS_FOR[ms_name]
        if sc != expected:
            bad.append("synthetic_class %r for motion_source %r must be %r "
                       "(CLAUDE.md constructed vs generated)" % (sc, ms_name, expected))

    if anchors_pth is not None:
        if not anchors_pth.is_file():
            bad.append("--anchors-pth %s does not exist" % anchors_pth)
        else:
            expected = sha256_of(anchors_pth)
            got = manifest.get("wholebody133_pth_sha256")
            if got != expected:
                bad.append("wholebody133_pth_sha256 %s does not match anchors main %s"
                           % (got, expected))

    shards = sorted(subset.glob("*.parquet"))
    if not shards:
        bad.append("no *.parquet shards under %s" % subset)
        return bad

    import pyarrow.parquet as pq

    for shard in shards:
        pf = pq.ParquetFile(shard)
        schema = pf.schema_arrow
        for col in REQUIRED_ROW_COLUMNS:
            if col not in schema.names:
                bad.append("%s missing column %r" % (shard.name, col))

        codecs = {pf.metadata.row_group(g).column(c).compression
                  for g in range(pf.num_row_groups)
                  for c in range(pf.metadata.num_columns)}
        if not codecs.issubset({"ZSTD"}):
            bad.append("%s uses codecs %s; RFD 2196 requires zstd" % (shard.name, codecs))

        if "anny_posed_vertices" in schema.names:
            first = pf.read_row_group(0, columns=["anny_posed_vertices"]).column(0)
            for row in first.to_pylist()[:1]:
                if row is None or len(row) != MAKEHUMAN_VERTEX_COUNT:
                    bad.append("%s anny_posed_vertices has length %s, not %d"
                               % (shard.name, "None" if row is None else len(row),
                                  MAKEHUMAN_VERTEX_COUNT))
        if "soma_pose" in schema.names:
            first = pf.read_row_group(0, columns=["soma_pose"]).column(0)
            for row in first.to_pylist()[:1]:
                if row is None or len(row) != SOMA_JOINT_COUNT:
                    bad.append("%s soma_pose has length %s, not %d"
                               % (shard.name, "None" if row is None else len(row),
                                  SOMA_JOINT_COUNT))

    readme = subset / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        if "configs:" not in text and "config_name:" not in text:
            bad.append("README.md is missing a HF-viewer configs: block")
    return bad


def check_corpus(root: pathlib.Path) -> list[str]:
    """Corpus-level gate: at least one subset carries motion_source=soma-library (or another
    entry in CONSTRUCTED_MOTION_SOURCES). CLAUDE.md generated-synthetic condition 3: a Kimodo-
    only corpus is not the sole distribution for a model deployed on real inputs, so at least
    one constructed-pose subset must be present before any training run consumes the corpus.
    """
    bad = []
    subsets = [d for d in sorted(root.iterdir())
               if d.is_dir() and (d / "manifest.json").is_file()]
    if not subsets:
        return bad
    sources = []
    for d in subsets:
        try:
            m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        ms = m.get("motion_source")
        sources.append(ms.get("name") if isinstance(ms, dict) else ms)
    if not any(s in CONSTRUCTED_MOTION_SOURCES for s in sources):
        bad.append("corpus has %d subset(s) with motion_source names %s but none is in "
                   "CONSTRUCTED_MOTION_SOURCES (%s). CLAUDE.md generated-synthetic condition "
                   "3: the corpus must carry at least one constructed-pose subset before any "
                   "training run consumes it" % (len(sources), sources,
                                                 tuple(CONSTRUCTED_MOTION_SOURCES)))
    return bad


def _plant_good_shard(root: pathlib.Path, pth_sha: str) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [{
        "image": {"bytes": b"\x89PNG", "path": "f0000_v0.png"},
        "camera": np.eye(4, dtype=np.float32).tolist(),
        "anny_posed_vertices": np.zeros((MAKEHUMAN_VERTEX_COUNT, 3), dtype=np.float32).tolist(),
        "keypoints_2d": np.zeros((133, 3), dtype=np.float32).tolist(),
        "soma_pose": np.zeros((SOMA_JOINT_COUNT, 3), dtype=np.float32).tolist(),
    }]
    pq.write_table(pa.Table.from_pylist(rows), root / "shard_00.parquet",
                   compression="zstd", row_group_size=100)
    manifest = {k: "seed" for k in REQUIRED_MANIFEST_KEYS}
    manifest["wholebody133_pth_sha256"] = pth_sha
    manifest["observed_soma_joint_count_raw"] = SOMA_JOINT_COUNT
    manifest["projection_check"] = [
        {"anchor": "kpt_%d" % i, "kind": "body_surface"}
        for i in range(WHOLEBODY133_ANCHOR_COUNT)
    ]
    manifest["motion_source"] = {"name": "soma-library"}
    manifest["synthetic_class"] = SYNTHETIC_CLASS_FOR["soma-library"]
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "README.md").write_text("---\nconfigs:\n- config_name: default\n  data_files: '*.parquet'\n---\n", encoding="utf-8")


def self_test() -> int:
    """Plant a passing subset, then break one rule at a time and assert the gate rejects."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)

        pth_file = root / "wholebody133.pth"
        pth_file.write_bytes(b"stub-pth-content-for-self-test")
        pth_sha = sha256_of(pth_file)

        good = root / "good"
        good.mkdir()
        _plant_good_shard(good, pth_sha)

        cases = []
        cases.append(("positive control passes", check_subset(good, pth_file), True))

        wrong_verts = root / "wrong_verts"
        wrong_verts.mkdir()
        _plant_good_shard(wrong_verts, pth_sha)
        rows = [{
            "image": {"bytes": b"", "path": "x"},
            "camera": np.eye(4, dtype=np.float32).tolist(),
            "anny_posed_vertices": np.zeros((18056, 3), dtype=np.float32).tolist(),
            "keypoints_2d": np.zeros((133, 3), dtype=np.float32).tolist(),
            "soma_pose": np.zeros((SOMA_JOINT_COUNT, 3), dtype=np.float32).tolist(),
        }]
        pq.write_table(pa.Table.from_pylist(rows), wrong_verts / "shard_00.parquet",
                       compression="zstd", row_group_size=100)
        cases.append(("18,056-vertex shard rejected", check_subset(wrong_verts, pth_file), False))

        missing = root / "missing_col"
        missing.mkdir()
        _plant_good_shard(missing, pth_sha)
        manifest = json.loads((missing / "manifest.json").read_text(encoding="utf-8"))
        del manifest["motion_source"]
        (missing / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        cases.append(("missing manifest column rejected", check_subset(missing, pth_file), False))

        stale = root / "stale_hash"
        stale.mkdir()
        _plant_good_shard(stale, "0" * 64)
        cases.append(("stale pth hash rejected", check_subset(stale, pth_file), False))

        wrong_j = root / "wrong_j"
        wrong_j.mkdir()
        _plant_good_shard(wrong_j, pth_sha)
        rows_j = [{
            "image": {"bytes": b"", "path": "x"},
            "camera": np.eye(4, dtype=np.float32).tolist(),
            "anny_posed_vertices": np.zeros((MAKEHUMAN_VERTEX_COUNT, 3), dtype=np.float32).tolist(),
            "keypoints_2d": np.zeros((133, 3), dtype=np.float32).tolist(),
            "soma_pose": np.zeros((66, 3), dtype=np.float32).tolist(),
        }]
        pq.write_table(pa.Table.from_pylist(rows_j), wrong_j / "shard_00.parquet",
                       compression="zstd", row_group_size=100)
        cases.append(("66-joint soma_pose rejected", check_subset(wrong_j, pth_file), False))

        uncomp = root / "uncompressed"
        uncomp.mkdir()
        _plant_good_shard(uncomp, pth_sha)
        rows_u = [{
            "image": {"bytes": b"", "path": "x"},
            "camera": np.eye(4, dtype=np.float32).tolist(),
            "anny_posed_vertices": np.zeros((MAKEHUMAN_VERTEX_COUNT, 3), dtype=np.float32).tolist(),
            "keypoints_2d": np.zeros((133, 3), dtype=np.float32).tolist(),
            "soma_pose": np.zeros((SOMA_JOINT_COUNT, 3), dtype=np.float32).tolist(),
        }]
        pq.write_table(pa.Table.from_pylist(rows_u), uncomp / "shard_00.parquet",
                       compression="snappy", row_group_size=100)
        cases.append(("non-zstd compression rejected", check_subset(uncomp, pth_file), False))

        no_configs = root / "no_configs"
        no_configs.mkdir()
        _plant_good_shard(no_configs, pth_sha)
        (no_configs / "README.md").write_text("# no configs block\n", encoding="utf-8")
        cases.append(("README without configs block rejected",
                      check_subset(no_configs, pth_file), False))

        unnamed_accuracy = root / "unnamed_accuracy"
        unnamed_accuracy.mkdir()
        _plant_good_shard(unnamed_accuracy, pth_sha)
        manifest = json.loads((unnamed_accuracy / "manifest.json").read_text(encoding="utf-8"))
        manifest["projection_accuracy_max_mm"] = 0.000
        (unnamed_accuracy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        cases.append(("unnamed accuracy field rejected (bone vs vertex)",
                      check_subset(unnamed_accuracy, pth_file), False))

        named_accuracy = root / "named_accuracy"
        named_accuracy.mkdir()
        _plant_good_shard(named_accuracy, pth_sha)
        manifest = json.loads((named_accuracy / "manifest.json").read_text(encoding="utf-8"))
        manifest["bone_projection_accuracy_max_mm"] = 0.000
        (named_accuracy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        cases.append(("accuracy field naming the quantity passes",
                      check_subset(named_accuracy, pth_file), True))

        no_proj = root / "no_projection_check"
        no_proj.mkdir()
        _plant_good_shard(no_proj, pth_sha)
        manifest = json.loads((no_proj / "manifest.json").read_text(encoding="utf-8"))
        del manifest["projection_check"]
        (no_proj / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        cases.append(("missing projection_check rejected",
                      check_subset(no_proj, pth_file), False))

        short_proj = root / "short_projection_check"
        short_proj.mkdir()
        _plant_good_shard(short_proj, pth_sha)
        manifest = json.loads((short_proj / "manifest.json").read_text(encoding="utf-8"))
        manifest["projection_check"] = manifest["projection_check"][:132]
        (short_proj / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        cases.append(("projection_check covering 132 of 133 rejected",
                      check_subset(short_proj, pth_file), False))

        bad_kind = root / "bad_kind_projection_check"
        bad_kind.mkdir()
        _plant_good_shard(bad_kind, pth_sha)
        manifest = json.loads((bad_kind / "manifest.json").read_text(encoding="utf-8"))
        manifest["projection_check"][0]["kind"] = "not_a_kind"
        (bad_kind / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        cases.append(("projection_check with unknown kind rejected",
                      check_subset(bad_kind, pth_file), False))

        unknown_ms = root / "unknown_motion_source"
        unknown_ms.mkdir()
        _plant_good_shard(unknown_ms, pth_sha)
        manifest = json.loads((unknown_ms / "manifest.json").read_text(encoding="utf-8"))
        manifest["motion_source"] = {"name": "not-in-enum"}
        (unknown_ms / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        cases.append(("motion_source not in enum rejected",
                      check_subset(unknown_ms, pth_file), False))

        wrong_class = root / "wrong_synthetic_class"
        wrong_class.mkdir()
        _plant_good_shard(wrong_class, pth_sha)
        manifest = json.loads((wrong_class / "manifest.json").read_text(encoding="utf-8"))
        manifest["motion_source"] = {"name": "kimodo"}
        manifest["synthetic_class"] = "constructed"
        (wrong_class / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        cases.append(("kimodo mislabelled as constructed rejected",
                      check_subset(wrong_class, pth_file), False))

        kimodo_only_root = root / "kimodo_only_corpus"
        kimodo_only_root.mkdir()
        for i in range(2):
            sub = kimodo_only_root / ("subset-%02d-kimodo" % i)
            sub.mkdir()
            _plant_good_shard(sub, pth_sha)
            m = json.loads((sub / "manifest.json").read_text(encoding="utf-8"))
            m["motion_source"] = {"name": "kimodo"}
            m["synthetic_class"] = SYNTHETIC_CLASS_FOR["kimodo"]
            (sub / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        corpus_bad = check_corpus(kimodo_only_root)
        cases.append(("corpus of only kimodo subsets rejected (condition 3)",
                      corpus_bad, False))

        mixed_root = root / "mixed_corpus"
        mixed_root.mkdir()
        for i, ms in enumerate(("soma-library", "kimodo")):
            sub = mixed_root / ("subset-%02d-%s" % (i, ms))
            sub.mkdir()
            _plant_good_shard(sub, pth_sha)
            m = json.loads((sub / "manifest.json").read_text(encoding="utf-8"))
            m["motion_source"] = {"name": ms}
            m["synthetic_class"] = SYNTHETIC_CLASS_FOR[ms]
            (sub / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        corpus_ok = check_corpus(mixed_root)
        cases.append(("corpus with one constructed + one generated passes condition 3",
                      corpus_ok, True))

        fails = []
        for label, result, expected_pass in cases:
            passed = len(result) == 0
            if passed == expected_pass:
                print("  ok  %s" % label)
            else:
                fails.append(label)
                print("  BAD %s: %s" % (label, result if result else "unexpectedly passed"))

        print("\n%d failed" % len(fails))
        return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", type=pathlib.Path, help="directory holding shards + manifest.json")
    ap.add_argument("--corpus", type=pathlib.Path,
                    help="directory holding subset-*/ subdirectories, each with manifest.json")
    ap.add_argument("--anchors-pth", type=pathlib.Path, default=None,
                    help="path to wholebody133.pth on anchors main; hash cross-check skipped if omitted")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    bad = []
    if args.subset is not None:
        bad.extend(check_subset(args.subset, args.anchors_pth))
    if args.corpus is not None:
        bad.extend(check_corpus(args.corpus))
    if args.subset is None and args.corpus is None:
        sys.exit("--subset or --corpus (or --self-test) required")

    for b in bad:
        print("  FAIL  %s" % b)
    print("%d problem(s)" % len(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
