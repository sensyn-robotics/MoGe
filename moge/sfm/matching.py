"""Correspondences via hloc: learned features + pair selection (sequential + retrieval
loop closure) + learned matching, plus track prep for BA.

Loop closure lives here: the retrieval pairs (``_run_hloc``, ``use_retrieval``) match
revisited places so bundle adjustment can undo odometry drift. Uses only
hloc.extract_features / pairs_from_* / match_features — none import pycolmap, so this is
independent of the pycolmap model code."""

from __future__ import annotations

from pathlib import Path

from moge.sfm.config import MoGe3SfMConfig


def _run_hloc(image_dir: Path, work_dir: Path, image_names: list[str], cfg: MoGe3SfMConfig):
    """Extract learned features (ALIKED), pick pairs (sequential window + retrieval loop
    closure), match with LightGlue. Returns (features_h5, matches_h5, pairs[list[(a,b)]])."""
    from hloc import extract_features, match_features, pairs_from_retrieval

    work_dir.mkdir(parents=True, exist_ok=True)
    feat_conf = extract_features.confs[cfg.feature_conf]
    match_conf = match_features.confs[cfg.matcher_conf]

    features = extract_features.main(feat_conf, image_dir, work_dir)

    pairs_path = work_dir / "pairs.txt"
    pair_set: set[tuple[str, str]] = set()

    # Sequential window (the office capture is a continuous walk).
    order = {n: i for i, n in enumerate(image_names)}
    for a in image_names:
        ia = order[a]
        for j in range(1, cfg.sequential_window + 1):
            ib = ia + j
            if ib < len(image_names):
                pair_set.add(tuple(sorted((a, image_names[ib]))))

    # Retrieval (NetVLAD) loop pairs.
    if cfg.use_retrieval:
        # Retrieval finds revisited places (loop-closure pairs) so BA can undo drift. OpenIBL
        # loads its weights via torch.hub, which refuses an "untrusted" repo non-interactively;
        # force trust_repo=True (we control the repo) so it runs headless.
        import torch.hub as _hub
        _orig_load = _hub.load
        _hub.load = lambda *a, **k: _orig_load(*a, **{"trust_repo": True, **k})
        retr_conf = extract_features.confs[cfg.retrieval_conf]
        retr_desc = extract_features.main(retr_conf, image_dir, work_dir)
        retr_pairs = work_dir / "pairs_retrieval.txt"
        pairs_from_retrieval.main(retr_desc, retr_pairs, num_matched=cfg.num_retrieval_pairs)
        for line in retr_pairs.read_text().splitlines():
            a, b = line.split()
            if a != b:
                pair_set.add(tuple(sorted((a, b))))

    pairs = sorted(pair_set)
    pairs_path.write_text("\n".join(f"{a} {b}" for a, b in pairs) + "\n")
    print(f"[moge3-sfm] {len(pairs)} pairs (seq window {cfg.sequential_window}"
          f"{' + retrieval' if cfg.use_retrieval else ''})")

    # hloc requires an explicit matches Path when `features` is a Path (not a name).
    matches_path = work_dir / f"{match_conf['output']}.h5"
    matches = match_features.main(match_conf, pairs_path, features=features, matches=matches_path)
    return features, matches, pairs


def _matches_to_kept(pairs, matches_h5, geoms, name_to_idx, cfg):
    """Per-pair inlier keypoint indices (kept for BA triangulation) straight from the matches,
    keeping only correspondences with valid MoGe 3D on both sides. No 3D-3D pose filtering —
    triangulation-angle + cheirality + reprojection filtering in bundle_adjust handles outliers."""
    from hloc.utils.io import get_matches
    kept: dict[tuple[int, int], tuple] = {}
    for a, b in pairs:
        ia, ib = name_to_idx[a], name_to_idx[b]
        m, _ = get_matches(matches_h5, a, b)
        if m.shape[0] < cfg.min_pair_inliers:
            continue
        v = geoms[ia].kp_valid[m[:, 0]] & geoms[ib].kp_valid[m[:, 1]]
        if v.sum() < cfg.min_pair_inliers:
            continue
        kept[(ia, ib)] = (m[v, 0], m[v, 1])
    return kept
