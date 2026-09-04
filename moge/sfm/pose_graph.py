"""Pose engine 2: pairwise metric pose + global pose graph (utils3d_moge).

For each matched pair, lift correspondences to metric 3D via each image's MoGe points,
solve a robust relative pose, and accumulate all edges into a global pose graph (GNC).
Keeps only the largest connected component — disconnected frames have no gauge and would
collapse to the origin. Fragile on small-baseline video; kept for non-sequential captures.
The convention is verified: edge [i,j] has x = points in node i frame, y = node j frame,
and the optimizer's world->node poses ARE COLMAP cam_from_world."""

from __future__ import annotations

import numpy as np

from moge.sfm.config import MoGe3SfMConfig
from moge.sfm.moge_infer import _infer_and_lift


def _build_edges(pairs, matches_h5, geoms, name_to_idx, cfg):
    """For each matched pair build a pose-graph edge from MoGe 3D correspondences.

    Uses the per-frame keypoint 3D precomputed in `_infer_and_lift` (geom.kp3d /
    kp_valid), indexed by match keypoint id. Returns (edges (E,2) int, moment tensors,
    kept_matches) where kept_matches maps (ia, ib) -> (idx_a, idx_b) inlier
    keypoint-index arrays for later triangulation."""
    import torch
    import utils3d_moge as u3d
    from hloc.utils.io import get_matches

    edges, CYX, CXX, CYY, MX, MY, W = [], [], [], [], [], [], []
    kept: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(0)

    for a, b in pairs:
        ia, ib = name_to_idx[a], name_to_idx[b]
        m, _ = get_matches(matches_h5, a, b)        # (M,2) keypoint indices
        if m.shape[0] < cfg.min_pair_inliers:
            continue
        pa, va = geoms[ia].kp3d[m[:, 0]], geoms[ia].kp_valid[m[:, 0]]
        pb, vb = geoms[ib].kp3d[m[:, 1]], geoms[ib].kp_valid[m[:, 1]]
        v = va & vb
        if v.sum() < cfg.min_pair_inliers:
            continue
        pa, pb, mv = pa[v], pb[v], m[v]
        # relative pose a->b in metric 3D; inliers filter mismatches + MoGe outliers.
        _, inl = u3d.np.solve_pose_ransac(
            pa, pb, mode=cfg.pose_mode, threshold=cfg.ransac_threshold, rng=rng
        )
        if inl.sum() < cfg.min_pair_inliers:
            continue
        x = torch.tensor(pa[inl])[None]             # node ia frame
        y = torch.tensor(pb[inl])[None]             # node ib frame
        cyx, cxx, cyy, mx, my, w = u3d.pt.pose_graph_edge_moments(x, y)
        edges.append([ia, ib])
        CYX.append(cyx); CXX.append(cxx); CYY.append(cyy); MX.append(mx); MY.append(my); W.append(w)
        kept[(ia, ib)] = (mv[inl, 0], mv[inl, 1])

    if not edges:
        raise RuntimeError("[moge3-sfm] no surviving pair edges — matching/pairing failed.")
    print(f"[moge3-sfm] {len(edges)} pose-graph edges kept")
    cat = lambda L: torch.cat(L, 0)
    moments = (cat(CYX), cat(CXX), cat(CYY), cat(MX), cat(MY), cat(W))
    return torch.tensor(edges), moments, kept


def _global_poses(num_nodes: int, edges, moments, cfg) -> np.ndarray:
    """Global pose-graph optimization → (N,4,4) rigid cam_from_world (w2c).

    pose_graph_optimization_gnc returns world→node poses. For 'similar' mode each node
    also carries scale s_i (x_node = s_i R_i p + t_i); we fold it out to a rigid w2c
    (R_i, t_i / s_i) so the COLMAP model is metric-rigid and BA refines from there."""
    import utils3d_moge as u3d

    cyx, cxx, cyy, mx, my, w = moments
    poses, _ = u3d.pt.pose_graph_optimization_gnc(
        num_nodes, edges, cyx, cxx, cyy, mx, my, w,
        mode=cfg.pose_mode, niter=cfg.pose_graph_niter, gnc_iters=cfg.gnc_iters,
    )
    poses = poses.detach().cpu().numpy()             # (N,4,4) world->node, possibly scaled
    out = np.zeros_like(poses)
    for i in range(num_nodes):
        R = poses[i, :3, :3]
        s = np.cbrt(max(np.linalg.det(R), 1e-12))    # per-node scale from det (similar mode)
        out[i] = np.eye(4)
        out[i, :3, :3] = R / s
        out[i, :3, 3] = poses[i, :3, 3] / s
    return out


def _largest_component(edges, num_nodes: int) -> list[int]:
    """Node indices of the largest connected component of the edge graph.

    The pose graph fixes one global gauge, so nodes NOT connected to it collapse to the
    origin. We keep only the largest component and treat the rest as unregistered — an
    honest coverage number, and no garbage origin-cameras poisoning triangulation/BA."""
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges.tolist():
        parent[find(int(a))] = find(int(b))
    comps: dict[int, list[int]] = {}
    for i in range(num_nodes):
        comps.setdefault(find(i), []).append(i)
    return sorted(max(comps.values(), key=len))


def _pose_graph_engine(image_paths, kpts, matches, pairs, name_to_idx, cfg):
    """Pose init from pairwise 3D correspondences + global pose graph (largest component)."""
    import torch
    geoms = _infer_and_lift(image_paths, kpts, cfg)
    edges, moments, kept = _build_edges(pairs, matches, geoms, name_to_idx, cfg)
    # Disconnected frames have no gauge relative to node 0 and would collapse to the origin.
    comp = _largest_component(edges, len(geoms))
    inset = set(comp)
    if len(comp) < len(geoms):
        print(f"[moge3-sfm] largest connected component: {len(comp)}/{len(geoms)} frames "
              f"registered ({len(geoms) - len(comp)} unconnected frames dropped)")
    remap = {orig: i for i, orig in enumerate(comp)}
    emask = torch.tensor([int(a) in inset and int(b) in inset for a, b in edges.tolist()])
    edges = torch.tensor([[remap[int(a)], remap[int(b)]] for a, b in edges[emask].tolist()])
    moments = tuple(m[emask] for m in moments)
    geoms = [geoms[i] for i in comp]
    kpts = [kpts[i] for i in comp]
    kept = {(remap[a], remap[b]): v for (a, b), v in kept.items() if a in inset and b in inset}
    poses_w2c = _global_poses(len(geoms), edges, moments, cfg)
    return geoms, kpts, kept, poses_w2c
