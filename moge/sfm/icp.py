"""Pose engine 1: scale-aware (Sim3) pose graph on MoGe metric clouds.

MoGe's per-frame *metric scale* drifts wildly (observed 0.44-5.9x across one room), so a
rigid SE(3) approach — chained odometry or an SE(3) pose graph — cannot recover a coherent
metric room: it collapses or balloons (tens of metres, non-planar). This engine builds a
**Sim3** pose graph whose nodes carry a per-frame scale, solved by utils3d's robust
GNC-TLS optimizer (validated: recovers per-node scale to 1e-15 and rejects wrong edges):

  * odometry edges (consecutive frames): point-to-point ICP WITH scaling aligns across the
    per-frame scale; its dense inlier correspondences are well-conditioned even at the cm
    baselines that made sparse-match 3D-3D collapse. Streamed (only the previous cloud kept).
  * loop-closure edges (retrieval pairs): learned-match correspondences lifted to MoGe 3D.
  * GNC 'similar': per-node rotation+translation+SCALE, with GNC-TLS rejecting outlier edges.
  * fold Sim3 -> metric cam_from_world (R, t/s) in node 0's scale gauge, then gravity-align
    (rotate so the averaged MoGe floor-normal points up) to enforce the planar-room prior."""

from __future__ import annotations

import numpy as np
from PIL import Image

from moge.sfm.config import MoGe3SfMConfig
from moge.sfm.moge_infer import _FrameGeom, _lift_frame, _resolve_moge_pretrained


def _to_pcd(points, mask, voxel):
    import open3d as o3d
    pts = points[mask & np.isfinite(points).all(-1)].astype(np.float64)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    return pcd.voxel_down_sample(voxel)


def _up_from_normals(normal, mask):
    """Per-frame 'up' (opposite gravity) in the CAMERA frame from MoGe normals.

    Horizontal surfaces (floor/ceiling/tables) dominate an indoor scene and share one axis
    (vertical), so the top eigenvector of sum(n nᵀ) is that axis. Sign is ambiguous; orient
    it upward (OpenCV camera y points down, so world-up has a negative y component)."""
    n = normal[mask & np.isfinite(normal).all(-1)].astype(np.float64)
    if n.shape[0] < 50:
        return None
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    _, V = np.linalg.eigh(n.T @ n)
    v = V[:, -1]
    return v if v[1] < 0 else -v          # y<0 == upward in the OpenCV camera frame


def _gravity_align(poses_w2c, up_cam, valid):
    """Rotate the whole reconstruction so the averaged floor-normal is world +Z.

    up_cam[i] is frame i's up in its camera frame; R_wc_i @ up_cam_i is it in world. Average
    the valid ones, rotate the gauge so mean up = +Z (horizontal floor / near-planar walk)."""
    ups = [np.linalg.inv(poses_w2c[i])[:3, :3] @ up_cam[i]
           for i in range(len(poses_w2c)) if valid[i]]
    if not ups:
        return poses_w2c
    g = np.mean(ups, 0)
    g /= np.linalg.norm(g) + 1e-9
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(g, z)
    s = np.linalg.norm(axis)
    if s < 1e-6:
        return poses_w2c
    axis /= s
    ang = np.arctan2(s, np.dot(g, z))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    Rw = np.eye(4); Rw[:3, :3] = R
    return [w2c @ np.linalg.inv(Rw) for w2c in poses_w2c]


def _icp_posegraph(image_paths, kpts, matches_h5, pairs, name_to_idx, cfg: MoGe3SfMConfig):
    """MoGe-3 inference + Sim3 pose graph (scale-aware) + gravity align. Returns
    (geoms, poses_w2c (N,4,4))."""
    import torch
    import open3d as o3d
    from hloc.utils.io import get_matches
    from moge.model import import_model_class_by_version
    import utils3d_moge as u3d
    r = o3d.pipelines.registration

    device = torch.device(cfg.device)
    model = (import_model_class_by_version(cfg.moge_version)
             .from_pretrained(_resolve_moge_pretrained(cfg)).to(device).eval())
    rng = np.random.default_rng(0)
    n = len(image_paths)

    edges, CYX, CXX, CYY, MX, MY, W = [], [], [], [], [], [], []

    def add_edge(a, b, x, y):
        xt = torch.tensor(np.asarray(x), dtype=torch.float64)[None]
        yt = torch.tensor(np.asarray(y), dtype=torch.float64)[None]
        cyx, cxx, cyy, mx, my, w = u3d.pt.pose_graph_edge_moments(xt, yt)
        edges.append([a, b])
        CYX.append(cyx); CXX.append(cxx); CYY.append(cyy); MX.append(mx); MY.append(my); W.append(w)

    # 1) MoGe inference -> compact geom + camera-frame up; STREAM odometry edges (only the
    #    previous downsampled cloud is kept). Point-to-point ICP WITH scaling gives a Sim3 that
    #    aligns across the per-frame scale, and its dense correspondences feed the edge moments.
    geoms: list[_FrameGeom] = []
    up_cam = []
    est = r.TransformationEstimationPointToPoint(with_scaling=True)
    prev_pcd, prev_rel = None, np.eye(4)
    n_odo = 0
    for i, q in enumerate(image_paths):
        rgb = np.array(Image.open(q).convert("RGB"))
        t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            pred = model.infer(t)
        points = pred["points"].float().cpu().numpy()
        mask = (pred["mask"].cpu().numpy().astype(bool) if "mask" in pred
                else np.isfinite(points).all(-1))
        K = u3d.np.denormalize_intrinsics(pred["intrinsics"].cpu().numpy(),
                                          (rgb.shape[0], rgb.shape[1]))
        kp3d, kp_valid, kp_rgb, fuse_xyz, fuse_rgb = _lift_frame(points, mask, rgb, kpts[i], cfg, rng)
        geoms.append(_FrameGeom(q.name, (rgb.shape[1], rgb.shape[0]), K,
                                kp3d, kp_valid, kp_rgb, fuse_xyz, fuse_rgb))
        up_cam.append(_up_from_normals(pred["normal"].cpu().numpy(), mask)
                      if "normal" in pred else None)

        pcd = _to_pcd(points, mask, cfg.icp_voxel)
        if prev_pcd is not None:
            T = prev_rel
            reg = None
            for dist in (cfg.icp_max_corr_dist, cfg.icp_max_corr_dist / 6.0):
                reg = r.registration_icp(pcd, prev_pcd, dist, T, est,
                                         r.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iter))
                T = np.asarray(reg.transformation)
            corr = np.asarray(reg.correspondence_set)
            if corr.shape[0] >= 3:
                prev_rel = T
                xa = np.asarray(prev_pcd.points)[corr[:, 1]]   # node i-1 frame
                yb = np.asarray(pcd.points)[corr[:, 0]]        # node i frame
            else:
                # no overlap found: weak identity prior keeps the odometry chain connected.
                pts = np.asarray(prev_pcd.points)[:20]
                xa, yb = pts, pts.copy()
            add_edge(i - 1, i, xa, yb)
            n_odo += 1
        prev_pcd = pcd
        if (i + 1) % 50 == 0:
            print(f"[moge3-sfm] MoGe inference + odometry {i + 1}/{n}")

    # 2) loop-closure edges from retrieval pairs (beyond the sequential window): learned-match
    #    correspondences lifted to MoGe 3D. GNC-TLS rejects the wrong ones, so no pre-gating.
    n_loop = 0
    for a, b in pairs:
        ia, ib = name_to_idx[a], name_to_idx[b]
        if abs(ia - ib) <= cfg.sequential_window:
            continue
        m, _ = get_matches(matches_h5, a, b)
        if m.shape[0] < cfg.min_loop_inliers:
            continue
        v = geoms[ia].kp_valid[m[:, 0]] & geoms[ib].kp_valid[m[:, 1]]
        if v.sum() < cfg.min_loop_inliers:
            continue
        add_edge(ia, ib, geoms[ia].kp3d[m[v, 0]], geoms[ib].kp3d[m[v, 1]])
        n_loop += 1
    print(f"[moge3-sfm] {n_odo} odometry + {n_loop} loop edges")
    if not edges:
        raise RuntimeError("[moge3-sfm] no pose-graph edges — matching/ICP failed.")

    # 3) robust Sim3 global optimization (per-node rotation+translation+scale; GNC-TLS rejects
    #    outlier edges). Returns world->node poses whose 3x3 block is s*R.
    cat = torch.cat
    poses, _ = u3d.pt.pose_graph_optimization_gnc(
        n, torch.tensor(edges), cat(CYX, 0), cat(CXX, 0), cat(CYY, 0),
        cat(MX, 0), cat(MY, 0), cat(W, 0),
        mode="similar", threshold=cfg.ransac_threshold,
        niter=cfg.pose_graph_niter, gnc_iters=cfg.gnc_iters)
    poses = poses.detach().cpu().numpy()

    # 4) fold Sim3 -> metric cam_from_world (R, t/s) in node 0's gauge; rescale each frame's MoGe
    #    fusion points to that gauge so the BA-fallback fusion is scale-consistent.
    poses_w2c = np.zeros((n, 4, 4))
    for i in range(n):
        sR = poses[i, :3, :3]
        s = np.cbrt(max(np.linalg.det(sR), 1e-12))
        poses_w2c[i] = np.eye(4)
        poses_w2c[i, :3, :3] = sR / s
        poses_w2c[i, :3, 3] = poses[i, :3, 3] / s
        geoms[i].fuse_xyz = (geoms[i].fuse_xyz / s).astype(np.float32)
    poses_w2c = list(poses_w2c)

    # 5) gravity align (rotate scene so averaged MoGe floor-normal = up -> planar room).
    if cfg.gravity_align and any(u is not None for u in up_cam):
        valid = [u is not None for u in up_cam]
        safe_up = [u if u is not None else np.array([0.0, -1.0, 0.0]) for u in up_cam]
        poses_w2c = _gravity_align(poses_w2c, safe_up, valid)

    print(f"[moge3-sfm] Sim3 pose graph optimized: {n} frames")
    return geoms, np.stack(poses_w2c)
