"""Triangulation + bundle adjustment for MoGe-3 SfM (pycolmap 4.x).

Given a `pycolmap.Reconstruction` whose images already hold the pose-graph poses
(`pipeline._base_reconstruction`) and the LightGlue inlier matches, build feature
tracks, triangulate each at those poses (rejecting zero-baseline / behind-camera
tracks), and run robust global bundle adjustment. BA on true 2D reprojection is what
corrects MoGe's monocular-depth bias — the accuracy lever the pose-graph init alone
cannot provide.

Robustness is essential: a plain BA (TRIVIAL loss) lets a few outlier tracks — bad
matches, or near-parallel triangulations — pull cameras to infinity (NO_CONVERGENCE,
astronomical reprojection error). So we (1) keep only well-conditioned tracks
(triangulation angle + cheirality), (2) use a Cauchy loss, (3) filter large-residual
points and re-BA, and (4) report failure to the caller (which falls back to the
pose-graph poses + MoGe point fusion) if BA still diverges.

pycolmap 4.0.4 APIs confirmed against the installed build: `recon.image(id).cam_from_world()`
(a method) `.matrix()` (3x4), `Camera.cam_from_img` (pixel->ray), `triangulate_point`,
`calculate_triangulation_angle`, `add_point3D(xyz, Track, color)->id`, `Track.add_element`,
`Image.set_point3D_for_point2D`, `bundle_adjustment(recon, opts)` (loss via
`opts.ceres.loss_function_type`), `recon.update_point_3d_errors`, `recon.delete_point3D`.
"""

from __future__ import annotations

import numpy as np
import pycolmap

_DIVERGED_REPROJ_PX = 50.0    # above this after BA, treat the model as diverged


def _union_find_tracks(kept: dict) -> list[list[tuple[int, int]]]:
    """Connected components of the match graph over (image_idx, keypoint_idx) nodes.

    `kept` maps (ia, ib) -> (idx_a[], idx_b[]) inlier keypoint indices. Returns a list
    of tracks, each a list of (image_idx, keypoint_idx) observations."""
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for (ia, ib), (ida, idb) in kept.items():
        for ka, kb in zip(ida, idb):
            parent[find((ia, int(ka)))] = find((ib, int(kb)))

    groups: dict[tuple[int, int], list] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return list(groups.values())


def _centers(P: list[np.ndarray]) -> list[np.ndarray]:
    """Camera centers in world from 3x4 cam_from_world matrices (C = -R^T t)."""
    return [(-p[:3, :3].T @ p[:3, 3]) for p in P]


def triangulate_and_ba(recon, geoms, kpts, kept: dict, cfg) -> bool:
    """Populate recon.points3D from triangulated tracks, then robust-BA in place.

    Returns True if BA converged to a sane reprojection error, False if it diverged
    (caller should fall back to the pose-graph poses + MoGe fusion). `kpts` is a
    per-image list of (Ki,2) keypoint pixel arrays (index == keypoint id)."""
    n = len(geoms)
    cam = recon.camera(1)
    P = [np.asarray(recon.image(i + 1).cam_from_world().matrix()) for i in range(n)]
    C = _centers(P)
    rays = [cam.cam_from_img(kpts[i].astype(np.float64)) for i in range(n)]  # normalized
    min_angle = np.deg2rad(cfg.min_tri_angle_deg)
    # Distance cap for triangulated points: near-parallel rays can land a point at ~infinity
    # yet still reproject within threshold. Cap at 5x the camera-cloud diagonal (scale-adaptive).
    Ca = np.stack(C)
    max_point_dist = 5.0 * float(np.linalg.norm(Ca.max(0) - Ca.min(0)))

    # 1. Register keypoints as image Point2D (index == keypoint index).
    for i in range(n):
        recon.image(i + 1).points2D = [pycolmap.Point2D(xy.astype(np.float64)) for xy in kpts[i]]

    added = weak = behind = 0
    for members in _union_find_tracks(kept):
        by_img: dict[int, int] = {}                 # one observation per image (first wins)
        for ii, kk in members:
            by_img.setdefault(ii, kk)
        if len(by_img) < 2:
            continue
        obs = list(by_img.items())
        # Triangulate from the widest-baseline pair; reject if under-angled.
        best = None
        for a in range(len(obs)):
            for b in range(a + 1, len(obs)):
                ia, ka = obs[a]; ib, kb = obs[b]
                d = float(np.linalg.norm(C[ia] - C[ib]))
                if best is None or d > best[0]:
                    best = (d, ia, ka, ib, kb)
        _, ia, ka, ib, kb = best
        xyz = pycolmap.triangulate_point(P[ia], P[ib], rays[ia][ka], rays[ib][kb])
        if xyz is None:
            weak += 1
            continue
        if pycolmap.calculate_triangulation_angle(C[ia], C[ib], xyz) < min_angle:
            weak += 1
            continue
        # cheirality: in front of both anchor cameras.
        if (P[ia][2] @ np.append(xyz, 1.0)) <= 0 or (P[ib][2] @ np.append(xyz, 1.0)) <= 0:
            behind += 1
            continue
        if np.linalg.norm(xyz - C[ia]) > max_point_dist:   # runaway triangulation
            weak += 1
            continue
        color = geoms[ia].kp_rgb[ka].astype(np.uint8)
        track = pycolmap.Track()
        for ii, kk in obs:
            track.add_element(pycolmap.TrackElement(ii + 1, int(kk)))
        pid = recon.add_point3D(np.asarray(xyz, dtype=np.float64), track, color)
        for ii, kk in obs:
            recon.image(ii + 1).set_point3D_for_point2D(int(kk), pid)
        added += 1

    print(f"[moge3-sfm] triangulated {added} points "
          f"(rejected {weak} low-angle, {behind} behind-camera)")
    if added < 8:
        print("[moge3-sfm] too few points to bundle-adjust — BA skipped")
        return False

    # 2. Robust BA (Cauchy), then filter large-residual points and re-BA.
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_rig_from_world = True
    opts.refine_points3D = True
    opts.refine_focal_length = cfg.refine_intrinsics
    opts.refine_principal_point = False
    opts.refine_extra_params = False
    opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    opts.ceres.loss_function_scale = 1.0

    for it in range(2):
        pycolmap.bundle_adjustment(recon, opts)
        recon.update_point_3d_errors()
        bad = [pid for pid, p in recon.points3D.items()
               if not np.isfinite(p.error) or p.error > cfg.max_reproj_px]
        for pid in bad:
            recon.delete_point3D(pid)
        if not bad:
            break
        print(f"[moge3-sfm] BA pass {it + 1}: filtered {len(bad)} points "
              f"(> {cfg.max_reproj_px}px), {recon.num_points3D()} remain")

    err = recon.compute_mean_reprojection_error()
    print(f"[moge3-sfm] bundle-adjusted {recon.num_reg_images()} images, "
          f"{recon.num_points3D()} points (mean reproj err {err:.3f}px)")
    if not np.isfinite(err) or err > _DIVERGED_REPROJ_PX or recon.num_points3D() < 8:
        print(f"[moge3-sfm] BA diverged (err {err:.1f}px) — caller falls back to pose-graph poses")
        return False
    return True
