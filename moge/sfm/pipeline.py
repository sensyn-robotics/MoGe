"""Structure-from-Motion from MoGe-3 metric geometry + learned matching.

MoGe-3 is a single-image model: per image it predicts a **metric** point map (OpenCV
camera frame) + intrinsics + mask, but **no cross-image poses**. This module recovers
the poses MoGe lacks and writes a standard COLMAP ``sparse/0`` model (cameras + poses +
points), usable by any COLMAP consumer (3D Gaussian Splatting, MVS, visualization).

## Pipeline
1. **MoGe-3 inference**: per image → metric ``points`` (H,W,3, camera frame), pixel
   ``intrinsics``, ``mask``, rgb (keypoints are lifted immediately; the full map is freed).
2. **Correspondences** (hloc): learned features (ALIKED by default) + LightGlue matches
   over a sequential window (+ optional retrieval loop pairs).
3. **Pairwise metric relative pose**: lift matched pixels to metric 3D via each image's
   MoGe ``points`` → ``utils3d.solve_pose_ransac`` per pair; keep inlier matches + edges.
4. **Global pose graph**: ``utils3d.pose_graph_optimization_gnc`` over all edges → world→node
   poses = COLMAP ``cam_from_world`` directly (edge ``[i,j]``: x=pts in node i frame, y=pts
   in node j frame). Keep only the largest connected component (unconnected frames would
   collapse to the gauge origin; they are reported as unregistered instead).
5. **Triangulation + bundle adjustment** (pycolmap): triangulate inlier tracks at the
   pose-graph poses, then BA refines poses+points on true 2D reprojection (corrects MoGe's
   monocular-depth bias). Falls back to MoGe-point fusion if BA diverges/is disabled.
6. **Export** ``<output>/sparse/0/``: one shared PINHOLE camera (median intrinsics), one
   Rig+Frame per image, written as both ``.bin`` and ``.txt``.

Dependencies beyond MoGe core (all permissive / MIT-compatible; install via the ``sfm``
extra): hloc (Apache-2.0), LightGlue (Apache-2.0), pycolmap/COLMAP (BSD-3-Clause), plus
ALIKED features (BSD-3-Clause via kornia, Apache-2.0). See ``moge/sfm/__init__.py`` for the
full dependency-license table. Do NOT wire in SuperPoint/SuperGlue — they are Magic Leap
NONCOMMERCIAL and would taint MoGe's MIT license.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import yaml
from PIL import Image


@dataclass
class MoGe3SfMConfig:
    # --- MoGe-3 model ---
    # HF hub id or local .pt for MoGe-3. None -> Ruicheng/moge-3-vitl (or a local
    # <repo>/pretrained/moge-3-vitl.pt if $MOGE_REPO_DIR is set).
    moge_pretrained: str | None = None
    moge_version: str = "v3"
    moge_repo: str | None = None
    device: str = "cuda"

    # --- correspondence front-end (hloc) ---
    max_keypoints: int = 4096          # learned keypoints per image (ALIKED)
    num_retrieval_pairs: int = 20      # retrieval top-k loop pairs per image (OpenIBL)
    sequential_window: int = 30        # match each frame to +/- N neighbours (wide → robust chain)
    use_retrieval: bool = True         # retrieval loop closure (helps coverage)

    # --- pose engine ---
    # 'icp': chain point-to-plane ICP of consecutive MoGe metric clouds (frame->previous,
    #        accumulated from frame 0), then global BA removes drift. Simple + robust on a
    #        continuous walk; needs no cross-image matching for the init.
    # 'pose_graph': pairwise 3D correspondences (from matches) -> global pose graph. Fragile
    #        on small-baseline video (cameras collapse); kept for non-sequential captures.
    pose_engine: str = "icp"
    icp_voxel: float = 0.03            # metres — voxel-downsample each cloud before ICP
    icp_max_corr_dist: float = 0.30    # metres — ICP correspondence radius (coarse pass; fine = /6)
    icp_max_iter: int = 60
    icp_min_fitness: float = 0.30      # reject ICP below this overlap fitness -> constant-velocity coast
    icp_max_motion: float = 1.00       # metres — reject implausibly large inter-frame motion

    # --- pairwise pose + global alignment (pose_graph engine only) ---
    # 'rigid': MoGe-3 is metric (scale baked in), so per-image scale drift is small and BA
    # on 2D reprojection makes the final model scale-consistent regardless of the init. Also
    # avoids a utils3d_moge bug in solve_pose(mode='similar') — s*R fails to broadcast
    # ((B,S) vs (B,S,3,3)) on the batched RANSAC path.
    pose_mode: str = "rigid"           # 'rigid' | 'similar' (latter is currently broken upstream)
    ransac_threshold: float = 0.05     # metres, solve_pose_ransac inlier threshold
    min_pair_inliers: int = 20         # drop a pair edge below this many 3D inliers
    gnc_iters: int = 20
    pose_graph_niter: int = 10

    # --- points3D + bundle adjustment ---
    bundle_adjust: bool = True         # triangulate tracks + pycolmap BA (accuracy lever)
    refine_intrinsics: bool = False    # let BA adjust the shared focal (off = trust MoGe)
    min_tri_angle_deg: float = 1.5     # reject near-parallel (low-baseline) triangulations
    max_reproj_px: float = 4.0         # drop points above this reprojection error, then re-BA
    points_max: int = 1_000_000
    fuse_per_frame: int = 1200         # BA-fallback: MoGe points sampled per frame to fuse

    # hloc conf names. Deliberately ALL PERMISSIVE (MIT-PR-clean): ALIKED (BSD-3) +
    # LightGlue (Apache-2.0) + OpenIBL (MIT). SuperPoint/SuperGlue are Magic Leap
    # NONCOMMERCIAL — never use them here, they'd taint an MIT contribution to MoGe.
    feature_conf: str = "aliked-n16"
    matcher_conf: str = "aliked+lightglue"
    retrieval_conf: str = "openibl"


# ----------------------------------------------------------------------------
# MoGe-3 inference
# ----------------------------------------------------------------------------

# HuggingFace hub id for MoGe-3 large. `from_pretrained` accepts a local .pt path OR
# a hub id; the image bakes this id into its HF cache (docker/moge3.Dockerfile) so no
# runtime download. Local dev uses the .pt under $MOGE_REPO_DIR/pretrained if present.
_MOGE3_HF_ID = "Ruicheng/moge-3-vitl"


def _resolve_moge_pretrained(cfg: MoGe3SfMConfig) -> str:
    if cfg.moge_pretrained:
        return cfg.moge_pretrained
    repo = cfg.moge_repo or os.environ.get("MOGE_REPO_DIR")
    if repo:
        p = Path(repo) / "pretrained" / "moge-3-vitl.pt"
        if p.exists():
            return str(p)
    return _MOGE3_HF_ID


def _sorted_images(image_dir: Path) -> list[Path]:
    """Deterministic image order (sorted .jpg then .png)."""
    d = str(image_dir)
    paths = glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png"))
    return [Path(p) for p in sorted(paths)]


@dataclass
class _FrameGeom:
    """Compact per-image MoGe geometry (metric, OpenCV camera frame).

    Only per-keypoint 3D + a small fusion subsample are kept — NOT the full (H,W,3)
    point map. Holding every frame's full map is a ~44 MB x N_frames RAM bomb that
    OOMs at the 1215-frame scale; keypoints (what the pose graph + triangulation
    actually consume) and a bounded fusion sample are ~0.15 MB/frame instead."""
    name: str
    size: tuple[int, int]        # (W, H) original resolution
    K: np.ndarray                # (3, 3) intrinsics in ORIGINAL pixels
    kp3d: np.ndarray             # (Ki, 3) MoGe metric 3D at each keypoint, camera frame
    kp_valid: np.ndarray         # (Ki,) bool — MoGe mask valid at that keypoint
    kp_rgb: np.ndarray           # (Ki, 3) uint8 colour at each keypoint
    fuse_xyz: np.ndarray         # (S, 3) camera-frame points for the BA-fallback fusion
    fuse_rgb: np.ndarray         # (S, 3) uint8 colours for those points


def _infer_and_lift(image_paths, kpts, cfg: MoGe3SfMConfig) -> list[_FrameGeom]:
    """Run MoGe-3 per image and immediately lift its keypoints to metric 3D, keeping
    only compact arrays (the full point map is freed each iteration).

    MoGe returns intrinsics NORMALIZED (cx=cy=0.5); we denormalize to original pixels.
    `points` are metric metres in the OpenCV camera frame (x right, y down, z forward).
    `kpts[i]` are that image's keypoint pixels (from hloc, index == keypoint id)."""
    import torch
    from moge.model import import_model_class_by_version
    import utils3d_moge as u3d

    device = torch.device(cfg.device)
    model = (
        import_model_class_by_version(cfg.moge_version)
        .from_pretrained(_resolve_moge_pretrained(cfg))
        .to(device)
        .eval()
    )
    rng = np.random.default_rng(0)
    out: list[_FrameGeom] = []
    for i, q in enumerate(image_paths):
        rgb = np.array(Image.open(q).convert("RGB"))
        h, w = rgb.shape[:2]
        t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            pred = model.infer(t)
        points = pred["points"].float().cpu().numpy()            # (ph,pw,3) metric cam frame
        mask = (pred["mask"].cpu().numpy().astype(bool)
                if "mask" in pred else np.isfinite(points).all(-1))
        K = u3d.np.denormalize_intrinsics(pred["intrinsics"].cpu().numpy(), (w, h))
        ph, pw = points.shape[:2]
        sx, sy = pw / w, ph / h

        # lift this frame's keypoints (original-pixel coords -> MoGe map grid)
        kp = kpts[i]
        kx = np.clip(np.round(kp[:, 0] * sx).astype(int), 0, pw - 1)
        ky = np.clip(np.round(kp[:, 1] * sy).astype(int), 0, ph - 1)
        kp3d = points[ky, kx]
        kp_valid = mask[ky, kx] & np.isfinite(kp3d).all(-1)
        okx = np.clip(kp[:, 0].astype(int), 0, w - 1)
        oky = np.clip(kp[:, 1].astype(int), 0, h - 1)
        kp_rgb = rgb[oky, okx]

        # bounded fusion subsample (BA-fallback only): random valid map pixels
        vy, vx = np.nonzero(mask & np.isfinite(points).all(-1))
        if vy.size:
            take = min(cfg.fuse_per_frame, vy.size)
            sel = rng.choice(vy.size, take, replace=False)
            fy, fx = vy[sel], vx[sel]
            fuse_xyz = points[fy, fx].astype(np.float32)
            fuse_rgb = rgb[np.clip((fy / sy).astype(int), 0, h - 1),
                           np.clip((fx / sx).astype(int), 0, w - 1)]
        else:
            fuse_xyz = np.zeros((0, 3), np.float32); fuse_rgb = np.zeros((0, 3), np.uint8)

        out.append(_FrameGeom(q.name, (w, h), K, kp3d, kp_valid, kp_rgb.astype(np.uint8),
                              fuse_xyz, fuse_rgb.astype(np.uint8)))
        if (i + 1) % 50 == 0:
            print(f"[moge3-sfm] MoGe inference + lift {i + 1}/{len(image_paths)}")
    print(f"[moge3-sfm] MoGe-3 done: {len(out)} frames")
    return out


# ----------------------------------------------------------------------------
# Correspondences (hloc: SuperPoint + NetVLAD retrieval + LightGlue)
# ----------------------------------------------------------------------------

def _run_hloc(image_dir: Path, work_dir: Path, image_names: list[str], cfg: MoGe3SfMConfig):
    """Extract SuperPoint features, pick pairs (retrieval + sequential), match with
    LightGlue. Returns (features_h5, matches_h5, pairs[list[(a,b)]]).

    Uses only hloc.extract_features / pairs_from_* / match_features — none import
    pycolmap, so this is independent of the pycolmap 4.x model code below."""
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


# ----------------------------------------------------------------------------
# Pairwise metric pose + global pose graph (utils3d_moge, convention verified)
# ----------------------------------------------------------------------------

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


# ----------------------------------------------------------------------------
# Pose engine 2: ICP odometry on MoGe metric clouds
# ----------------------------------------------------------------------------

def _lift_frame(points, mask, rgb, kp, cfg, rng) -> "_FrameGeom":
    """Compact per-frame geometry (keypoint 3D + fusion subsample) — the ICP and pose-graph
    engines both need this for triangulation/BA/fallback. `points` is MoGe's (ph,pw,3)."""
    h, w = rgb.shape[:2]
    ph, pw = points.shape[:2]
    sx, sy = pw / w, ph / h
    kx = np.clip(np.round(kp[:, 0] * sx).astype(int), 0, pw - 1)
    ky = np.clip(np.round(kp[:, 1] * sy).astype(int), 0, ph - 1)
    kp3d = points[ky, kx]
    kp_valid = mask[ky, kx] & np.isfinite(kp3d).all(-1)
    kp_rgb = rgb[np.clip(kp[:, 1].astype(int), 0, h - 1), np.clip(kp[:, 0].astype(int), 0, w - 1)]
    vy, vx = np.nonzero(mask & np.isfinite(points).all(-1))
    if vy.size:
        sel = rng.choice(vy.size, min(cfg.fuse_per_frame, vy.size), replace=False)
        fy, fx = vy[sel], vx[sel]
        fuse_xyz = points[fy, fx].astype(np.float32)
        fuse_rgb = rgb[np.clip((fy / sy).astype(int), 0, h - 1), np.clip((fx / sx).astype(int), 0, w - 1)]
    else:
        fuse_xyz, fuse_rgb = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return kp3d, kp_valid, kp_rgb.astype(np.uint8), fuse_xyz, fuse_rgb.astype(np.uint8)


def _icp_odometry(image_paths, kpts, cfg: MoGe3SfMConfig):
    """MoGe-3 inference + ICP odometry: chain point-to-plane ICP of consecutive metric clouds
    (frame i registered to frame i-1) into cam_from_world poses, accumulated from frame 0.
    Returns (geoms, poses_w2c (N,4,4)). Streaming — only the previous cloud is kept."""
    import torch
    import open3d as o3d
    from moge.model import import_model_class_by_version
    import utils3d_moge as u3d

    device = torch.device(cfg.device)
    model = (import_model_class_by_version(cfg.moge_version)
             .from_pretrained(_resolve_moge_pretrained(cfg)).to(device).eval())
    rng = np.random.default_rng(0)

    def to_pcd(points, mask):
        pts = points[mask & np.isfinite(points).all(-1)].astype(np.float64)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd = pcd.voxel_down_sample(cfg.icp_voxel)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.icp_voxel * 3, max_nn=30))
        return pcd

    geoms: list[_FrameGeom] = []
    poses: list[np.ndarray] = []
    prev_pcd = None
    cfw = np.eye(4)                      # cam_from_world of the previous frame (frame 0 = world)
    prev_rel = np.eye(4)                 # last ACCEPTED frame_i->frame_{i-1} transform (constant-velocity prior)
    n_fallback = 0
    for i, q in enumerate(image_paths):
        rgb = np.array(Image.open(q).convert("RGB"))
        t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            pred = model.infer(t)
        points = pred["points"].float().cpu().numpy()
        mask = (pred["mask"].cpu().numpy().astype(bool) if "mask" in pred
                else np.isfinite(points).all(-1))
        K = u3d.np.denormalize_intrinsics(pred["intrinsics"].cpu().numpy(), (rgb.shape[1], rgb.shape[0]))
        kp3d, kp_valid, kp_rgb, fuse_xyz, fuse_rgb = _lift_frame(points, mask, rgb, kpts[i], cfg, rng)
        geoms.append(_FrameGeom(q.name, (rgb.shape[1], rgb.shape[0]), K,
                                kp3d, kp_valid, kp_rgb, fuse_xyz, fuse_rgb))

        pcd = to_pcd(points, mask)
        if prev_pcd is not None:
            # ICP source=frame i, target=frame i-1: T maps frame-i coords -> frame-(i-1) coords.
            # Init from the previous accepted motion (constant velocity), coarse then fine radius.
            T = prev_rel.copy()
            reg = None
            for dist in (cfg.icp_max_corr_dist, cfg.icp_max_corr_dist / 6.0):
                reg = o3d.pipelines.registration.registration_icp(
                    pcd, prev_pcd, dist, T,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.icp_max_iter))
                T = np.asarray(reg.transformation)
            # Gate: a bad registration (low overlap fitness or implausibly large motion) would jump
            # this camera and, since odometry is a chain, everything downstream. Reject it and coast
            # on the constant-velocity prior instead.
            if reg.fitness < cfg.icp_min_fitness or np.linalg.norm(T[:3, 3]) > cfg.icp_max_motion:
                T = prev_rel
                n_fallback += 1
            else:
                prev_rel = T
            cfw = np.linalg.inv(T) @ cfw
        poses.append(cfw.copy())
        prev_pcd = pcd
        if (i + 1) % 50 == 0:
            print(f"[moge3-sfm] MoGe inference + ICP {i + 1}/{len(image_paths)}")
    print(f"[moge3-sfm] ICP odometry: {n_fallback} constant-velocity fallbacks (bad/implausible ICP)")
    print(f"[moge3-sfm] ICP odometry done: {len(geoms)} frames")
    return geoms, np.stack(poses)


# ----------------------------------------------------------------------------
# COLMAP model: poses + points3D, optional bundle adjustment, export
# ----------------------------------------------------------------------------

def _shared_pinhole_camera(geoms: list[_FrameGeom]) -> pycolmap.Camera:
    """One PINHOLE camera (median of per-frame intrinsics). A shared camera matches a
    single-lens video capture and is required by COLMAP loaders that key images by camera."""
    sizes = {g.size for g in geoms}
    if len(sizes) != 1:
        raise RuntimeError(f"MoGe3 SfM expects a uniform image size, got {sizes}.")
    W, H = sizes.pop()
    med = np.median(np.stack([[g.K[0, 0], g.K[1, 1], g.K[0, 2], g.K[1, 2]] for g in geoms]), 0)
    cam = pycolmap.Camera()
    cam.camera_id = 1
    cam.model = pycolmap.CameraModelId.PINHOLE
    cam.width, cam.height = W, H
    cam.params = np.asarray(med, dtype=np.float64)
    print(f"[moge3-sfm] shared PINHOLE fx={med[0]:.1f} fy={med[1]:.1f} cx={med[2]:.1f} cy={med[3]:.1f}")
    return cam


def _base_reconstruction(geoms, poses_w2c, cam) -> pycolmap.Reconstruction:
    """Reconstruction with the shared camera + one Rig/Frame/Image per view at the
    pose-graph poses (pycolmap 4.x Rig/Frame construction)."""
    recon = pycolmap.Reconstruction()
    recon.add_camera(cam)
    rig = pycolmap.Rig(); rig.rig_id = 1; rig.add_ref_sensor(cam.sensor_id)
    recon.add_rig(rig)
    for i, g in enumerate(geoms):
        w2c = poses_w2c[i]
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(w2c[:3, :3]), w2c[:3, 3])
        image = pycolmap.Image(); image.image_id = i + 1; image.camera_id = 1
        frame = pycolmap.Frame(); frame.frame_id = i + 1; frame.rig_id = 1
        frame.add_data_id(image.data_id); frame.rig_from_world = cam_from_world
        recon.add_frame(frame)
        image.frame_id = i + 1; image.name = g.name
        recon.add_image(image)
    return recon


def _fuse_moge_points(recon, geoms, poses_w2c, cfg):
    """Fallback points3D (no BA): fuse the per-frame MoGe subsamples into world, like
    a COLMAP init cloud (empty tracks; consumers re-triangulate/densify)."""
    pts_chunks, col_chunks = [], []
    for i, g in enumerate(geoms):
        if g.fuse_xyz.shape[0] == 0:
            continue
        c2w = np.linalg.inv(poses_w2c[i])
        Xw = (c2w[:3, :3] @ g.fuse_xyz.T + c2w[:3, 3:4]).T
        pts_chunks.append(Xw.astype(np.float32))
        col_chunks.append(g.fuse_rgb)
    pts = np.concatenate(pts_chunks, 0); cols = np.concatenate(col_chunks, 0)
    fin = np.isfinite(pts).all(1); pts, cols = pts[fin], cols[fin]
    if pts.shape[0] > cfg.points_max:
        sel = np.random.default_rng(0).choice(pts.shape[0], cfg.points_max, replace=False)
        pts, cols = pts[sel], cols[sel]
    for xyz, rgb in zip(pts, cols):
        recon.add_point3D(xyz.astype(np.float64), pycolmap.Track(), rgb.astype(np.uint8))
    print(f"[moge3-sfm] fused {recon.num_points3D()} MoGe init points")


def _write_sparse0(recon, output_dir: Path):
    sparse_out = output_dir / "sparse" / "0"
    sparse_out.mkdir(parents=True, exist_ok=True)
    recon.write_binary(str(sparse_out))
    recon.write_text(str(sparse_out))
    print(f"[moge3-sfm] wrote {sparse_out} ({recon.num_reg_images()} images, "
          f"{recon.num_points3D()} points; text + binary)")


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


def _matches_to_kept(pairs, matches_h5, geoms, name_to_idx, cfg):
    """Per-pair inlier keypoint indices (kept for BA triangulation) straight from the matches,
    keeping only correspondences with valid MoGe 3D on both sides. No 3D-3D pose filtering —
    triangulation-angle + cheirality + reprojection filtering in bundle_adjust handles outliers."""
    from hloc.utils.io import get_matches
    kept: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
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


def run_moge3_sfm(image_dir: Path, output_dir: Path, cfg: MoGe3SfMConfig | None = None):
    """MoGe-3 metric SfM → COLMAP sparse/0. See module docstring for the pipeline."""
    cfg = cfg or MoGe3SfMConfig()
    image_dir = Path(image_dir).resolve()
    output_dir = Path(output_dir).resolve()

    image_paths = _sorted_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"no .jpg/.png images in {image_dir}")
    names = [p.name for p in image_paths]
    name_to_idx = {n: i for i, n in enumerate(names)}

    # hloc gives the 2D matches BA triangulates (both engines); MoGe supplies the geometry.
    features, matches, pairs = _run_hloc(image_dir, output_dir / "hloc", names, cfg)
    from hloc.utils.io import get_keypoints
    kpts = [get_keypoints(features, n) for n in names]

    if cfg.pose_engine == "icp":
        geoms, poses_w2c = _icp_odometry(image_paths, kpts, cfg)
        kept = _matches_to_kept(pairs, matches, geoms, name_to_idx, cfg)
    elif cfg.pose_engine == "pose_graph":
        geoms, kpts, kept, poses_w2c = _pose_graph_engine(
            image_paths, kpts, matches, pairs, name_to_idx, cfg)
    else:
        raise ValueError(f"unknown pose_engine {cfg.pose_engine!r} (want 'icp' | 'pose_graph')")

    cam = _shared_pinhole_camera(geoms)
    recon = _base_reconstruction(geoms, poses_w2c, cam)

    ba_ok = False
    if cfg.bundle_adjust:
        # Triangulate tracks at the init poses, then robust pycolmap BA refines poses + points
        # on 2D reprojection (removes odometry drift; corrects MoGe depth bias).
        from moge.sfm.bundle_adjust import triangulate_and_ba
        ba_ok = triangulate_and_ba(recon, geoms, kpts, kept, cfg)
        if not ba_ok:
            print("[moge3-sfm] BA unusable — falling back to init poses + MoGe point fusion")
            recon = _base_reconstruction(geoms, poses_w2c, cam)
    if not ba_ok:
        _fuse_moge_points(recon, geoms, poses_w2c, cfg)

    _write_sparse0(recon, output_dir)
    return recon


def load_config(config_path: Path | None) -> MoGe3SfMConfig:
    """MoGe3SfMConfig from the `sfm:` block of a YAML file (or defaults if None)."""
    kwargs: dict = {}
    if config_path is not None and Path(config_path).exists():
        kwargs = (yaml.safe_load(Path(config_path).read_text()) or {}).get("sfm", {})
    return MoGe3SfMConfig(**kwargs)
