"""MoGe-3 SfM configuration + YAML loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


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
    # 'icp': globally-optimized ICP POSE GRAPH. Consecutive-frame ICP = odometry edges +
    #        retrieval loop-closure edges (3D-3D-initialised, ICP-refined) -> Open3D global
    #        optimization (robust line process closes loops, spreads drift, rejects bad edges)
    #        -> gravity align (rotate so MoGe floor-normal = up, enforcing the planar-room prior).
    #        Replaces pure odometry chaining, which drifted into a collapsed, non-planar blob.
    # 'pose_graph': pairwise 3D correspondences only -> global pose graph. Fragile on small
    #        baselines (cameras collapse); kept for non-sequential captures.
    pose_engine: str = "icp"
    icp_voxel: float = 0.03            # metres — voxel-downsample each cloud for the fine ICP refine
    coarse_voxel: float = 0.15         # metres — coarse downsample (~1-2k pts) for FPFH + FGR global reg
    icp_max_corr_dist: float = 0.30    # metres — ICP correspondence radius (coarse pass; fine = /6)
    icp_max_iter: int = 60
    icp_min_fitness: float = 0.30      # odometry edge below this overlap fitness -> low-weight + coast
    icp_max_motion: float = 1.00       # metres — odometry translation above this -> coast + down-weight
    scale_normalize: bool = True       # pin each MoGe cloud to a common scale (median depth) — MoGe's
                                       # per-frame metric scale drifts, ballooning the rigid reconstruction

    # --- pose-graph loop closure + global optimization (icp engine) ---
    min_loop_inliers: int = 30         # 3D-3D match inliers required to attempt a loop edge
    icp_loop_min_fitness: float = 0.40 # keep a loop edge only above this ICP overlap fitness
    icp_loop_max_rmse: float = 0.08    # metres — and below this ICP inlier RMSE (reject bad loops)
    posegraph_prune_threshold: float = 0.25  # Open3D line-process edge-prune threshold
    gravity_align: bool = True         # rotate the solved scene so averaged MoGe floor-normal = world up

    # --- pairwise pose + global alignment (pose_graph engine only) ---
    # 'rigid': MoGe-3 is metric (scale baked in), so per-image scale drift is small and BA
    # on 2D reprojection makes the final model scale-consistent regardless of the init. Also
    # avoids a utils3d_moge bug in solve_pose(mode='similar') — s*R fails to broadcast
    # ((B,S) vs (B,S,3,3)) on the batched RANSAC path.
    pose_mode: str = "rigid"           # 'rigid' | 'similar' (latter is currently broken upstream)
    ransac_threshold: float = 0.05     # metres, solve_pose_ransac inlier threshold (pose_graph engine)
    # icp-engine rough registration derives ROTATION from the depth-free 2D essential matrix (immune
    # to MoGe per-keypoint depth noise that pulls a 3D-3D fit onto the wrong wall -> >30 deg poses);
    # MoGe depth supplies only the metric scale. Threshold is in CALIBRATED (normalized) coords.
    epipolar_thresh: float = 0.002     # ~2 px at f~=1000; essential-matrix RANSAC inlier gate
    min_pair_inliers: int = 20         # drop a pair edge below this many essential inliers
    min_pair_inlier_ratio: float = 0.5 # AND drop it below this inlier FRACTION of the matches
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


def load_config(config_path: Path | None) -> MoGe3SfMConfig:
    """MoGe3SfMConfig from the `sfm:` block of a YAML file (or defaults if None)."""
    kwargs: dict = {}
    if config_path is not None and Path(config_path).exists():
        kwargs = (yaml.safe_load(Path(config_path).read_text()) or {}).get("sfm", {})
    return MoGe3SfMConfig(**kwargs)
