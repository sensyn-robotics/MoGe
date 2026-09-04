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


def load_config(config_path: Path | None) -> MoGe3SfMConfig:
    """MoGe3SfMConfig from the `sfm:` block of a YAML file (or defaults if None)."""
    kwargs: dict = {}
    if config_path is not None and Path(config_path).exists():
        kwargs = (yaml.safe_load(Path(config_path).read_text()) or {}).get("sfm", {})
    return MoGe3SfMConfig(**kwargs)
