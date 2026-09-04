"""Structure-from-Motion from MoGe-3 metric geometry + learned matching (orchestrator).

MoGe-3 is a single-image model: per image it predicts a **metric** point map (OpenCV
camera frame) + intrinsics + mask, but **no cross-image poses**. This package recovers
the poses MoGe lacks and writes a standard COLMAP ``sparse/0`` model (cameras + poses +
points), usable by any COLMAP consumer (3D Gaussian Splatting, MVS, visualization).

## Pipeline (this module wires the stages; each stage lives in its own module)
1. **MoGe-3 inference** (``moge_infer``): per image → metric ``points`` (H,W,3, camera
   frame), pixel ``intrinsics``, ``mask``, rgb (keypoints lifted immediately; map freed).
2. **Correspondences** (``matching``): learned features (ALIKED) + LightGlue over a
   sequential window + optional retrieval **loop-closure** pairs.
3. **Pose init** — one of two engines:
   - ``icp`` (``icp``): chain point-to-plane ICP of consecutive MoGe metric clouds.
   - ``pose_graph`` (``pose_graph``): pairwise metric relative pose → global pose graph.
4. **Triangulation + bundle adjustment** (``bundle_adjust``): triangulate inlier tracks at
   the init poses, then BA refines poses+points on true 2D reprojection (corrects MoGe's
   monocular-depth bias; removes odometry drift). Falls back to MoGe-point fusion if BA
   diverges/is disabled.
5. **Export** (``colmap_export``) ``<output>/sparse/0/``: one shared PINHOLE camera
   (median intrinsics), one Rig+Frame per image, written as both ``.bin`` and ``.txt``.

Dependencies beyond MoGe core (all permissive / MIT-compatible; install via the ``sfm``
extra): hloc (Apache-2.0), LightGlue (Apache-2.0), pycolmap/COLMAP (BSD-3-Clause), Open3D
(MIT), plus ALIKED features (BSD-3-Clause via kornia, Apache-2.0). See ``moge/sfm/__init__.py``
for the full dependency-license table. Do NOT wire in SuperPoint/SuperGlue — they are Magic
Leap NONCOMMERCIAL and would taint MoGe's MIT license.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

# Re-exported so `from moge.sfm.pipeline import MoGe3SfMConfig, load_config` keeps working.
from moge.sfm.config import MoGe3SfMConfig, load_config
from moge.sfm.matching import _matches_to_kept, _run_hloc
from moge.sfm.icp import _icp_odometry
from moge.sfm.pose_graph import _pose_graph_engine
from moge.sfm.colmap_export import (
    _base_reconstruction,
    _fuse_moge_points,
    _shared_pinhole_camera,
    _write_sparse0,
)

__all__ = ["MoGe3SfMConfig", "load_config", "run_moge3_sfm"]


def _sorted_images(image_dir: Path) -> list[Path]:
    """Deterministic image order (sorted .jpg then .png)."""
    d = str(image_dir)
    paths = glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png"))
    return [Path(p) for p in sorted(paths)]


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
