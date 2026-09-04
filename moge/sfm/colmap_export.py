"""Build the COLMAP model (shared PINHOLE camera + Rig/Frame poses) and write sparse/0.

Also holds the BA-fallback point fusion: when bundle adjustment is off or diverges, fuse
the per-frame MoGe subsamples into world as an init cloud (empty tracks), so the model
still ships a points3D."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap

from moge.sfm.moge_infer import _FrameGeom


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
