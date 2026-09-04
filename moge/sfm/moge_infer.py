"""MoGe-3 per-image inference + lifting keypoints to metric 3D.

MoGe-3 is single-image: per image it predicts a metric point map (OpenCV camera
frame) + intrinsics + mask, but no cross-image poses. This module runs the model and
produces the compact `_FrameGeom` (per-keypoint 3D + a small fusion subsample) that the
pose engines, triangulation, and BA-fallback consume — never the full (H,W,3) map, which
is a RAM bomb at the 1215-frame scale."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from moge.sfm.config import MoGe3SfMConfig

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


def _lift_frame(points, mask, rgb, kp, cfg, rng):
    """Compact per-frame geometry (keypoint 3D + fusion subsample) — the ICP and pose-graph
    engines both need this for triangulation/BA/fallback. `points` is MoGe's (ph,pw,3).
    Returns (kp3d, kp_valid, kp_rgb, fuse_xyz, fuse_rgb)."""
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
        # denormalize_intrinsics wants size as (height, width); passing (w, h) transposes
        # the intrinsics (swaps fx/fy and cx/cy) and produces a broken camera.
        K = u3d.np.denormalize_intrinsics(pred["intrinsics"].cpu().numpy(), (h, w))
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
