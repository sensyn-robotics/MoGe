"""MoGe-3 Structure-from-Motion — recover camera poses + a COLMAP model from images.

MoGe-3 predicts excellent per-image metric geometry but no cross-image poses. This
subpackage adds the pose recovery (learned matching -> per-pair metric relative pose ->
global pose graph -> triangulation + bundle adjustment) and writes a standard COLMAP
``sparse/0`` model. See ``moge.sfm.pipeline`` for the full pipeline.

Install the extra dependencies with::

    uv sync --extra sfm

## Dependency licenses — all permissive, compatible with MoGe's MIT license

| Dependency        | Role                        | License                 |
|-------------------|-----------------------------|-------------------------|
| MoGe-3            | metric per-image geometry   | MIT                     |
| utils3d           | pose solvers / geometry     | MIT                     |
| pycolmap / COLMAP | model, triangulation, BA    | BSD-3-Clause            |
| hloc              | matching orchestration      | Apache-2.0              |
| kornia            | ALIKED/DISK implementation  | Apache-2.0              |
| ALIKED (default)  | learned keypoints           | BSD-3-Clause            |
| LightGlue         | learned matcher             | Apache-2.0              |
| DISK (alt)        | learned keypoints           | Apache-2.0              |
| OpenIBL / MegaLoc | retrieval (loop closure)    | MIT                     |

**Excluded on purpose:** SuperPoint / SuperGlue (Magic Leap) are **NONCOMMERCIAL research
use only** and are NOT used here — wiring them in would taint MoGe's MIT license. The
defaults (ALIKED + LightGlue + OpenIBL) are all permissive.
"""

from moge.sfm.pipeline import MoGe3SfMConfig, load_config, run_moge3_sfm

__all__ = ["MoGe3SfMConfig", "load_config", "run_moge3_sfm"]
