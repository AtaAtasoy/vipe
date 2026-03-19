"""
Convert ViPE artifact outputs to the 4D scene format expected by the
agentic-cinematography scene-processing scripts.

Output directory structure
--------------------------
output_dir/
  frame_{idx:04d}.ply              world-space point cloud (H×W points, pixel-aligned)
  frame_{idx:04d}.png              RGB frame (uint8)
  frame_{idx:04d}.npy              metric depth map (float32, H×W)   [--no_depth_npy to skip]
  conf_{idx}.npy                   depth reliability (float32, H×W, 0.0 or 1.0)
  enlarged_dynamic_mask_{idx}.png  binary foreground mask (instance>0 → 255)  [--no_masks to skip]
  pred_traj.txt                    TUM pose: "ts tx ty tz qw qx qy qz" per line
  pred_intrinsics.txt              camera K: "fx 0 cx 0 fy cy 0 0 1" per line (9 values)

Usage
-----
python scripts/vipe_to_scene.py \\
    --artifacts_dir vipe_results/ \\
    --artifact_name my_video \\
    --output_dir scene_output/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Make sure the vipe package is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vipe.utils.cameras import CameraType
from vipe.utils.depth import reliable_depth_mask_range
from vipe.utils.io import (
    ArtifactPath,
    read_depth_artifacts,
    read_instance_artifacts,
    read_intrinsics_artifacts,
    read_pose_artifacts,
    read_rgb_artifacts,
)


# ──────────────────────────────────────────────────────────────────────────────
# Core per-frame processing
# ──────────────────────────────────────────────────────────────────────────────


def _unproject_depth(
    depth: torch.Tensor,
    intr: torch.Tensor,
    camera_type: CameraType,
    orig_H: int | None = None,
    orig_W: int | None = None,
) -> np.ndarray:
    """
    Unproject a depth map into camera-space 3D points (H, W, 3).

    Mirrors the logic in vipe/utils/viser.py lines 255-272 exactly so that
    point clouds match what the interactive viewer displays.

    Args:
        depth:  (H, W) metric depth in metres (may be subsampled).
        intr:   intrinsics tensor (fx, fy, cx, cy, ...) in original pixel units.
        camera_type: PINHOLE, MEI, PANORAMA, etc.
        orig_H: original (pre-subsampling) height — if provided the pixel coordinates
                are scaled to match the original image grid so the intrinsics stay correct.
        orig_W: original (pre-subsampling) width.

    Returns:
        (H, W, 3) float32 array of camera-space XYZ.  NaN/inf depth maps to (0,0,0).
    """
    H, W = depth.shape
    camera_model = camera_type.build_camera_model(intr)

    if orig_H is not None and orig_W is not None and (orig_H != H or orig_W != W):
        # Generate pixel coordinates in the original image grid (subsampled rows/cols)
        stride_v = orig_H / H
        stride_u = orig_W / W
        disp_v, disp_u = torch.meshgrid(
            torch.arange(H, dtype=torch.float32) * stride_v,
            torch.arange(W, dtype=torch.float32) * stride_u,
            indexing="ij",
        )
    else:
        disp_v, disp_u = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )

    if camera_type == CameraType.PANORAMA:
        # Normalise to [0, 1] using the *original* grid dimensions
        full_H = orig_H if orig_H is not None else H
        full_W = orig_W if orig_W is not None else W
        disp_v = disp_v / (full_H - 1)
        disp_u = disp_u / (full_W - 1)

    disp = torch.ones_like(disp_v)
    pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
    rays = pts[..., :3].numpy()  # (H, W, 3)
    if camera_type != CameraType.PANORAMA:
        rays = rays / rays[..., 2:3]  # normalise so z=1

    depth_np = depth.numpy()
    # Replace NaN/inf with 0 so they map to the camera origin (filtered by conf later)
    depth_safe = np.where(np.isfinite(depth_np), depth_np, 0.0)
    pcd_cam = rays * depth_safe[..., None]  # (H, W, 3)
    return pcd_cam.astype(np.float32)


def _cam_to_world(pcd_cam: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """
    Transform (H, W, 3) camera-space points to world space using the 4×4 c2w matrix.

    Returns (H*W, 3) float32 array.
    """
    H, W, _ = pcd_cam.shape
    flat = pcd_cam.reshape(-1, 3).astype(np.float64)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    world = flat @ R.T + t
    return world.astype(np.float32)


def _save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """
    Save an (N, 3) point cloud + (N, 3) uint8 RGB as a PLY file.

    Uses the same plyfile library that postprocess_scene.py uses for loading.
    """
    N = points.shape[0]
    vertex_data = np.empty(
        N,
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]

    el = PlyElement.describe(vertex_data, "vertex")
    PlyData([el]).write(str(path))


def _make_conf(depth: torch.Tensor) -> np.ndarray:
    """
    Build a float32 confidence map from the depth reliability mask.

    Unreliable pixels and NaN/inf depth pixels both get conf=0.0.
    All other pixels get conf=1.0.

    The default conf_threshold in postprocess_scene.py is 0.1, so anything
    above that passes — setting reliable pixels to 1.0 is fine.
    """
    rel = reliable_depth_mask_range(depth).numpy().astype(np.float32)
    # Also zero out positions where depth itself is invalid
    rel[~np.isfinite(depth.numpy())] = 0.0
    return rel


def _c2w_to_tum_line(frame_idx: int, c2w: np.ndarray) -> str:
    """
    Convert a 4×4 cam-to-world matrix to a TUM trajectory line.

    The custom TUM convention used by postprocess_scene.py / create_motion_overlay.py:
        timestamp  tx  ty  tz  qw  qx  qy  qz
    (qw comes first among the quaternion components, unlike standard TUM where qw is last).
    """
    t = c2w[:3, 3]
    qxyzw = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # scipy returns (x, y, z, w)
    qw, qx, qy, qz = qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]
    return f"{frame_idx:.6f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f}"


def _intr_to_K(intr: torch.Tensor) -> np.ndarray:
    """Build a 3×3 K matrix from a [fx, fy, cx, cy, ...] intrinsics vector."""
    fx, fy, cx, cy = intr[0].item(), intr[1].item(), intr[2].item(), intr[3].item()
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


# ──────────────────────────────────────────────────────────────────────────────
# Main conversion routine
# ──────────────────────────────────────────────────────────────────────────────


def vipe_to_scene(
    artifacts_dir: Path,
    artifact_name: str,
    output_dir: Path,
    save_depth_npy: bool = True,
    save_masks: bool = True,
    stride: int = 4,
) -> None:
    artifact = ArtifactPath(artifacts_dir, artifact_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pose and intrinsics ────────────────────────────────────────────
    pose_inds, pose_traj = read_pose_artifacts(artifact.pose_path)
    c2w_matrices = pose_traj.matrix().numpy()  # (N, 4, 4)

    intr_inds, intrinsics, camera_types = read_intrinsics_artifacts(
        artifact.intrinsics_path, artifact.camera_type_path
    )

    # Build per-frame lookups (frame_idx → data)
    pose_by_idx = {int(idx): c2w_matrices[i] for i, idx in enumerate(pose_inds)}
    # Intrinsics may be one global set or per-frame; align by intr_inds
    intr_by_idx = {int(idx): (intrinsics[i], camera_types[i]) for i, idx in enumerate(intr_inds)}

    # ── Depth iterator ──────────────────────────────────────────────────────
    depth_by_idx: dict[int, torch.Tensor] = {}
    try:
        for didx, depth in read_depth_artifacts(artifact.depth_path):
            depth_by_idx[int(didx)] = depth
    except FileNotFoundError:
        print(f"[warn] No depth artifacts found at {artifact.depth_path} — skipping depth.")

    # ── RGB iterator ────────────────────────────────────────────────────────
    rgb_by_idx: dict[int, np.ndarray] = {}
    try:
        for ridx, rgb in read_rgb_artifacts(artifact.rgb_path):
            rgb_by_idx[int(ridx)] = (rgb.numpy() * 255).astype(np.uint8)
    except FileNotFoundError:
        print(f"[warn] No RGB artifacts found at {artifact.rgb_path}.")

    # ── Instance mask iterator ──────────────────────────────────────────────
    mask_by_idx: dict[int, np.ndarray] = {}
    if save_masks and artifact.mask_path.exists():
        for midx, mask in read_instance_artifacts(artifact.mask_path):
            mask_by_idx[int(midx)] = mask.numpy()

    # ── Per-frame export ────────────────────────────────────────────────────
    frame_indices = sorted(pose_by_idx.keys())
    traj_lines: list[str] = []
    K_lines: list[str] = []

    print(f"Exporting {len(frame_indices)} frames to {output_dir} …")
    for frame_idx in tqdm(frame_indices):
        c2w = pose_by_idx[frame_idx]

        # Intrinsics: fall back to last available if this frame has none
        intr_key = frame_idx if frame_idx in intr_by_idx else max(k for k in intr_by_idx if k <= frame_idx)
        intr, camera_type = intr_by_idx[intr_key]

        # TUM trajectory line
        traj_lines.append(_c2w_to_tum_line(frame_idx, c2w))

        # Intrinsics K matrix line
        K = _intr_to_K(intr)
        K_lines.append(" ".join(f"{v:.8f}" for v in K.flatten()))

        # ── RGB ──────────────────────────────────────────────────────────
        rgb_np = rgb_by_idx.get(frame_idx)
        if rgb_np is not None:
            H, W = rgb_np.shape[:2]
            Image.fromarray(rgb_np).save(output_dir / f"frame_{frame_idx:04d}.png")
            rgb_sub = rgb_np[::stride, ::stride]  # (H', W', 3) subsampled
        else:
            H, W = None, None
            rgb_sub = None

        # ── Depth + point cloud ──────────────────────────────────────────
        depth = depth_by_idx.get(frame_idx)
        if depth is not None:
            if H is None:
                H, W = depth.shape

            depth_sub = depth[::stride, ::stride]  # (H', W') subsampled

            # Confidence map (saved at subsampled resolution for mask alignment)
            conf = _make_conf(depth_sub)
            np.save(output_dir / f"conf_{frame_idx}.npy", conf)

            # Depth NPY (raw metric depth at original resolution, in metres)
            if save_depth_npy:
                np.save(output_dir / f"frame_{frame_idx:04d}.npy", depth.numpy().astype(np.float32))

            # Unproject subsampled depth → world-space point cloud.
            # Pass the original H/W so pixel coordinates are correct for the intrinsics.
            pcd_cam = _unproject_depth(depth_sub, intr, camera_type, orig_H=H, orig_W=W)  # (H', W', 3)
            pcd_world = _cam_to_world(pcd_cam, c2w)  # (H'*W', 3)

            # Colors at subsampled resolution
            if rgb_sub is not None:
                colors = rgb_sub.reshape(-1, 3)
            else:
                Hs, Ws = depth_sub.shape
                colors = np.full((Hs * Ws, 3), 200, dtype=np.uint8)

            _save_ply(output_dir / f"frame_{frame_idx:04d}.ply", pcd_world, colors)

        # ── Dynamic / foreground mask (saved at subsampled resolution) ────
        if save_masks:
            raw_mask = mask_by_idx.get(frame_idx)
            if raw_mask is not None:
                mask_sub = raw_mask[::stride, ::stride]
                binary = ((mask_sub > 0).astype(np.uint8) * 255)
                cv2.imwrite(str(output_dir / f"enlarged_dynamic_mask_{frame_idx}.png"), binary)

    # ── Write trajectory and intrinsics text files ──────────────────────────
    (output_dir / "pred_traj.txt").write_text("\n".join(traj_lines) + "\n")
    (output_dir / "pred_intrinsics.txt").write_text("\n".join(K_lines) + "\n")

    print(f"Done. Scene written to {output_dir}")
    print(f"  Trajectory : pred_traj.txt  ({len(traj_lines)} poses)")
    print(f"  Intrinsics : pred_intrinsics.txt")
    ply_count = len(list(output_dir.glob("frame_*.ply")))
    print(f"  PLY files  : {ply_count}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Convert ViPE outputs to the 4D scene format used by the "
                    "agentic-cinematography scene-processing scripts."
    )
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        required=True,
        help="Root ViPE output directory (contains pose/, depth/, rgb/, intrinsics/ sub-dirs).",
    )
    parser.add_argument(
        "--artifact_name",
        type=str,
        required=True,
        help="Artifact stem name (e.g. 'my_video'), matching the filenames inside the sub-dirs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write the scene files into.",
    )
    parser.add_argument(
        "--no_depth_npy",
        action="store_true",
        help="Skip saving per-frame depth .npy files (saves disk space).",
    )
    parser.add_argument(
        "--no_masks",
        action="store_true",
        help="Skip exporting instance masks as enlarged_dynamic_mask_*.png.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Spatial subsampling stride for point clouds and masks (default: 4). "
             "stride=1 keeps all pixels; stride=4 keeps every 4th row and column "
             "(16× fewer points, ~16× smaller PLY files).",
    )
    args = parser.parse_args()

    if not args.artifacts_dir.is_dir():
        parser.error(f"artifacts_dir does not exist: {args.artifacts_dir}")

    vipe_to_scene(
        artifacts_dir=args.artifacts_dir,
        artifact_name=args.artifact_name,
        output_dir=args.output_dir,
        save_depth_npy=not args.no_depth_npy,
        save_masks=not args.no_masks,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
