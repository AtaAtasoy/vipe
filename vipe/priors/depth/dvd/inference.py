# Adapted from DVD/test_script/test_single_video.py
# https://github.com/EnVision-Research/DVD

import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_scale_and_shift(curr_frames, ref_frames, mask=None):
    """Computes scale and shift for overlap alignment."""
    if mask is None:
        mask = np.ones_like(ref_frames)

    a_00 = np.sum(mask * curr_frames * curr_frames)
    a_01 = np.sum(mask * curr_frames)
    a_11 = np.sum(mask)
    b_0 = np.sum(mask * curr_frames * ref_frames)
    b_1 = np.sum(mask * ref_frames)

    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        scale = (a_11 * b_0 - a_01 * b_1) / det
        shift = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        scale, shift = 1.0, 0.0

    return scale, shift


def resize_for_training_scale(video_tensor, target_h=480, target_w=640):
    B, T, C, H, W = video_tensor.shape
    ratio = max(target_h / H, target_w / W)
    new_H = int(np.ceil(H * ratio))
    new_W = int(np.ceil(W * ratio))

    # Align to 16
    new_H = (new_H + 15) // 16 * 16
    new_W = (new_W + 15) // 16 * 16

    if new_H == H and new_W == W:
        return video_tensor, (H, W)

    video_reshape = video_tensor.view(B * T, C, H, W)
    resized = F.interpolate(video_reshape, size=(new_H, new_W), mode="bilinear", align_corners=False)
    resized = resized.view(B, T, C, new_H, new_W)
    return resized, (H, W)


def resize_depth_back(depth_np, orig_size):
    orig_H, orig_W = orig_size
    depth_tensor = torch.from_numpy(depth_np).permute(0, 3, 1, 2).float()
    depth_tensor = F.interpolate(depth_tensor, size=(orig_H, orig_W), mode="bilinear", align_corners=False)
    return depth_tensor.permute(0, 2, 3, 1).cpu().numpy()


def pad_time_mod4(video_tensor):
    """Pads the temporal dimension to satisfy 4n+1 requirement."""
    B, T, C, H, W = video_tensor.shape
    remainder = T % 4
    if remainder != 1:
        pad_len = (4 - remainder + 1) % 4
        pad_frames = video_tensor[:, -1:, :, :, :].repeat(1, pad_len, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, pad_frames], dim=1)
    return video_tensor, T


def get_window_index(T, window_size, overlap):
    if T <= window_size:
        return [(0, T)]
    res = [(0, window_size)]
    start = window_size - overlap
    while start < T:
        end = start + window_size
        if end < T:
            res.append((start, end))
            start += window_size - overlap
        else:
            start = max(0, T - window_size)
            res.append((start, T))
            break
    return res


def generate_depth_sliced(model, input_rgb, window_size=45, overlap=9, scale_only=False):
    B, T, C, H, W = input_rgb.shape
    depth_windows = get_window_index(T, window_size, overlap)
    logger.info(f"DVD depth windows: {depth_windows}")

    depth_res_list = []

    # 1. Inference per window
    for start, end in depth_windows:
        logger.debug(f"DVD inference window [{start}:{end}]")
        _input_rgb_slice = input_rgb[:, start:end]

        # Ensure 4n+1 padding
        _input_rgb_slice, origin_T = pad_time_mod4(_input_rgb_slice)
        _input_frame = _input_rgb_slice.shape[1]
        _input_height, _input_width = _input_rgb_slice.shape[-2:]

        outputs = model.pipe(
            prompt=[""] * B,
            negative_prompt=[""] * B,
            mode=model.args.mode,
            height=_input_height,
            width=_input_width,
            num_frames=_input_frame,
            batch_size=B,
            input_image=_input_rgb_slice[:, 0],
            extra_images=_input_rgb_slice,
            extra_image_frame_index=torch.ones([B, _input_frame]).to(model.pipe.device),
            input_video=_input_rgb_slice,
            cfg_scale=1,
            seed=0,
            tiled=False,
            denoise_step=model.args.denoise_step,
        )
        # Drop the padded frames
        depth_res_list.append(outputs["depth"][:, :origin_T])

    # 2. Overlap Alignment
    depth_list_aligned = None
    prev_end = None

    for i, (t, (start, end)) in enumerate(zip(depth_res_list, depth_windows)):
        if i == 0:
            depth_list_aligned = t
            prev_end = end
            continue

        curr_start = start
        real_overlap = prev_end - curr_start

        if real_overlap > 0:
            ref_frames = depth_list_aligned[:, -real_overlap:]
            curr_frames = t[:, :real_overlap]

            if scale_only:
                scale = np.sum(curr_frames * ref_frames) / (np.sum(curr_frames * curr_frames) + 1e-6)
                shift = 0.0
            else:
                scale, shift = compute_scale_and_shift(curr_frames, ref_frames)

            scale = np.clip(scale, 0.7, 1.5)

            aligned_t = t * scale + shift
            aligned_t[aligned_t < 0] = 0

            # Smooth blending
            alpha = np.linspace(0, 1, real_overlap, dtype=np.float32).reshape(1, real_overlap, 1, 1, 1)
            smooth_overlap = (1 - alpha) * ref_frames + alpha * aligned_t[:, :real_overlap]

            depth_list_aligned = np.concatenate(
                [depth_list_aligned[:, :-real_overlap], smooth_overlap, aligned_t[:, real_overlap:]], axis=1
            )
        else:
            depth_list_aligned = np.concatenate([depth_list_aligned, t], axis=1)

        prev_end = end

    # Crop to original length
    return depth_list_aligned[:, :T]
