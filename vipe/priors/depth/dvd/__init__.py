# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import numpy as np
import torch

from vipe.utils.misc import unpack_optional

from ..base import DepthEstimationInput, DepthEstimationModel, DepthEstimationResult, DepthType

logger = logging.getLogger(__name__)

# Path to the DVD submodule root
_DVD_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "DVD")


class DVDDepthModel(DepthEstimationModel):
    """
    DVD: Depth from Video Diffusion
    https://github.com/EnVision-Research/DVD

    Adapts a pre-trained video diffusion model into a deterministic single-pass depth regressor.
    Outputs affine-invariant inverse depth (relative disparity).
    """

    def __init__(self, window_size: int = 45, overlap: int = 9) -> None:
        super().__init__()

        # Add DVD to sys.path so its internal imports resolve
        dvd_root = os.path.abspath(_DVD_ROOT)
        if dvd_root not in sys.path:
            sys.path.insert(0, dvd_root)

        try:
            from accelerate import Accelerator
            from omegaconf import OmegaConf
            from safetensors.torch import load_file
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"DVD dependencies not found ({e}). Install via: pip install accelerate omegaconf safetensors"
            ) from e

        try:
            from examples.wanvideo.model_training.WanTrainingModule import WanTrainingModule
        except ModuleNotFoundError as e:
            if "examples" in str(e):
                raise RuntimeError(
                    f"DVD submodule not found ({e}). Make sure DVD is cloned: git submodule update --init DVD"
                ) from e
            else:
                raise RuntimeError(
                    f"DVD dependency missing ({e}). Install DVD requirements: pip install -r DVD/requirements.txt"
                ) from e

        # Resolve checkpoint directory
        ckpt_dir = os.environ.get("DVD_CKPT_DIR", None)
        if ckpt_dir is None:
            # Auto-download from HuggingFace
            from huggingface_hub import snapshot_download

            ckpt_dir = snapshot_download(repo_id="FayeHongfeiZhang/DVD", revision="main")
            logger.info(f"DVD checkpoint downloaded to: {ckpt_dir}")

        # Load model config
        config_path = os.path.join(dvd_root, "ckpt", "model_config.yaml")
        yaml_args = OmegaConf.load(config_path)

        # Initialize model
        accelerator = Accelerator()
        self.model = WanTrainingModule(
            accelerator=accelerator,
            model_id_with_origin_paths=yaml_args.model_id_with_origin_paths,
            trainable_models=None,
            use_gradient_checkpointing=False,
            lora_rank=yaml_args.lora_rank,
            lora_base_model=yaml_args.lora_base_model,
            args=yaml_args,
        )

        # Load checkpoint weights
        ckpt_path = os.path.join(ckpt_dir, "model.safetensors")
        state_dict = load_file(ckpt_path, device="cpu")
        dit_state_dict = {k.replace("pipe.dit.", ""): v for k, v in state_dict.items() if "pipe.dit." in k}
        self.model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
        self.model.merge_lora_layer()
        self.model = self.model.to("cuda")

        self.window_size = window_size
        self.overlap = overlap

    @property
    def depth_type(self) -> DepthType:
        return DepthType.AFFINE_DISP

    def estimate(self, src: DepthEstimationInput) -> DepthEstimationResult:
        from .inference import generate_depth_sliced, resize_depth_back, resize_for_training_scale

        frame_list: list[np.ndarray] = unpack_optional(src.video_frame_list)

        # Convert list of HWC float32 [0,1] arrays → [1, T, C, H, W] tensor
        frames_np = np.stack(frame_list, axis=0)  # [T, H, W, 3]
        input_rgb = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()  # [T, C, H, W]
        input_rgb = input_rgb.unsqueeze(0)  # [1, T, C, H, W]

        # Resize to meet DVD minimum dimensions (480x640, aligned to 16)
        input_rgb, orig_size = resize_for_training_scale(input_rgb)

        # Run inference
        with torch.no_grad():
            depth = generate_depth_sliced(self.model, input_rgb, self.window_size, self.overlap)

        # depth shape: [1, T, H, W, 3] → remove batch, resize, then extract single channel
        depth = depth[0]  # [T, H, W, 3]
        depth = resize_depth_back(depth, orig_size)  # [T, H_orig, W_orig, 3]
        depth = np.mean(depth, axis=-1)  # [T, H_orig, W_orig]

        depths = torch.from_numpy(depth).float().cuda()
        return DepthEstimationResult(relative_inv_depth=depths)
