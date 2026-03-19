# DVD (Depth from Video Diffusion) Integration

DVD is a video depth estimation model that adapts a pre-trained video diffusion model (Wan2.1-T2V-1.3B) into a deterministic single-pass depth regressor. It outputs affine-invariant relative depth and is integrated as a video depth model in the `AdaptiveDepthProcessor`, where its output is aligned to metric depth via scale+shift fitting.

## Running the Pipeline with DVD

```bash
python run.py pipeline=dvd streams=raw_mp4_stream streams.base_path=VIDEO.mp4
```

Or via the CLI:

```bash
vipe infer VIDEO.mp4 --output vipe_results --visualize
# (after setting pipeline=dvd in config or override)
```

## Additional Dependencies

Install these in the `vipe` conda environment:

```bash
conda activate vipe
pip install accelerate diffusers peft sentencepiece tabulate modelscope
pip install "transformers>=4.45.0,<5.0.0"
```

The DVD checkpoint is auto-downloaded from HuggingFace (`FayeHongfeiZhang/DVD`) on first run. To download manually:

```bash
huggingface-cli download FayeHongfeiZhang/DVD --revision main --local-dir DVD/ckpt
```

Set `DVD_CKPT_DIR` environment variable to use a custom checkpoint location.

## Files Created

- **`vipe/priors/depth/dvd/__init__.py`** — `DVDDepthModel` class implementing the `DepthEstimationModel` interface with `DepthType.AFFINE_DISP`. Handles model loading (with HuggingFace auto-download), adds the DVD submodule to `sys.path` for imports, and converts between VIPE's frame list format and DVD's tensor format.

- **`vipe/priors/depth/dvd/inference.py`** — Adapted inference helpers from `DVD/test_script/test_single_video.py`: sliding window generation (`generate_depth_sliced`), temporal padding (`pad_time_mod4`), overlap alignment (`compute_scale_and_shift`), and resize utilities. Isolated from the DVD submodule to avoid fragile cross-module imports.

- **`configs/pipeline/dvd.yaml`** — Pipeline config inheriting from `default.yaml`, overriding `post.depth_align_model` to `"adaptive_unidepth-l_dvd"`.

## Files Modified

- **`vipe/pipeline/processors.py`** — Extended `AdaptiveDepthProcessor.__init__` to accept `"dvd"` as a video model option alongside `"svda"` and `"vda"`. When selected, it instantiates `DVDDepthModel` instead of `VideoDepthAnythingDepthModel`.

- **`vipe/priors/depth/__init__.py`** — Registered `"dvd"` in the `make_depth_model()` factory function.

## Architecture

DVD is integrated as a **video depth model** (not a SLAM keyframe depth model) because it outputs relative depth rather than metric depth. The integration follows the same pattern as VideoDepthAnything:

```
AdaptiveDepthProcessor
  |
  +-- metric depth model (unidepth-l) --> per-frame metric depth
  |
  +-- video depth model (DVD) ----------> temporally consistent relative depth
  |
  +-- align_inv_depth_to_depth() -------> aligned metric depth output
```

The `AdaptiveDepthProcessor` first evaluates SLAM map coverage. If coverage is sufficient, it uses the SLAM map as a prompt for PriorDA; otherwise, it falls back to the metric depth model. In both cases, DVD's video depth is aligned to the metric result via affine fitting with momentum-based smoothing.
