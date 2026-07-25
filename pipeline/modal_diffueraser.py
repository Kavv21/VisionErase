"""Modal.com serverless DiffuEraser refinement.

DiffuEraser (https://github.com/lixiaowen-xw/DiffuEraser) is a diffusion video
inpainter that takes an existing inpainting result as its "priori" and refines
it. We feed it our ProPainter output rather than letting it run its own bundled
ProPainter, so the ROI cropping and post-processing in
workers/inpainting/chunk_tasks.py are preserved and we don't pay for the prior
twice.

Its real entry point takes three MP4 *files* — video, mask video and priori —
not frame directories, and sizes output with a single max_img_size (longest
side) rather than a width/height pair. See DiffuEraser/run_diffueraser.py and
DiffuEraser/diffueraser/diffueraser.py::forward.
"""
import modal

app = modal.App("visionerase-diffueraser")

# Versions are DiffuEraser's own pins (requirements.txt). diffusers in
# particular must stay at 0.29.2 — the repo subclasses pipeline internals that
# moved in later releases.
diffueraser_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git", "libgl1", "libglib2.0-0", "ffmpeg"])
    .pip_install([
        "torch==2.3.1",
        "torchvision==0.18.1",
        "diffusers==0.29.2",
        "transformers==4.41.1",
        "accelerate==0.25.0",
        "peft==0.13.2",
        "einops==0.8.0",
        "numpy==1.26.4",
        "opencv-python-headless==4.9.0.80",
        "pillow==10.4.0",
        "imageio==2.34.1",
        "scipy==1.13.1",
        "av==14.0.1",
        "tqdm==4.66.4",
    ])
    .run_commands(
        "git clone https://github.com/lixiaowen-xw/DiffuEraser /DiffuEraser",
    )
)

weights_volume = modal.Volume.from_name("diffueraser-weights", create_if_missing=True)


@app.function(
    image=diffueraser_image.pip_install("huggingface_hub==0.24.1"),
    volumes={"/weights": weights_volume},
    timeout=3600,
)
def download_weights(propainter_files: dict[str, bytes] | None = None) -> str:
    """Populate the weights volume, pulling straight from HuggingFace.

    Runs inside Modal rather than uploading ~10GB from a laptop. Only the
    ProPainter checkpoints come from the caller, since those live in this repo's
    model_weights and are not on the Hub as a single snapshot.

    Skips SD1.5's unet/ and vae/: DiffuEraser passes both to from_pretrained as
    already-constructed objects, so those folders are never read from disk.
    """
    import os

    from huggingface_hub import hf_hub_download, snapshot_download

    def size_gb(path):
        total = 0
        for root, _, files in os.walk(path):
            for name in files:
                fp = os.path.join(root, name)
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
        return total / 1e9

    print("[1/5] lixiaowen/diffuEraser (brushnet + unet_main)")
    snapshot_download(
        "lixiaowen/diffuEraser",
        local_dir="/weights/diffuEraser",
        allow_patterns=["brushnet/*", "unet_main/*"],
    )

    print("[2/5] stable-diffusion-v1-5 (scheduler/tokenizer/text_encoder only)")
    snapshot_download(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        local_dir="/weights/stable-diffusion-v1-5",
        allow_patterns=[
            "model_index.json", "scheduler/*", "tokenizer/*",
            "text_encoder/*.json", "text_encoder/*.safetensors",
            "feature_extractor/*",
            "safety_checker/*.json", "safety_checker/*.safetensors",
        ],
    )

    print("[3/5] stabilityai/sd-vae-ft-mse")
    snapshot_download(
        "stabilityai/sd-vae-ft-mse",
        local_dir="/weights/sd-vae-ft-mse",
        allow_patterns=["*.json", "diffusion_pytorch_model.safetensors"],
    )

    print("[4/5] PCM_Weights sd15 2-step LoRA")
    os.makedirs("/weights/PCM_Weights/sd15", exist_ok=True)
    src = hf_hub_download(
        "wangfuyun/PCM_Weights", "sd15/pcm_sd15_smallcfg_2step_converted.safetensors"
    )
    dst = "/weights/PCM_Weights/sd15/pcm_sd15_smallcfg_2step_converted.safetensors"
    if not os.path.exists(dst):
        import shutil
        shutil.copy(src, dst)

    print("[5/5] propainter checkpoints from the caller")
    os.makedirs("/weights/propainter", exist_ok=True)
    for name, blob in (propainter_files or {}).items():
        path = f"/weights/propainter/{name}"
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(blob)

    weights_volume.commit()
    report = "\n".join(
        f"  {d}: {size_gb(f'/weights/{d}'):.2f} GB"
        for d in sorted(os.listdir("/weights"))
        if os.path.isdir(f"/weights/{d}")
    )
    total = f"TOTAL {size_gb('/weights'):.2f} GB"
    print(report)
    print(total)
    return f"{report}\n{total}"

# DiffuEraser refuses clips shorter than this (diffueraser.py raises on
# n_total_frames < 22). Our chunks are 40 frames, but a trailing chunk can be
# shorter, so callers must check before paying for a container.
MIN_FRAMES = 22


@app.function(
    image=diffueraser_image,
    gpu="A10G",
    memory=20480,
    timeout=900,
    volumes={"/weights": weights_volume},
)
def refine_with_diffueraser(
    frames_bytes: list[bytes],
    masks_bytes: list[bytes],
    propainter_bytes: list[bytes],
    fps: float = 24.0,
    max_img_size: int = 960,
    mask_dilation_iter: int = 8,
) -> bytes:
    """Refine a ProPainter result with DiffuEraser, returning MP4 bytes.

    frames/masks/propainter are per-frame encoded images (JPEG, PNG, JPEG).
    They are muxed into the three MP4s DiffuEraser's API actually wants.
    """
    import os
    import subprocess
    import sys
    import tempfile
    import time

    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, "/DiffuEraser")

    T = len(frames_bytes)
    if T < MIN_FRAMES:
        raise ValueError(f"DiffuEraser needs >= {MIN_FRAMES} frames, got {T}")

    def _decode(buf, flag):
        return cv2.imdecode(np.frombuffer(buf, np.uint8), flag)

    frames = [_decode(b, cv2.IMREAD_COLOR) for b in frames_bytes]
    masks = [_decode(b, cv2.IMREAD_GRAYSCALE) for b in masks_bytes]
    priors = [_decode(b, cv2.IMREAD_COLOR) for b in propainter_bytes]

    H, W = frames[0].shape[:2]
    print(f"DiffuEraser: {T} frames at {W}x{H}, max_img_size={max_img_size}")
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"GPU free {free_b/1e9:.1f}GB of {total_b/1e9:.1f}GB at entry")

    # read_mask() compares mask fps against the video's with `!=`, and reads the
    # two files through different libraries (cv2 vs torchvision). A fractional
    # rate like 30000/1001 can round differently between them and abort the run,
    # so everything is muxed at one clean integer rate. Frame order and count —
    # all that actually matters here — are untouched.
    enc_fps = max(1, int(round(fps)))

    def _write_mp4(images, path, is_mask=False):
        """Mux BGR (or gray) frames into an MP4 at the shared encode rate."""
        h, w = images[0].shape[:2]
        pix = "gray" if is_mask else "bgr24"
        # Masks go out lossless: read_mask thresholds with `mask > 0`, so the
        # 1-3 value noise a lossy encode leaves in black regions would be read
        # as mask and bleed the inpainting across the whole frame.
        quality = ["-qp", "0"] if is_mask else ["-crf", "12"]
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
             "-s", f"{w}x{h}", "-pix_fmt", pix, "-r", str(enc_fps), "-i", "pipe:0",
             "-vcodec", "libx264", *quality, "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        for img in images:
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg mux failed for {path}: {err[-400:]}")

    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "video.mp4")
        mask_path = os.path.join(tmp, "mask.mp4")
        priori_path = os.path.join(tmp, "priori.mp4")
        output_path = os.path.join(tmp, "diffueraser_result.mp4")

        # The mask must be a video at the *same* fps as the source, else
        # read_mask resamples and misaligns it against the frames.
        _write_mp4(frames, video_path)
        _write_mp4(masks, mask_path, is_mask=True)
        _write_mp4(priors, priori_path)

        from diffueraser.diffueraser import DiffuEraser

        # load_lora_weights() resolves "weights/PCM_Weights" relatively, so the
        # process has to run from a directory where that path exists.
        os.chdir("/DiffuEraser")
        if not os.path.islink("/DiffuEraser/weights"):
            if os.path.isdir("/DiffuEraser/weights"):
                import shutil
                shutil.rmtree("/DiffuEraser/weights")
            os.symlink("/weights", "/DiffuEraser/weights")

        started = time.time()
        model = DiffuEraser(
            torch.device("cuda"),
            "weights/stable-diffusion-v1-5",
            "weights/sd-vae-ft-mse",
            "weights/diffuEraser",
            ckpt="2-Step",
        )
        load_sec = time.time() - started
        print(f"Models loaded in {load_sec:.1f}s, "
              f"GPU free {torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

        # video_length is in SECONDS (read_video passes it as end_pts and then
        # computes int(video_length * fps)) — not a frame count. Round up so the
        # last frame is not truncated by int() flooring.
        video_length_sec = (T + 1) / float(enc_fps)

        torch.cuda.reset_peak_memory_stats()
        infer_started = time.time()
        model.forward(
            video_path,
            mask_path,
            priori_path,
            output_path,
            max_img_size=max_img_size,
            video_length=video_length_sec,
            mask_dilation_iter=mask_dilation_iter,
            guidance_scale=None,
        )
        infer_sec = time.time() - infer_started
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"DiffuEraser inference {infer_sec:.1f}s for {T} frames, "
              f"peak VRAM {peak_gb:.1f}GB (torch allocator), "
              f"free {torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

        if not os.path.exists(output_path):
            raise RuntimeError("DiffuEraser produced no output file")

        # It writes at its own working resolution; restore the caller's size so
        # the chunk still composites into the full-res frame.
        cap = cv2.VideoCapture(output_path)
        out_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LANCZOS4)
            out_frames.append(frame)
        cap.release()

        if len(out_frames) < T:
            # Short returns are padded from the prior rather than failing the
            # chunk — the caller's frame count must survive stitching.
            print(f"WARNING: got {len(out_frames)} frames, padding to {T}")
            out_frames.extend(priors[len(out_frames):T])
        out_frames = out_frames[:T]

        final_path = os.path.join(tmp, "final.mp4")
        _write_mp4(out_frames, final_path)
        with open(final_path, "rb") as f:
            data = f.read()

    torch.cuda.empty_cache()
    print(f"Returning {len(data)/1e6:.1f}MB")
    return data
