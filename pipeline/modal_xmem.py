"""
Modal.com serverless XMem++ mask tracking.
Runs on T4 GPU (16GB VRAM) — no local VRAM needed.
"""
import modal
import numpy as np

xmem_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git", "libgl1", "libglib2.0-0"])
    .pip_install([
        "torch==2.3.0",
        "torchvision==0.18.0",
        "opencv-python-headless",
        "numpy",
        "pillow",
        "progressbar2",
        "gdown",
        "gitpython",
        "hickle",
    ])
    .run_commands(  # force rebuild 20260713_121922
        "git clone https://github.com/hkchengrex/XMem.git /XMem",
    )
)

app = modal.App("visionerase-xmem", image=xmem_image)
weights_volume = modal.Volume.from_name("xmem-weights", create_if_missing=True)


@app.function(volumes={"/weights": weights_volume}, timeout=300)
def upload_xmem_weights(xmem_bytes: bytes):
    """Upload XMem checkpoint to Modal volume. Run once."""
    import os
    os.makedirs("/weights", exist_ok=True)
    with open("/weights/XMem.pth", "wb") as f:
        f.write(xmem_bytes)
    print(f"XMem.pth uploaded: {len(xmem_bytes)/1e6:.1f}MB")
    weights_volume.commit()


@app.function(
    gpu="T4",
    volumes={"/weights": weights_volume},
    timeout=600,
    memory=8192,
)
def track_masks(
    frames_bytes: list[bytes],   # JPEG-encoded RGB frames
    seed_mask_bytes: bytes,      # PNG-encoded seed mask (0/255)
) -> bytes:                      # numpy .npy bytes (T, H, W) uint8
    """
    Run XMem++ mask tracking on T4 GPU.
    Input:  JPEG frames + PNG seed mask from SAM2
    Output: numpy array (T, H, W) uint8 tracked masks
    """
    import sys, io, numpy as np, torch
    from PIL import Image

    sys.path.insert(0, "/XMem")
    device = torch.device("cuda")

    # Decode frames
    frames = [np.array(Image.open(io.BytesIO(b)).convert("RGB"))
              for b in frames_bytes]
    seed_mask = np.array(Image.open(io.BytesIO(seed_mask_bytes)).convert("L"))
    seed_mask_binary = (seed_mask > 127).astype(np.uint8)

    T = len(frames)
    H, W = frames[0].shape[:2]
    print(f"Tracking {T} frames at {W}x{H} on {torch.cuda.get_device_name(0)}")
    print(f"Seed mask coverage: {seed_mask_binary.mean()*100:.1f}%")

    # Load XMem
    from model.network import XMem as XMemNetwork
    from inference.inference_core import InferenceCore
    from inference.interact.interactive_utils import (
        image_to_torch, index_numpy_to_one_hot_torch
    )

    xmem_config = {
        'top_k': 30,
        'mem_every': 5,
        'deep_update_every': -1,
        'enable_long_term': True,
        'enable_long_term_count_usage': True,
        'num_prototypes': 128,
        'min_mid_term_frames': 5,
        'max_mid_term_frames': 10,
        'max_long_term_elements': 10000,
    }

    network = XMemNetwork(
        xmem_config,
        '/weights/XMem.pth'
    ).cuda().eval()
    processor = InferenceCore(network, config=xmem_config)
    processor.set_all_labels(list(range(1, 2)))

    print("XMem loaded. Running tracking...")
    print(f"Seed mask: shape={seed_mask_binary.shape} min={seed_mask_binary.min()} max={seed_mask_binary.max()} coverage={seed_mask_binary.mean()*100:.1f}%")

    tracked_masks = []
    with torch.no_grad():
        for i, frame_rgb in enumerate(frames):
            frame_torch, _ = image_to_torch(frame_rgb, device='cuda')
            if i == 0:
                mask_torch = index_numpy_to_one_hot_torch(
                    seed_mask_binary, 2
                ).cuda()
                output_prob = processor.step(frame_torch, mask_torch[1:2])
            else:
                output_prob = processor.step(frame_torch)

            if i == 0:
                print(f"  output_prob shape: {output_prob.shape} min={output_prob.min():.3f} max={output_prob.max():.3f}")
            out_mask = (output_prob[1].cpu().numpy() > 0.5).astype(np.uint8) * 255  # [1] = foreground, [0] = background
            if i == 0: print(f"  out_mask coverage: {out_mask.mean()/255*100:.1f}%")
            tracked_masks.append(out_mask)

            if i % 10 == 0:
                print(f"  Frame {i}/{T} tracked")

    tracked_array = np.stack(tracked_masks)  # (T, H, W)
    print(f"Tracking complete. Shape: {tracked_array.shape}")

    buf = io.BytesIO()
    np.save(buf, tracked_array)
    return buf.getvalue()


@app.local_entrypoint()
def upload_weights_local():
    """Run with: modal run pipeline/modal_xmem.py"""
    import os
    weights_path = os.path.expanduser(
        "~/visionerase/model_weights/XMem.pth"
    )
    print(f"Reading {weights_path}...")
    with open(weights_path, "rb") as f:
        xmem_bytes = f.read()
    print(f"Uploading {len(xmem_bytes)/1e6:.1f}MB...")
    upload_xmem_weights.remote(xmem_bytes)
    print("Done — XMem weights uploaded to Modal volume")
    print("Now run: modal deploy pipeline/modal_xmem.py")
