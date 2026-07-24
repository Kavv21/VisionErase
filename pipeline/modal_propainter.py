"""
Modal.com serverless ProPainter inpainting function.
Inference loop matches inference_propainter.py exactly.
"""
import modal
import numpy as np

propainter_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git", "libgl1", "libglib2.0-0", "ffmpeg"])
    .pip_install([
        "torch==2.3.0",
        "torchvision==0.18.0",
        "opencv-python-headless",
        "numpy",
        "scipy",
        "scikit-image",
        "pillow",
        "matplotlib",
        "timm",
        "einops",
        "addict",
        "yapf",
        "imageio-ffmpeg",
        "av",
        "tqdm",
    ])
    .run_commands(  # rebuilt 20260714_090000
        "git clone https://github.com/sczhou/ProPainter.git /ProPainter",
        "mv /ProPainter/model /ProPainter/propainter_model",
        # cuDNN's grid sampler rejects RAFT CorrBlock's huge batch dim
        # (B*H/8*W/8) with CUDNN_STATUS_NOT_SUPPORTED; the inputs are
        # already contiguous, so disable cuDNN for this one call.
        "sed -i 's/    img = F.grid_sample(img, grid, align_corners=True)/    with torch.backends.cudnn.flags(enabled=False):\\n        img = F.grid_sample(img.contiguous(), grid.contiguous(), align_corners=True)/' /ProPainter/RAFT/utils/utils.py",
        "grep -q 'cudnn.flags(enabled=False)' /ProPainter/RAFT/utils/utils.py && grep -n -A1 'cudnn.flags' /ProPainter/RAFT/utils/utils.py && python3 -m py_compile /ProPainter/RAFT/utils/utils.py && echo 'bilinear_sampler patch verified'",
        "find /ProPainter -name '*.py' | xargs sed -i 's/from model\\./from propainter_model./g'",
        "find /ProPainter -name '*.py' | xargs sed -i 's/import model\\./import propainter_model./g'",
    )
)

app = modal.App("visionerase-propainter", image=propainter_image)
weights_volume = modal.Volume.from_name("propainter-weights", create_if_missing=True)


@app.function(volumes={"/weights": weights_volume}, timeout=600)
def upload_weights(propainter_bytes, raft_bytes, flow_bytes):
    import os
    os.makedirs("/weights/propainter", exist_ok=True)
    with open("/weights/propainter/ProPainter.pth", "wb") as f:
        f.write(propainter_bytes)
    with open("/weights/propainter/raft-things.pth", "wb") as f:
        f.write(raft_bytes)
    with open("/weights/propainter/recurrent_flow_completion.pth", "wb") as f:
        f.write(flow_bytes)
    print(f"Uploaded: {len(propainter_bytes)/1e6:.0f}MB + {len(raft_bytes)/1e6:.0f}MB + {len(flow_bytes)/1e6:.0f}MB")
    weights_volume.commit()


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx   = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


@app.function(
    gpu="A10G",
    volumes={"/weights": weights_volume},
    timeout=600,
    memory=16384,
)
def inpaint_frames(frames_bytes, masks_bytes, fps=24.0):
    import sys, os, cv2, torch, numpy as np, tempfile, io, subprocess
    from PIL import Image
    from scipy.ndimage import binary_dilation

    sys.path.insert(0, "/ProPainter")
    device    = torch.device("cuda")
    model_dir = "/weights/propainter"

    # Decode
    frames = [np.array(Image.open(io.BytesIO(b)).convert("RGB")) for b in frames_bytes]
    masks  = [(np.array(Image.open(io.BytesIO(b)).convert("L")) > 127).astype(np.uint8) * 255
              for b in masks_bytes]

    T = len(frames)
    H, W = frames[0].shape[:2]
    print(f"Processing {T} frames at {W}x{H} on {torch.cuda.get_device_name(0)}")

    # Keep full-resolution originals for the final composite
    frames_orig = [f.copy() for f in frames]
    out_size = (W, H)

    # Cap resolution
    MAX_W, MAX_H = 854, 480  # A10G 24GB can handle this resolution
    scale    = min(MAX_W / W, MAX_H / H, 1.0)
    W_new = (int(W * scale) // 8) * 8
    H_new = (int(H * scale) // 8) * 8
    if (W_new, H_new) != (W, H):
        frames = [cv2.resize(f, (W_new, H_new)) for f in frames]
        masks  = [cv2.resize(m, (W_new, H_new), interpolation=cv2.INTER_NEAREST) for m in masks]
        W, H   = W_new, H_new
        print(f"Resized to {W}x{H}")

    torch.cuda.empty_cache()

    # Load models
    from propainter_model.modules.flow_comp_raft import RAFT_bi
    from propainter_model.recurrent_flow_completion import RecurrentFlowCompleteNet
    from propainter_model.propainter import InpaintGenerator
    from core.utils import to_tensors

    print("Loading models...")
    fix_raft = RAFT_bi(
        model_path=f"{model_dir}/raft-things.pth", device=device
    ).to(device).eval()
    fix_flow = RecurrentFlowCompleteNet(
        f"{model_dir}/recurrent_flow_completion.pth"
    ).to(device)
    for p in fix_flow.parameters():
        p.requires_grad = False
    fix_flow.eval()
    model = InpaintGenerator(
        init_weights=False,
        model_path=f"{model_dir}/ProPainter.pth"
    ).to(device).eval()
    print(f"Models loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Keep original frames for blending
    frames_inp = [f.copy() for f in frames]

    # Dilate masks
    flow_masks_pil    = []
    masks_dilated_pil = []
    for m in masks:
        fm = binary_dilation(m > 0, iterations=8).astype(np.uint8) * 255
        dm = binary_dilation(m > 0, iterations=4).astype(np.uint8) * 255
        flow_masks_pil.append(Image.fromarray(fm))
        masks_dilated_pil.append(Image.fromarray(dm))

    # Tensors (1, T, C, H, W)
    frames_t    = to_tensors()([Image.fromarray(f) for f in frames]).unsqueeze(0).to(device)
    frames_t    = frames_t * 2 - 1  # [0,1] -> [-1,1]; masks stay in [0,1]
    flow_mask_t = to_tensors()(flow_masks_pil).unsqueeze(0).to(device)
    mask_dil_t  = to_tensors()(masks_dilated_pil).unsqueeze(0).to(device)

    b, video_length, _, h, w = frames_t.size()
    masked_frames = frames_t * (1 - mask_dil_t)

    RAFT_ITERS      = 10  # reduced for memory
    NEIGHBOR_LENGTH = 20  # wider window for the larger overlapping chunks
    REF_STRIDE      = 15
    SUBVIDEO_LENGTH = 80

    # RAFT builds its correlation volume over the whole clip at once:
    # (T-1) x (H*W/64)^2 x 4 bytes. At 854x480 that is ~7.3GB for T=50 but
    # ~11.9GB for T=80, which OOMs the A10G alongside the loaded models.
    # CHUNK_SIZE in workers/segmentation/tasks.py must stay within this bound.

    with torch.no_grad():
        print("Computing optical flow...")
        frames_t = frames_t.contiguous()  # fix cuDNN non-contiguous error
        gt_flows_bi = fix_raft(frames_t, iters=RAFT_ITERS)

        print("Completing flow...")
        pred_flows_bi, _ = fix_flow.forward_bidirect_flow(gt_flows_bi, flow_mask_t)
        pred_flows_bi     = fix_flow.combine_flow(gt_flows_bi, pred_flows_bi, flow_mask_t)

        print("Image propagation...")
        prop_imgs, updated_local_masks = model.img_propagation(
            masked_frames, pred_flows_bi, mask_dil_t, "nearest"
        )
        updated_frames = (
            frames_t * (1 - mask_dil_t) +
            prop_imgs.view(b, video_length, 3, h, w) * mask_dil_t
        )
        updated_masks = updated_local_masks.view(b, video_length, 1, h, w)
        torch.cuda.empty_cache()

    # Transformer inpainting loop
    print("Transformer inpainting...")
    comp_frames     = [None] * video_length
    neighbor_stride = NEIGHBOR_LENGTH // 2
    ref_num = SUBVIDEO_LENGTH // REF_STRIDE if video_length > SUBVIDEO_LENGTH else -1

    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [
            i for i in range(
                max(0, f - neighbor_stride),
                min(video_length, f + neighbor_stride + 1)
            )
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, REF_STRIDE, ref_num)

        selected_imgs         = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks        = mask_dil_t[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (
            pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
            pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :],
        )

        print(f"  f={f}: neighbors={len(neighbor_ids)} refs={len(ref_ids)} "
              f"imgs={selected_imgs.shape} flows={selected_pred_flows_bi[0].shape}")

        with torch.no_grad():
            l_t      = len(neighbor_ids)
            pred_img = model(
                selected_imgs,
                selected_pred_flows_bi,
                selected_masks,
                selected_update_masks,
                l_t,
            )
            pred_img     = pred_img.view(-1, 3, h, w)
            pred_img     = (pred_img + 1) / 2
            pred_np      = np.clip(pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8)
            binary_masks = mask_dil_t[0, neighbor_ids].cpu().permute(
                0, 2, 3, 1
            ).numpy().astype(np.uint8)

            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = (pred_np[i].astype(np.float32) * binary_masks[i] +
                       frames_inp[idx].astype(np.float32) * (1 - binary_masks[i])
                       ).clip(0, 255).astype(np.uint8)
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = (
                        comp_frames[idx].astype(np.float32) * 0.5 +
                        img.astype(np.float32) * 0.5
                    ).astype(np.uint8)

    # Fill any None frames
    for i in range(video_length):
        if comp_frames[i] is None:
            comp_frames[i] = frames_inp[i]

    # Composite only the inpainted patch into the full-resolution originals
    mask_dil_np = np.stack([
        (mask_dil_t[0, i, 0].cpu().numpy() > 0.5).astype(np.float32)
        for i in range(video_length)
    ])  # (T, H_small, W_small)

    final_frames = []
    for i in range(video_length):
        orig = frames_orig[i]  # (H_orig, W_orig, 3) uint8
        if comp_frames[i] is not None:
            patch = cv2.resize(
                comp_frames[i], out_size, interpolation=cv2.INTER_LINEAR
            )
            m = cv2.resize(
                mask_dil_np[i], out_size, interpolation=cv2.INTER_NEAREST
            )[:, :, None]  # (H, W, 1)
            frame_out = (patch.astype(np.float32) * m +
                         orig.astype(np.float32) * (1 - m)
                         ).clip(0, 255).astype(np.uint8)
        else:
            frame_out = orig
        final_frames.append(frame_out)
    comp_frames = final_frames

    print(f"Writing {len(comp_frames)} frames...")
    out_h, out_w = comp_frames[0].shape[:2]
    raw_frames = b""
    for frame in comp_frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        raw_frames += bgr.tobytes()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-vcodec", "libx264",
        "-profile:v", "high",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        tmp_path
    ], input=raw_frames, check=True)

    with open(tmp_path, "rb") as f:
        video_bytes = f.read()
    os.unlink(tmp_path)
    print(f"Done. {len(video_bytes)/1e6:.1f}MB")
    return video_bytes


@app.local_entrypoint()
def upload_weights_local():
    import os
    model_dir = os.path.expanduser("~/visionerase/model_weights/propainter")
    with open(f"{model_dir}/ProPainter.pth", "rb") as f: pp = f.read()
    with open(f"{model_dir}/raft-things.pth", "rb") as f: raft = f.read()
    with open(f"{model_dir}/recurrent_flow_completion.pth", "rb") as f: flow = f.read()
    print(f"Uploading {len(pp)/1e6:.0f}MB + {len(raft)/1e6:.0f}MB + {len(flow)/1e6:.0f}MB...")
    upload_weights.remote(pp, raft, flow)
    print("Done")
