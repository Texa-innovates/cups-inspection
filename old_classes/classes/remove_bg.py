# =========================
# OFFLINE BG REMOVE (GPU) + SAVE WHITE BMP
# Input : /home/texa_innovates/chromo_gpu/capture_remove/CAM1..CAM4  (or any folders inside)
# Output: /home/texa_innovates/chromo_gpu/bgremove_img_1/<same folders>/*_white.bmp
# - Auto-detects camera folders (no hardcoding)
# - Skips missing folders safely
# - Skips if output exists (optional)
# =========================

import os
import io
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from rembg import remove, new_session
import onnxruntime as ort

# ---------------------------
# CONFIG
# ---------------------------
CAPTURES_DIR   = Path("/home/texa_innovates/chromo_gpu/capture_with_bg")
OUTDIR         = Path("/home/texa_innovates/chromo_gpu/bgremove_img_1")

EXTS           = (".bmp", ".png", ".jpg", ".jpeg")
SKIP_IF_EXISTS = True          # skip if *_white.bmp exists
PRINT_EVERY    = 25            # progress print interval
MODEL_NAME     = "isnet-general-use"
# ---------------------------


def fatal(msg: str, code: int = 1):
    print("❌", msg)
    raise SystemExit(code)


# Debug providers
print("ONNXRuntime providers:", ort.get_available_providers())

# ✅ Avoid TensorRT EP (common libcublas issues). Prefer CUDA, fallback CPU.
BG_SESSION = new_session(
    MODEL_NAME,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

OUTDIR.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS])


def remove_background_white_bmp(input_image_path: Path, out_dir: Path, session=BG_SESSION) -> Optional[str]:
    """
    Removes background and saves result with WHITE background in BMP format.
    Output file: <name>_white.bmp
    Returns output path string or None.
    """
    input_image_path = Path(input_image_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = input_image_path.stem
    out_white = out_dir / f"{base}_white.bmp"

    if SKIP_IF_EXISTS and out_white.exists():
        return str(out_white)

    # Quick validation
    test = cv2.imread(str(input_image_path), cv2.IMREAD_UNCHANGED)
    if test is None:
        print(f"⚠️ Skip unreadable (OpenCV): {input_image_path}")
        return None

    # Read bytes -> PIL RGBA
    try:
        data = input_image_path.read_bytes()
        pil_img = Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception as e:
        print(f"⚠️ Skip unreadable (PIL): {input_image_path.name} | {e}")
        return None

    # rembg inference
    try:
        out_obj = remove(pil_img, session=session)
    except Exception as e:
        print(f"❌ rembg failed: {input_image_path.name} | {e}")
        return None

    # Normalize output
    try:
        if isinstance(out_obj, bytes):
            out_pil = Image.open(io.BytesIO(out_obj)).convert("RGBA")
        else:
            out_pil = out_obj.convert("RGBA")
    except Exception as e:
        print(f"⚠️ Output decode failed: {input_image_path.name} | {e}")
        return None

    # Composite to WHITE background (RGBA -> RGB)
    rgba = np.array(out_pil, dtype=np.uint8)        # (H,W,4)
    rgb  = rgba[:, :, :3].astype(np.float32)
    a    = (rgba[:, :, 3:4].astype(np.float32)) / 255.0

    white  = np.ones_like(rgb, dtype=np.float32) * 255.0
    merged = (rgb * a + white * (1.0 - a)).astype(np.uint8)  # (H,W,3) RGB

    # Save as BMP (BGR)
    ok = cv2.imwrite(str(out_white), cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))
    if not ok:
        print(f"⚠️ Failed to write: {out_white}")
        return None

    return str(out_white)


def detect_cam_folders(root: Path) -> List[str]:
    """
    Auto-detect all subfolders inside CAPTURES_DIR.
    Prioritizes CAM1..CAM4 if they exist.
    """
    if not root.exists():
        fatal(f"CAPTURES_DIR not found: {root}")

    folders = [p.name for p in root.iterdir() if p.is_dir()]
    if not folders:
        fatal(f"No folders found inside: {root}")

    # If CAM1..CAM4 exist, use those in order; else use everything found
    preferred = [f"CAM{i}" for i in range(1, 5)]
    if all((root / p).exists() for p in preferred):
        return preferred

    return sorted(folders)


def process_camera_folder(cam_name: str):
    in_dir  = CAPTURES_DIR / cam_name
    out_dir = OUTDIR / cam_name

    if not in_dir.exists():
        print(f"⚠️ Missing folder, skipped: {in_dir}")
        return

    imgs = list_images(in_dir)
    total = len(imgs)
    if total == 0:
        print(f"⚠️ No images found in: {in_dir}")
        return

    print(f"\n=== Processing {cam_name}: {total} images ===")
    ok_count = 0
    fail_count = 0
    t0 = time.time()

    for i, img_path in enumerate(imgs, 1):
        out = remove_background_white_bmp(img_path, out_dir)
        if out is None:
            fail_count += 1
        else:
            ok_count += 1

        if (i % PRINT_EVERY) == 0 or i == total:
            dt = time.time() - t0
            print(f"[{cam_name}] {i}/{total} | ok={ok_count} fail={fail_count} | {dt:.1f}s")

    dt = time.time() - t0
    print(f"✅ Done {cam_name}: ok={ok_count}, fail={fail_count}, time={dt:.1f}s")
    print(f"➡️ Output folder: {out_dir}")


if __name__ == "__main__":
    cam_folders = detect_cam_folders(CAPTURES_DIR)
    print("✅ Detected camera folders:", cam_folders)

    grand_t0 = time.time()
    for cam in cam_folders:
        process_camera_folder(cam)

    print(f"\n🎉 All done. Total time: {(time.time() - grand_t0):.1f}s")
    print(f"📁 Saved in: {OUTDIR}")
