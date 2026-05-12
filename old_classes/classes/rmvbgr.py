import os
import io
from pathlib import Path
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image

from rembg import remove, new_session
import onnxruntime as ort

# -------------------------------------------------
# SHOW PROVIDERS
# -------------------------------------------------
print("ONNXRuntime providers:", ort.get_available_providers())

# -------------------------------------------------
# USE LIGHTWEIGHT MODEL
# -------------------------------------------------
BG_MODEL_NAME = "u2netp"     # <<< FAST MODEL HERE

# -------------------------------------------------
# GPU ONLY (REMOVE TensorRT)
# -------------------------------------------------
BG_SESSION = new_session(
    BG_MODEL_NAME,
    providers=[
        "CUDAExecutionProvider",   # Primary
        "CPUExecutionProvider"     # Fallback
    ],
)

# -------------------------------------------------
# PER-IMAGE GPU PROCESS
# -------------------------------------------------
def process_one_image(input_image_path, output_folder, session=BG_SESSION):
    t0 = perf_counter()
    
    input_image_path = Path(input_image_path)
    os.makedirs(output_folder, exist_ok=True)

    # 1) read image
    img = cv2.imread(str(input_image_path))
    if img is None:
        return input_image_path.name, None

    base = input_image_path.stem
    out_white = Path(output_folder) / f"{base}_white.png"
    out_trans = Path(output_folder) / f"{base}_transparent.png"

    # 2) PIL conversion
    data = input_image_path.read_bytes()
    pil_img = Image.open(io.BytesIO(data)).convert("RGBA")

    # 3) GPU rembg using u2netp
    out_bytes = remove(pil_img, session=session)

    # normalize output
    out_pil = (
        Image.open(io.BytesIO(out_bytes)).convert("RGBA")
        if isinstance(out_bytes, bytes)
        else out_bytes.convert("RGBA")
    )

    # save transparent
    # out_pil.save(out_trans)

    # merge to white background
    rgba = np.array(out_pil)
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = (rgba[:, :, 3:4].astype(np.float32)) / 255.0

    white = np.ones_like(rgb) * 255.0
    merged = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
    
    cv2.imwrite(str(out_white), cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))

    t1 = perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0

    print(f"✅ {input_image_path.name} | {elapsed_ms:.2f} ms")
    return input_image_path.name, elapsed_ms


# -------------------------------------------------
# RUN EXACTLY 4 THREADS
# -------------------------------------------------
FOLDER = Path("/home/texa_innovates/chromo_gpu/capture_bg")
OUTDIR = Path("/home/texa_innovates/chromo_gpu/capture")
OUTDIR.mkdir(parents=True, exist_ok=True)

exts = (".bmp", ".png", ".jpg", ".jpeg")
images = sorted([p for p in FOLDER.iterdir() if p.suffix.lower() in exts])

images = images

print(f"\n🚀 Using model: {BG_MODEL_NAME}")
print(f"🚀 Running 4 threads in parallel\n")

batch_start = perf_counter()

results = []
with ThreadPoolExecutor(max_workers=4) as exe:
    futs = [exe.submit(process_one_image, img, OUTDIR, BG_SESSION) for img in images]
    for f in as_completed(futs):
        results.append(f.result())

batch_end = perf_counter()

valid_times = [ms for _, ms in results if ms]

print("\n📊 SUMMARY")
for name, ms in results:
    print(f"• {name}: {ms:.2f} ms")

if valid_times:
    avg = sum(valid_times) / len(valid_times)
    wall = (batch_end - batch_start) * 1000.0
    print(f"\n➡ Total wall time for 4 images: {wall:.2f} ms")
    print(f"➡ Average per image: {avg:.2f} ms\n")
