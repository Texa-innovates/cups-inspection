import os
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from rembg import remove, new_session
import onnxruntime as ort
import time
from concurrent.futures import ThreadPoolExecutor

# -------------------------------------------------
# 1) Show which providers are available (debug)
# -------------------------------------------------
print("ONNXRuntime providers:", ort.get_available_providers())

# -------------------------------------------------
# 2) Create a global GPU rembg session
# -------------------------------------------------
# High-quality model for general objects
BG_MODEL_NAME = "isnet-general-use"

# Prefer TensorRT -> CUDA -> CPU
BG_SESSION = new_session(
    BG_MODEL_NAME,
    providers=[
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)



def remove_background_from_image(bgr_img, session=BG_SESSION):
    
    if bgr_img is None:
        print("❌ Invalid image")
        return None

    # ---- 1) OpenCV BGR → PIL RGBA ----
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGBA")

    # ---- 2) Remove background using shared GPU session ----
    try:
        out = remove(pil_img, session=session)
    except Exception as e:
        print("❌ rembg failed:", e)
        return None

    # ---- 3) Normalize output to RGBA PIL ----
    if isinstance(out, bytes):
        out_pil = Image.open(io.BytesIO(out)).convert("RGBA")
    else:
        out_pil = out.convert("RGBA")

    rgba = np.array(out_pil)

    # ---- 4) Merge onto white background ----
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0

    white = np.ones_like(rgb) * 255.0
    merged = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)

    white_bg_img = cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)

    return white_bg_img

import cv2
img_path1 = cv2.imread(r"/home/texa_innovates/chromo_gpu/captures/cam_0_4.png")
img_path2 = cv2.imread(r"/home/texa_innovates/chromo_gpu/captures/cam_0_4.png")
img_path3 = cv2.imread(r"/home/texa_innovates/chromo_gpu/captures/cam_0_4.png")

import time

imgs = [img_path1, img_path2, img_path3, img_path3]

# ---- Sequential ----
t0 = time.perf_counter()
for im in imgs:
    remove_background_from_image(im)
t1 = time.perf_counter()
print(f"Sequential total: {(t1 - t0)*1000:.1f} ms")

# ---- Threads (same session) ----
from concurrent.futures import ThreadPoolExecutor

t2 = time.perf_counter()
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = [ex.submit(remove_background_from_image, im) for im in imgs]
    _ = [f.result() for f in futs]
t3 = time.perf_counter()
print(f"Threads (same session) total: {(t3 - t2)*1000:.1f} ms")


# print(result)