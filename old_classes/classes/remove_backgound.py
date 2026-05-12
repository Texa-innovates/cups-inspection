# import os
# import io
# from pathlib import Path

# import cv2
# import numpy as np
# from PIL import Image

# from rembg import remove, new_session
# import onnxruntime as ort

# # -------------------------------------------------
# # 1) Show which providers are available (debug)
# # -------------------------------------------------
# print("ONNXRuntime providers:", ort.get_available_providers())

# # -------------------------------------------------
# # 2) Create a global GPU rembg session
# # -------------------------------------------------
# # High-quality model for general objects
# BG_MODEL_NAME = "isnet-general-use"

# # Prefer TensorRT -> CUDA -> CPU
# BG_SESSION = new_session(
#     BG_MODEL_NAME,
#     providers=[
#         "TensorrtExecutionProvider",
#         "CUDAExecutionProvider",
#         "CPUExecutionProvider",
#     ],
# )


# # -------------------------------------------------
# # 3) Helpers
# # -------------------------------------------------
# def load_images_from_folder(folder_path):
#     folder = Path(folder_path)
#     exts = (".bmp", ".png", ".jpg", ".jpeg")
#     return sorted(
#         [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
#     )


# def remove_background(input_image_path, output_folder="testing_img", session=BG_SESSION):
#     input_image_path = Path(input_image_path)
#     os.makedirs(output_folder, exist_ok=True)

#     # ---- 1) Quick validation with OpenCV (skip corrupt files) ----
#     test = cv2.imread(str(input_image_path))
#     if test is None:
#         print(f"⚠️ Skip (OpenCV can't read): {input_image_path.name}")
#         return None

#     base_name = input_image_path.stem
#     out_white = Path(output_folder) / f"{base_name}_white.png"
#     out_trans = Path(output_folder) / f"{base_name}_transparent.png"

#     # ---- 2) Read bytes and convert to PIL safely ----
#     data = input_image_path.read_bytes()
#     try:
#         pil_img = Image.open(io.BytesIO(data)).convert("RGBA")
#     except Exception as e:
#         print(f"⚠️ Skip (PIL can't read): {input_image_path.name} | {e}")
#         return None

#     # ---- 3) Remove background with shared GPU session ----
#     try:
#         out_bytes = remove(pil_img, session=session)  # GPU / TensorRT here
#     except Exception as e:
#         print(f"❌ rembg failed: {input_image_path.name} | {e}")
#         return None

#     # out_bytes may be bytes OR PIL depending on version; normalize:
#     if isinstance(out_bytes, bytes):
#         out_pil = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
#     else:
#         out_pil = out_bytes.convert("RGBA")

#     # ---- 4) Save transparent PNG ----
#     # out_pil.save(out_trans)

#     # ---- 5) Save white background ----
#     rgba = np.array(out_pil, dtype=np.uint8)  # (H,W,4)
#     rgb = rgba[:, :, :3].astype(np.float32)
#     alpha = (rgba[:, :, 3:4].astype(np.float32)) / 255.0

#     white = np.ones_like(rgb, dtype=np.float32) * 255.0
#     merged = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)

#     cv2.imwrite(str(out_white), cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))

#     print(f"✅ Saved: {out_trans.name} & {out_white.name}")
#     return str(out_white), str(out_trans)


# # # ---------------- RUN ----------------
# FOLDER = r"/home/texa_innovates/chromo_gpu/captures"
# OUTDIR = r"/home/texa_innovates/chromo/bgremove_img"
# import time
# # os.makedirs(OUTDIR, exist_ok=True)
# from concurrent.futures import ThreadPoolExecutor

# img_path1 = r"/home/texa_innovates/chromo_gpu/captures/cam_0_4.png"
# img_path2 = r"/home/texa_innovates/chromo_gpu/captures/cam_0_4.png"
# img_path3 = r"/home/texa_innovates/chromo_gpu/captures/cam_0_4.png"
# start = time.time()
# remove_background(img_path1)
# end = time.time()
# print(f"second:{end-start:2f}")

# start = time.time()

# with ThreadPoolExecutor(max_workers=4) as excute:
#     value = [
#         excute.submit(remove_background,img_path1),
#         excute.submit(remove_background,img_path2),
#         excute.submit(remove_background,img_path3),
#         excute.submit(remove_background,img_path3),
#     ]
#     result = [r.result() for r in value ]
# end = time.time()
# print(f"second:{end-start:2f}")

# print(result)

# # for img_path in load_images_from_folder(FOLDER):
# #     remove_background(img_path, output_folder=OUTDIR)



import os
import io
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from rembg import remove, new_session
import onnxruntime as ort

# ---------------------------
# CONFIG
# ---------------------------
CAPTURES_DIR = Path("/home/texa_innovates/chromo_gpu/capture_remove")
OUTDIR       = Path("/home/texa_innovates/chromo_gpu/bgremove_img_1")

CAM_FOLDERS  = ["camera1", "camera2", "camera3", "camera4"]
    # ✅ process only these
EXTS         = (".bmp", ".png", ".jpg", ".jpeg")

SKIP_IF_EXISTS = True                      # ✅ skip if output already exists
PRINT_EVERY    = 25                        # progress print
MODEL_NAME     = "isnet-general-use"       # good quality
# ---------------------------


# Debug: providers available
print("ONNXRuntime providers:", ort.get_available_providers())

# ✅ Avoid TensorRT EP (your system missing libcublas.so.12)
# Prefer CUDA, fallback CPU
BG_SESSION = new_session(
    MODEL_NAME,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

OUTDIR.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS])


def remove_background_white_bmp(input_image_path: Path, out_dir: Path, session=BG_SESSION):
    """
    Removes background and saves result with WHITE background in BMP format.
    Output file: <name>_white.bmp
    Returns output path string or None.
    """
    input_image_path = Path(input_image_path)
    base = input_image_path.stem
    out_white = out_dir / f"{base}_white.bmp"   # ✅ BMP ONLY

    if SKIP_IF_EXISTS and out_white.exists():
        return str(out_white)

    # Quick validation
    test = cv2.imread(str(input_image_path), cv2.IMREAD_UNCHANGED)
    if test is None:
        print(f"⚠️ Skip unreadable: {input_image_path}")
        return None

    # Read bytes -> PIL RGBA
    try:
        data = input_image_path.read_bytes()
        pil_img = Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception as e:
        print(f"⚠️ PIL read failed: {input_image_path.name} | {e}")
        return None

    # rembg inference
    try:
        out_obj = remove(pil_img, session=session)
    except Exception as e:
        print(f"❌ rembg failed: {input_image_path.name} | {e}")
        return None

    # Normalize output
    if isinstance(out_obj, bytes):
        out_pil = Image.open(io.BytesIO(out_obj)).convert("RGBA")
    else:
        out_pil = out_obj.convert("RGBA")

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


def process_camera_folder(cam_name: str):
    in_dir = CAPTURES_DIR / cam_name
    out_dir = OUTDIR / cam_name
    out_dir.mkdir(parents=True, exist_ok=True)

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
    grand_t0 = time.time()

    for cam in CAM_FOLDERS:
        process_camera_folder(cam)

    print(f"\n🎉 All done. Total time: {(time.time() - grand_t0):.1f}s")
    print(f"📁 Saved in: {OUTDIR}")
