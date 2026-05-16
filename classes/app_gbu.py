# ---------- Torch / ML ----------
import joblib, torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os, time, json, threading, warnings, sys
from datetime import datetime
from time import perf_counter_ns
from typing import Dict, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from main_window import now_str,CAM_INDICES
import platform
import io
from path import JOBID_JSON_FILE,BAD_IMG_SAVE
from classes.database import insert_defect,upsert_cup_entry
# ---------- PyQt5 ----------
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QFrame, QSplitter, QSizePolicy, QScrollArea,QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
# ---------- Modbus ----------
from pymodbus.client.sync import ModbusTcpClient

# ---------- rembg (GPU ONLY) ----------
from rembg import remove, new_session
import onnxruntime as ort

BG_MODEL_NAME = "isnet-general-use"
# GPU-only providers: remove CPUExecutionProvider completely
BG_SESSION = new_session(
    BG_MODEL_NAME,
    providers=["CUDAExecutionProvider"]
)

# =========================
# QT PLATFORM INIT
# =========================
def init_qt_platform():
    system = platform.system()
    if system == "Linux":
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    else:
        os.environ.pop("QT_QPA_PLATFORM", None)

    try:
        if getattr(sys, "frozen", False):
            base = sys._MEIPASS
            qt_plugin_path = os.path.join(base, "cv2", "qt", "plugins", "platforms")
        else:
            import cv2 as _cv2
            
            base = os.path.dirname(_cv2.__file__)
            qt_plugin_path = os.path.join(base, "qt", "plugins", "platforms")

        if os.path.isdir(qt_plugin_path):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugin_path
            print(f"[QT] Using platform plugins at: {qt_plugin_path}")
        else:
            os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
            print(f"[QT] Plugin path not found, using default search paths. Tried: {qt_plugin_path}")
    except Exception as e:
        print(f"[QT] Error while setting platform plugin path: {e}")

init_qt_platform()

# # ---------- FAISS GPU (kNN on GPU) ----------
# try:
#     import faiss
#     import faiss.contrib.torch_utils  # enables CUDA torch tensors with faiss
# except Exception as e:
#     fatal_exit(f"FAISS-GPU not available/import failed: {e}\nInstall faiss-gpu / correct CUDA build.")

# =========================
# CONFIG
# =========================
SERVER_IP    = "192.168.3.1"
PORT         = 507
UNIT_ID      = 1
READ_OFFSET  = 1
WRITE_OFFSET = 2
POLL_SEC     = 0.005
RESET_DELAY  = 0.050

# CAM_INDICES  = (2, 6, 4, 0)
CAM_W, CAM_H = 1280, 720
TARGET_FPS   = 15
BUFFERSIZE   = 1

# POLY_JSON = os.path.join("polygons", "polys_20251205_125802_320733.json")

WARMUP_ITERS  = 3

SAVE_ASYNC    = False
SAVE_ROOT     = "captures"
SAVE_FORMAT   = ".png"
SAVE_IMG      = True


# ============ Cameras (CPU) ============
class CameraStream:
    def __init__(self, index: int, width: int, height: int, fps: int):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L
        cap = cv2.VideoCapture(self.index, backend)
        # if not cap.isOpened():
        #     fatal_exit(f"Camera {self.index} failed to open.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFERSIZE)
        except Exception:
            pass

        self.cap = cap
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.latest = frame

    def get_latest(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.3)
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass

class MultiCam:
    def __init__(self, indices, w, h, fps):
        self.cams = {i: CameraStream(i, w, h, fps) for i in indices}
    def start_all(self):
        for cam in self.cams.values():
            cam.start()
    def stop_all(self):
        for cam in self.cams.values():
            cam.stop()
    def get_all_latest(self) -> Dict[int, Optional[np.ndarray]]:
        return {i: cam.get_latest() for i, cam in self.cams.items()}

# # ================= Polygons & masks ==================
# def load_polygons(json_path: str):
#     if not (json_path and os.path.exists(json_path)):
#         return None
#     try:
#         with open(json_path, "r") as f:
#             jd = json.load(f)
#         return jd.get("polygons", None)
#     except Exception:
#         return None

# def precompute_masks(polys: dict, size: Tuple[int,int]) -> Dict[int, np.ndarray]:
#     if not polys: return {}
#     W, H = size
#     out = {}
#     for k, pts in polys.items():
#         try:
#             cam = int(k)
#             mask = np.zeros((H, W), dtype=np.uint8)
#             cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
#             out[cam] = mask
#         except Exception:
#             continue
#     return out

# def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
#     if img is None or mask is None:
#         return img
#     out = np.full_like(img, 255, dtype=img.dtype)
#     out[mask == 255] = img[mask == 255]
#     return out

# ================= rembg (GPU-only; fail hard) ==================
def remove_background_from_image(bgr_img, session=BG_SESSION):
    if bgr_img is None:
        fatal_exit("Invalid image for rembg")

    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGBA")

    # GPU-only: if CUDA provider not working, this will raise -> fatal
    out = remove(pil_img, session=session)

    if isinstance(out, bytes):
        out_pil = Image.open(io.BytesIO(out)).convert("RGBA")
    else:
        out_pil = out.convert("RGBA")

    rgba = np.array(out_pil)
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0

    white = np.ones_like(rgb) * 255.0
    merged = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)

    return cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)

# ============== QImage helper ==============
def qimage_from_bgr(bgr: np.ndarray) -> QImage:
    h, w, _ = bgr.shape
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qi = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return qi.copy()

# ================= EfficientNet-B7 Feature Model (GPU) ==================
class EffB7_Feature(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.backbone = m.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return torch.flatten(x, 1)  # [B,2560]

# ================= GPU-only Predictor (FAISS-GPU kNN) ==================
class B7GpuCpuKnnPredictor:
    """
    - EfficientNet-B7 forward: GPU ONLY
    - kNN distance: CPU (sklearn object from artifact["knn"])
    Artifact must include: img_size, embed_mean, embed_std, knn
    """
    def __init__(self, artifact_path: str, fixed_threshold: float, warmup_iters: int = 3):
        if not torch.cuda.is_available():
            fatal_exit("CUDA required for EfficientNet-B7 (GPU-only for B7).")

        self.fixed_threshold = float(fixed_threshold)
        self.device = torch.device("cuda")

        art = joblib.load(artifact_path)
        missing = [k for k in ("img_size", "embed_mean", "embed_std", "knn") if k not in art]
        if missing:
            fatal_exit(f"Artifact missing keys {missing}. Your artifact must contain sklearn 'knn'.")

        self.img_size = int(art["img_size"])
        self.mean = np.asarray(art["embed_mean"], dtype=np.float32)
        self.std  = np.asarray(art["embed_std"], dtype=np.float32)
        self.knn  = art["knn"]  # sklearn NearestNeighbors (CPU)

        # Model GPU only
        self.model = EffB7_Feature().to(self.device).eval().to(memory_format=torch.channels_last)

        # Preprocess (CPU)
        self.tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

        # Warmup GPU
        if warmup_iters > 0:
            dummy = torch.randn(4, 3, self.img_size, self.img_size, device=self.device).to(memory_format=torch.channels_last)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                _ = self.model(dummy)
            torch.cuda.synchronize()

    def _prep_one(self, bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self.tf(pil)  # CPU tensor

    @torch.no_grad()
    def predict_batch_from_bgr_list(self, bgr_list: List[np.ndarray]) -> List[Tuple[str, float]]:
        # CPU preprocessing
        tensors = [self._prep_one(b) for b in bgr_list]
        x = torch.stack(tensors, dim=0).to(self.device, non_blocking=True).to(memory_format=torch.channels_last)

        # GPU forward
        with torch.amp.autocast("cuda", dtype=torch.float16):
            emb = self.model(x)  # (N,2560) on GPU

        # Move embeddings to CPU for sklearn kNN
        emb_np = emb.float().detach().cpu().numpy().astype(np.float32)
        emb_z = (emb_np - self.mean) / self.std

        # CPU kNN distance (sklearn)
        dists, _ = self.knn.kneighbors(emb_z, n_neighbors=self.knn.n_neighbors, return_distance=True)
        scores = dists.mean(axis=1)

        out = []
        for s in scores:
            sv = float(s)
            label = "GOOD" if sv <= self.fixed_threshold else "BAD"
            out.append((label, sv))
        return out

# ===== BAD IMAGE SAVE CONFIG =====
BAD_SAVE_ROOT = BAD_IMG_SAVE
os.makedirs(BAD_SAVE_ROOT, exist_ok=True)

# ============== Worker ==============
class PLCWorker(QThread):
    sig_plc_read  = pyqtSignal(int, str)
    sig_plc_write = pyqtSignal(int, str)
    sig_cam_res   = pyqtSignal(int, str, float, str)
    sig_cam_vis   = pyqtSignal(int, QImage)
    sig_overall   = pyqtSignal(int, str, int)
    sig_log       = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        require_cuda_or_exit()
        require_rembg_cuda_or_exit()
        self._running = True

        self.mc = MultiCam(CAM_INDICES, CAM_W, CAM_H, TARGET_FPS)
        self.mc.start_all()

        # polys = load_polygons(POLY_JSON)
        # self.masks = precompute_masks(polys, (CAM_W, CAM_H)) if polys else {}
        ARTIFACT_PATH, FIXED_THRESHOLD = load_runtime_config()
        # GPU-only predictor (FAISS-GPU)
        self.pred = B7GpuCpuKnnPredictor(
            artifact_path=ARTIFACT_PATH,
            fixed_threshold=FIXED_THRESHOLD,
            warmup_iters=WARMUP_ITERS
        )

        self.count = 1
        self.cup_count = 0


    def stop(self):
        self._running = False

    def run(self):
        try:
            client = ModbusTcpClient(SERVER_IP, port=PORT)
            if not ensure_connected(client):
                self.sig_log.emit(f"[{now_str()}] ❌ Could not connect {SERVER_IP}:{PORT}")
                self.mc.stop_all()
                return

            self.sig_log.emit(f"[{now_str()}] ✅ Connected {SERVER_IP}:{PORT}")
            self.sig_log.emit(f"[{now_str()}] ✅ GPU REQUIRED MODE: rembg+EffB7+FAISS-GPU kNN")

            last = 0
            while self._running:
                rr = client.read_holding_registers(READ_OFFSET, count=1, unit=UNIT_ID)
                if rr is None or rr.isError():
                    time.sleep(POLL_SEC)
                    continue

                val = int(rr.registers[0])
                self.sig_plc_read.emit(val, now_str())

                if val == 1 and last != 1:
                    self.cup_count += 1
                    update_cup_count_in_job_setup(self.cup_count)
                    time.sleep(0.18)

                    t0 = perf_counter_ns()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                    latest = self.mc.get_all_latest()

                    imgs: List[np.ndarray] = []
                    cams_ok: List[int] = []
                    previews: Dict[int, np.ndarray] = {}

                    for cam in CAM_INDICES:
                        img = latest.get(cam, None)
                        if img is None:
                            fatal_exit(f"No frame for cam {cam} (GPU-only mode stops).")

                        if SAVE_IMG:
                            os.makedirs("test_img", exist_ok=True)
                            cv2.imwrite(f"test_img/cam_{cam}_{self.count}.png", img)
                            self.count += 1

                        # Optional mask:
                        # if cam in self.masks:
                        #     img = apply_mask(img, self.masks[cam])

                        # GPU-only rembg (fatal on failure)
                        img = remove_background_from_image(img)

                        previews[cam] = img
                        imgs.append(img)
                        cams_ok.append(cam)

                    t1 = perf_counter_ns()

                    for cam in CAM_INDICES:
                        self.sig_cam_vis.emit(cam, qimage_from_bgr(previews[cam]))

                    # GPU-only prediction + GPU-only kNN distance
                    batch_labels_scores = self.pred.predict_batch_from_bgr_list(imgs)

                    t2 = perf_counter_ns()

                    any_bad = False
                    for cam, (label, score) in zip(cams_ok, batch_labels_scores):
                        self.sig_cam_res.emit(cam, label, score, f"{ts}{SAVE_FORMAT}")
                        
                        if label == "BAD":
                            any_bad = True
                            out_path=BAD_SAVE_ROOT
                            try:
                                fname = f"BAD_cam{cam}_{ts}.png"
                                out_path = os.path.join(BAD_SAVE_ROOT, fname)
                                cv2.imwrite(out_path, previews[cam])
                            except Exception as e:
                                self.sig_log.emit(f"[WARN] Failed to save BAD image cam {cam}: {e}")
                                # ✅ Insert into defect_table (even if save failed, path may be empty)
                            try:
                                insert_defect(
                                    cup_count=self.cup_count,
                                    camara_angle=str(cam),
                                    img_path=out_path,
                                    defect_type="BAD"
                                )
                                self.sig_log.emit(f"[DB] defect_table inserted: count={self.cup_count}, cam={cam}, path={out_path}")
                            except Exception as e:
                                self.sig_log.emit(f"[DB ERROR] insert_defect failed: {e}")

                    # ✅ ONE status per CUP (after all 4 cameras are checked)
                    try:
                        upsert_cup_entry(cup_count=self.cup_count)
                    except Exception as e:
                        self.sig_log.emit(f"[DB ERROR] upsert_cup_entry failed: {e}")

                    result = 2  if any_bad else 1# keep your logic
                    ok1 = plc_write_value(client, result)
                    if ok1: self.sig_plc_write.emit(result, now_str())
                    time.sleep(RESET_DELAY)
                    ok2 = plc_write_value(client, 0)
                    if ok2: self.sig_plc_write.emit(0, now_str())
                    print(self.cup_count)
                    self.sig_overall.emit(result, now_str(), self.cup_count)

                    t4 = perf_counter_ns()
                    grab_ms  = (t1 - t0) / 1e6
                    pred_ms  = (t2 - t1) / 1e6
                    total_ms = (t4 - t0) / 1e6
                    self.sig_log.emit(f"[TIMING] grab+rembg:{grab_ms:.1f} ms | predict+knn(GPU):{pred_ms:.1f} ms | total:{total_ms:.1f} ms")

                    last = 1
                elif val != 1:
                    last = 0

                time.sleep(POLL_SEC)

            try: client.close()
            except Exception: pass
            self.mc.stop_all()
            self.sig_log.emit(f"[{now_str()}] 👋 Worker stopped.")

        except Exception as e:
            fatal_exit(f"GPU-only mode crash: {e}")
            
# =========================
# HARD GPU REQUIREMENT
# =========================
def fatal_exit(msg: str, code: int = 1):
    print("❌ FATAL:", msg)
    try:
        import os as _os
        _os._exit(code)  # hard kill (best for QThread situations)
    except Exception:
        sys.exit(code)

def require_cuda_or_exit():
    if not torch.cuda.is_available():
        fatal_exit("CUDA GPU is NOT available. This app is GPU-required (no CPU mode).")

def require_rembg_cuda_or_exit():
    prov = ort.get_available_providers()
    if "CUDAExecutionProvider" not in prov:
        fatal_exit(f"rembg needs CUDAExecutionProvider but it's not available. Providers={prov}")
        
JOB_SETUP_JSON_FILE = JOBID_JSON_FILE
_job_setup_lock = threading.Lock()

def load_runtime_config():
    """
    Loads ARTIFACT_PATH and FIXED_THRESHOLD from job_setup.json
    """
    if not os.path.exists(JOB_SETUP_JSON_FILE):
        raise FileNotFoundError("job_setup.json not found. Please configure Job ID Setup.")

    with open(JOB_SETUP_JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        artifact_path = data["threshold_path"]["value"]
        threshold = float(data["threshold"]["value"])
    except Exception as e:
        raise ValueError(f"Invalid job_setup.json format: {e}")

    if not artifact_path or not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

    return artifact_path, threshold

# ARTIFACT_PATH, FIXED_THRESHOLD = load_runtime_config()


def read_job_setup_json() -> dict:
    if not os.path.exists(JOB_SETUP_JSON_FILE):
        return {}
    try:
        with open(JOB_SETUP_JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def update_job_setup_json(key_path: List[str], value):
    """
    key_path example: ["cup_count", "value"]
    """
    os.makedirs(os.path.dirname(JOB_SETUP_JSON_FILE), exist_ok=True)

    with _job_setup_lock:
        data = read_job_setup_json()

        # walk / create nested dicts
        cur = data
        for k in key_path[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]

        cur[key_path[-1]] = value

        with open(JOB_SETUP_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

def update_cup_count_in_job_setup(cup_count: int):
    update_job_setup_json(["cup_count", "value"], int(cup_count))
    update_job_setup_json(["cup_count", "updated_at"], datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# =========================
# Helpers
# =========================

def ensure_connected(client: ModbusTcpClient) -> bool:
    try:
        return client.connect()
    except Exception:
        return False

def plc_write_value(client: ModbusTcpClient, value: int) -> bool:
    try:
        wr = client.write_register(WRITE_OFFSET, int(value), unit=UNIT_ID)
        return (wr is not None) and (not wr.isError())
    except Exception:
        return False