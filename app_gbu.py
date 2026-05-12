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
from classes.controller_popup import ControllerPopup
from classes.settings_popup import SettingsPopup
import platform
import io
from classes.database import insert_defect,upsert_cup_entry
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

require_cuda_or_exit()

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

# ---------- PyQt5 ----------
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QFrame, QSplitter, QSizePolicy, QScrollArea
)
from PyQt5.QtGui import QImage, QPixmap

# ---------- Modbus ----------
from pymodbus.client.sync import ModbusTcpClient

# ---------- rembg (GPU ONLY) ----------
from rembg import remove, new_session
import onnxruntime as ort

BG_MODEL_NAME = "isnet-general-use"

def require_rembg_cuda_or_exit():
    prov = ort.get_available_providers()
    if "CUDAExecutionProvider" not in prov:
        fatal_exit(f"rembg needs CUDAExecutionProvider but it's not available. Providers={prov}")

require_rembg_cuda_or_exit()

# GPU-only providers: remove CPUExecutionProvider completely
BG_SESSION = new_session(
    BG_MODEL_NAME,
    providers=["CUDAExecutionProvider"]
)


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

CAM_INDICES  = (2, 6, 4, 0)
CAM_W, CAM_H = 1280, 720
TARGET_FPS   = 15
BUFFERSIZE   = 1

POLY_JSON = os.path.join("polygons", "polys_20251205_125802_320733.json")

# ARTIFACT_PATH   = "oneclass_bg_visa.joblib"
# FIXED_THRESHOLD = 75.0

WARMUP_ITERS  = 3

SAVE_ASYNC    = False
SAVE_ROOT     = "captures"
SAVE_FORMAT   = ".png"
SAVE_IMG      = True



JOB_SETUP_JSON_FILE = os.path.join("config", "job_setup.json")
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

ARTIFACT_PATH, FIXED_THRESHOLD = load_runtime_config()


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
def now_str():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

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

def precompute_masks(polys: dict, size: Tuple[int,int]) -> Dict[int, np.ndarray]:
    if not polys: return {}
    W, H = size
    out = {}
    for k, pts in polys.items():
        try:
            cam = int(k)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
            out[cam] = mask
        except Exception:
            continue
    return out

def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if img is None or mask is None:
        return img
    out = np.full_like(img, 255, dtype=img.dtype)
    out[mask == 255] = img[mask == 255]
    return out

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
BAD_SAVE_ROOT = "bad_images"
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
        self._running = True

        self.mc = MultiCam(CAM_INDICES, CAM_W, CAM_H, TARGET_FPS)
        self.mc.start_all()

        # polys = load_polygons(POLY_JSON)
        # self.masks = precompute_masks(polys, (CAM_W, CAM_H)) if polys else {}

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

# Brand / theme
WINE_RED = "#5a191f"
PANEL_BG = "#5a191f"

class StatusCard(QFrame):
    def __init__(self, cam_id: int):
        super().__init__()
        self.cam_id = cam_id

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            QFrame {{
                background:{PANEL_BG};
                border:1px solid #444;
                border-radius:10px;
                padding:6px;
            }}
            QLabel#CamBadge {{
                background: #5a191f;
                border-radius: 12px;
                padding: 6px 10px;
                font-weight: 700;
            }}
        """)

        root = QHBoxLayout()
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(4)

        self.preview = QLabel()
        self.preview.setStyleSheet("background:#000; border:1px solid #222;")
        self.preview.setScaledContents(True)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.res = QLabel("—")
        self.score_lbl = QLabel("Score: —")

        left.addWidget(self.preview, 1)
        left.addWidget(self.res)
        left.addWidget(self.score_lbl)

        side = QVBoxLayout()
        side.addStretch(1)
        self.badge = QLabel(f"Cam {cam_id}")
        self.badge.setObjectName("CamBadge")
        side.addWidget(self.badge, alignment=Qt.AlignRight)
        side.addStretch(1)

        side_wrap = QWidget()
        side_wrap.setLayout(side)
        side_wrap.setFixedWidth(76)

        root.addLayout(left, 1)
        root.addWidget(side_wrap, 0)
        self.setLayout(root)

    def set_result(self, label: str, score: float, fname: str):
        self.res.setText(f"Result: {label}")
        self.score_lbl.setText(f"Score: {score:.3f}  (thr {FIXED_THRESHOLD:.2f})")
        if label == "BAD":
            self.setStyleSheet(f"""
                QFrame {{ background:{PANEL_BG}; border:2px solid #ff4d4f; border-radius:10px; padding:6px; }}
                QLabel#CamBadge {{ background:#5a191f; border-radius:12px; padding:6px 10px; font-weight:700; }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame {{ background:{PANEL_BG}; border:2px solid #52c41a; border-radius:10px; padding:6px; }}
                QLabel#CamBadge {{ background:#5a191f; border-radius:12px; padding:6px 10px; font-weight:700; }}
            """)

    def set_preview(self, qimg: QImage):
        self.preview.setPixmap(QPixmap.fromImage(qimg))

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.settings_popup = None
        self.controller_popup = None

        self.setWindowTitle("𝔗𝔢𝔵𝔞-ℭ𝔲𝔭 𝔦𝔫𝔰𝔭𝔢𝔠𝔱𝔦𝔬𝔫 𝔭𝔯𝔬")
        self.resize(980, 600)
        self.setStyleSheet(f"QWidget {{ background: {WINE_RED}; color:#f5f5f5; }}")

        header = QHBoxLayout()
        header.setContentsMargins(10, 6, 10, 6)
        header.setSpacing(10)

        self.logo = QLabel()
        self.logo.setFixedSize(125, 40)
        self.logo.setScaledContents(True)
        self.logo.setPixmap(QPixmap("assets/logo-tr.png"))
        self.logo.setStyleSheet("margin-left:10px;padding:5px;border-radius:5px;background-color:white;")

        self.title_lbl = QLabel("Cup Inspection Pro")
        self.title_lbl.setAlignment(Qt.AlignCenter)
        self.title_lbl.setStyleSheet("font-size: 25px; font-weight: 700;")

        self.lbl_score = QLabel("Count : 0")
        self.lbl_score.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_score.setStyleSheet("font-size: 20px; font-weight: 600;")

        header.addWidget(self.logo)
        header.addStretch(1)
        header.addWidget(self.title_lbl)
        header.addStretch(1)
        header.addWidget(self.lbl_score)

        status_bar = QHBoxLayout()
        status_bar.setContentsMargins(10, 2, 10, 4)
        status_bar.setSpacing(20)

        self.lbl_read = QLabel("Read 40001: —")
        self.lbl_write = QLabel("Write 40002: —")
        for lbl in (self.lbl_read, self.lbl_write):
            lbl.setStyleSheet("font-size:12px;")

        status_bar.addWidget(self.lbl_read)
        status_bar.addWidget(self.lbl_write)
        status_bar.addStretch(1)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_reset = QPushButton("Reset")
        self.btn_settings = QPushButton("Settings")
        self.btn_controller = QPushButton("Controller")

        self.btn_stop.setEnabled(False)
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_controller.clicked.connect(self.open_controller)

        for b in (self.btn_start, self.btn_stop, self.btn_reset, self.btn_settings, self.btn_controller):
            b.setFixedHeight(34)
            b.setMinimumWidth(110)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(6, 2, 6, 2)
        ctrl.setSpacing(5)
        ctrl.addStretch(1)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        ctrl.addWidget(self.btn_reset)
        ctrl.addSpacing(10)
        ctrl.addWidget(self.btn_settings)
        ctrl.addWidget(self.btn_controller)
        ctrl.addStretch(1)

        self.cards: Dict[int, StatusCard] = {}
        grid = QGridLayout()
        grid.setSpacing(6)

        idx = 0
        for r in range(2):
            for c in range(2):
                cam = CAM_INDICES[idx]
                card = StatusCard(cam)
                self.cards[cam] = card
                grid.addWidget(card, r, c)
                idx += 1

        grid_wrap = QWidget()
        grid_wrap.setLayout(grid)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(grid_wrap)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(scroll)

        foot = QHBoxLayout()
        foot.setContentsMargins(6, 2, 6, 6)
        self.lbl_overall = QLabel("Overall: —")
        self.lbl_overall.setStyleSheet("font-weight:600;")
        foot.addWidget(self.lbl_overall)
        foot.addStretch(1)

        v = QVBoxLayout()
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)
        v.addLayout(header)
        v.addLayout(status_bar)
        v.addWidget(splitter, 1)
        v.addLayout(ctrl)
        v.addLayout(foot)
        self.setLayout(v)

        self.worker = PLCWorker()
        self.worker.sig_plc_read.connect(self.on_plc_read)
        self.worker.sig_plc_write.connect(self.on_plc_write)
        self.worker.sig_cam_res.connect(self.on_cam_res)
        self.worker.sig_cam_vis.connect(self.on_cam_vis)
        self.worker.sig_overall.connect(self.on_overall)
        self.worker.sig_log.connect(self.append_log)

        self.btn_start.clicked.connect(self.start_worker)
        self.btn_stop.clicked.connect(self.stop_worker)

        self._resize_previews()

    def _resize_previews(self):
        grid_height_hint = int(max(180, self.height() * 0.38))
        per_row = max(120, (grid_height_hint - 36) // 2)
        prev_h = per_row
        prev_w = int(prev_h * 16 / 9)
        max_w = (self.width() - 64) // 2
        if prev_w > max_w:
            prev_w = max_w
            prev_h = int(prev_w * 9 / 16)
        for card in self.cards.values():
            card.preview.setFixedSize(QSize(prev_w, prev_h))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._resize_previews()

    def open_settings(self):
        if self.settings_popup is None or not self.settings_popup.isVisible():
            self.settings_popup = SettingsPopup(self)
            self.settings_popup.show()
        else:
            self.settings_popup.raise_()
            self.settings_popup.activateWindow()

    def open_controller(self):
        if self.controller_popup is None or not self.controller_popup.isVisible():
            self.controller_popup = ControllerPopup(self)
            self.controller_popup.show()
        else:
            self.controller_popup.raise_()
            self.controller_popup.activateWindow()

    def on_plc_read(self, value: int, t: str):
        self.lbl_read.setText(f"Read 40001: {value} @ {t}")

    def on_plc_write(self, value: int, t: str):
        self.lbl_write.setText(f"Write 40002: {value} @ {t}")

    def on_cam_res(self, cam: int, label: str, score: float, fname: str):
        self.cards[cam].set_result(label, score, fname)

    def on_cam_vis(self, cam: int, qimg: QImage):
        self.cards[cam].set_preview(qimg)

    def on_overall(self, result: int, t: str,cup_count):
        txt = "ALL GOOD (wrote 1→0)" if result == 1 else "ANY BAD (wrote 2→0)"
        self.lbl_overall.setText(f"Overall: {txt} @ {t}")
        self.lbl_score.setText(f"Count : {cup_count}")

    def append_log(self, line: str):
        print(line)

    def start_worker(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.append_log(f"[{now_str()}] ▶️ Start")
        self.worker.start()

    def stop_worker(self):
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.append_log(f"[{now_str()}] ⏹ Stop requested")
        self.worker.stop()

    def closeEvent(self, e):
        try:
            if self.worker.isRunning():
                self.worker.stop()
                self.worker.wait(1500)
        except Exception:
            pass
        e.accept()

def main():
    from classes.database import create_database_and_tables
    create_database_and_tables()
    app = QApplication(sys.argv)
    w = MainWindow()
    w.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
