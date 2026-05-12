# =========================
# LIVE CAPTURE + SAVE BMP (NO REMBG, NO PREDICTION)
# Folder: capture_remove/CAM1..CAM4
# Auto camera mapping (indices may change)
# =========================

import os, sys, time, json, threading, platform, glob, subprocess, re
from datetime import datetime
from typing import Dict, Optional, List

import numpy as np
import cv2

# ---------- Torch (GPU REQUIRED) ----------
import torch

# =========================
# HARD GPU REQUIREMENT
# =========================
def fatal_exit(msg: str, code: int = 1):
    print("❌ FATAL:", msg)
    try:
        import os as _os
        _os._exit(code)
    except Exception:
        sys.exit(code)

def require_cuda_or_exit():
    if not torch.cuda.is_available():
        fatal_exit("CUDA GPU is NOT available. This app is GPU-required (no CPU mode).")

require_cuda_or_exit()

# =========================
# QT PLATFORM INIT (PyInstaller safe-ish)
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


# =========================
# CONFIG
# =========================
SERVER_IP    = "192.168.3.1"
PORT         = 507
UNIT_ID      = 1
READ_OFFSET  = 1          # 40001
POLL_SEC     = 0.005
TRIGGER_DELAY_SEC = 0.18
TARGET_FPS   = 15
BUFFERSIZE   = 1

CAM_W, CAM_H = 1280, 720

CAPTURE_ROOT = "capture_with_bg"
SAVE_FORMAT  = ".bmp"

# Persistent mapping (stable)
CAM_MAP_FILE = os.path.join("config", "cam_map.json")

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

def _run(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception:
        return ""

def _v4l2_usb_id(dev: str) -> str:
    """
    Stable identity for a camera using v4l2-ctl output.
    Needs: sudo apt install v4l-utils
    """
    out = _run(["v4l2-ctl", "-D", "-d", dev])
    m = re.search(r"Bus info\s*:\s*(.+)", out)
    if m:
        return m.group(1).strip()
    m = re.search(r"Card type\s*:\s*(.+)", out)
    if m:
        return m.group(1).strip()
    return dev

def _opencv_can_open(index: int) -> bool:
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_V4L) if os.name != "nt" else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(index, backend)
    ok = cap.isOpened()
    if ok:
        ret, _ = cap.read()
        ok = bool(ret)
    try:
        cap.release()
    except Exception:
        pass
    return ok

def scan_cameras(max_index: int = 50) -> List[dict]:
    cams = []
    if os.name == "nt":
        for idx in range(max_index + 1):
            if _opencv_can_open(idx):
                cams.append({"index": idx, "device": f"IDX{idx}", "usb_id": f"IDX{idx}"})
        return cams

    devs = sorted(glob.glob("/dev/video*"), key=lambda p: int(re.findall(r"\d+", p)[0]))
    for dev in devs:
        idx = int(re.findall(r"\d+", dev)[0])
        if idx > max_index:
            continue
        if _opencv_can_open(idx):
            cams.append({"index": idx, "device": dev, "usb_id": _v4l2_usb_id(dev)})
    return cams

def load_cam_map() -> dict:
    if not os.path.exists(CAM_MAP_FILE):
        return {}
    try:
        with open(CAM_MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def save_cam_map(m: dict):
    os.makedirs(os.path.dirname(CAM_MAP_FILE), exist_ok=True)
    with open(CAM_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)

def resolve_cam_map(num_cams: int = 4) -> Dict[str, dict]:
    cams_now = scan_cameras(max_index=50)
    if len(cams_now) == 0:
        fatal_exit("No cameras detected by OpenCV.")

    saved = load_cam_map()
    now_by_usb = {c["usb_id"]: c for c in cams_now}
    resolved_map: Dict[str, dict] = {}

    # 1) resolve using saved usb_id
    if saved:
        for i in range(1, num_cams + 1):
            key = f"CAM{i}"
            usb_id = (saved.get(key) or {}).get("usb_id", "")
            if usb_id and usb_id in now_by_usb:
                resolved_map[key] = now_by_usb[usb_id]

    # 2) rebuild if not enough
    if len(resolved_map) < num_cams:
        cams_sorted = sorted(cams_now, key=lambda x: x["index"])
        resolved_map = {}
        for i, cam in enumerate(cams_sorted[:num_cams], start=1):
            resolved_map[f"CAM{i}"] = cam
        save_cam_map(resolved_map)

    return resolved_map

# Resolve camera mapping at startup
CAM_MAP = resolve_cam_map(num_cams=4)
print("[CAM] Mapping:")
for k in sorted(CAM_MAP.keys()):
    print(f"  {k}: index={CAM_MAP[k]['index']} usb_id={CAM_MAP[k]['usb_id']} dev={CAM_MAP[k]['device']}")

# Create folders CAM1..CAM4
os.makedirs(CAPTURE_ROOT, exist_ok=True)
for i in range(1, 5):
    os.makedirs(os.path.join(CAPTURE_ROOT, f"CAM{i}"), exist_ok=True)

# ============== QImage helper ==============
def qimage_from_bgr(bgr: np.ndarray) -> QImage:
    h, w, _ = bgr.shape
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qi = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return qi.copy()

# ============ Cameras (CPU live) ============
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
        backend = cv2.CAP_DSHOW if os.name == "nt" else getattr(cv2, "CAP_V4L2", cv2.CAP_V4L)
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            print(f"[WARN] Camera index {self.index} failed to open")

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
            try:
                self.cap.release()
            except Exception:
                pass

class MultiCam:
    def __init__(self, cam_map: Dict[str, dict], w, h, fps):
        self.cam_map = cam_map
        self.cams = {k: CameraStream(v["index"], w, h, fps) for k, v in cam_map.items()}

    def start_all(self):
        for cam in self.cams.values():
            cam.start()

    def stop_all(self):
        for cam in self.cams.values():
            cam.stop()

    def get_latest_for(self, cam_key: str) -> Optional[np.ndarray]:
        return self.cams[cam_key].get_latest()

# ============== Worker (PLC trigger read only) ==============
class PLCWorker(QThread):
    sig_plc_read  = pyqtSignal(int, str)
    sig_cam_vis   = pyqtSignal(str, QImage)
    sig_log       = pyqtSignal(str)
    sig_count     = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self.mc = MultiCam(CAM_MAP, CAM_W, CAM_H, TARGET_FPS)
        self.mc.start_all()
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
            self.sig_log.emit(f"[{now_str()}] ✅ MODE: AUTO CAM MAP + SAVE BMP (NO REMBG, NO PREDICTION)")

            last = 0
            while self._running:
                rr = client.read_holding_registers(READ_OFFSET, count=1, unit=UNIT_ID)
                if rr is None or rr.isError():
                    time.sleep(POLL_SEC)
                    continue

                val = int(rr.registers[0])
                self.sig_plc_read.emit(val, now_str())

                if val == 1 and last != 1:
                    time.sleep(TRIGGER_DELAY_SEC)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                    for cam_key in ["CAM1", "CAM2", "CAM3", "CAM4"]:
                        img = self.mc.get_latest_for(cam_key)
                        if img is None:
                            self.sig_log.emit(f"[WARN] {cam_key} no frame (index={CAM_MAP[cam_key]['index']}) - skipped")
                            continue

                        cam_dir = os.path.join(CAPTURE_ROOT, cam_key)
                        os.makedirs(cam_dir, exist_ok=True)

                        real_idx = CAM_MAP[cam_key]["index"]
                        fname = f"{ts}_IDX{real_idx}{SAVE_FORMAT}"
                        save_path = os.path.join(cam_dir, fname)

                        ok = cv2.imwrite(save_path, img)
                        if ok:
                            self.sig_log.emit(f"[SAVE] {cam_key} (idx={real_idx}) -> {save_path}")
                        else:
                            self.sig_log.emit(f"[WARN] Failed to save: {save_path}")

                        self.sig_cam_vis.emit(cam_key, qimage_from_bgr(img))

                    self.cup_count += 1
                    self.sig_count.emit(self.cup_count)
                    self.sig_log.emit(f"[{now_str()}] ✅ Cup {self.cup_count} saved")
                    last = 1
                elif val != 1:
                    last = 0

                time.sleep(POLL_SEC)

            try:
                client.close()
            except Exception:
                pass

            self.mc.stop_all()
            self.sig_log.emit(f"[{now_str()}] 👋 Worker stopped.")

        except Exception as e:
            fatal_exit(f"Crash in capture-only mode: {e}")

# Brand / theme
WINE_RED = "#5a191f"
PANEL_BG = "#5a191f"

class StatusCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
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
        self.preview = QLabel()
        self.preview.setStyleSheet("background:#000; border:1px solid #222;")
        self.preview.setScaledContents(True)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.preview, 1)

        side = QVBoxLayout()
        side.addStretch(1)
        self.badge = QLabel(title)
        self.badge.setObjectName("CamBadge")
        side.addWidget(self.badge, alignment=Qt.AlignRight)
        side.addStretch(1)

        side_wrap = QWidget()
        side_wrap.setLayout(side)
        side_wrap.setFixedWidth(90)

        root.addLayout(left, 1)
        root.addWidget(side_wrap, 0)
        self.setLayout(root)

    def set_preview(self, qimg: QImage):
        self.preview.setPixmap(QPixmap.fromImage(qimg))

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("𝔗𝔢𝔵𝔞-ℭ𝔲𝔭 𝔦𝔫𝔰𝔭𝔢𝔠𝔱𝔦𝔬𝔫 𝔭𝔯𝔬 (CAPTURE ONLY)")
        self.resize(980, 600)
        self.setStyleSheet(f"QWidget {{ background: {WINE_RED}; color:#f5f5f5; }}")

        header = QHBoxLayout()
        header.setContentsMargins(10, 6, 10, 6)

        self.logo = QLabel()
        self.logo.setFixedSize(125, 40)
        self.logo.setScaledContents(True)
        try:
            self.logo.setPixmap(QPixmap("assets/logo-tr.png"))
        except Exception:
            pass
        self.logo.setStyleSheet("margin-left:10px;padding:5px;border-radius:5px;background-color:white;")

        self.title_lbl = QLabel("PLC Trigger → Capture → Save BMP (NO REMBG)")
        self.title_lbl.setAlignment(Qt.AlignCenter)
        self.title_lbl.setStyleSheet("font-size: 20px; font-weight: 700;")

        self.lbl_count = QLabel("Count : 0")
        self.lbl_count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_count.setStyleSheet("font-size: 20px; font-weight: 600;")

        header.addWidget(self.logo)
        header.addStretch(1)
        header.addWidget(self.title_lbl)
        header.addStretch(1)
        header.addWidget(self.lbl_count)

        status_bar = QHBoxLayout()
        status_bar.setContentsMargins(10, 2, 10, 4)
        self.lbl_read = QLabel("Read 40001: —")
        self.lbl_read.setStyleSheet("font-size:12px;")
        status_bar.addWidget(self.lbl_read)
        status_bar.addStretch(1)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        for b in (self.btn_start, self.btn_stop):
            b.setFixedHeight(34)
            b.setMinimumWidth(110)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(6, 2, 6, 2)
        ctrl.addStretch(1)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        ctrl.addStretch(1)

        self.cards: Dict[str, StatusCard] = {}
        grid = QGridLayout()
        grid.setSpacing(6)

        for pos, cam_key in enumerate(["CAM1", "CAM2", "CAM3", "CAM4"]):
            r = pos // 2
            c = pos % 2
            idx = CAM_MAP[cam_key]["index"]
            title = f"{cam_key} (idx={idx})"
            card = StatusCard(title)
            self.cards[cam_key] = card
            grid.addWidget(card, r, c)

        grid_wrap = QWidget()
        grid_wrap.setLayout(grid)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(grid_wrap)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(scroll)

        v = QVBoxLayout()
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)
        v.addLayout(header)
        v.addLayout(status_bar)
        v.addWidget(splitter, 1)
        v.addLayout(ctrl)
        self.setLayout(v)

        self.worker = PLCWorker()
        self.worker.sig_plc_read.connect(self.on_plc_read)
        self.worker.sig_cam_vis.connect(self.on_cam_vis)
        self.worker.sig_log.connect(self.append_log)
        self.worker.sig_count.connect(self.on_count)

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

    def on_plc_read(self, value: int, t: str):
        self.lbl_read.setText(f"Read 40001: {value} @ {t}")

    def on_cam_vis(self, cam_key: str, qimg: QImage):
        if cam_key in self.cards:
            self.cards[cam_key].set_preview(qimg)

    def on_count(self, c: int):
        self.lbl_count.setText(f"Count : {c}")

    def append_log(self, line: str):
        print(line)

    def start_worker(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.append_log(f"[{now_str()}] ▶️ Start (auto camera mapping)")
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
    app = QApplication(sys.argv)
    w = MainWindow()
    w.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
