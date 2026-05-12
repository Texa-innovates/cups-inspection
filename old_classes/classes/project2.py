import sys
import os
import json
import cv2
import time
import math
import threading
import numpy as np
from datetime import datetime
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout,
    QFileDialog, QDialog, QMessageBox, QComboBox, QLineEdit, QFormLayout
)

# ------------------------- Config -------------------------
CAM_INDICES = (2 , 6 , 4 , 0)    # adjust to your 4 e-con camera indices
CAM_W, CAM_H, CAM_FPS = 1280, 720, 15
SAVE_ROOT = "captures"
JSON_DIR = "polygons"

os.makedirs(SAVE_ROOT, exist_ok=True)
for i in CAM_INDICES:
    os.makedirs(os.path.join(SAVE_ROOT, f"camera{i}"), exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# --------------------- Camera Worker ----------------------
class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage, int)  # preview signal

    def __init__(self, cam_index, width, height, fps, cond=None, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cond = cond
        self._running = threading.Event()
        self._running.clear()
        self._cap = None
        self._lock = threading.Lock()
        self._last_bgr = None  # numpy frame cache

    def open_camera(self):
        cap = cv2.VideoCapture(
            self.cam_index,
            cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L
        )
        if not cap.isOpened():
            print(f"❌ Camera {self.cam_index} failed to open.")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap
        return True

    def run(self):
        if not self.open_camera():
            return
        self._running.set()
        # continuous streaming, but we also support sync snapshot via cond.notify_all()
        while self._running.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            # cache last frame
            with self._lock:
                self._last_bgr = frame.copy()
            # preview
            qimg = self._to_qimage(frame)
            self.frame_ready.emit(qimg, self.cam_index)

            # check if a sync wake happened; we simply keep frames fresh
            # main thread will fetch latest cached frames
            time.sleep(1.0 / max(1, self.fps))
        self._release()

    def get_latest_frame(self):
        with self._lock:
            if self._last_bgr is None:
                return None
            return self._last_bgr.copy()

    def stop(self):
        self._running.clear()
        if self.cond:
            with self.cond:
                self.cond.notify_all()
        self.wait(1000)

    def _release(self):
        try:
            if self._cap and self._cap.isOpened():
                self._cap.release()
        except Exception as e:
            print("Error releasing camera:", e)

    @staticmethod
    def _to_qimage(frame_bgr: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

# -------------------- Polygon Drawer ----------------------
class PolygonCanvas(QLabel):
    """Image canvas that lets user click to define polygon."""
    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setPixmap(pixmap)
        self.base_pix = pixmap
        self.points = []  # list of QPoint
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.points.append(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace and self.points:
            self.points.pop()
            self.update()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            # finish handled by dialog
            pass
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.points:
            return
        painter = QPainter(self)
        pen = QPen(Qt.green, 2)
        painter.setPen(pen)
        # draw poly lines
        for i in range(1, len(self.points)):
            painter.drawLine(self.points[i-1], self.points[i])
        # close hint
        if len(self.points) >= 3:
            painter.drawLine(self.points[-1], self.points[0])
        # draw points
        for p in self.points:
            painter.drawEllipse(p, 3, 3)

    def get_polygon(self, img_w, img_h):
        """Return polygon as list of (x,y) in image coordinates (scaled if label scaled)."""
        if not self.points or len(self.points) < 3:
            return None
        # Map label coords to image coords based on how pixmap was scaled to label size
        # We rendered pixmap as-is; ensure label size equals pixmap size to keep 1:1.
        # Here we assume no scaling; but to be safe clamp to image bounds.
        poly = []
        for pt in self.points:
            x = int(np.clip(pt.x(), 0, img_w - 1))
            y = int(np.clip(pt.y(), 0, img_h - 1))
            poly.append([x, y])
        return poly

class PolygonDialog(QDialog):
    """Walks user through drawing polygons for 4 images; returns dict cam_idx -> polygon list."""
    def __init__(self, frames_by_cam: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw Polygons — Left click add, Backspace undo, Enter to proceed")
        self.frames_by_cam = frames_by_cam  # cam_idx -> BGR np array
        self.cam_ids = sorted(frames_by_cam.keys())
        self.curr_idx = 0
        self.polygons = {}  # str(cam_idx) -> [[x,y],...]
    

        self.label = None
        self.info = QLabel("")
        self.next_btn = QPushButton("Next (Enter)")
        self.next_btn.clicked.connect(self.next_camera)

        self._build_ui()
        self._load_current()

    def _build_ui(self):
        v = QVBoxLayout()
        self.label = PolygonCanvas(QPixmap(4, 4))  # will be replaced
        v.addWidget(self.label)
        v.addWidget(self.info)
        v.addWidget(self.next_btn)
        self.setLayout(v)
        self.resize(300, 200)

    def _load_current(self):
        cam = self.cam_ids[self.curr_idx]
        bgr = self.frames_by_cam[cam]
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        # ensure label is exact size of image for 1:1 mapping
        self.label.setFixedSize(QSize(1000, 700))
        self.label.setPixmap(pix)
        self.label.base_pix = pix
        self.label.points = []
        self.info.setText(f"Camera {cam}: click to add points; Backspace=undo; Enter or 'Next' to continue.")

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.next_camera()
        else:
            super().keyPressEvent(event)

    def next_camera(self):
        # save polygon
        cam = self.cam_ids[self.curr_idx]
        bgr = self.frames_by_cam[cam]
        h, w = bgr.shape[:2]
        poly = self.label.get_polygon(w, h)
        if poly is None:
            QMessageBox.warning(self, "Polygon required", f"Please draw a polygon for camera {cam}.")
            return
        self.polygons[str(cam)] = poly

        # move next
        self.curr_idx += 1
        if self.curr_idx >= len(self.cam_ids):
            self.accept()
        else:
            self._load_current()

    @staticmethod
    def apply_mask_keep_inside_white_out(bgr: np.ndarray, polygon_xy: list) -> np.ndarray:
        h, w = bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([polygon_xy], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        # create white image
        out = np.ones_like(bgr, dtype=np.uint8) * 255
        # copy inside polygon
        out[mask == 255] = bgr[mask == 255]
        return out

# --------------------- Main UI Window ---------------------
class MultiCamWindow(QWidget):
    frame_captured = pyqtSignal(dict)  # cam_idx -> BGR np (sync snapshots)

    def __init__(self, cam_indices=CAM_INDICES, cam_w=CAM_W, cam_h=CAM_H, cam_fps=CAM_FPS):
        super().__init__()
        self.setWindowTitle("4-Camera Live + Polygon Cutting — PyQt5")
        self.cam_indices = list(cam_indices)
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.cam_fps = cam_fps

        self.sync_cond = threading.Condition()
        self.workers = {}
        self.preview_labels = {}
        self.pixmaps = {}

        self._build_ui()
        self._create_workers()
        for w in self.workers.values():
            w.start()
        self.refresh_json_list()

    def _build_ui(self):
        main = QVBoxLayout()
        grid = QGridLayout()
        idx = 0
        for r in range(2):
            for c in range(2):
                cam = self.cam_indices[idx]
                lbl = QLabel(f"Cam {cam}")
                lbl.setFixedSize(QSize(320, 180))
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("background:#111;color:#ddd;border:1px solid #333;")
                grid.addWidget(lbl, r, c)
                self.preview_labels[cam] = lbl
                idx += 1
        main.addLayout(grid)

        # controls
        row1 = QHBoxLayout()
        self.poly_cut_btn = QPushButton("Polygon Cutting")
        self.poly_capture_btn = QPushButton("Poly Capture (apply JSON)")
        self.quit_btn = QPushButton("Quit")

        row1.addWidget(self.poly_cut_btn)
        row1.addWidget(self.poly_capture_btn)
        row1.addStretch()
        row1.addWidget(self.quit_btn)
        main.addLayout(row1)

        # JSON chooser
        row2 = QHBoxLayout()
        self.json_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh JSON List")
        self.new_json_name = QLineEdit()
        self.new_json_name.setPlaceholderText("Optional: name for new JSON (without .json)")
        row2.addWidget(QLabel("Polygon JSON:"))
        row2.addWidget(self.json_combo)
        row2.addWidget(self.refresh_btn)
        row2.addWidget(self.new_json_name)
        main.addLayout(row2)

        self.setLayout(main)

        # signals
        self.quit_btn.clicked.connect(self.close)
        self.poly_cut_btn.clicked.connect(self.on_polygon_cutting)
        self.poly_capture_btn.clicked.connect(self.on_poly_capture)
        self.refresh_btn.clicked.connect(self.refresh_json_list)

    def _create_workers(self):
        for idx in self.cam_indices:
            w = CameraWorker(idx, self.cam_w, self.cam_h, self.cam_fps, cond=self.sync_cond)
            w.frame_ready.connect(self.on_frame_ready)
            self.workers[idx] = w

    def on_frame_ready(self, qimg: QImage, cam_index: int):
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview_labels[cam_index].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_labels[cam_index].setPixmap(pix)
        self.pixmaps[cam_index] = pix

    # --------- Helper: get near-simultaneous frames ----------
    def get_sync_frames(self, wait_ms=150) -> dict:
        # Since workers run continuously and cache last frames,
        # we "sync" by sampling nearly at once.
        frames = {}
        # small barrier: sleep a tiny moment so all workers refreshed
        time.sleep(wait_ms / 1000.0)
        for cam in self.cam_indices:
            frm = self.workers[cam].get_latest_frame()
            if frm is None:
                raise RuntimeError(f"Camera {cam} has no frame yet.")
            frames[cam] = frm
        return frames

    # ---------------- Polygon Cutting flow -------------------
    def on_polygon_cutting(self):
        try:
            frames = self.get_sync_frames(wait_ms=120)
        except Exception as e:
            QMessageBox.critical(self, "Capture failed", str(e))
            return

        dlg = PolygonDialog(frames, self)
        if dlg.exec_() != QDialog.Accepted:
            return

        # We have polygons for each camera; apply mask & save
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        poly_dict = {
            "created_at": datetime.now().astimezone().isoformat(),
            "camera_indices": list(self.cam_indices),
            "polygons": dlg.polygons,  # keys are strings of camera indices
        }

        # Save masked images and JSON
        for cam_str, poly in dlg.polygons.items():
            cam = int(cam_str)
            bgr = frames[cam]
            masked = PolygonDialog.apply_mask_keep_inside_white_out(bgr, poly)
            outpath = os.path.join(SAVE_ROOT, f"camera{cam}", f"polycut_{ts}.png")
            cv2.imwrite(outpath, masked)

        # Decide JSON file name
        custom = self.new_json_name.text().strip()
        if custom:
            base = "".join(ch for ch in custom if ch.isalnum() or ch in ("-", "_"))
            json_name = f"{base}.json"
        else:
            json_name = f"polys_{ts}.json"

        json_path = os.path.join(JSON_DIR, json_name)
        with open(json_path, "w") as f:
            json.dump(poly_dict, f, indent=2)
        QMessageBox.information(self, "Saved", f"Masked images saved.\nPolygons saved to:\n{json_path}")
        self.refresh_json_list(select_name=json_name)

    # ----------------- Poly Capture flow ---------------------
    def on_poly_capture(self):
        # load selected JSON
        jname = self.json_combo.currentText().strip()
        if not jname:
            QMessageBox.warning(self, "Pick JSON", "Please select a polygon JSON.")
            return
        jpath = os.path.join(JSON_DIR, jname)
        if not os.path.exists(jpath):
            QMessageBox.warning(self, "Missing JSON", f"File not found:\n{jpath}")
            return
        try:
            with open(jpath, "r") as f:
                jd = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "JSON error", str(e))
            return

        polys = jd.get("polygons", {})
        # capture fresh frames (any time)
        try:
            frames = self.get_sync_frames(wait_ms=80)
        except Exception as e:
            QMessageBox.critical(self, "Capture failed", str(e))
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for cam in self.cam_indices:
            key = str(cam)
            if key not in polys:
                QMessageBox.warning(self, "Missing polygon", f"No polygon for camera {cam} in {jname}. Skipping.")
                continue
            poly = polys[key]
            bgr = frames[cam]
            masked = PolygonDialog.apply_mask_keep_inside_white_out(bgr, poly)
            outpath = os.path.join(SAVE_ROOT, f"camera{cam}", f"polycap_{ts}.png")
            cv2.imwrite(outpath, masked)

        QMessageBox.information(self, "Saved", f"Poly capture saved to camera folders with timestamp {ts}.")

    # ----------------- JSON list helpers ---------------------
    def refresh_json_list(self, select_name=None):
        self.json_combo.blockSignals(True)
        self.json_combo.clear()
        files = [f for f in os.listdir(JSON_DIR) if f.lower().endswith(".json")]
        files.sort()
        self.json_combo.addItems(files)
        if select_name and select_name in files:
            self.json_combo.setCurrentText(select_name)
        self.json_combo.blockSignals(False)

    def closeEvent(self, e):
        for w in self.workers.values():
            try:
                w.stop()
            except Exception:
                pass
        e.accept()

# ------------------------- main ---------------------------
def main():
    app = QApplication(sys.argv)
    win = MultiCamWindow()
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
