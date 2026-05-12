import sys
import os
import json
import cv2
import time
import threading
import numpy as np
from datetime import datetime

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QGridLayout, QVBoxLayout, QHBoxLayout,
    QDialog, QMessageBox, QComboBox, QLineEdit
)

# ------------------------- Config -------------------------
# You can keep duplicates here; UI + logic will still work.
# NOTE: If you set all to (0,0,0,0), OpenCV may still conflict because it's the SAME camera device.
CAM_INDICES = (0, 1, 2, 3)   # change to your real 4 camera indices (recommended unique)
CAM_W, CAM_H, CAM_FPS = 1280, 720, 15

SAVE_ROOT = "captures"
JSON_DIR = "polygons"

os.makedirs(SAVE_ROOT, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# Save folders by SLOT, so no overwrite even if camera index repeats
for slot in range(4):
    os.makedirs(os.path.join(SAVE_ROOT, f"camera{slot}"), exist_ok=True)


# --------------------- Camera Worker ----------------------
class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage, int)  # (qimg, slot_id)

    def __init__(self, slot_id: int, cam_index: int, width: int, height: int, fps: int, parent=None):
        super().__init__(parent)
        self.slot_id = slot_id
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.fps = fps

        self._running = threading.Event()
        self._running.clear()

        self._cap = None
        self._lock = threading.Lock()
        self._last_bgr = None

    def open_camera(self) -> bool:
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L
        cap = cv2.VideoCapture(self.cam_index, backend)
        if not cap.isOpened():
            print(f"❌ SLOT {self.slot_id}: Camera index {self.cam_index} failed to open.")
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
        while self._running.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self._lock:
                self._last_bgr = frame.copy()

            qimg = self._to_qimage(frame)
            self.frame_ready.emit(qimg, self.slot_id)

            time.sleep(1.0 / max(1, self.fps))

        self._release()

    def get_latest_frame(self):
        with self._lock:
            if self._last_bgr is None:
                return None
            return self._last_bgr.copy()

    def stop(self):
        self._running.clear()
        self.wait(1000)

    def _release(self):
        try:
            if self._cap is not None and self._cap.isOpened():
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.base_pix = None
        self.points = []  # list[QPoint]

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)  # IMPORTANT for Backspace

    def set_image(self, pixmap: QPixmap):
        self.base_pix = pixmap
        self.setPixmap(pixmap)
        self.points = []
        self.setFixedSize(pixmap.size())   # 1:1 mapping
        self.update()
        self.setFocus()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.points.append(event.pos())
            self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace and self.points:
            self.points.pop()
            self.update()
            return
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
            painter.drawLine(self.points[i - 1], self.points[i])

        # close hint
        if len(self.points) >= 3:
            painter.drawLine(self.points[-1], self.points[0])

        # draw points
        for p in self.points:
            painter.drawEllipse(p, 3, 3)

    def get_polygon(self, img_w, img_h):
        if len(self.points) < 3:
            return None
        poly = []
        for pt in self.points:
            x = int(np.clip(pt.x(), 0, img_w - 1))
            y = int(np.clip(pt.y(), 0, img_h - 1))
            poly.append([x, y])
        return poly


class PolygonDialog(QDialog):
    """Draw polygons for 4 frames; returns dict slot_id(str) -> [[x,y],...]"""
    def __init__(self, frames_by_slot: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw Polygons — Left click add | Backspace undo | Enter/Next to proceed")

        self.frames_by_slot = frames_by_slot  # slot -> BGR
        self.slots = sorted(frames_by_slot.keys())
        self.curr_idx = 0

        self.polygons = {}  # str(slot) -> [[x,y],...]

        self.canvas = PolygonCanvas()
        self.info = QLabel("")
        self.next_btn = QPushButton("Next (Enter)")
        self.next_btn.clicked.connect(self.next_slot)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.info)
        layout.addWidget(self.next_btn)
        self.setLayout(layout)

        self._load_current()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.next_slot()
            return
        super().keyPressEvent(event)

    def _load_current(self):
        slot = self.slots[self.curr_idx]
        bgr = self.frames_by_slot[slot]
        h, w = bgr.shape[:2]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)

        self.canvas.set_image(pix)
        self.info.setText(f"SLOT {slot}: click points | Backspace undo | Enter/Next to continue")

        # resize dialog to fit image but keep within screen-ish limits
        self.resize(min(w + 60, 1500), min(h + 140, 950))

    def next_slot(self):
        slot = self.slots[self.curr_idx]
        bgr = self.frames_by_slot[slot]
        h, w = bgr.shape[:2]

        poly = self.canvas.get_polygon(w, h)
        if poly is None:
            QMessageBox.warning(self, "Polygon required", f"Please draw a polygon for SLOT {slot}.")
            return

        self.polygons[str(slot)] = poly

        self.curr_idx += 1
        if self.curr_idx >= len(self.slots):
            self.accept()
        else:
            self._load_current()

    @staticmethod
    def apply_mask_keep_inside_white_out(bgr: np.ndarray, polygon_xy: list) -> np.ndarray:
        h, w = bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([polygon_xy], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)

        out = np.ones_like(bgr, dtype=np.uint8) * 255
        out[mask == 255] = bgr[mask == 255]
        return out


# --------------------- Main UI Window ---------------------
class MultiCamWindow(QWidget):
    def __init__(self, cam_indices=CAM_INDICES, cam_w=CAM_W, cam_h=CAM_H, cam_fps=CAM_FPS):
        super().__init__()
        self.setWindowTitle("4-Camera Live + Polygon Cutting — PyQt5")

        self.cam_indices = list(cam_indices)  # index per slot
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.cam_fps = cam_fps

        self.workers = {}         # slot -> worker
        self.preview_labels = {}  # slot -> QLabel

        # Capture undo/redo stacks
        self._cap_undo_stack = []  # {"ts":..., "paths":[...], "json":...}
        self._cap_redo_stack = []

        self._build_ui()
        self._create_workers()

        for w in self.workers.values():
            w.start()

        self.refresh_json_list()
        self._update_capture_undo_redo_buttons()

    def _build_ui(self):
        main = QVBoxLayout()

        # ---- 2x2 preview grid by SLOT ----
        grid = QGridLayout()
        slot = 0
        for r in range(2):
            for c in range(2):
                cam_index = self.cam_indices[slot]
                lbl = QLabel(f"SLOT {slot}\n(cam idx {cam_index})")
                lbl.setFixedSize(QSize(320, 180))
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("background:#111;color:#ddd;border:1px solid #333;")
                grid.addWidget(lbl, r, c)
                self.preview_labels[slot] = lbl
                slot += 1
        main.addLayout(grid)

        # ---- controls row ----
        row1 = QHBoxLayout()
        self.poly_cut_btn = QPushButton("Polygon Cutting")
        self.poly_capture_btn = QPushButton("Poly Capture (apply JSON)")

        self.btn_poly_undo = QPushButton("Undo")
        self.btn_poly_redo = QPushButton("Forward")

        self.quit_btn = QPushButton("Quit")

        row1.addWidget(self.poly_cut_btn)
        row1.addWidget(self.poly_capture_btn)
        row1.addWidget(self.btn_poly_undo)
        row1.addWidget(self.btn_poly_redo)
        row1.addStretch()
        row1.addWidget(self.quit_btn)
        main.addLayout(row1)

        # ---- JSON chooser row ----
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

        # ---- signals ----
        self.quit_btn.clicked.connect(self.close)
        self.poly_cut_btn.clicked.connect(self.on_polygon_cutting)
        self.poly_capture_btn.clicked.connect(self.on_poly_capture)
        self.refresh_btn.clicked.connect(self.refresh_json_list)

        self.btn_poly_undo.clicked.connect(self.on_poly_capture_undo)
        self.btn_poly_redo.clicked.connect(self.on_poly_capture_redo)

    def _create_workers(self):
        for slot in range(4):
            cam_index = self.cam_indices[slot]
            w = CameraWorker(slot, cam_index, self.cam_w, self.cam_h, self.cam_fps)
            w.frame_ready.connect(self.on_frame_ready)
            self.workers[slot] = w

    def on_frame_ready(self, qimg: QImage, slot_id: int):
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview_labels[slot_id].size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.preview_labels[slot_id].setPixmap(pix)

    # --------- Helper: get near-simultaneous frames ----------
    def get_sync_frames(self, wait_ms=150) -> dict:
        time.sleep(wait_ms / 1000.0)
        frames = {}
        for slot in range(4):
            frm = self.workers[slot].get_latest_frame()
            if frm is None:
                raise RuntimeError(f"SLOT {slot} has no frame yet.")
            frames[slot] = frm
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

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        poly_dict = {
            "created_at": datetime.now().astimezone().isoformat(),
            "slot_to_camera_index": {str(s): int(self.cam_indices[s]) for s in range(4)},
            "polygons": dlg.polygons,  # keys are strings of slot ids
        }

        # Save masked images per slot
        for slot_str, poly in dlg.polygons.items():
            slot = int(slot_str)
            bgr = frames[slot]
            masked = PolygonDialog.apply_mask_keep_inside_white_out(bgr, poly)
            outpath = os.path.join(SAVE_ROOT, f"camera{slot}", f"polycut_{ts}.png")
            cv2.imwrite(outpath, masked)

        # Decide JSON file name
        custom = self.new_json_name.text().strip()
        if custom:
            base = "".join(ch for ch in custom if ch.isalnum() or ch in ("-", "_"))
            json_name = f"{base}.json"
        else:
            json_name = f"polys_{ts}.json"

        json_path = os.path.join(JSON_DIR, json_name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(poly_dict, f, indent=2)

        QMessageBox.information(self, "Saved", f"Masked images saved.\nPolygons saved to:\n{json_path}")
        self.refresh_json_list(select_name=json_name)

    # ----------------- Poly Capture flow ---------------------
    def on_poly_capture(self):
        jname = self.json_combo.currentText().strip()
        if not jname:
            QMessageBox.warning(self, "Pick JSON", "Please select a polygon JSON.")
            return

        jpath = os.path.join(JSON_DIR, jname)
        if not os.path.exists(jpath):
            QMessageBox.warning(self, "Missing JSON", f"File not found:\n{jpath}")
            return

        try:
            with open(jpath, "r", encoding="utf-8") as f:
                jd = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "JSON error", str(e))
            return

        polys = jd.get("polygons", {})

        try:
            frames = self.get_sync_frames(wait_ms=80)
        except Exception as e:
            QMessageBox.critical(self, "Capture failed", str(e))
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        saved_paths = []

        for slot in range(4):
            key = str(slot)
            if key not in polys:
                QMessageBox.warning(self, "Missing polygon", f"No polygon for SLOT {slot} in {jname}. Skipping.")
                continue

            poly = polys[key]
            bgr = frames[slot]
            masked = PolygonDialog.apply_mask_keep_inside_white_out(bgr, poly)

            outpath = os.path.join(SAVE_ROOT, f"camera{slot}", f"polycap_{ts}.png")
            cv2.imwrite(outpath, masked)
            saved_paths.append(outpath)

        if saved_paths:
            self._cap_undo_stack.append({"ts": ts, "paths": saved_paths, "json": jname})
            self._cap_redo_stack.clear()
            self._update_capture_undo_redo_buttons()

        QMessageBox.information(self, "Saved", f"Poly capture saved with timestamp {ts}.")

    # ----------------- Undo / Forward for Poly Capture -----------------
    def _update_capture_undo_redo_buttons(self):
        self.btn_poly_undo.setEnabled(len(self._cap_undo_stack) > 0)
        self.btn_poly_redo.setEnabled(len(self._cap_redo_stack) > 0)

    def on_poly_capture_undo(self):
        if not self._cap_undo_stack:
            QMessageBox.information(self, "Undo", "Nothing to undo.")
            return

        last = self._cap_undo_stack.pop()
        deleted = 0

        for p in last.get("paths", []):
            try:
                if os.path.exists(p):
                    os.remove(p)
                    deleted += 1
            except Exception:
                pass

        self._cap_redo_stack.append(last)
        self._update_capture_undo_redo_buttons()

        QMessageBox.information(self, "Undo", f"Deleted {deleted} images.\nTimestamp: {last.get('ts')}")

    def on_poly_capture_redo(self):
        if not self._cap_redo_stack:
            QMessageBox.information(self, "Forward", "Nothing to redo.")
            return

        item = self._cap_redo_stack.pop()
        jname = item.get("json", "").strip()
        ts = item.get("ts")

        if not jname or not ts:
            QMessageBox.warning(self, "Redo failed", "Redo item is missing json/ts.")
            return

        jpath = os.path.join(JSON_DIR, jname)
        if not os.path.exists(jpath):
            QMessageBox.warning(self, "Redo failed", f"JSON not found:\n{jpath}")
            return

        try:
            with open(jpath, "r", encoding="utf-8") as f:
                jd = json.load(f)
            polys = jd.get("polygons", {})
        except Exception as e:
            QMessageBox.critical(self, "JSON error", str(e))
            return

        try:
            frames = self.get_sync_frames(wait_ms=80)
        except Exception as e:
            QMessageBox.critical(self, "Capture failed", str(e))
            return

        saved_paths = []
        for slot in range(4):
            key = str(slot)
            if key not in polys:
                continue

            poly = polys[key]
            bgr = frames[slot]
            masked = PolygonDialog.apply_mask_keep_inside_white_out(bgr, poly)

            outpath = os.path.join(SAVE_ROOT, f"camera{slot}", f"polycap_{ts}.png")
            cv2.imwrite(outpath, masked)
            saved_paths.append(outpath)

        item["paths"] = saved_paths
        self._cap_undo_stack.append(item)
        self._update_capture_undo_redo_buttons()

        QMessageBox.information(self, "Forward", f"Re-saved {len(saved_paths)} images.\nTimestamp: {ts}")

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
