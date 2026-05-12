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
    QFileDialog, QDialog, QMessageBox, QComboBox, QLineEdit, QFormLayout, QCheckBox
)

# ------------------------- Config -------------------------
CAM_INDICES = (4, 6, 2, 0)    # adjust to your 4 e-con camera indices
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
        while self._running.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self._lock:
                self._last_bgr = frame.copy()

            qimg = self._to_qimage(frame)
            self.frame_ready.emit(qimg, self.cam_index)

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

        # grid options (optional)
        self.show_grid = False
        self.grid_cols = 10
        self.grid_rows = 8

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.points.append(event.pos())
            self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Backspace and self.points:
            self.points.pop()
            self.update()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)

        # --- draw grid (optional) ---
        if self.show_grid:
            pen_grid = QPen(Qt.yellow, 1, Qt.DotLine)
            painter.setPen(pen_grid)

            w = self.width()
            h = self.height()

            # vertical lines
            for i in range(1, self.grid_cols):
                x = int(i * w / self.grid_cols)
                painter.drawLine(x, 0, x, h)

            # horizontal lines
            for j in range(1, self.grid_rows):
                y = int(j * h / self.grid_rows)
                painter.drawLine(0, y, w, y)

            # center cross (stronger)
            pen_center = QPen(Qt.yellow, 2)
            painter.setPen(pen_center)
            painter.drawLine(w // 2, 0, w // 2, h)
            painter.drawLine(0, h // 2, w, h // 2)

        # --- draw polygon points/lines ---
        if not self.points:
            return

        pen = QPen(Qt.green, 2)
        painter.setPen(pen)

        for i in range(1, len(self.points)):
            painter.drawLine(self.points[i - 1], self.points[i])

        if len(self.points) >= 3:
            painter.drawLine(self.points[-1], self.points[0])

        for p in self.points:
            painter.drawEllipse(p, 3, 3)

    def get_polygon(self, img_w, img_h):
        if not self.points or len(self.points) < 3:
            return None
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
        self.frames_by_cam = frames_by_cam
        self.cam_ids = sorted(frames_by_cam.keys())
        self.curr_idx = 0
        self.polygons = {}

        self.label = None
        self.info = QLabel("")
        self.next_btn = QPushButton("Next (Enter)")
        self.next_btn.clicked.connect(self.next_camera)

        # optional grid controls inside polygon page
        self.grid_chk = QCheckBox("Show Grid (for alignment)")
        self.grid_chk.setChecked(True)
        self.grid_combo = QComboBox()
        self.grid_combo.addItems(["8x6", "10x8", "12x9", "16x12"])
        self.grid_combo.setCurrentText("10x8")
        self.grid_chk.stateChanged.connect(self._apply_grid_settings)
        self.grid_combo.currentTextChanged.connect(self._apply_grid_settings)

        self._build_ui()
        self._load_current()

    def _build_ui(self):
        v = QVBoxLayout()
        self.label = PolygonCanvas(QPixmap(4, 4))
        v.addWidget(self.label)

        grid_row = QHBoxLayout()
        grid_row.addWidget(self.grid_chk)
        grid_row.addWidget(QLabel("Grid:"))
        grid_row.addWidget(self.grid_combo)
        grid_row.addStretch()
        v.addLayout(grid_row)

        v.addWidget(self.info)
        v.addWidget(self.next_btn)
        self.setLayout(v)
        self.resize(1100, 820)

    def _apply_grid_settings(self):
        self.label.show_grid = self.grid_chk.isChecked()
        txt = self.grid_combo.currentText().strip()
        try:
            c, r = txt.split("x")
            self.label.grid_cols = max(2, int(c))
            self.label.grid_rows = max(2, int(r))
        except Exception:
            pass
        self.label.update()

    def _load_current(self):
        cam = self.cam_ids[self.curr_idx]
        bgr = self.frames_by_cam[cam]
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)

        self.label.setFixedSize(QSize(1000, 700))
        self.label.setPixmap(pix)
        self.label.base_pix = pix
        self.label.points = []

        self._apply_grid_settings()

        self.info.setText(
            f"Camera {cam}: click to add points; Backspace=undo; Enter or 'Next' to continue."
        )

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.next_camera()
        else:
            super().keyPressEvent(event)

    def next_camera(self):
        cam = self.cam_ids[self.curr_idx]
        bgr = self.frames_by_cam[cam]
        h, w = bgr.shape[:2]
        poly = self.label.get_polygon(w, h)
        if poly is None:
            QMessageBox.warning(self, "Polygon required", f"Please draw a polygon for camera {cam}.")
            return
        self.polygons[str(cam)] = poly

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
        out = np.ones_like(bgr, dtype=np.uint8) * 255
        out[mask == 255] = bgr[mask == 255]
        return out
    

class ZoomViewer(QDialog):
    def __init__(self, parent=None, title="Viewer"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.setModal(True)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background:#000;")

        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.label)
        self.setLayout(lay)

        self._orig_pix = None          # original pixmap
        self._scale = 1.0              # zoom scale
        self._min_scale = 0.1
        self._max_scale = 12.0

        self._panning = False
        self._pan_start = QPoint(0, 0)
        self._offset = QPoint(0, 0)    # pan offset in label coords

    def set_pixmap(self, pix: QPixmap):
        self._orig_pix = pix
        self._offset = QPoint(0, 0)
        self._scale = self._calc_fit_scale()
        self._render()

    def _calc_fit_scale(self):
        if self._orig_pix is None or self.label.width() == 0 or self.label.height() == 0:
            return 1.0

        sw = self.label.width() / self._orig_pix.width()
        sh = self.label.height() / self._orig_pix.height()
        return min(sw, sh)

    def showEvent(self, e):
        super().showEvent(e)
        self.showMaximized()
        self._render()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._scale = self._calc_fit_scale()
        self._offset = QPoint(0, 0)
        self._render()

    def wheelEvent(self, e):
        if self._orig_pix is None:
            return
        delta = e.angleDelta().y()
        if delta == 0:
            return

        factor = 1.15 if delta > 0 else (1 / 1.15)
        new_scale = float(np.clip(self._scale * factor, self._min_scale, self._max_scale))

        # keep zoom centered around cursor
        cursor_pos = e.position().toPoint()
        old_scale = self._scale
        self._scale = new_scale

        if old_scale != 0:
            k = self._scale / old_scale
            # adjust offset so that cursor stays roughly on same content point
            self._offset = QPoint(
                int(cursor_pos.x() - k * (cursor_pos.x() - self._offset.x())),
                int(cursor_pos.y() - k * (cursor_pos.y() - self._offset.y()))
            )

        self._render()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._panning = True
            self._pan_start = e.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._panning:
            delta = e.pos() - self._pan_start
            self._pan_start = e.pos()
            self._offset = self._offset + delta
            self._render()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, e):
        self._scale = self._calc_fit_scale()
        self._offset = QPoint(0, 0)
        self._render()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(e)

    def _render(self):
        if self._orig_pix is None:
            return

        # scale
        scaled = self._orig_pix.scaled(
            int(self._orig_pix.width() * self._scale),
            int(self._orig_pix.height() * self._scale),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # draw onto a black canvas same size as label, using offset for panning
        canvas = QPixmap(self.label.size())
        canvas.fill(Qt.black)

        painter = QPainter(canvas)
        x = (canvas.width() - scaled.width()) // 2 + self._offset.x()
        y = (canvas.height() - scaled.height()) // 2 + self._offset.y()
        painter.drawPixmap(x, y, scaled)
        painter.end()

        self.label.setPixmap(canvas)

from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence




# --------------------- Main UI Window ---------------------
class MultiCamWindow(QWidget):
    def __init__(self, cam_indices=CAM_INDICES, cam_w=CAM_W, cam_h=CAM_H, cam_fps=CAM_FPS):
        super().__init__()
        self.setWindowTitle("4-Camera Live + Polygon Cutting — PyQt5")
        self.cam_indices = list(cam_indices)
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.cam_fps = cam_fps

        # --- Grid options for preview ---
        self.show_grid_preview = True
        self.grid_cols = 16 #10
        self.grid_rows = 12 #8

        self.sync_cond = threading.Condition()
        self.workers = {}
        self.preview_labels = {}
        self.pixmaps = {}

        self._build_ui()
        self._create_workers()
        for w in self.workers.values():
            w.start()
        self.refresh_json_list()
       
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self.redo)


    def _push_undo(self):
        # Save a snapshot BEFORE changing points
        self.undo_stack.append([QPoint(p) for p in self.points])
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append([QPoint(p) for p in self.points])
        self.points = self.undo_stack.pop()
        self.update()

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append([QPoint(p) for p in self.points])
        self.points = self.redo_stack.pop()
        self.update()

    def clear_polygon(self):
        if not self.points:
            return
        self._push_undo()
        self.points = []
        self.update()

    def _build_ui(self):
        main = QVBoxLayout()
        grid = QGridLayout()
        idx = 0
        for r in range(2):
            for c in range(2):
                cam = self.cam_indices[idx]
                lbl = QLabel(f"Cam {cam}")
                lbl.mousePressEvent = lambda ev, cam_id=cam: self.open_fullscreen(cam_id)
                lbl.setFixedSize(QSize(520, 300))
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("background:#111;color:#ddd;border:1px solid #333;")
                grid.addWidget(lbl, r, c)
                self.preview_labels[cam] = lbl
                idx += 1
        main.addLayout(grid)

        # controls row
        row1 = QHBoxLayout()
        self.poly_cut_btn = QPushButton("Polygon Cutting")
        self.poly_capture_btn = QPushButton("Poly Capture (apply JSON)")
        self.quit_btn = QPushButton("Quit")
        row1.addWidget(self.poly_cut_btn)
        row1.addWidget(self.poly_capture_btn)
        row1.addStretch()
        row1.addWidget(self.quit_btn)
        main.addLayout(row1)

        # JSON chooser row
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

        # --- NEW: grid controls row ---
        row3 = QHBoxLayout()
        self.grid_chk = QCheckBox("Show Grid (Preview)")
        self.grid_chk.setChecked(True)
        self.grid_combo = QComboBox()
        self.grid_combo.addItems(["8x6", "10x8", "12x9", "16x12", "20x15"])
        self.grid_combo.setCurrentText("10x8")
        row3.addWidget(self.grid_chk)
        row3.addWidget(QLabel("Grid:"))
        row3.addWidget(self.grid_combo)
        row3.addStretch()
        main.addLayout(row3)

        self.setLayout(main)

        # signals
        self.quit_btn.clicked.connect(self.close)
        self.poly_cut_btn.clicked.connect(self.on_polygon_cutting)
        self.poly_capture_btn.clicked.connect(self.on_poly_capture)
        self.refresh_btn.clicked.connect(self.refresh_json_list)

        self.grid_chk.stateChanged.connect(self._update_grid_settings)
        self.grid_combo.currentTextChanged.connect(self._update_grid_settings)

    def _update_grid_settings(self):
        self.show_grid_preview = self.grid_chk.isChecked()
        txt = self.grid_combo.currentText().strip()
        try:
            c, r = txt.split("x")
            self.grid_cols = max(2, int(c))
            self.grid_rows = max(2, int(r))
        except Exception:
            pass

    def open_fullscreen(self, cam_id: int):
        # Use latest displayed pixmap (with grid overlay)
        pix = self.pixmaps.get(cam_id, None)
        if pix is None:
            QMessageBox.warning(self, "No frame", f"No frame available yet for camera {cam_id}.")
            return

        dlg = ZoomViewer(self, title=f"Camera {cam_id} — Fullscreen Zoom")
        dlg.set_pixmap(pix)
        dlg.exec_()

    def _create_workers(self):
        for idx in self.cam_indices:
            w = CameraWorker(idx, self.cam_w, self.cam_h, self.cam_fps, cond=self.sync_cond)
            w.frame_ready.connect(self.on_frame_ready)
            self.workers[idx] = w

    def _draw_grid_on_pixmap(self, pix: QPixmap) -> QPixmap:
        """Overlay grid lines on pixmap (for angle alignment)."""
        if not self.show_grid_preview:
            return pix

        out = QPixmap(pix)
        painter = QPainter(out)

        # dotted grid
        pen_grid = QPen(Qt.yellow, 1, Qt.DotLine)
        painter.setPen(pen_grid)

        w = out.width()
        h = out.height()

        for i in range(1, self.grid_cols):
            x = int(i * w / self.grid_cols)
            painter.drawLine(x, 0, x, h)

        for j in range(1, self.grid_rows):
            y = int(j * h / self.grid_rows)
            painter.drawLine(0, y, w, y)

        # center cross thicker
        pen_center = QPen(Qt.yellow, 2)
        painter.setPen(pen_center)
        painter.drawLine(w // 2, 0, w // 2, h)
        painter.drawLine(0, h // 2, w, h // 2)

        painter.end()
        return out

    def on_frame_ready(self, qimg: QImage, cam_index: int):
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview_labels[cam_index].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        pix = self._draw_grid_on_pixmap(pix)  # <-- grid overlay here
        self.preview_labels[cam_index].setPixmap(pix)
        self.pixmaps[cam_index] = pix

    def get_sync_frames(self, wait_ms=150) -> dict:
        frames = {}
        time.sleep(wait_ms / 1000.0)
        for cam in self.cam_indices:
            frm = self.workers[cam].get_latest_frame()
            if frm is None:
                raise RuntimeError(f"Camera {cam} has no frame yet.")
            frames[cam] = frm
        return frames

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
            "camera_indices": list(self.cam_indices),
            "polygons": dlg.polygons,
        }

        for cam_str, poly in dlg.polygons.items():
            cam = int(cam_str)
            bgr = frames[cam]
            masked = PolygonDialog.apply_mask_keep_inside_white_out(bgr, poly)
            outpath = os.path.join(SAVE_ROOT, f"camera{cam}", f"polycut_{ts}.png")
            cv2.imwrite(outpath, masked)

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
            with open(jpath, "r") as f:
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


def main():
    app = QApplication(sys.argv)
    win = MultiCamWindow()
    win.resize(1200, 900)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
