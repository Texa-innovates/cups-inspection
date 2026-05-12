#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-Camera Viewer (Windows-friendly) with BMP capture and toast popup.
- Fixed indices: 0,1,2,3  (edit CAM_INDEXES if needed)
- Uses DirectShow/MSMF on Windows, default backend elsewhere
- Saves BMPs to ./captures/camera#/ with timestamped filenames
- Shows animated toast popup "Captured Image X" in bottom-left
- Updated: uses QThread per camera for smoother, non-blocking capture.
"""

import sys, cv2, platform
from pathlib import Path
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (
    Qt,
    QRect,
    QEasingCurve,
    QPropertyAnimation,
    QThread,
    pyqtSignal,
)

# ----- UI colors -----
APP_BG      = "#111418"
CARD_BG     = "#1a1f24"
ACCENT      = "#3aa675"
TEXT_LIGHT  = "#f2f5f7"
TEXT_DIM    = "#aab2bd"
BORDER      = "#2a3036"

# ----- Config -----
CAM_INDEXES = [0,2,4,6]     # <- set your 4 fixed indices here
FRAME_SIZE  = (1280, 720)      # request size; camera may cho")
IMAGE_EXT   = ".bmp"           # BMP as requested

# ----- Helpers -----
def choose_backend():
    if platform.system().lower().startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    return [None]

def cvimg_to_qpix(img):
    if img is None:
        return QtGui.QPixmap()
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg)

# ----- Camera Thread -----
class CameraThread(QThread):
    """
    Background thread for a single camera.
    Continuously grabs frames and emits them.
    """
    frame_ready = pyqtSignal(int, object)  # index, frame (np.ndarray)
    status = pyqtSignal(int, bool)         # index, opened_ok

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self._running = True
        self.cap = None

    def _open_camera(self):
        opened = False
        for be in choose_backend():
            try:
                if be is None:
                    cap = cv2.VideoCapture(int(self.index))
                else:
                    cap = cv2.VideoCapture(int(self.index), be)
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_SIZE[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
                    cap.set(cv2.CAP_PROP_FPS, FPS_REQUEST)
                    self.cap = cap
                    opened = True
                    break
            except Exception:
                pass
        self.status.emit(self.index, opened)
        return opened

    def run(self):
        if not self._open_camera():
            return
        # main capture loop
        while self._running:
            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            if ok and frame is not None:
                self.frame_ready.emit(self.index, frame)
            # small sleep to avoid hogging CPU; approx fps control
            self.msleep(10)

    def stop(self):
        self._running = False
        self.wait()
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None

# ----- Toast Popup -----
class ToastPopup(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(50, 150, 90, 220);
                border-radius: 10px;
                padding: 10px 16px;
            }
            QLabel {
                color: white;
                font-size: 13px;
                font-weight: 600;
            }
        """)
        layout = QtWidgets.QHBoxLayout(self)
        self.label = QtWidgets.QLabel("Captured Image")
        layout.addWidget(self.label)
        self.anim = None
        self.timer = None

    def show_message(self, message, parent: QtWidgets.QWidget):
        self.label.setText(message)
        self.adjustSize()

        pg = parent.geometry()
        target_x = pg.left() + 20
        target_y = pg.bottom() - self.height() - 20

        start_rect = QRect(
            pg.left() - self.width(),
            target_y,
            self.width(),
            self.height()
        )
        end_rect = QRect(
            target_x,
            target_y,
            self.width(),
            self.height()
        )

        self.setGeometry(start_rect)
        self.show()

        if self.anim and self.anim.state() == QPropertyAnimation.Running:
            self.anim.stop()
        if self.timer:
            self.timer.stop()

        self.anim = QPropertyAnimation(self, b"geometry")
        self.anim.setDuration(250)
        self.anim.setStartValue(start_rect)
        self.anim.setEndValue(end_rect)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.anim.start()

        from PyQt5.QtCore import QTimer
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(lambda: self.hide_with_slide(pg, end_rect))
        self.timer.start(2000)

    def hide_with_slide(self, pg, current_rect):
        hide_rect = QRect(
            current_rect.x(),
            pg.bottom() + self.height(),
            self.width(),
            self.height()
        )
        self.anim = QPropertyAnimation(self, b"geometry")
        self.anim.setDuration(250)
        self.anim.setStartValue(current_rect)
        self.anim.setEndValue(hide_rect)
        self.anim.setEasingCurve(QEasingCurve.InCubic)
        self.anim.finished.connect(self.hide)
        self.anim.start()

# ----- Camera Box -----
class CamBox(QtWidgets.QFrame):
    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.last_frame = None  # updated from CameraThread

        self.setStyleSheet(
            f"QFrame{{background:{CARD_BG}; "
            f"border:1px solid {BORDER}; border-radius:12px;}}"
        )
        self.setMinimumSize(360, 270)

        self.name = QtWidgets.QLabel(f"Camera {index}")
        self.name.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:12px; padding:6px;"
        )

        self.view = QtWidgets.QLabel("no signal")
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setStyleSheet(f"color:{TEXT_DIM};")

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(6)
        v.addWidget(self.name)
        v.addWidget(self.view, 1)

    def set_pix(self, img):
        if img is None:
            return
        self.view.setPixmap(
            cvimg_to_qpix(img).scaled(
                self.view.width(),
                self.view.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def set_frame(self, frame):
        """Called from main thread when a new frame is available."""
        self.last_frame = frame
        self.set_pix(frame)

    def mark_no_camera(self):
        self.view.setText("no camera")

# ----- Title Bar -----
class TitleBar(QtWidgets.QWidget):
    def __init__(self, parent=None, title="4-Camera Viewer"):
        super().__init__(parent)
        self._parent = parent
        self.setFixedHeight(44)

        self.title_lbl = QtWidgets.QLabel(title)
        self.title_lbl.setStyleSheet(
            f"color:{TEXT_LIGHT}; font-size:14px; font-weight:600;"
        )

        self.min_btn   = QtWidgets.QPushButton("—")
        self.max_btn   = QtWidgets.QPushButton("▢")
        self.close_btn = QtWidgets.QPushButton("✕")

        for b in (self.min_btn, self.max_btn, self.close_btn):
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedSize(36, 28)
            b.setStyleSheet(f"""
                QPushButton {{
                    color: {TEXT_LIGHT}; background: transparent;
                    border: 1px solid {BORDER};
                    border-radius: 6px; font-size:13px;
                }}
                QPushButton:hover {{ background: {BORDER}; }}
            """)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 8, 8)
        lay.addWidget(self.title_lbl, 1)
        lay.addWidget(self.min_btn)
        lay.addWidget(self.max_btn)
        lay.addWidget(self.close_btn)

        self.min_btn.clicked.connect(
            lambda: self._parent.showMinimized()
        )
        self.max_btn.clicked.connect(
            lambda: self._parent.showNormal()
            if self._parent.isMaximized()
            else self._parent.showMaximized()
        )
        self.close_btn.clicked.connect(QtWidgets.QApplication.quit)

        self._drag_pos = None

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPos()

    def mouseMoveEvent(self, e):
        if self._drag_pos and self._parent:
            delta = e.globalPos() - self._drag_pos
            self._parent.move(self._parent.pos() + delta)
            self._drag_pos = e.globalPos()

    def mouseReleaseEvent(self, e):
        self._drag_pos = None

# ----- Main Viewer -----
class Viewer(QtWidgets.QWidget):
    def __init__(self, cam_indexes=CAM_INDEXES):
        super().__init__()
        self.setWindowTitle("4-Camera Viewer")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)

        # Cam boxes + map
        self.cams = [CamBox(i) for i in cam_indexes[:4]]
        self.cam_map = {c.index: c for c in self.cams}

        self.titlebar = TitleBar(self, "4-Camera Viewer")

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(12)
        grid.setContentsMargins(12, 12, 12, 12)
        if len(self.cams) >= 1:
            grid.addWidget(self.cams[0], 0, 0)
        if len(self.cams) >= 2:
            grid.addWidget(self.cams[1], 0, 1)
        if len(self.cams) >= 3:
            grid.addWidget(self.cams[2], 1, 0)
        if len(self.cams) >= 4:
            grid.addWidget(self.cams[3], 1, 1)

        self.btn_capture = QtWidgets.QPushButton("Capture BMP")
        self.btn_capture.setCursor(Qt.PointingHandCursor)
        self.btn_capture.setFixedHeight(44)
        self.btn_capture.setStyleSheet(f"""
            QPushButton {{
                background:{ACCENT}; color:{TEXT_LIGHT};
                font-size:15px; font-weight:600;
                border:none; border-radius:10px;
                padding:6px 14px;
            }}
            QPushButton:hover {{ filter: brightness(1.05); }}
        """)
        self.btn_capture.clicked.connect(self.capture_all)

        status_lbl = QtWidgets.QLabel(
            "Tip: If a panel says ‘no camera’, check USB port/bandwidth."
        )
        status_lbl.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:12px; padding:4px;"
        )

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(status_lbl, 1)
        ctrl.addWidget(self.btn_capture, 0)

        card = QtWidgets.QFrame()
        card.setStyleSheet(
            f"QFrame {{ background:{APP_BG}; border:1px solid {BORDER}; "
            f"border-radius:14px; }}"
        )
        v = QtWidgets.QVBoxLayout(card)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)
        v.addLayout(grid, 1)
        v.addLayout(ctrl, 0)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        root.addWidget(self.titlebar, 0)
        root.addWidget(card, 1)

        # Camera threads
        self.threads = []
        for c in self.cams:
            th = CameraThread(c.index, self)
            th.frame_ready.connect(self.on_frame_ready)
            th.status.connect(self.on_camera_status)
            th.start()
            self.threads.append(th)

        self.resize(1000, 740)
        self.center_on_screen()

        # Toast + capture counter
        self.toast = ToastPopup(self)
        self.capture_count = 0

    def center_on_screen(self):
        ag = QtWidgets.QApplication.desktop().availableGeometry(self)
        w, h = 1000, 740
        x = (ag.width() - w) // 2
        y = (ag.height() - h) // 2
        self.setGeometry(x, y, w, h)

    # ---- Slots for camera threads ----
    @QtCore.pyqtSlot(int, object)
    def on_frame_ready(self, index, frame):
        cam = self.cam_map.get(index)
        if cam is not None:
            cam.set_frame(frame)

    @QtCore.pyqtSlot(int, bool)
    def on_camera_status(self, index, opened):
        if not opened:
            cam = self.cam_map.get(index)
            if cam is not None:
                cam.mark_no_camera()

    # ---- Capture BMP ----
    def capture_all(self):
        SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.capture_count += 1

        for i, c in enumerate(self.cams, start=1):
            cam_dir = SAVE_ROOT / f"camera{i}"
            cam_dir.mkdir(parents=True, exist_ok=True)

            frame = c.last_frame
            if frame is None:
                (cam_dir / f"cam{i}_{ts}_FAILED.txt").write_text(
                    "No frame at capture time"
                )
                continue

            out = cam_dir / f"cam{i}_{ts}{IMAGE_EXT}"
            try:
                cv2.imwrite(str(out), frame)
            except Exception as e:
                (cam_dir / f"cam{i}_{ts}_ERROR.txt").write_text(
                    f"Save error: {e}"
                )

        # Show toast popup
        self.toast.show_message(f"Captured Image {self.capture_count}", self)

    def closeEvent(self, e):
        # stop threads
        for th in self.threads:
            try:
                th.stop()
            except Exception:
                pass
        return super().closeEvent(e)

# ----- Main Entry -----
def main():
    app = QtWidgets.QApplication(sys.argv)
    f = app.font()
    f.setPointSize(11)
    app.setFont(f)
    win = Viewer(CAM_INDEXES)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
