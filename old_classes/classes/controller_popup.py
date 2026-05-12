# controller_popup.py
# ------------------------------------------------------------
# Controller popup enhancements:
# - Check PLC: themed status card (primary color) + pulse/shake animations
# - Check Camera: progressive scan + scanner bar + row fade-in animations
# - 4 cam live preview:
#     * fixed standard size (no maximize/minimize)
#     * proper 2x2 layout (no overlap)
#     * tile shimmer until first frame
#     * fade-in on first frame
#     * premium breathing glow border
#     * CLICK a tile -> fullscreen that camera
#       - ESC or Close -> return to 4-cam grid
#       - if no camera frame -> fullscreen will NOT open
#     * fullscreen uses STANDARD view (no zoom/crop/upscale)
# ------------------------------------------------------------

import os
import sys
import math
import platform

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QColor


# --- Silence OpenCV logs as much as possible ---
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
try:
    cv2.setLogLevel(0)  # some builds
except Exception:
    pass


# -------- pymodbus import (supports old/new layouts) ----------
try:
    # pymodbus 3.x
    from pymodbus.client import ModbusTcpClient
except Exception:
    # pymodbus 2.5.x
    from pymodbus.client.sync import ModbusTcpClient


# ----------------------------- Theme -----------------------------
BRAND = "#70212a"        # primary
BRAND_DARK = "#5a191f"
BG = "#fdf6f7"           # project soft bg
SURFACE = "#ffffff"
INK = "#201a1c"
MUTED = "#786a70"
LINE = "#ece3e7"
GOOD = "#1f7a1f"
BAD = "#a11d1d"
DARK_PANEL = "#0f0f11"

TINT_OK = "#eaf7ee"
TINT_BAD = "#fdecef"
TINT_BRAND = "#fdf2f5"


POPUP_QSS = f"""
/* --- Hard reset to prevent parent-app stylesheet bleed --- */
* {{ background-color: transparent; }}
QWidget {{ background-color: transparent; color: {INK}; }}
QLabel {{ background-color: transparent; color: {INK}; }}
QDialog {{ background-color: {BG}; }}
QDialog {{
    background: {BG};
}}
QLabel#title {{
    font-size: 18px;
    font-weight: 800;
    color: {INK};
}}
QLabel#subtitle {{
    font-size: 12px;
    color: {MUTED};
}}
QFrame#card {{
    border: 1px solid {LINE};
    border-radius: 16px;
    background: {SURFACE};
}}
QFrame#primaryCard {{
    border: 1px solid {BRAND_DARK};
    border-radius: 16px;
    background: {BRAND};
}}
QLabel#primaryText {{
    color: white;
    font-size: 14px;
    font-weight: 800;
}}
QLabel#secondaryText {{
    color: rgba(255,255,255,0.92);
    font-size: 12px;
    font-weight: 600;
}}
QLabel#valueText {{
    color: {INK};
    font-size: 13px;
    font-weight: 700;
}}
QLabel#badge {{
    border-radius: 12px;
    padding: 6px 10px;
    font-weight: 800;
}}
QPushButton {{
    padding: 10px 14px;
    border-radius: 12px;
    border: 1px solid {LINE};
    background: {SURFACE};
    color: {INK};
    font-weight: 700;
}}
QPushButton:hover {{
    background: {TINT_BRAND};
}}
QPushButton#danger {{
    background: {BRAND};
    color: white;
    border: 1px solid {BRAND};
}}
QPushButton#danger:hover {{
    background: {BRAND_DARK};
    border: 1px solid {BRAND_DARK};
}}
QListWidget {{
    border-radius: 14px;
    border: 1px solid {LINE};
    background: {SURFACE};
    padding: 8px;
}}
QListWidget::item {{
    padding: 10px 12px;
    border-radius: 12px;
}}
QListWidget::item:selected {{
    background: {TINT_BRAND};
}}
QLabel#camBox {{
    background: {DARK_PANEL};
    border: 1px solid {LINE};
    border-radius: 16px;
}}
"""


# ----------------------------- Helpers -----------------------------
def _choose_backend():
    """
    Keep it SIMPLE and CLEAN:
    - Windows -> DSHOW only (no MSMF fallback spam)
    - Linux -> default
    """
    if platform.system().lower().startswith("win"):
        return [cv2.CAP_DSHOW]
    return [None]


def _cvimg_to_qpix_fill(img, target_w: int, target_h: int) -> QtGui.QPixmap:
    """
    FILL view (used for 4-cam grid):
    - No black bars
    - Center crop
    """
    if img is None or target_w <= 0 or target_h <= 0:
        return QtGui.QPixmap()

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    qimg = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    pm = QtGui.QPixmap.fromImage(qimg)

    pm = pm.scaled(target_w, target_h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

    x = max(0, (pm.width() - target_w) // 2)
    y = max(0, (pm.height() - target_h) // 2)
    return pm.copy(x, y, target_w, target_h)


def _cvimg_to_qpix_standard(img, target_w: int, target_h: int, allow_upscale: bool = False) -> QtGui.QPixmap:
    """
    STANDARD view (no zoom / no crop):
    - Keep aspect ratio
    - Never crops (so no zoom effect)
    - If allow_upscale=False: do NOT enlarge beyond original resolution
      (shows actual camera resolution; fullscreen will have borders)
    """
    if img is None or target_w <= 0 or target_h <= 0:
        return QtGui.QPixmap()

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    qimg = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    pm = QtGui.QPixmap.fromImage(qimg)

    # keep native size if window is bigger and no upscale allowed
    if (not allow_upscale) and (target_w >= w) and (target_h >= h):
        return pm

    return pm.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ==================================================
# Small Animation Helper (shake)
# ==================================================
def apply_shake(widget: QtWidgets.QWidget, distance=8, duration_ms=380):
    if widget is None:
        return
    start_pos = widget.pos()
    anim = QPropertyAnimation(widget, b"pos", widget)
    anim.setDuration(duration_ms)
    anim.setEasingCurve(QEasingCurve.InOutSine)

    anim.setKeyValueAt(0.00, start_pos)
    anim.setKeyValueAt(0.15, start_pos + QtCore.QPoint(-distance, 0))
    anim.setKeyValueAt(0.30, start_pos + QtCore.QPoint(distance, 0))
    anim.setKeyValueAt(0.45, start_pos + QtCore.QPoint(-distance, 0))
    anim.setKeyValueAt(0.60, start_pos + QtCore.QPoint(distance, 0))
    anim.setKeyValueAt(0.75, start_pos + QtCore.QPoint(-distance, 0))
    anim.setKeyValueAt(1.00, start_pos)

    widget._shake_anim = anim
    anim.start()


# ==================================================
# CLICKABLE CAMERA LABEL
# ==================================================
class ClickableCamLabel(QtWidgets.QLabel):
    clicked = pyqtSignal(int)

    def __init__(self, cam_idx: int, parent=None):
        super().__init__(parent)
        self.cam_idx = int(cam_idx)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.cam_idx)
        super().mousePressEvent(e)


# ==================================================
# FULLSCREEN SINGLE CAM VIEW (ESC to exit)
# ==================================================
class FullscreenCamView(QtWidgets.QDialog):
    def __init__(self, parent, cam_idx: int):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(POPUP_QSS)
        self.setWindowTitle(f"Camera {cam_idx}")
        self.setModal(True)
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)

        self.cam_idx = int(cam_idx)
        self._last_frame = None

        self._resize_debounce = QTimer(self)
        self._resize_debounce.setSingleShot(True)
        self._resize_debounce.timeout.connect(self._redraw)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        top = QtWidgets.QHBoxLayout()
        top.setSpacing(10)

        title = QtWidgets.QLabel(f"Camera {cam_idx} - Fullscreen")
        title.setObjectName("title")
        title.setStyleSheet("font-size:16px;")
        top.addWidget(title, 1, Qt.AlignLeft)

        hint = QtWidgets.QLabel("ESC to exit")
        hint.setObjectName("subtitle")
        top.addWidget(hint, 0, Qt.AlignRight)

        root.addLayout(top)

        self.view = QtWidgets.QLabel()
        self.view.setObjectName("camBox")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setText("")
        root.addWidget(self.view, 1)

        btnrow = QtWidgets.QHBoxLayout()
        btnrow.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setObjectName("danger")
        close_btn.clicked.connect(self.close)
        btnrow.addWidget(close_btn)
        root.addLayout(btnrow)

        QtCore.QTimer.singleShot(0, self.showFullScreen)

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(e)

    def update_frame(self, frame):
        self._last_frame = frame
        self._redraw()

    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        self._resize_debounce.start(60)

    def _redraw(self):
        if self._last_frame is None:
            return
        w = max(10, self.view.width())
        h = max(10, self.view.height())

        # STANDARD = no zoom/crop/upscale (true camera resolution)
        pm = _cvimg_to_qpix_standard(self._last_frame, w, h, allow_upscale=False)
        self.view.setPixmap(pm)
        self.view.setAlignment(Qt.AlignCenter)


# ==================================================
# CAMERA STATUS (progressive scan + animations)
# ==================================================
class CameraStatusDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, max_index: int = 10):
        super().__init__(parent)
        # Ensure this dialog paints its own background (avoid parent stylesheet bleed)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setWindowTitle("Camera Status")
        self.setModal(True)
        self.setMinimumSize(620, 480)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(POPUP_QSS)

        self.max_index = int(max_index)
        self._connected = set()
        self._scan_i = 0
        self._row_widgets = {}
        self._row_items = {}

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QtWidgets.QLabel("Check Camera")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        self.head_card = QtWidgets.QFrame()
        self.head_card.setObjectName("card")
        self.head_card.setMinimumHeight(92)
        head = QtWidgets.QVBoxLayout(self.head_card)
        head.setContentsMargins(14, 12, 14, 12)
        head.setSpacing(8)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)

        self.icon = QtWidgets.QLabel("🔍")
        self.icon.setStyleSheet("font-size:22px;")
        row.addWidget(self.icon, 0, Qt.AlignLeft)

        self.status_text = QtWidgets.QLabel("Preparing scan…")
        self.status_text.setStyleSheet(f"color:{INK}; font-weight:900; font-size:13px;")
        row.addWidget(self.status_text, 1, Qt.AlignLeft)

        self.badge = QtWidgets.QLabel("RUNNING")
        self.badge.setObjectName("badge")
        self.badge.setStyleSheet(f"background:{TINT_BRAND}; color:{BRAND}; border:1px solid {LINE};")
        row.addWidget(self.badge, 0, Qt.AlignRight)

        head.addLayout(row)

        self.scan_track = QtWidgets.QFrame()
        self.scan_track.setStyleSheet(f"background:{LINE}; border-radius:10px;")
        self.scan_track.setFixedHeight(16)

        self.scan_bar = QtWidgets.QFrame(self.scan_track)
        self.scan_bar.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f" stop:0 {BRAND_DARK}, stop:0.5 {BRAND}, stop:1 {BRAND_DARK});"
            f" border-radius:10px;"
        )
        self.scan_bar.setGeometry(0, 0, 140, 16)

        head.addWidget(self.scan_track)
        root.addWidget(self.head_card)

        self.list = QtWidgets.QListWidget()
        self.list.setSpacing(6)
        root.addWidget(self.list, 1)

        self.summary = QtWidgets.QLabel("")
        self.summary.setObjectName("subtitle")
        self.summary.setAlignment(Qt.AlignRight)
        root.addWidget(self.summary)

        btnrow = QtWidgets.QHBoxLayout()
        btnrow.addStretch(1)

        self.btn_rescan = QtWidgets.QPushButton("Rescan")
        self.btn_rescan.clicked.connect(self.start_scan)
        btnrow.addWidget(self.btn_rescan)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.setObjectName("danger")
        btn_close.clicked.connect(self.close)
        btnrow.addWidget(btn_close)
        root.addLayout(btnrow)

        self._scan_anim = QtCore.QPropertyAnimation(self.scan_bar, b"pos", self)
        self._scan_anim.setDuration(900)
        self._scan_anim.setEasingCurve(QtCore.QEasingCurve.InOutSine)
        self._scan_anim.setLoopCount(-1)

        self._icon_anim = QtCore.QVariantAnimation(self)
        self._icon_anim.setDuration(800)
        self._icon_anim.setLoopCount(-1)
        self._icon_anim.setStartValue(0)
        self._icon_anim.setEndValue(3)
        self._icon_anim.valueChanged.connect(self._icon_tick)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._scan_step)

        self._build_rows()
        QtCore.QTimer.singleShot(80, self.start_scan)

    def _icon_tick(self, v):
        icons = ["🔍", "🔎", "📷", "🔍"]
        self.icon.setText(icons[int(v) % len(icons)])

    def _start_scan_anim(self):
        w = self.scan_track.width()
        barw = self.scan_bar.width()
        self._scan_anim.stop()
        self._scan_anim.setStartValue(QtCore.QPoint(0, 0))
        self._scan_anim.setEndValue(QtCore.QPoint(max(0, w - barw), 0))
        self._scan_anim.start()

    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        self.scan_bar.setFixedHeight(self.scan_track.height())
        self.scan_bar.setFixedWidth(140)
        self._start_scan_anim()

    def _make_row_widget(self, cam_idx: int) -> QtWidgets.QWidget:
        w = QtWidgets.QFrame()
        w.setObjectName("card")
        w.setStyleSheet(f"QFrame#card{{border:1px solid {LINE}; border-radius:14px; background:{SURFACE};}}")

        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(10)

        left = QtWidgets.QLabel(f"Camera {cam_idx}")
        left.setStyleSheet(f"color:{INK}; font-weight:900; font-size:13px;")
        lay.addWidget(left, 0, Qt.AlignLeft)

        lay.addStretch(1)

        mid = QtWidgets.QLabel("PENDING…")
        mid.setObjectName("mid")
        mid.setStyleSheet(f"color:{MUTED}; font-weight:900; font-size:13px;")
        lay.addWidget(mid, 0, Qt.AlignCenter)

        lay.addStretch(1)

        right = QtWidgets.QLabel("⏳")
        right.setObjectName("right")
        right.setStyleSheet("font-size:16px;")
        lay.addWidget(right, 0, Qt.AlignRight)

        eff = QtWidgets.QGraphicsOpacityEffect(w)
        eff.setOpacity(0.55)
        w.setGraphicsEffect(eff)
        w._opacity = eff

        return w

    def _build_rows(self):
        self.list.clear()
        self._row_widgets.clear()
        self._row_items.clear()

        for i in range(self.max_index):
            item = QtWidgets.QListWidgetItem(self.list)
            item.setSizeHint(QtCore.QSize(0, 56))
            self.list.addItem(item)

            widget = self._make_row_widget(i)
            self.list.setItemWidget(item, widget)

            self._row_items[i] = item
            self._row_widgets[i] = widget

    def start_scan(self):
        self._connected = set()
        self._scan_i = 0

        self.badge.setText("RUNNING")
        self.badge.setStyleSheet(f"background:{TINT_BRAND}; color:{BRAND}; border:1px solid {LINE};")
        self.status_text.setText("Scanning camera ports…")
        self.icon.setText("🔍")

        for i in range(self.max_index):
            w = self._row_widgets[i]
            w.setStyleSheet(f"QFrame#card{{border:1px solid {LINE}; border-radius:14px; background:{SURFACE};}}")
            w._opacity.setOpacity(0.55)

            mid = w.findChild(QtWidgets.QLabel, "mid")
            right = w.findChild(QtWidgets.QLabel, "right")
            if mid:
                mid.setText("PENDING…")
                mid.setStyleSheet(f"color:{MUTED}; font-weight:900; font-size:13px;")
            if right:
                right.setText("⏳")

        self._start_scan_anim()
        self._icon_anim.start()

        self._timer.stop()
        self._timer.start(180)

    def _scan_step(self):
        if self._scan_i >= self.max_index:
            self._timer.stop()
            self._scan_anim.stop()
            self._icon_anim.stop()

            cnt = len(self._connected)
            if cnt == 0:
                self.icon.setText("🚫")
                self.status_text.setText("No camera detected")
                self.badge.setText("FAILED")
                self.badge.setStyleSheet(f"background:{TINT_BAD}; color:{BAD}; border:1px solid {LINE};")
                apply_shake(self.head_card)
            else:
                self.icon.setText("✅")
                self.status_text.setText("Camera scan completed")
                self.badge.setText("DONE")
                self.badge.setStyleSheet(f"background:{TINT_OK}; color:{GOOD}; border:1px solid {LINE};")

            self.summary.setText(f"Detected Cameras: {cnt} / {self.max_index}")
            return

        idx = self._scan_i
        self.status_text.setText(f"Scanning camera {idx} …")

        opened = False
        for be in _choose_backend():
            try:
                cap = cv2.VideoCapture(idx) if be is None else cv2.VideoCapture(idx, be)
                if cap is not None and cap.isOpened():
                    opened = True
                    cap.release()
                    break
                if cap:
                    cap.release()
            except Exception:
                pass

        w = self._row_widgets[idx]
        mid = w.findChild(QtWidgets.QLabel, "mid")
        right = w.findChild(QtWidgets.QLabel, "right")

        if opened:
            self._connected.add(idx)
            w.setStyleSheet(f"QFrame#card{{border:1px solid {LINE}; border-radius:14px; background:{TINT_OK};}}")
            if mid:
                mid.setText("CONNECTED")
                mid.setStyleSheet(f"color:{GOOD}; font-weight:900; font-size:13px;")
            if right:
                right.setText("✅")
        else:
            w.setStyleSheet(f"QFrame#card{{border:1px solid {LINE}; border-radius:14px; background:{TINT_BAD};}}")
            if mid:
                mid.setText("NOT CONNECTED")
                mid.setStyleSheet(f"color:{BAD}; font-weight:900; font-size:13px;")
            if right:
                right.setText("❌")

        anim = QtCore.QPropertyAnimation(w._opacity, b"opacity", w)
        anim.setDuration(220)
        anim.setStartValue(0.55)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        w._fade_anim = anim
        anim.start()

        self.summary.setText(f"Detected Cameras: {len(self._connected)} / {self.max_index}")
        self._scan_i += 1


# ==================================================
# PLC STATUS (primary card + pulse + shake)
# ==================================================
PLC_IP = "192.168.1.5"
PLC_PORT = 502
UNIT_ID = 1
D512_ADDRESS = 512
D0_ADDRESS = 0


def create_plc_client():
    return ModbusTcpClient(PLC_IP, port=PLC_PORT)


def read_register(client, address: int):
    try:
        resp = client.read_holding_registers(int(address), 1, unit=UNIT_ID)
        if resp is None or resp.isError():
            return None
        return int(resp.registers[0])
    except Exception:
        return None


class PLCStatusDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Ensure this dialog paints its own background (avoid parent stylesheet bleed)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setWindowTitle("PLC Status")
        self.setModal(True)
        self.setMinimumSize(560, 380)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(POPUP_QSS)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QtWidgets.QLabel("Check PLC")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        self.primary = QtWidgets.QFrame()
        self.primary.setObjectName("primaryCard")
        p = QtWidgets.QVBoxLayout(self.primary)
        p.setContentsMargins(16, 14, 16, 14)
        p.setSpacing(8)

        self.icon = QtWidgets.QLabel("⏳")
        self.icon.setAlignment(Qt.AlignCenter)
        self.icon.setStyleSheet("font-size:28px; color:white;")
        p.addWidget(self.icon)

        self.status_text = QtWidgets.QLabel("Checking connection...")
        self.status_text.setObjectName("primaryText")
        self.status_text.setAlignment(Qt.AlignCenter)
        p.addWidget(self.status_text)

        self.sub_text = QtWidgets.QLabel("")
        self.sub_text.setObjectName("secondaryText")
        self.sub_text.setAlignment(Qt.AlignCenter)
        p.addWidget(self.sub_text)

        root.addWidget(self.primary)

        self.values = QtWidgets.QFrame()
        self.values.setObjectName("card")
        v = QtWidgets.QVBoxLayout(self.values)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)

        self.machine = QtWidgets.QLabel("Machine State: -")
        self.machine.setObjectName("valueText")
        self.machine.setAlignment(Qt.AlignCenter)

        self.d0 = QtWidgets.QLabel("D0 Value: -")
        self.d0.setObjectName("valueText")
        self.d0.setAlignment(Qt.AlignCenter)

        v.addWidget(self.machine)
        v.addWidget(self.d0)
        root.addWidget(self.values)

        btnrow = QtWidgets.QHBoxLayout()
        btnrow.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.setObjectName("danger")
        btn_close.clicked.connect(self.close)
        btnrow.addWidget(btn_close)
        root.addLayout(btnrow)

        self._pulse = True
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._pulse_primary)
        self._timer.start(420)

        QTimer.singleShot(50, self.check_plc)

    def _pulse_primary(self):
        self._pulse = not self._pulse
        if self._pulse:
            self.primary.setStyleSheet(f"background:{BRAND}; border:1px solid {BRAND_DARK}; border-radius:16px;")
        else:
            self.primary.setStyleSheet(f"background:{BRAND_DARK}; border:1px solid {BRAND_DARK}; border-radius:16px;")

    def check_plc(self):
        client = create_plc_client()

        ok = False
        try:
            ok = bool(client.connect())
        except Exception:
            ok = False

        if not ok:
            self.icon.setText("⚠️")
            self.status_text.setText("PLC: NOT CONNECTED")
            self.sub_text.setText("Check network / IP / power")
            self._timer.stop()
            self.primary.setStyleSheet(f"background:{BRAND_DARK}; border:1px solid {BRAND_DARK}; border-radius:16px;")
            apply_shake(self.primary)
            try:
                client.close()
            except Exception:
                pass
            return

        self.icon.setText("✅")
        self.status_text.setText("PLC: CONNECTED")
        self.sub_text.setText(f"IP: {PLC_IP}:{PLC_PORT}")
        self._timer.stop()
        self.primary.setStyleSheet(f"background:{BRAND}; border:1px solid {BRAND_DARK}; border-radius:16px;")

        d512 = read_register(client, D512_ADDRESS)
        if d512 == 1:
            self.machine.setText("Machine State: ON ✅")
        elif d512 == 0:
            self.machine.setText("Machine State: OFF ⏸️")
        else:
            self.machine.setText("Machine State: UNKNOWN ⚠️")

        d0 = read_register(client, D0_ADDRESS)
        if d0 is None:
            self.d0.setText("D0 Value: READ ERROR ❌")
        else:
            self.d0.setText(f"D0 Value: {d0}")

        try:
            client.close()
        except Exception:
            pass


# ==================================================
# CAMERA THREAD (raw live)
# ==================================================
class CameraThread(QThread):
    frame_ready = pyqtSignal(int, object)
    status = pyqtSignal(int, bool)

    def __init__(self, cam_index: int, width: int = 640, height: int = 480, parent=None):
        super().__init__(parent)
        self.cam_index = int(cam_index)
        self.width = int(width)
        self.height = int(height)
        self._running = True
        self._cap = None

    def stop(self):
        self._running = False

    def _open(self):
        for be in _choose_backend():
            try:
                cap = cv2.VideoCapture(self.cam_index) if be is None else cv2.VideoCapture(self.cam_index, be)
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self._cap = cap
                    self.status.emit(self.cam_index, True)
                    return True
                if cap:
                    cap.release()
            except Exception:
                pass

        self.status.emit(self.cam_index, False)
        return False

    def run(self):
        if not self._open():
            return
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                self.frame_ready.emit(self.cam_index, frame)
            self.msleep(10)
        try:
            if self._cap:
                self._cap.release()
        except Exception:
            pass


# ==================================================
# 4 CAM LIVE PREVIEW (fixed size + click-to-fullscreen)
# ==================================================
class Live4CamPopup(QtWidgets.QDialog):
    WIN_W = 980
    WIN_H = 720
    TILE_W = 440
    TILE_H = 280

    def __init__(self, cam_indexes=(0,2,4,6), parent=None):
        super().__init__(parent)

        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setWindowTitle("4 Cam Live Preview")
        self.setModal(True)
        self.setFixedSize(self.WIN_W, self.WIN_H)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(POPUP_QSS)

        self._threads = []
        self._labels = {}
        self._got_first = set()
        self._fade_anims = {}
        self._last_frame = {}
        self._fullscreen = None

        self._shimmer_phase = 0.0
        self._shimmer_timer = QTimer(self)
        self._shimmer_timer.timeout.connect(self._tick_shimmer)

        self._glow_phase = 0.0
        self._glow_timer = QTimer(self)
        self._glow_timer.timeout.connect(self._tick_glow)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        head_card = QtWidgets.QFrame()
        head_card.setObjectName("card")
        head_card.setMinimumHeight(78)
        head = QtWidgets.QVBoxLayout(head_card)
        head.setContentsMargins(14, 12, 14, 12)
        head.setSpacing(8)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)

        live_icon = QtWidgets.QLabel("🔴")
        live_icon.setStyleSheet("font-size:18px;")
        row.addWidget(live_icon, 0, Qt.AlignLeft)

        live_text = QtWidgets.QLabel("4 Cam Live Preview")
        live_text.setStyleSheet(f"color:{INK}; font-weight:900; font-size:14px;")
        row.addWidget(live_text, 1, Qt.AlignLeft)

        live_badge = QtWidgets.QLabel("LIVE")
        live_badge.setObjectName("badge")
        live_badge.setStyleSheet(f"background:{TINT_BRAND}; color:{BRAND}; border:1px solid {LINE};")
        row.addWidget(live_badge, 0, Qt.AlignRight)

        head.addLayout(row)

        self.head_track = QtWidgets.QFrame()
        self.head_track.setStyleSheet(f"background:{LINE}; border-radius:10px;")
        self.head_track.setFixedHeight(14)

        self.head_bar = QtWidgets.QFrame(self.head_track)
        self.head_bar.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f" stop:0 {BRAND_DARK}, stop:0.5 {BRAND}, stop:1 {BRAND_DARK});"
            f" border-radius:10px;"
        )
        self.head_bar.setGeometry(0, 0, 180, 14)

        head.addWidget(self.head_track)
        root.addWidget(head_card)

        self.grid_frame = QtWidgets.QFrame()
        self.grid_frame.setObjectName("card")
        grid = QtWidgets.QGridLayout(self.grid_frame)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        for pos, cam_idx in enumerate(cam_indexes[:4]):
            r, c = divmod(pos, 2)

            lbl = ClickableCamLabel(cam_idx)
            lbl.setObjectName("camBox")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(self.TILE_W, self.TILE_H)
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            lbl.clicked.connect(self._on_tile_clicked)

            eff = QtWidgets.QGraphicsOpacityEffect(lbl)
            eff.setOpacity(0.55)
            lbl.setGraphicsEffect(eff)
            lbl._opacity = eff

            grid.addWidget(lbl, r, c)
            self._labels[int(cam_idx)] = lbl

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        root.addWidget(self.grid_frame, 1)

        btnrow = QtWidgets.QHBoxLayout()
        btnrow.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setObjectName("danger")
        close_btn.clicked.connect(self.close)
        btnrow.addWidget(close_btn)
        root.addLayout(btnrow)

        self._start_header_anim()
        self._shimmer_timer.start(40)
        self._glow_timer.start(50)

        self._start_threads(cam_indexes[:4])

    def _start_header_anim(self):
        w = self.head_track.width()
        barw = self.head_bar.width()
        self._head_anim = QtCore.QPropertyAnimation(self.head_bar, b"pos", self)
        self._head_anim.setDuration(1200)
        self._head_anim.setEasingCurve(QtCore.QEasingCurve.InOutSine)
        self._head_anim.setLoopCount(-1)
        self._head_anim.setStartValue(QtCore.QPoint(0, 0))
        self._head_anim.setEndValue(QtCore.QPoint(max(0, w - barw), 0))
        self._head_anim.start()

    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        try:
            self._head_anim.stop()
        except Exception:
            pass
        self.head_bar.setFixedHeight(self.head_track.height())
        self.head_bar.setFixedWidth(180)
        self._start_header_anim()

    def _tick_glow(self):
        self._glow_phase += 0.05
        if self._glow_phase > 6.28:
            self._glow_phase = 0.0
        t = (1.0 + math.sin(self._glow_phase)) * 0.5
        alpha = int(25 + 35 * t)

        self.grid_frame.setStyleSheet(
            f"""
            QFrame#card {{
                background: {SURFACE};
                border-radius: 16px;
                border: 2px solid rgba(112, 33, 42, {alpha});
            }}
            """
        )

    def _tick_shimmer(self):
        self._shimmer_phase += 0.03
        if self._shimmer_phase > 6.28:
            self._shimmer_phase = 0.0

        t = (1.0 + math.sin(self._shimmer_phase)) * 0.5
        base = QColor(DARK_PANEL)
        hi = QColor("#18181b")
        mix = QColor(
            int(base.red() * (1 - t) + hi.red() * t),
            int(base.green() * (1 - t) + hi.green() * t),
            int(base.blue() * (1 - t) + hi.blue() * t),
        )

        for cam_idx, lbl in self._labels.items():
            if cam_idx in self._got_first:
                continue
            lbl.setStyleSheet(
                f"QLabel#camBox{{background:{mix.name()}; border:1px solid {LINE}; border-radius:16px;}}"
            )

    def _start_threads(self, cam_indexes):
        self._stop_threads()
        for cam_idx in cam_indexes:
            th = CameraThread(cam_idx, width=640, height=480, parent=self)
            th.frame_ready.connect(self._on_frame)
            th.status.connect(self._on_status)
            th.start()
            self._threads.append(th)

    def _on_status(self, cam_idx: int, ok: bool):
        lbl = self._labels.get(int(cam_idx))
        if not lbl:
            return
        if not ok:
            lbl.setText("⚠️")
            lbl.setStyleSheet(
                f"QLabel#camBox{{background:{DARK_PANEL}; border:1px solid {LINE}; border-radius:16px;}}"
            )
            apply_shake(lbl)

    def _fade_in_tile(self, cam_idx: int):
        lbl = self._labels.get(int(cam_idx))
        if not lbl:
            return
        if cam_idx in self._fade_anims:
            return
        anim = QtCore.QPropertyAnimation(lbl._opacity, b"opacity", lbl)
        anim.setDuration(320)
        anim.setStartValue(0.55)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._fade_anims[cam_idx] = anim
        anim.start()

    def _on_frame(self, cam_idx: int, frame):
        cam_idx = int(cam_idx)
        lbl = self._labels.get(cam_idx)
        if not lbl:
            return

        self._last_frame[cam_idx] = frame

        if cam_idx not in self._got_first:
            self._got_first.add(cam_idx)
            lbl.setText("")
            lbl.setStyleSheet(
                f"QLabel#camBox{{background:{DARK_PANEL}; border:1px solid {LINE}; border-radius:16px;}}"
            )
            self._fade_in_tile(cam_idx)

        w = max(10, lbl.width())
        h = max(10, lbl.height())
        lbl.setPixmap(_cvimg_to_qpix_fill(frame, w, h))

        if self._fullscreen is not None and self._fullscreen.isVisible():
            if getattr(self._fullscreen, "cam_idx", -1) == cam_idx:
                self._fullscreen.update_frame(frame)

    def _on_tile_clicked(self, cam_idx: int):
        cam_idx = int(cam_idx)

        if cam_idx not in self._got_first:
            return
        if cam_idx not in self._last_frame:
            return

        if self._fullscreen is not None and self._fullscreen.isVisible():
            return

        fs = FullscreenCamView(self, cam_idx)
        self._fullscreen = fs
        fs.update_frame(self._last_frame[cam_idx])
        fs.finished.connect(self._on_fullscreen_closed)
        fs.show()

    def _on_fullscreen_closed(self):
        self._fullscreen = None

    def _stop_threads(self):
        for th in self._threads:
            try:
                th.stop()
                th.wait(1500)
            except Exception:
                pass
        self._threads = []

    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            self._shimmer_timer.stop()
        except Exception:
            pass
        try:
            self._glow_timer.stop()
        except Exception:
            pass
        self._stop_threads()
        super().closeEvent(e)


# ==================================================
# CONTROLLER POPUP
# ==================================================
class ControllerPopup(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Ensure this dialog paints its own background (avoid parent stylesheet bleed)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setWindowTitle("Controller")
        self.setMinimumSize(420, 350)
        self.setModal(False)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(POPUP_QSS)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Controller")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel("Quick health checks and live preview tools.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        self.btn_plc = QtWidgets.QPushButton("Check PLC")
        self.btn_cam = QtWidgets.QPushButton("Check Camera")
        self.btn_live4 = QtWidgets.QPushButton("4 cam live preview")

        self.btn_plc.clicked.connect(self._on_check_plc)
        self.btn_cam.clicked.connect(self._on_check_camera)
        self.btn_live4.clicked.connect(self._on_live_preview)

        layout.addWidget(self.btn_plc)
        layout.addWidget(self.btn_cam)
        layout.addWidget(self.btn_live4)

        layout.addStretch(1)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setObjectName("danger")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def _on_check_plc(self):
        dlg = PLCStatusDialog(self)
        dlg.exec_()

    def _on_check_camera(self):
        dlg = CameraStatusDialog(self, max_index=10)
        dlg.exec_()

    def _on_live_preview(self):
        dlg = Live4CamPopup(cam_indexes=(0, 1, 2, 3), parent=self)
        dlg.exec_()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = ControllerPopup()
    w.show()
    sys.exit(app.exec_())
