# ---------- PyQt5 ----------
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QFrame, QSplitter, QSizePolicy, QScrollArea,QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from typing import Dict
from classes.controller_popup import ControllerPopup
from classes.settings_popup import SettingsPopup
from datetime import datetime

CAM_INDICES = (2, 6, 4, 0)
def now_str():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

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
        self.score_lbl.setText(f"Score: {score:.3f}")
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
        
        try:
            from classes.app_gbu import PLCWorker
            self.worker = PLCWorker()

        except Exception as e:
            QMessageBox.critical(self, "Worker Start Error", str(e))
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return

        self.worker.sig_plc_read.connect(self.on_plc_read)
        self.worker.sig_plc_write.connect(self.on_plc_write)
        self.worker.sig_cam_res.connect(self.on_cam_res)
        self.worker.sig_cam_vis.connect(self.on_cam_vis)
        self.worker.sig_overall.connect(self.on_overall)
        self.worker.sig_log.connect(self.append_log)
        
        self.worker.start()

    def stop_worker(self):
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.append_log(f"[{now_str()}] ⏹ Stop requested")
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)

    def closeEvent(self, e):
        try:
            if self.worker and self.worker.isRunning():
                self.worker.stop()
                self.worker.wait(1500)
        except Exception:
            pass
        e.accept()