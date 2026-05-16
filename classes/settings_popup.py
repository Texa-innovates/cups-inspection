import sqlite3
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QComboBox, QFrame, QTimeEdit,
    QWidget, QStackedWidget, QLineEdit, QListWidget, QMessageBox, QTableWidget, QTableWidgetItem,
    QFileDialog
)
from path import JOBID_JSON_FILE,MODEL_PATH
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTime
from PyQt5.QtGui import QIntValidator
from classes.database import (fetch_shifts, delete_shift_by_id, insert_shift,
                      get_saved_machine,save_machine,delete_machine,insert_jobid,fetch_jobids,set_active_jobid  )
import os
from path import DB_PATH

JOB_SETUP_JSON_FILE = JOBID_JSON_FILE

import json

def save_job_setup_to_json(job_id: str, threshold: int, threshold_path: str):
    """
    Saves Job ID Setup values into ONE JSON file as 3 separate dictionaries.
    File: config/job_setup.json
    """
    os.makedirs(os.path.dirname(JOB_SETUP_JSON_FILE), exist_ok=True)

    payload = {
        "job_id": {
            "value": job_id or "",
            "locked": True
        },
        "threshold": {
            "value": int(threshold),
            "locked": True
        },
        "threshold_path": {
            "value": threshold_path or "",
            "locked": True
        }
    }

    with open(JOB_SETUP_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_job_setup_from_json() -> dict:
    """
    Reads config/job_setup.json and returns dict.
    If file missing/corrupt -> returns empty dict.
    """
    try:
        with open(JOB_SETUP_JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


class TickToggleSwitch(QtWidgets.QAbstractButton):
    """Modern pill toggle with animated knob + check icon (ON=locked, OFF=editable)."""

    def __init__(
        self,
        parent=None,
        w: int = 10,
        h: int = 8,
        bg_off: QtGui.QColor = QtGui.QColor("#B63838"),
        bg_on: QtGui.QColor = QtGui.QColor("#1A6326"),
        knob: QtGui.QColor = QtGui.QColor("#FFFFFF"),
        check_color: QtGui.QColor = QtGui.QColor("#1A6326"),
    ):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)

        self._w = int(w)
        self._h = int(h)
        self._margin = 4
        self._bg_off = bg_off
        self._bg_on = bg_on
        self._knob = knob
        self._check_color = check_color

        self._t = 0.0  # 0=OFF, 1=ON
        self._press_scale = 1.0

        self._anim = QtCore.QPropertyAnimation(self, b"t", self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        self._press_anim = QtCore.QPropertyAnimation(self, b"pressScale", self)
        self._press_anim.setDuration(120)
        self._press_anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        self.setFixedSize(50, 30)
        self.toggled.connect(self._on_toggled)

    # ---------- animation properties ----------
    def getT(self):
        return self._t

    def setT(self, v):
        self._t = float(v)
        self.update()

    t = QtCore.pyqtProperty(float, fget=getT, fset=setT)

    def getPressScale(self):
        return self._press_scale

    def setPressScale(self, v):
        self._press_scale = float(v)
        self.update()

    pressScale = QtCore.pyqtProperty(float, fget=getPressScale, fset=setPressScale)

    # ---------- behavior ----------
    def _on_toggled(self, checked: bool):
        self._anim.stop()
        self._anim.setStartValue(self._t)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def setChecked(self, checked: bool):
        super().setChecked(checked)
        self._t = 1.0 if checked else 0.0
        self.update()

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._press_anim.stop()
            self._press_anim.setStartValue(self._press_scale)
            self._press_anim.setEndValue(0.97)
            self._press_anim.start()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        self._press_anim.stop()
        self._press_anim.setStartValue(self._press_scale)
        self._press_anim.setEndValue(1.0)
        self._press_anim.start()
        super().mouseReleaseEvent(e)

    # ---------- painting ----------
    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        p.save()
        cx, cy = self.width() / 2, self.height() / 2
        p.translate(cx, cy)
        p.scale(self._press_scale, self._press_scale)
        p.translate(-cx, -cy)

        def lerp(a, b, t):
            return a + (b - a) * t

        bg = QtGui.QColor(
            int(lerp(self._bg_off.red(), self._bg_on.red(), self._t)),
            int(lerp(self._bg_off.green(), self._bg_on.green(), self._t)),
            int(lerp(self._bg_off.blue(), self._bg_on.blue(), self._t)),
        )

        r = self.height() / 2
        rect = QtCore.QRectF(0, 0, self.width(), self.height())

        # shadow/glow (stronger when ON)
        p.setPen(Qt.NoPen)
        shadow_alpha = int(35 + 85 * self._t)
        p.setBrush(QtGui.QColor(0, 0, 0, shadow_alpha))
        p.drawRoundedRect(rect.adjusted(2, 4, -2, -1), r, r)

        # pill
        p.setBrush(bg)
        p.drawRoundedRect(rect.adjusted(2, 2, -2, -3), r, r)

        # knob
        knob_d = self.height() - 2 * self._margin
        x0 = self._margin
        x1 = self.width() - self._margin - knob_d
        knob_x = x0 + (x1 - x0) * self._t
        knob_rect = QtCore.QRectF(knob_x, self._margin, knob_d, knob_d)

        # knob shadow
        p.setBrush(QtGui.QColor(0, 0, 0, 55))
        p.drawEllipse(knob_rect.adjusted(1.5, 2.5, 1.5, 2.5))

        # knob circle
        p.setBrush(self._knob)
        p.drawEllipse(knob_rect)

# ---------- lock/unlock icon like sample (filled lock / outline unlock) ----------
        cxk = knob_rect.center().x()
        cyk = knob_rect.center().y()
        S = knob_d * 0.52  # overall icon scale (tune 0.48~0.58)

        # Colors:
        # - Filled icon (LOCK) usually looks best with dark fill + white keyhole
        # - Outline icon (UNLOCK) uses stroke only
        lock_fill = QtGui.QColor(25, 25, 25)          # solid lock color (like sample)
        unlock_stroke = QtGui.QColor(25, 25, 25)      # outline color
        keyhole_color = QtGui.QColor(255, 255, 255)   # keyhole "cutout" (white)

        stroke = max(2.0, knob_d * 0.10)

        def draw_lock_icon(open_state: bool, filled: bool, opacity: float):
            p.save()
            p.setOpacity(opacity)
            p.setRenderHint(QtGui.QPainter.Antialiasing, True)

            # Body geometry
            body_w = S * 0.92
            body_h = S * 0.68
            body_x = cxk - body_w / 2
            body_y = cyk - body_h / 2 + S * 0.12
            body_r = body_h * 0.20

            body_rect = QtCore.QRectF(body_x, body_y, body_w, body_h)

            # Shackle geometry
            sh_w = body_w * 0.62
            sh_h = S * 0.60
            sh_x = cxk - sh_w / 2
            sh_y = body_y - sh_h * 0.58

            # Unlocked look: shift shackle slightly right and open it
            if open_state:
                sh_x += sh_w * 0.22

            sh_rect = QtCore.QRectF(sh_x, sh_y, sh_w, sh_h)

            # --- draw filled lock ---
            if filled:
                # Body fill
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(lock_fill)
                p.drawRoundedRect(body_rect, body_r, body_r)

                # Shackle fill: draw thick stroke-style shackle using pen+path but filled look
                pen = QtGui.QPen(lock_fill)
                pen.setWidthF(stroke * 1.35)
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)

                sh = QtGui.QPainterPath()
                # Right leg up -> arc -> left leg down (locked),
                # unlocked: left leg stops early (gap)
                right_leg_end = QtCore.QPointF(sh_rect.right(), body_y + body_h * 0.10)
                sh.moveTo(right_leg_end)
                sh.lineTo(sh_rect.right(), sh_rect.center().y())
                sh.arcTo(sh_rect, 0, 180)

                if not open_state:
                    sh.lineTo(sh_rect.left(), body_y + body_h * 0.10)
                else:
                    sh.lineTo(sh_rect.left(), body_y - sh_h * 0.06)

                p.drawPath(sh)

                # Keyhole (white cutout)
                p.setPen(Qt.NoPen)
                p.setBrush(keyhole_color)
                # small circle + small rectangle (classic keyhole)
                kh_cy = body_rect.center().y() + body_h * 0.02
                r = body_w * 0.06
                p.drawEllipse(QtCore.QPointF(cxk, kh_cy - r * 0.2), r, r)
                p.drawRoundedRect(QtCore.QRectF(cxk - r * 0.55, kh_cy, r * 1.1, r * 1.35), r*0.35, r*0.35)

            # --- draw outline lock ---
            else:
                p.setBrush(Qt.NoBrush)
                pen = QtGui.QPen(unlock_stroke)
                pen.setWidthF(stroke)
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
                p.setPen(pen)

                # Body outline
                p.drawRoundedRect(body_rect, body_r, body_r)

                # Shackle outline (smooth)
                sh = QtGui.QPainterPath()
                right_leg_end = QtCore.QPointF(sh_rect.right(), body_y + body_h * 0.10)
                sh.moveTo(right_leg_end)
                sh.lineTo(sh_rect.right(), sh_rect.center().y())
                sh.arcTo(sh_rect, 0, 180)

                if not open_state:
                    sh.lineTo(sh_rect.left(), body_y + body_h * 0.10)
                else:
                    sh.lineTo(sh_rect.left(), body_y - sh_h * 0.06)

                p.drawPath(sh)

                # Keyhole outline (simple)
                kh_cy = body_rect.center().y() + body_h * 0.02
                r = body_w * 0.06
                p.drawEllipse(QtCore.QPointF(cxk, kh_cy - r * 0.2), r, r)
                p.drawRoundedRect(QtCore.QRectF(cxk - r * 0.55, kh_cy, r * 1.1, r * 1.35), r*0.35, r*0.35)

            p.restore()

        # Cross-fade between styles using self._t
        # ON (locked) => filled, closed
        # OFF (unlocked) => outline, open
        draw_lock_icon(open_state=True,  filled=False, opacity=0.15 + 0.85 * (1.0 - self._t))  # outline unlock
        draw_lock_icon(open_state=False, filled=True,  opacity=0.15 + 0.85 * self._t)          # filled lock
        p.restore()
        p.end()


class ShiftCreateDialog(QDialog):
    def __init__(self, conn, parent=None):
        super().__init__(parent)
        self.conn = conn
        self.setWindowTitle("Create Shift")
        self.setFixedSize(260, 160)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        lbl_shift=QLabel("Shift Name")
        lbl_start = QLabel("Start Time")
        lbl_end = QLabel("End Time")
        
        self.shift_combo = QComboBox()
        self.shift_combo.addItems(["A", "B", "C"])
        self.shift_combo.setFixedHeight(28)
        self.time_start = QTimeEdit(QTime(6, 0))
        self.time_end = QTimeEdit(QTime(14, 0))

        for t in (self.time_start, self.time_end):
            t.setDisplayFormat("HH:mm")
            t.setFixedHeight(28)

        form.addWidget(lbl_shift, 0, 0)
        form.addWidget(self.shift_combo, 0, 1)

        form.addWidget(lbl_start, 1, 0)
        form.addWidget(self.time_start, 1, 1)

        form.addWidget(lbl_end, 2, 0)
        form.addWidget(self.time_end, 2, 1)

        layout.addLayout(form)

        btn_submit = QPushButton("Submit")
        btn_submit.setFixedHeight(32)
        btn_submit.clicked.connect(self.on_submit)

        layout.addStretch(1)
        layout.addWidget(btn_submit, alignment=Qt.AlignRight)

    def get_values(self):
        shift = self.shift_combo.currentText().strip()
        return shift, self.time_start.time(), self.time_end.time()
    
    def _duration_minutes(self, start_t: QTime, end_t: QTime) -> int:
        s = start_t.hour() * 60 + start_t.minute()
        e = end_t.hour() * 60 + end_t.minute()
        return e - s
    
    def on_submit(self):
        shift, start_t, end_t = self.get_values()

        if start_t == end_t:
            QMessageBox.warning(self, "Invalid", "Start and End time cannot be same.")
            return

        dur = self._duration_minutes(start_t, end_t)

        # ✅ MUST BE < 8 hours (strict)
        if dur >= 8 * 60:
            QMessageBox.warning(self, "Invalid", "Shift duration must be less than 8 hours.")
            return
        try:
            insert_shift(self.conn, shift, start_t, end_t)  # database.py
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "DB Error", str(e))
    
class MachineAddDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Machine")
        self.setFixedSize(260, 140)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        lbl_machine = QLabel("Machine Number")
        self.txt_machine = QLineEdit()
        self.txt_machine.setValidator(QIntValidator())
        self.txt_machine.setFixedHeight(28)
        self.txt_machine.setPlaceholderText("Enter machine number")

        form.addWidget(lbl_machine, 0, 0)
        form.addWidget(self.txt_machine, 0, 1)

        layout.addLayout(form)

        btn_submit = QPushButton("Submit")
        btn_submit.setFixedHeight(32)
        btn_submit.clicked.connect(self.accept)

        layout.addStretch(1)
        layout.addWidget(btn_submit, alignment=Qt.AlignRight)

    def get_machine_number(self):
        return self.txt_machine.text().strip()



class SettingsPopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(680, 440)
        self.setModal(False)
        self.conn = sqlite3.connect(DB_PATH)
        # ensure_shift_table(self.conn)


        # ================= ROOT =================
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # ==================================================
        # HEADER (Title + Create)
        # ==================================================
        header = QHBoxLayout()

        title = QLabel("Settings")
        title.setStyleSheet("font-size:18px; font-weight:600;")

        # self.btn_create = QPushButton("Create")
        # self.btn_create.setFixedHeight(30)
        # self.btn_create.setMinimumWidth(90)

        header.addWidget(title)
        header.addStretch(1)
        # header.addWidget(self.btn_create)

        # ==================================================
        # TAB BUTTONS
        # ==================================================
        tabs = QHBoxLayout()
        tabs.setSpacing(8)

        self.btn_quick = QPushButton("Job ID Setup")
        self.btn_program = QPushButton("Job ID Create")
        self.btn_timing = QPushButton("Shifts Create")
        self.btn_data = QPushButton("Machine")

        for b in (self.btn_quick, self.btn_program, self.btn_timing, self.btn_data):
            b.setCheckable(True)
            b.setFixedHeight(34)
            b.setMinimumWidth(130)

        self.btn_quick.setChecked(True)

        tabs.addWidget(self.btn_quick)
        tabs.addWidget(self.btn_program)
        tabs.addWidget(self.btn_timing)
        tabs.addWidget(self.btn_data)
        tabs.addStretch(1)

        # ==================================================
        # CONTENT AREA (STACKED)
        # ==================================================
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_job_setup())
        self.stack.addWidget(self._build_job_id())
        self.stack.addWidget(self._build_shift_timings())  
        self.stack.addWidget(self._build_machine())

        # ==================================================
        # FOOTER (Save / Exit)
        # ==================================================
        footer = QHBoxLayout()
        footer.addStretch(1)

        self.btn_save = QPushButton("Save")
        self.btn_exit = QPushButton("Exit")

        for b in (self.btn_save, self.btn_exit):
            b.setFixedHeight(34)
            b.setMinimumWidth(100)

        self.btn_exit.clicked.connect(self.close)
        # Save is only shown for Job ID Setup tab, so we keep this handler specific.
        self.btn_save.clicked.connect(self._job_setup_on_save)

        footer.addWidget(self.btn_save)
        footer.addWidget(self.btn_exit)

        # ==================================================
        # ASSEMBLE
        # ==================================================
        root.addLayout(header)
        root.addLayout(tabs)
        root.addWidget(self.stack, 1)
        root.addLayout(footer)

        # ==================================================
        # TAB CONNECTIONS
        # ==================================================
        self.btn_quick.clicked.connect(lambda: self.switch_tab(0))
        self.btn_program.clicked.connect(lambda: self.switch_tab(1))
        self.btn_timing.clicked.connect(lambda: self.switch_tab(2))
        self.btn_data.clicked.connect(lambda: self.switch_tab(3))

        # Initialize correct visibility for default page
        self.switch_tab(0)  


    # ==================================================
    # TAB SWITCH HANDLER
    # ==================================================
    def switch_tab(self, index: int):
        self.stack.setCurrentIndex(index)

        self.btn_quick.setChecked(index == 0)
        self.btn_program.setChecked(index == 1)
        self.btn_timing.setChecked(index == 2)
        self.btn_data.setChecked(index == 3)

        # ================= SAVE / EXIT VISIBILITY =================
        # NOTE: during early UI construction, btn_save/btn_exit may not exist yet.
        if hasattr(self, "btn_save") and hasattr(self, "btn_exit"):
            if index == 0:
                # Job ID Setup → show
                self.btn_save.show()
                self.btn_exit.show()
            else:
                # Job ID Create / Machine / Others → hide
                self.btn_save.hide()
                self.btn_exit.hide()


    # ==================================================
    # Job ID Setup - helpers (DO NOT change other tab logic)
    # ==================================================
    def _job_setup_update_save_state(self):
        """Enable Save only when all 3 selections are locked (switch ON)."""
        # During widget construction, Save button might not be created yet.
        if not hasattr(self, "btn_save"):
            return
        try:
            ok = bool(self.sw_job.isChecked() and self.sw_threshold.isChecked() and self.sw_browse.isChecked())
        except Exception:
            ok = False
        self.btn_save.setEnabled(ok)

    def _job_setup_toggle_job_lock(self, locked: bool):
        """Lock/unlock the Job ID combo."""
        if locked:
            if self.cmb_job.currentIndex() < 0:
                QMessageBox.warning(self, "Missing", "Please select a Job ID before locking.")
                self.sw_job.setChecked(False)
                return
        self.cmb_job.setEnabled(not locked)
        self._job_setup_update_save_state()

    def _job_setup_toggle_threshold_lock(self, locked: bool):
        """Lock/unlock threshold input. Threshold must be integer only."""
        if locked:
            text = self.txt_threshold.text().strip()
            if text == "" or not text.isdigit():
                QMessageBox.warning(self, "Invalid", "Threshold must be an integer.")
                self.sw_threshold.setChecked(False)
                return
        self.txt_threshold.setEnabled(not locked)
        self._job_setup_update_save_state()

    def _job_setup_toggle_browse_lock(self, locked: bool):
        """Lock/unlock browse threshold file selection."""
        if locked:
            if not self._job_setup_selected_path:
                QMessageBox.warning(self, "Missing", "Please browse and select a file before locking.")
                self.sw_browse.setChecked(False)
                return
        self.btn_browse_threshold.setEnabled(not locked)
        self._job_setup_update_save_state()

    def _job_setup_browse_threshold_file(self):
        """Open file dialog and show selected path near the button."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Threshold File", MODEL_PATH, "Joblib Files (*.joblib);;All Files (*.*)")
        if path:
            self._job_setup_selected_path = path
            self.lbl_browse_path.setText(path)
            self.lbl_browse_path.setToolTip(path)
        else:
            # user cancelled; keep old selection
            pass
        self._job_setup_update_save_state()

    def _job_setup_get_material_brand(self, jobid_name: str):
        """Fetch material + module(brand) from DB for display after Save."""
        if not jobid_name:
            return "", ""
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT material, module FROM jobid_table WHERE jobid_name=? ORDER BY id DESC LIMIT 1", (jobid_name,))
            row = cur.fetchone()
            if not row:
                return "", ""
            material = row[0] or ""
            brand = row[1] or ""
            return material, brand
        except Exception:
            return "", ""

    def _job_setup_on_save(self):
        """Save the locked values to JSON file (only after all 3 are locked)."""

        # Must be locked before save
        if not (self.sw_job.isChecked() and self.sw_threshold.isChecked() and self.sw_browse.isChecked()):
            QMessageBox.warning(self, "Not Ready", "Please lock Job ID, Threshold and Browse Threshold before saving.")
            return

        jobid_selected = self.cmb_job.currentText().strip()
        thr_text = self.txt_threshold.text().strip()    

        if not jobid_selected:
            QMessageBox.warning(self, "Missing", "Job ID is missing.")
            return
        if thr_text == "" or not thr_text.isdigit():
            QMessageBox.warning(self, "Invalid", "Threshold must be an integer.")
            return
        if not self._job_setup_selected_path:
            QMessageBox.warning(self, "Missing", "Browse Threshold file path is missing.")
            return

        threshold_value = int(thr_text)

        # ✅ save JSON
        try:
            save_job_setup_to_json(jobid_selected, threshold_value, self._job_setup_selected_path)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save JSON:\n{e}")
            return

        # ✅ update DB status: only one ACTIVE
        try:
            set_active_jobid(self.conn, jobid_selected)
        except Exception as e:
            QMessageBox.critical(self, "DB Error", f"Failed to set ACTIVE job:\n{e}")
            return

        # ✅ refresh Job ID Create table if refresh function exists
        if hasattr(self, "refresh_jobid_table"):
            try:
                self.refresh_jobid_table()
            except Exception as e:
                print("refresh_jobid_table failed:", e)


        QMessageBox.information(self, "Saved", f"Active Job set: {jobid_selected}")
        
    def _job_setup_load_from_json(self):
        """
        Load last saved Job Setup from config/job_setup.json and restore:
        - JobID selection
        - threshold value
        - browse path label
        - all 3 switches ON (locked)
        - disable inputs (locked state)
        """
        data = load_job_setup_from_json()
        if not data:
            return

        jobid = (data.get("job_id") or {}).get("value", "") or ""
        thr = (data.get("threshold") or {}).get("value", "")
        path = (data.get("threshold_path") or {}).get("value", "") or ""

        # nothing to restore
        if not (jobid or str(thr).strip() or path):
            return

        # ---- set values without triggering popups ----
        # block signals for switches while setting
        self.sw_job.blockSignals(True)
        self.sw_threshold.blockSignals(True)
        self.sw_browse.blockSignals(True)

        self.cmb_job.blockSignals(True)
        self.txt_threshold.blockSignals(True)
        self.btn_browse_threshold.blockSignals(True)

        try:
            # Job ID in combo? else add it
            if jobid:
                idx = self.cmb_job.findText(jobid)
                if idx < 0:
                    self.cmb_job.addItem(jobid)
                    idx = self.cmb_job.findText(jobid)
                self.cmb_job.setCurrentIndex(idx)

            # Threshold
            if thr != "" and thr is not None:
                self.txt_threshold.setText(str(thr))

            # Path
            if path:
                self._job_setup_selected_path = path
                self.lbl_browse_path.setText(path)
                self.lbl_browse_path.setToolTip(path)

            # Turn ON switches (locked)
            self.sw_job.setChecked(True)
            self.sw_threshold.setChecked(True)
            self.sw_browse.setChecked(True)

        finally:
            self.cmb_job.blockSignals(False)
            self.txt_threshold.blockSignals(False)
            self.btn_browse_threshold.blockSignals(False)

            self.sw_job.blockSignals(False)
            self.sw_threshold.blockSignals(False)
            self.sw_browse.blockSignals(False)

        # ---- apply locked UI state ----
        self.cmb_job.setEnabled(False)
        self.txt_threshold.setEnabled(False)
        self.btn_browse_threshold.setEnabled(False)

        # Save button should be enabled
        self._job_setup_update_save_state()

        # After Save -> show material & brand
        material, brand = self._job_setup_get_material_brand(jobid)
        if material or brand:
            self.lbl_material_brand.setText(f"Material name : {material} \nBrand name :{brand}")
        else:
            self.lbl_material_brand.setText(f"Material name : (Not Found) \nBrand name : (Not Found)")


    # ==================================================
    # Job ID Setup UI
    # ==================================================
    def _build_job_setup(self):
        w = QWidget()

        # Compact input styling
        w.setStyleSheet("""
            QLineEdit, QComboBox {
                padding: 2px 6px;
            }
        """)

        # ---------------- state (job setup) ----------------
        self._job_setup_selected_path = ""

        main_layout = QVBoxLayout(w)
        main_layout.setSpacing(14)

        form = QGridLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(16)
        form.setColumnStretch(0, 0)   # labels
        form.setColumnStretch(1, 1)   # inputs
        form.setColumnStretch(2, 0)   # lock switch

        INPUT_H = 30
        INPUT_W = 400

        # ---------------- Select Job ID ----------------
        lbl_job = QLabel("Select Job ID")
        self.cmb_job = QComboBox()
        self.cmb_job.setEditable(False)
        self.cmb_job.setFixedHeight(INPUT_H)
        self.cmb_job.setFixedWidth(INPUT_W)

        # Load Job IDs from DB
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT jobid_name FROM jobid_table ORDER BY id DESC")
            job_ids = [row[0] for row in cursor.fetchall()]
            self.cmb_job.addItems(job_ids)
        except Exception as e:
            QMessageBox.critical(self, "DB Error", f"Failed to load Job IDs:\n{e}")

        # start unselected (user must lock explicitly)
        if self.cmb_job.count() > 0:
            self.cmb_job.setCurrentIndex(-1)

        self.sw_job = TickToggleSwitch(w=90, h=42)
        self.sw_job.setChecked(False)

        # ---------------- Threshold (integer only) ----------------
        lbl_threshold = QLabel("Threshold")
        self.txt_threshold = QLineEdit()
        self.txt_threshold.setPlaceholderText("Enter threshold value")
        self.txt_threshold.setFixedHeight(INPUT_H)
        self.txt_threshold.setFixedWidth(INPUT_W)
        self.txt_threshold.setValidator(QIntValidator(0, 10**9, self))

        self.sw_threshold = TickToggleSwitch(w=90, h=42)
        self.sw_threshold.setChecked(False)

        # ---------------- Browse Threshold ----------------
        lbl_browse = QLabel("Browse Threshold")
        browse_wrap = QHBoxLayout()
        browse_wrap.setSpacing(10)
        browse_wrap.setContentsMargins(0, 0, 0, 0)

        self.btn_browse_threshold = QPushButton("Browse Threshold")
        self.btn_browse_threshold.setFixedHeight(INPUT_H)
        self.btn_browse_threshold.setMinimumWidth(160)

        self.lbl_browse_path = QLabel("No file selected")
        self.lbl_browse_path.setStyleSheet("color:#faf0f0;")
        self.lbl_browse_path.setMinimumWidth(220)
        self.lbl_browse_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_browse_path.setToolTip("")

        browse_wrap.addWidget(self.btn_browse_threshold)
        browse_wrap.addWidget(self.lbl_browse_path, 1)

        browse_container = QWidget()
        browse_container.setLayout(browse_wrap)
        browse_container.setFixedWidth(INPUT_W)

        self.sw_browse = TickToggleSwitch(w=90, h=42)
        self.sw_browse.setChecked(False)

        # ---------------- layout rows ----------------
        form.addWidget(lbl_job, 0, 0)
        form.addWidget(self.cmb_job, 0, 1, alignment=Qt.AlignLeft)
        form.addWidget(self.sw_job, 0, 2, alignment=Qt.AlignLeft)

        form.addWidget(lbl_threshold, 1, 0)
        form.addWidget(self.txt_threshold, 1, 1, alignment=Qt.AlignLeft)
        form.addWidget(self.sw_threshold, 1, 2, alignment=Qt.AlignLeft)

        form.addWidget(lbl_browse, 2, 0)
        form.addWidget(browse_container, 2, 1, alignment=Qt.AlignLeft)
        form.addWidget(self.sw_browse, 2, 2, alignment=Qt.AlignLeft)

        main_layout.addLayout(form)

        # ---------------- NOTE ----------------
        self.lbl_jobsetup_note = QLabel(
            "Note: Select Job ID, Threshold and Browse Threshold. Lock each one using the switch (✅). Then Save will be enabled."
        )
        self.lbl_jobsetup_note.setWordWrap(True)
        self.lbl_jobsetup_note.setStyleSheet("color:#444; font-size:12px;")
        main_layout.addWidget(self.lbl_jobsetup_note)

        # ---------------- AFTER SAVE: show material & brand ----------------
        self.lbl_material_brand = QLabel("")
        self.lbl_material_brand.setStyleSheet("color:#faf0f0; font-size:13px; font-weight:600;")
        self.lbl_material_brand.setWordWrap(True)
        main_layout.addWidget(self.lbl_material_brand)

        main_layout.addStretch(1)

        # ---------------- connections ----------------
        self.btn_browse_threshold.clicked.connect(self._job_setup_browse_threshold_file)
        self.sw_job.toggled.connect(self._job_setup_toggle_job_lock)
        self.sw_threshold.toggled.connect(self._job_setup_toggle_threshold_lock)
        self.sw_browse.toggled.connect(self._job_setup_toggle_browse_lock)

        # initial state
        self._job_setup_update_save_state()
        # ✅ Restore last saved config (locks + values)
        self._job_setup_load_from_json()
        return w
    
    # ==================================================
    # Job ID Create UI 
    # ==================================================
    def _build_job_id(self):
        w = QWidget()

        # Compact input styling
        w.setStyleSheet("""
            QLineEdit, QComboBox {
                padding: 2px 6px;
            }
        """)

        main_layout = QVBoxLayout(w)
        main_layout.setSpacing(18)

        # ================= FORM =================
        form = QGridLayout()
        form.setHorizontalSpacing(40)
        form.setVerticalSpacing(16)
        form.setColumnStretch(0, 0)   # labels
        form.setColumnStretch(1, 1)   # inputs

        INPUT_H = 30
        INPUT_W = 400

        # ---- Job ID ----
        lbl_job_id = QLabel("Job ID")
        txt_job_id = QLineEdit()
        txt_job_id.setFixedHeight(INPUT_H)
        txt_job_id.setFixedWidth(INPUT_W)

        # ---- Machine Number (auto from Machine tab / DB) ----
        lbl_machine = QLabel("Machine Number")
        self.txt_job_machine = QLineEdit()
        self.txt_job_machine.setFixedHeight(INPUT_H)
        self.txt_job_machine.setFixedWidth(INPUT_W)
        self.txt_job_machine.setPlaceholderText("Auto-filled from Machine tab")
        self.txt_job_machine.setReadOnly(True)

        # ---- Material ----
        lbl_material = QLabel("Material")
        txt_material = QLineEdit()
        txt_material.setFixedHeight(INPUT_H)
        txt_material.setFixedWidth(INPUT_W)

        # ---- Brand (you called it module in DB, keeping your meaning) ----
        lbl_brand = QLabel("Brand")
        txt_brand = QLineEdit()
        txt_brand.setFixedHeight(INPUT_H)
        txt_brand.setFixedWidth(INPUT_W)

        # ---- Add widgets to form ----
        form.addWidget(lbl_job_id, 1, 0)
        form.addWidget(txt_job_id, 1, 1, alignment=Qt.AlignLeft)

        form.addWidget(lbl_machine, 2, 0)
        form.addWidget(self.txt_job_machine, 2, 1, alignment=Qt.AlignLeft)

        form.addWidget(lbl_material, 3, 0)
        form.addWidget(txt_material, 3, 1, alignment=Qt.AlignLeft)

        form.addWidget(lbl_brand, 4, 0)
        form.addWidget(txt_brand, 4, 1, alignment=Qt.AlignLeft)

        main_layout.addLayout(form)

        # ================= CREATE BUTTON =================
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)

        btn_create = QPushButton("Create")
        btn_create.setFixedHeight(34)
        btn_create.setMinimumWidth(120)

        btn_layout.addWidget(btn_create)

        main_layout.addLayout(btn_layout)

        # ================= TABLE (shows saved job ids) =================
        self.jobid_table = QTableWidget()
        self.jobid_table.setColumnCount(6)
        self.jobid_table.setHorizontalHeaderLabels(
            ["ID", "Job ID", "Machine", "Material", "Module", "Status"]
        )
        self.jobid_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.jobid_table.setSelectionMode(QTableWidget.SingleSelection)
        self.jobid_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.jobid_table.verticalHeader().setVisible(False)
        self.jobid_table.horizontalHeader().setStretchLastSection(True)

        main_layout.addWidget(self.jobid_table, 1)

        def refresh_jobid_table():
            try:
                rows = fetch_jobids(self.conn)
            except Exception as e:
                QMessageBox.critical(self, "DB Error", f"fetch_jobids failed:\n{e}")
                rows = []

            self.jobid_table.setRowCount(0)

            for rid, jobid_name, machine_no, material, module, status in rows:
                r = self.jobid_table.rowCount()
                self.jobid_table.insertRow(r)
                self.jobid_table.setItem(r, 0, QTableWidgetItem(str(rid)))
                self.jobid_table.setItem(r, 1, QTableWidgetItem(jobid_name or ""))
                self.jobid_table.setItem(r, 2, QTableWidgetItem(machine_no or ""))
                self.jobid_table.setItem(r, 3, QTableWidgetItem(material or ""))
                self.jobid_table.setItem(r, 4, QTableWidgetItem(module or ""))
                self.jobid_table.setItem(r, 5, QTableWidgetItem(status or ""))

            self.jobid_table.setColumnHidden(0, True)  # hide ID
        self.refresh_jobid_table = refresh_jobid_table
        def on_create():
            jobid_name = txt_job_id.text().strip()
            machine_no = self.txt_job_machine.text().strip()
            material   = txt_material.text().strip()
            module     = txt_brand.text().strip()

            if not jobid_name:
                QMessageBox.warning(self, "Missing", "Enter Job ID.")
                return
            if not machine_no:
                QMessageBox.warning(self, "Missing", "Machine number not set. Add machine in Machine tab.")
                return

            try:
                insert_jobid(self.conn, jobid_name, machine_no, material, module, status="DEACTIVE")
                refresh_jobid_table()
                QMessageBox.information(self, "Saved", "Job ID saved to DB.")

                # clear inputs (machine stays)
                txt_job_id.clear()
                txt_material.clear()
                txt_brand.clear()

            except Exception as e:
                QMessageBox.critical(self, "DB Error", str(e))

        btn_create.clicked.connect(on_create)

        # ================= AUTO-FILL MACHINE FROM DB =================
        try:
            saved = get_saved_machine(self.conn)  # from machinenumber_table
            if saved:
                self.txt_job_machine.setText(str(saved))
        except Exception:
            pass

        # Load table initially
        refresh_jobid_table()

        return w



    # ==================================================
    # Shift Timing UI 
    # ==================================================
    def _build_shift_timings(self):
        w = QWidget()
        main = QVBoxLayout(w)
        main.setSpacing(12)

        # ================= HEADER =================
        header = QHBoxLayout()

        title = QLabel("Shift Timings")
        title.setStyleSheet("font-size:18px; font-weight:600;")

        btn_create = QPushButton("Create")
        btn_delete = QPushButton("Delete")

        for b in (btn_create, btn_delete):
            b.setFixedHeight(32)
            b.setMinimumWidth(90)

        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(btn_create)
        header.addWidget(btn_delete)

        main.addLayout(header)

        # ================= SHIFT TABLE (DB) =================
        self.shift_table = QTableWidget()
        self.shift_table.setColumnCount(4)
        self.shift_table.setHorizontalHeaderLabels(["ID", "Shift", "Start", "End"])
        self.shift_table.setSelectionBehavior(self.shift_table.SelectRows)
        self.shift_table.setSelectionMode(self.shift_table.SingleSelection)
        self.shift_table.setEditTriggers(self.shift_table.NoEditTriggers)
        self.shift_table.verticalHeader().setVisible(False)
        self.shift_table.horizontalHeader().setStretchLastSection(True)

        main.addWidget(self.shift_table, 1)

        # # ================= SHIFT LIST =================
        # self.shift_list = QListWidget()
        # self.shift_list.setSelectionMode(QListWidget.SingleSelection)
        # self.shift_list.setStyleSheet("""
        #     QListWidget {
        #         border: 1px solid #cfcfcf;
        #         border-radius: 6px;
        #     }
        #     QListWidget::item {
        #         padding: 8px;
        #     }
        #     QListWidget::item:selected {
        #         background: #dbe9ff;
        #     }
        # # """)

        # main.addWidget(self.shift_list, 1)

        def refresh_shift_table():
            rows = fetch_shifts(self.conn)
            self.shift_table.setRowCount(0)

            for shift_id, shift, start_s, end_s in rows:
                r = self.shift_table.rowCount()
                self.shift_table.insertRow(r)
                self.shift_table.setItem(r, 0, QTableWidgetItem(str(shift_id)))
                self.shift_table.setItem(r, 1, QTableWidgetItem(shift))
                self.shift_table.setItem(r, 2, QTableWidgetItem(start_s))
                self.shift_table.setItem(r, 3, QTableWidgetItem(end_s))

            self.shift_table.setColumnHidden(0, True)  # hide ID but keep it for delete

        refresh_shift_table()

        # ================= CREATE ACTION =================
        def create_shift():
            dlg = ShiftCreateDialog(self.conn, self) 
            if dlg.exec_() != QDialog.Accepted:
                return
            refresh_shift_table()
            shift, start, end = dlg.get_values()

            entry = f"Shift: {shift}   |   Start: {start.toString('HH:mm')}   |   End: {end.toString('HH:mm')}"
            self.shift_list.addItem(entry)

            # Convert QTime → minutes for comparison
            new_start = start.hour() * 60 + start.minute()
            new_end = end.hour() * 60 + end.minute()

            # Safety check (UI-only)
            if new_start >= new_end:
                QMessageBox.warning(
                    self,
                    "Invalid Time Range",
                    "End time must be after start time."
                )
                return

            # -------- OVERLAP CHECK --------
            for i in range(self.shift_list.count()):
                text = self.shift_list.item(i).text()

                # Extract times from text: "Start: HH:mm | End: HH:mm"
                try:
                    start_str = text.split("Start:")[1].split("|")[0].strip()
                    end_str = text.split("End:")[1].strip()

                    h1, m1 = map(int, start_str.split(":"))
                    h2, m2 = map(int, end_str.split(":"))

                    exist_start = h1 * 60 + m1
                    exist_end = h2 * 60 + m2
                except Exception:
                    continue  # skip malformed entries safely

                # Overlap condition
                if new_start < exist_end and new_end > exist_start:
                    QMessageBox.warning(
                        self,
                        "Shift Overlap",
                        "This shift timing overlaps with an existing shift."
                    )
                    return  # ❌ BLOCK CREATION

            # -------- ADD IF VALID --------
            entry = f"Start: {start.toString('HH:mm')}   |   End: {end.toString('HH:mm')}"
            self.shift_list.addItem(entry)

        btn_create.clicked.connect(create_shift)

        # # ================= DELETE ACTION =================
        # def delete_shift():
        #     row = self.shift_list.currentRow()
        #     if row < 0:
        #         return

        #     shift_id_item = self.shift_table.item(row, 0)
        #     if not shift_id_item:
        #         return

        #     shift_id = int(shift_id_item.text())
        #     delete_shift_by_id(self.conn, shift_id)
        #     refresh_shift_table()


    # ==================================================
    # Machine UI 
    # ==================================================
    def _build_shift_timings(self):
        w = QWidget()
        main = QVBoxLayout(w)
        main.setSpacing(12)

        # ================= HEADER =================
        header = QHBoxLayout()

        title = QLabel("Shift Timings")
        title.setStyleSheet("font-size:18px; font-weight:600;")

        btn_create = QPushButton("Create")
        btn_delete = QPushButton("Delete")

        for b in (btn_create, btn_delete):
            b.setFixedHeight(32)
            b.setMinimumWidth(90)

        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(btn_create)
        header.addWidget(btn_delete)

        main.addLayout(header)

        # ================= SHIFT TABLE (DB) =================
        self.shift_table = QTableWidget()
        self.shift_table.setColumnCount(4)
        self.shift_table.setHorizontalHeaderLabels(["ID", "Shift", "Start", "End"])
        self.shift_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.shift_table.setSelectionMode(QTableWidget.SingleSelection)
        self.shift_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.shift_table.verticalHeader().setVisible(False)
        self.shift_table.horizontalHeader().setStretchLastSection(True)

        main.addWidget(self.shift_table, 1)

        def refresh_shift_table():
            rows = fetch_shifts(self.conn)
            self.shift_table.setRowCount(0)

            for shift_id, shift, start_s, end_s in rows:
                r = self.shift_table.rowCount()
                self.shift_table.insertRow(r)
                self.shift_table.setItem(r, 0, QTableWidgetItem(str(shift_id)))
                self.shift_table.setItem(r, 1, QTableWidgetItem(shift))
                self.shift_table.setItem(r, 2, QTableWidgetItem(start_s))
                self.shift_table.setItem(r, 3, QTableWidgetItem(end_s))

            self.shift_table.setColumnHidden(0, True)  # keep ID for delete

        refresh_shift_table()

        # ================= CREATE ACTION =================
        def create_shift():
            dlg = ShiftCreateDialog(self.conn, self)
            if dlg.exec_() != QDialog.Accepted:
                return
            refresh_shift_table()

        btn_create.clicked.connect(create_shift)

        # ================= DELETE ACTION =================
        def delete_shift():
            row = self.shift_table.currentRow()
            if row < 0:
                QMessageBox.information(self, "Select", "Select one shift row to delete.")
                return

            shift_id_item = self.shift_table.item(row, 0)  # hidden column
            if not shift_id_item:
                return

            shift_id = int(shift_id_item.text())
            delete_shift_by_id(self.conn, shift_id)
            refresh_shift_table()

        btn_delete.clicked.connect(delete_shift)

        return w

    def _build_machine(self):
        w = QWidget()
        main = QVBoxLayout(w)
        main.setSpacing(12)

        header = QHBoxLayout()
        title = QLabel("Machine")
        title.setStyleSheet("font-size:18px; font-weight:600;")

        btn_add = QPushButton("Add")
        btn_delete = QPushButton("Delete")

        for b in (btn_add, btn_delete):
            b.setFixedHeight(32)
            b.setMinimumWidth(90)

        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(btn_add)
        header.addWidget(btn_delete)
        main.addLayout(header)

        self.machine_list = QListWidget()
        self.machine_list.setSelectionMode(QListWidget.SingleSelection)
        self.machine_list.setStyleSheet("""
            QListWidget { border: 1px solid #cfcfcf; border-radius: 6px; }
            QListWidget::item { padding: 10px; }
            QListWidget::item:selected { background: #dbe9ff; }
        """)
        main.addWidget(self.machine_list, 1)

        # ---------- LOAD from DB on open ----------
        saved = get_saved_machine(self.conn)
        if saved:
            self.machine_list.addItem(f"Machine Number: {saved}")
            if hasattr(self, "txt_job_machine"):
                self.txt_job_machine.setText(saved)

        # ---------- ADD ----------
        def add_machine():
            if self.machine_list.count() > 0:
                QMessageBox.information(self, "Info", "Only one machine allowed.")
                return

            dlg = MachineAddDialog(self)
            if dlg.exec_() == QDialog.Accepted:
                machine_no = dlg.get_machine_number()
                if machine_no:
                    try:
                        save_machine(self.conn, machine_no)   # ✅ save permanently
                    except Exception as e:
                        QMessageBox.critical(self, "DB Error", str(e))
                        return

                    self.machine_list.clear()
                    self.machine_list.addItem(f"Machine Number: {machine_no}")

                    if hasattr(self, "txt_job_machine"):
                        self.txt_job_machine.setText(machine_no)

        btn_add.clicked.connect(add_machine)

        # ---------- DELETE ----------
        def delete_machine_ui():
            row = self.machine_list.currentRow()
            if row < 0:
                return

            try:
                delete_machine(self.conn)   # ✅ delete from DB
            except Exception as e:
                QMessageBox.critical(self, "DB Error", str(e))
                return

            self.machine_list.clear()
            if hasattr(self, "txt_job_machine"):
                self.txt_job_machine.clear()

        btn_delete.clicked.connect(delete_machine_ui)

        return w

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SettingsPopup()
    w.show()
    sys.exit(app.exec_())
    