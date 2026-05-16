import sqlite3
from PyQt5.QtCore import Qt, QTime
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QGridLayout, QLabel, QTimeEdit, QPushButton, QMessageBox
from datetime import datetime
from path import DB_PATH

def create_database_and_tables():
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ---------------- TABLE 1 : jobid_table ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobid_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            jobid_name TEXT UNIQUE,
            machine_no TEXT,
            material TEXT,
            module TEXT,
            status TEXT,
            created_at TEXT DEFAULT (datetime('now'))
            )
    """)

    # ---------------- TABLE 2 : start_timing_table ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS start_timing_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            start_time TEXT,
            end_time TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # ---------------- TABLE 3 : shift_table ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shift_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shift TEXT,
            shift_start_time TEXT,
            shift_end_time TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # ---------------- TABLE 4 : defect_table ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS defect_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            count INTEGER,
            camara_angle TEXT,
            img_path TEXT,
            defect_type TEXT
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # ---------------- TABLE 5 : cup_entry ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cup_entry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cup_count INTEGER,
            shift_id INTEGER DEFAULT 0,
            shift_count INTEGER DEFAULT 0,
            job_id INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)


    # ---------------- TABLE 6 : machinenumber_table ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS machinenumber_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_number TEXT UNIQUE,
            created_at TEXT DEFAULT (datetime('now'))
            )""")

    conn.commit()
    conn.close()

    print("✅ SQLite database and all 5 tables created successfully")


def insert_shift(conn,shift: str,start_qtime: QTime,end_qtime: QTime):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO shift_table
        (shift, shift_start_time, shift_end_time)
        VALUES (?, ?, ?)
    """, (
        shift,
        start_qtime.toString("HH:mm"),
        end_qtime.toString("HH:mm")
    ))
    conn.commit()

def fetch_shifts(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, shift, shift_start_time, shift_end_time FROM shift_table ORDER BY id ASC")
    return cur.fetchall()

def delete_shift_by_id(conn, shift_id: int):
    cur = conn.cursor()
    cur.execute("DELETE FROM shift_table WHERE id = ?", (shift_id,))
    conn.commit()


def get_saved_machine(conn):
    cur = conn.cursor()
    cur.execute("SELECT machine_number FROM machinenumber_table ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    return row[0] if row else None

def save_machine(conn, machine_no: str):
    cur = conn.cursor()
    # keep ONLY one machine (your UI rule)
    cur.execute("DELETE FROM machinenumber_table")
    cur.execute("INSERT INTO machinenumber_table(machine_number) VALUES(?)", (machine_no,))
    conn.commit()

def delete_machine(conn):
    cur = conn.cursor()
    cur.execute("DELETE FROM machinenumber_table")
    conn.commit()


def insert_jobid(conn, jobid_name: str, machine_no: str, material: str, module: str, status: str = "DEACTIVE"):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO jobid_table (jobid_name, machine_no, material, module, status)
        VALUES (?, ?, ?, ?, ?)
    """, (jobid_name, machine_no, material, module, status))
    conn.commit()

def fetch_jobids(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT id, jobid_name, machine_no, material, module, status
        FROM jobid_table
        ORDER BY id DESC
    """)
    return cur.fetchall()

def set_active_jobid(conn, active_jobid_name: str):
    """
    Make ONLY this jobid ACTIVE and all others DEACTIVE.
    """
    cur = conn.cursor()

    # Deactivate all rows
    cur.execute("UPDATE jobid_table SET status='DEACTIVE'")

    # Activate the selected jobid
    cur.execute(
        "UPDATE jobid_table SET status='ACTIVE' WHERE jobid_name=?",
        (active_jobid_name,)
    )

    conn.commit()
    

def insert_defect(cup_count: int, camara_angle: str, img_path: str, defect_type: str = "BAD") -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO defect_table (count, camara_angle, img_path, defect_type)
        VALUES (?, ?, ?, ?)
    """, (int(cup_count), str(camara_angle), str(img_path), str(defect_type)))
    conn.commit()
    conn.close()

def _parse_time(tstr: str):
    tstr = (tstr or "").strip()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(tstr, fmt).time()
        except Exception:
            pass
    return None

def get_current_shift_id() -> int:
    now_t = datetime.now().time()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, shift_start_time, shift_end_time FROM shift_table")
    rows = cur.fetchall()
    conn.close()

    for sid, st_s, en_s in rows:
        st = _parse_time(st_s)
        en = _parse_time(en_s)
        if not st or not en:
            continue

        # normal shift
        if st <= en:
            if st <= now_t < en:
                return int(sid)
        else:
            # night shift (start > end)
            if now_t >= st or now_t < en:
                return int(sid)

    return 0

def get_total_shift_count() -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM shift_table")
    n = cur.fetchone()[0]
    conn.close()
    return int(n)

def get_active_job_pk_id() -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id
        FROM jobid_table
        WHERE status = 'ACTIVE'
        ORDER BY id DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else 0

def upsert_cup_entry(cup_count):
    shift_id = get_current_shift_id()
    shift_count = get_total_shift_count()
    job_id = get_active_job_pk_id()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Find last row for same context (job_id + shift_id + shift_count)
    cur.execute("""
        SELECT id
        FROM cup_entry
        WHERE job_id = ? AND shift_id = ? AND shift_count = ?
        ORDER BY id DESC
        LIMIT 1
    """, (job_id, shift_id, shift_count))
    row = cur.fetchone()

    if row:
        cup_entry_id = int(row[0])
        # Update same row continuously
        cur.execute("""
            UPDATE cup_entry
            SET cup_count = ?, created_at = ?
            WHERE id = ?
        """, (str(cup_count), created_at, cup_entry_id))
    else:
        # Insert new row if context changed
        cur.execute("""
            INSERT INTO cup_entry (cup_count, shift_id, shift_count, job_id, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (str(cup_count), shift_id, shift_count, job_id, created_at))

    conn.commit()
    conn.close()
