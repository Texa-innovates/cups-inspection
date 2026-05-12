import sqlite3
from PyQt5.QtCore import Qt, QTime
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QGridLayout, QLabel, QTimeEdit, QPushButton, QMessageBox

def create_database_and_tables():
    
    conn = sqlite3.connect("production.db")
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
            created_at TEXT DEFAULT (datetime('now')
                   
        )
    """)

    # ---------------- TABLE 2 : start_timing_table ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS start_timing_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            start_time TEXT,
            end_time TEXT,
            created_at TEXT DEFAULT (datetime('now')
        )
    """)

    # ---------------- TABLE 3 : shift_table ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shift_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shift TEXT,
            shift_start_time TEXT,
            shift_end_time TEXT,
            created_at TEXT DEFAULT (datetime('now')
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
            created_at TEXT DEFAULT (datetime('now')
        )
    """)

    # ---------------- TABLE 5 : machinenumber_table ----------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS machinenumber_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_number TEXT UNIQUE,
            created_at TEXT DEFAULT (datetime('now')
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
    conn = sqlite3.connect("production.db")
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO defect_table (count, camara_angle, img_path, defect_type)
        VALUES (?, ?, ?, ?)
    """, (int(cup_count), str(camara_angle), str(img_path), str(defect_type)))
    conn.commit()