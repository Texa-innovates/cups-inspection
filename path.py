import os
import sys

def app_path(*parts):
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base, *parts)


def run_path(*parts):

    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))

    run_dir = os.path.join(base, "Cup_Data")
    os.makedirs(run_dir, exist_ok=True)

    final_path = os.path.join(run_dir, *parts)

    folder = (
        final_path
        if os.path.splitext(final_path)[1] == ""
        else os.path.dirname(final_path)
    )

    os.makedirs(folder, exist_ok=True)

    return final_path

JOBID_JSON_FILE = run_path("job_setup.json")

MODEL_PATH = run_path("model", "oneclass_heritage_new.joblib")

BAD_IMG_SAVE = run_path("bad_images")

DB_PATH=run_path("production.db")