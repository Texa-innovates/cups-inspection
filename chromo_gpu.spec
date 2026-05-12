# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs
)

# ============================================================
# FORCE-BUNDLE: PyQt5 + QtWebEngine + sklearn + scipy + opencv
# ============================================================

# --------------------------
# Hidden imports (forced)
# --------------------------
hiddenimports = []

# PyQt (UI + WebEngine)
hiddenimports += collect_submodules("PyQt5")
hiddenimports += collect_submodules("PyQt5.QtWebEngineWidgets")
hiddenimports += collect_submodules("PyQt5.QtWebEngineCore")
hiddenimports += collect_submodules("PyQt5.QtWebChannel")
hiddenimports += collect_submodules("PyQt5.QtWebEngine")

# OpenCV + common deps
hiddenimports += ["cv2", "numpy", "joblib"]

# sklearn + scipy (force core modules)
hiddenimports += [
    "sklearn",
    "sklearn.neighbors",
    "sklearn.metrics",
    "sklearn.utils",
    "sklearn.preprocessing",
    "sklearn.pipeline",
    "sklearn.base",
    "sklearn.linear_model",
    "sklearn.svm",
    "sklearn.tree",
    "sklearn.ensemble",
    "scipy",
    "scipy.sparse",
    "scipy.linalg",
    "scipy.special",
    "threadpoolctl",
]

# --------------------------
# Excludes (avoid optional deps like dask)
# --------------------------
excludes = [
    "dask",
    "distributed",
]

# --------------------------
# Binaries (native .so libs)
# --------------------------
binaries = []
binaries += collect_dynamic_libs("cv2")
binaries += collect_dynamic_libs("numpy")
binaries += collect_dynamic_libs("sklearn")
binaries += collect_dynamic_libs("scipy")

# --------------------------
# Data files (package data)
# --------------------------
datas = []
datas += collect_data_files("cv2")
datas += collect_data_files("sklearn")
datas += collect_data_files("scipy")

# Your project runtime data
datas += [
    ("assets", "assets"),
    ("config", "config"),
    ("production.db", "."),
    ("predition", "predition"),
    ("bad_images", "bad_images"),
]

block_cipher = None


a = Analysis(
    ["app_gbu.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="chromo_gpu",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,   # keep terminal logs (set False if you want no console)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="chromo_gpu",
)

