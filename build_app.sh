#!/bin/bash

set -e

# ==========================================
# TEXA - CHROMO GPU BUILD SCRIPT
# ==========================================

PROJECT_DIR="/home/texa_innovates/chromo_gpu"
VENV_DIR="$PROJECT_DIR/venv"
APP_NAME="chromo_gpu"
SPEC_FILE="$PROJECT_DIR/chromo_gpu.spec"

echo "=========================================="
echo " TEXA CHROMO GPU BUILD STARTED"
echo "=========================================="

cd "$PROJECT_DIR"

# ------------------------------------------
# Activate virtual environment
# ------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ venv folder not found: $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "✅ Virtual environment activated"

# ------------------------------------------
# Check pyinstaller
# ------------------------------------------
if ! command -v pyinstaller >/dev/null 2>&1; then
    echo "⚠️ PyInstaller not found in venv. Installing..."
    pip install pyinstaller
fi

# ------------------------------------------
# Clean old build folders
# ------------------------------------------
echo "🧹 Cleaning old build files..."
rm -rf build
rm -rf dist
rm -rf __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} +

echo "✅ Old build files removed"

# ------------------------------------------
# Optional: remove old spec if rebuilding from py
# ------------------------------------------
# rm -f "$SPEC_FILE"

# ------------------------------------------
# Build using spec file if present
# ------------------------------------------
if [ -f "$SPEC_FILE" ]; then
    echo "🚀 Building with spec file: $SPEC_FILE"
    pyinstaller --clean --noconfirm "$SPEC_FILE"
else
    echo "⚠️ Spec file not found. Building directly from app_gbu.py"
    pyinstaller --clean --noconfirm \
        --name "$APP_NAME" \
        --onedir \
        --windowed \
        app_gbu.py
fi

# ------------------------------------------
# Check output
# ------------------------------------------
if [ -d "$PROJECT_DIR/dist/$APP_NAME" ]; then
    echo "✅ Build completed successfully"
    echo "📁 Output folder: $PROJECT_DIR/dist/$APP_NAME"
else
    echo "❌ Build failed: output folder not found"
    exit 1
fi

# ------------------------------------------
# Make launcher executable if exists
# ------------------------------------------
if [ -f "$PROJECT_DIR/dist/$APP_NAME/$APP_NAME" ]; then
    chmod +x "$PROJECT_DIR/dist/$APP_NAME/$APP_NAME"
    echo "✅ Executable permission added"
fi

echo "=========================================="
echo " BUILD FINISHED SUCCESSFULLY"
echo "=========================================="
