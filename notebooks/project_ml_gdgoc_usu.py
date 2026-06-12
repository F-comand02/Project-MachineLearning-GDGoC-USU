"""Streamlit entrypoint.

Deployment log sebelumnya menjalankan file ini sebagai *main module*.
Agar tidak terjadi redirect loop/slow loading, file ini langsung
menjalankan UI dari `app.py`.

Catatan: kita tidak meng-copy UI; cukup import dan eksekusi.
"""

import importlib.util
from pathlib import Path

# Cari app.py di project root (satu level di atas folder notebooks)
ROOT_DIR = Path(__file__).resolve().parent.parent
APP_PY = ROOT_DIR / "app.py"

spec = importlib.util.spec_from_file_location("student_performance_app", APP_PY)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load app.py at: {APP_PY}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


