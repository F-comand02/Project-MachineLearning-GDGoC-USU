"""Streamlit entrypoint.

Deployment log sebelumnya menjalankan file ini sebagai *main module*.
Agar tidak terjadi redirect loop/slow loading, file ini langsung
menjalankan UI dari `app.py`.

Catatan: kita tidak meng-copy UI; cukup import dan eksekusi.
"""

import importlib.util
from pathlib import Path

# Cari UI file (app.py) di dalam folder notebooks
ROOT_DIR = Path(__file__).resolve().parent.parent
APP_PY = Path(__file__).resolve().parent / "app.py"


spec = importlib.util.spec_from_file_location("student_performance_app", APP_PY)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot create import spec for app.py at: {APP_PY}")

if not APP_PY.exists():
    raise FileNotFoundError(f"UI file not found: {APP_PY}")


module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


