from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "基础版_step51_precise_inbound_fix.py"
MODULE_NAME = "tg_business_ai_step51_wsgi"

spec = importlib.util.spec_from_file_location(MODULE_NAME, SOURCE_FILE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load source module from {SOURCE_FILE}")

module = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = module
spec.loader.exec_module(module)

app = module.create_production_app()
