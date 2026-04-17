from __future__ import annotations

import importlib.util
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / '基础版_step51_precise_inbound_fix.py'
MODULE_NAME = 'tg_business_ai_step51'

spec = importlib.util.spec_from_file_location(MODULE_NAME, SOURCE_FILE)
if spec is None or spec.loader is None:
    raise RuntimeError(f'Failed to load source module from {SOURCE_FILE}')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# When imported by a WSGI server, expose the Flask app.
# When executed directly, avoid building the app twice.
if __name__ != '__main__':
    app = module.create_production_app()


def main() -> None:
    module.main()


if __name__ == '__main__':
    main()
