from pathlib import Path
import importlib.util
import sys

BASE_DIR = Path(__file__).resolve().parent
TARGET = BASE_DIR / "基础版_step11.py"

if not TARGET.exists():
    raise FileNotFoundError(f"找不到主代码文件: {TARGET}")

spec = importlib.util.spec_from_file_location("app_main_module", TARGET)
if spec is None or spec.loader is None:
    raise ImportError(f"无法为目标文件创建模块规范: {TARGET}")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

app = module.create_production_app()
