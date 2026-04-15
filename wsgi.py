import importlib.util
from pathlib import Path

module_path = Path(__file__).with_name('基础版_step11.py')
spec = importlib.util.spec_from_file_location('tg_business_ai_step11', module_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)
app = module.create_production_app()
