"""Install the author's required evaluator into the selected Python environment."""
from pathlib import Path
import shutil
import pycocotools
root = Path(__file__).resolve().parents[1]
for name in ('coco.py', 'cocoeval.py'):
    dest = Path(pycocotools.__file__).parent/name
    backup = dest.with_suffix('.py.before_hupr')
    if not backup.exists(): shutil.copy2(dest, backup)
    shutil.copy2(root/'misc'/name, dest)
    print(dest)
