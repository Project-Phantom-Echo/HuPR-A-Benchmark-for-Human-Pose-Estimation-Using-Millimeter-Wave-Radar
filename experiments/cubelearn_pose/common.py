"""Paths and provenance for the isolated CubeLearn pose experiment."""
from pathlib import Path
import hashlib
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'local/cubelearn-pose-deps'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_hashes(cubelearn):
    paths = [*Path(__file__).parent.glob('*.py'), ROOT/'config/mscsa_prgcn.yaml',
             ROOT/'misc/utils.py',ROOT/'misc/coco.py',ROOT/'misc/cocoeval.py',
             Path(cubelearn)/'network.py',Path(cubelearn)/'network_har.py']
    return {str(p.resolve()):digest(p) for p in paths}

def check_evaluator():
    import pycocotools
    for name in ('coco.py','cocoeval.py'):
        assert digest(Path(pycocotools.__file__).parent/name)==digest(ROOT/'misc'/name), 'Use HuPR author evaluator'
