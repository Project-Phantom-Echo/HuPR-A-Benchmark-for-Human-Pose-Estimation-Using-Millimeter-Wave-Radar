"""Run the verified cached HuPR pipeline from any working directory."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'local/data')
    p.add_argument('--cache', type=Path, default=ROOT/'local/normalized-cache')
    p.add_argument('--epochs', type=int, default=10, help='Total epochs including any resumed epochs')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run', default='release-seed0', help='Directory name under logs/')
    p.add_argument('--eval', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check', action='store_true', help='Validate assets/configuration without training or evaluation')
    a = p.parse_args()
    if a.epochs < 1 or a.workers < 1 or Path(a.run).name != a.run or a.run in ('.','..'):
        p.error('Use positive epochs/workers and a simple run name')
    a.data, a.cache = a.data.resolve(), a.cache.resolve()
    import json
    import yaml
    cfg = yaml.safe_load((ROOT/'config/mscsa_prgcn.yaml').read_text())
    cfg['DATASET']['dataDir'] = str(a.data)
    cfg['TRAINING']['epochs'] = a.epochs
    cfg['SETUP']['numWorkers'] = a.workers
    for phase in ('train','val','test'):
        if not (a.data/f'hrnet_annot_{phase}.json').is_file():
            raise FileNotFoundError(f'Missing {phase} annotations in {a.data}')
        for seq in cfg['DATASET'][phase+'Name']:
            cache = a.cache/f'single_{seq}'
            receipt = json.loads((cache/'complete.json').read_text())
            if receipt['frames'] != cfg['DATASET']['duration']:
                raise ValueError(f'Wrong frame count in {cache}')
            for view in ('hori','vert'):
                if not (cache/f'{view}.npy').is_file(): raise FileNotFoundError(cache/view)
    os.environ['HUPR_NORMALIZED_CACHE'] = str(a.cache)
    import hashlib, pycocotools
    for name in ("coco.py", "cocoeval.py"):
        installed = Path(pycocotools.__file__).parent/name
        if hashlib.sha256(installed.read_bytes()).digest() != hashlib.sha256((ROOT/"misc"/name).read_bytes()).digest():
            raise RuntimeError("Install the author evaluator: python reproduction/install_evaluator.py")
    from main import obj
    from tools import Runner
    from tools.base import BaseRunner
    from resume_support import install
    if a.check:
        print('Configuration, imports, annotations and all cache receipts OK; no training started.')
        return
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('A GPU allocation is required')
    run = ROOT/'logs'/a.run
    if a.eval and not (run/'model_best.pth').is_file(): raise FileNotFoundError(run/'model_best.pth')
    if a.resume and not (run/'resume.pth').is_file(): raise FileNotFoundError(run/'resume.pth')
    if not a.eval and not a.resume and run.exists() and any(run.iterdir()):
        raise RuntimeError('Run already exists; use --resume or a new --run name')
    os.chdir(ROOT)
    (ROOT/'logs').mkdir(exist_ok=True)
    (ROOT/'visualization').mkdir(exist_ok=True)
    install(BaseRunner)
    args = argparse.Namespace(seed=a.seed,dir=a.run,visDir='none',gpuIDs=[0],eval=a.eval,sampling_ratio=1,keypoints=False)
    runner = Runner(args, obj(cfg))
    if a.eval:
        runner.loadModelWeight('model_best')
        runner.eval(visualization=False)
    else:
        if a.resume: runner.loadModelWeight('checkpoint')
        runner.train()

if __name__ == '__main__':
    main()
