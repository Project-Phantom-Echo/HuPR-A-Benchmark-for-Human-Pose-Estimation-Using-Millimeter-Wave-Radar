"""Generate author-format COCO ground truth from released HuPR annotations."""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.base import generateGTAnnot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['train', 'val', 'test'], action='append')
    args = parser.parse_args()
    phases = args.phase or ['val', 'test']
    sources = {phase:(args.annotations/f'hrnet_annot_{phase}.json').resolve()
               for phase in phases}
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    args.output.mkdir(parents=True, exist_ok=False)
    data_config = yaml.safe_load((ROOT/'config/mscsa_prgcn.yaml').read_text())['DATASET']
    cfg = SimpleNamespace(DATASET=SimpleNamespace(**{**data_config, 'dataDir':str(args.output)}))
    for phase, source in sources.items():
        (args.output/source.name).symlink_to(source)
        with contextlib.redirect_stdout(io.StringIO()):
            generateGTAnnot(cfg, phase)
        target = args.output/f'{phase}_gt.json'
        count = len(json.loads(target.read_text())['images'])
        print(target, count, 'frames')


if __name__ == '__main__':
    main()
