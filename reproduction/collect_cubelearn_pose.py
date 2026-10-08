"""Collect completed pose comparisons and verify their checkpoint provenance."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    selection = json.loads((args.campaign/'continuation-selection.json').read_text())
    rows = []
    for name in ['dft', selection['chosen']]:
        run = args.campaign/name
        assert json.loads((run/'complete.json').read_text())['epochs'] == 10
        history = json.loads((run/'history.json').read_text())
        assert [row['epoch'] for row in history] == list(range(1, 11))
        expected = max(history, key=lambda row: (row['validation']['AP'], -row['validation']['loss']))
        test = json.loads((run/'test.json').read_text())
        best = torch.load(run/'best.pth', map_location='cpu')
        latest = torch.load(run/'latest.pth', map_location='cpu')
        assert test['epoch'] == best['epoch'] == expected['epoch']
        assert latest['epoch'] == 10 and len(latest['history']) == 10
        assert all(torch.isfinite(value).all() for value in latest['model'].values())
        assert all(torch.isfinite(value).all() for value in best['model'].values())
        manifest = json.loads((run/'manifest.json').read_text())
        for path, expected_hash in manifest['source_sha256'].items():
            assert digest(Path(path)) == expected_hash, path
        predictions = json.loads((run/'test_predictions.json').read_text())
        gt = json.loads((Path(manifest['arguments']['gt_root'])/'test_gt.json').read_text())
        ids = [row['image_id'] for row in predictions]
        assert len(ids) == len(set(ids)) == test['samples'] == 12600
        assert set(ids) == {row['id'] for row in gt['images']}
        row = dict(arm=name, test=test, selected_validation=expected['validation'],
                   completed_epochs=10, selected_epoch=best['epoch'],
                   completed_epoch_training_seconds=sum(record['train_seconds'] for record in history),
                   completed_epoch_validation_seconds=sum(record['val_seconds'] for record in history),
                   best_checkpoint_sha256=digest(run/'best.pth'),
                   latest_checkpoint_sha256=digest(run/'latest.pth'),
                   test_predictions_sha256=digest(run/'test_predictions.json'),
                   source_hashes_verified=True, test_image_ids_verified=True)
        rows.append(row)
        dest = args.output/name
        dest.mkdir(parents=True, exist_ok=True)
        for filename in ['manifest.json', 'history.json', 'test.json', 'complete.json']:
            shutil.copy2(run/filename, dest/filename)
        (dest/'result.json').write_text(json.dumps(row, indent=2)+'\n')
        print(name, 'test AP', 100*test['AP'], 'selected epoch', best['epoch'])
    for filename in ['continuation-selection.json', 'continuation-submission.json',
                     'annotation-equivalence.json', 'ground-truth-generation-check.json',
                     'environment.json', 'memory-pressure.json']:
        shutil.copy2(args.campaign/filename, args.output/filename)
    for filename in ['pilot-submission.json', 'memory-reallocation.json',
                     'learned-memory-reallocation.json', 'scheduler-adjustments.json',
                     'gpu-utilization-snapshot.json', 'launcher-memory-default.json']:
        shutil.copy2(args.campaign/filename, args.output/filename)
    pilot = args.output/'cubelearn-1e4'
    pilot.mkdir(exist_ok=True)
    for filename in ['manifest.json', 'history.json', 'complete.json']:
        shutil.copy2(args.campaign/'cubelearn-1e4'/filename, pilot/filename)
    for name in ['benchmark-dft', 'benchmark-learned', 'dft-benchmark-gpu07']:
        dest = args.output/name
        dest.mkdir(exist_ok=True)
        for filename in ['manifest.json', 'benchmark.json']:
            shutil.copy2(args.campaign/name/filename, dest/filename)
    (args.output/'results.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
