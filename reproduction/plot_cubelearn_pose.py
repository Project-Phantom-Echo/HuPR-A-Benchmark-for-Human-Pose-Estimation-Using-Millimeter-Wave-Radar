"""Plot CubeLearn pose histories without changing any experiment source or state."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for run in args.run:
        history = json.loads((run/'history.json').read_text())
        epochs = [row['epoch'] for row in history]
        axes[0].plot(epochs, [row['train_loss'] for row in history], 'o-', label=run.name)
        axes[1].plot(epochs, [row['validation']['loss'] for row in history], 'o-', label=run.name)
        axes[2].plot(epochs, [100*row['validation']['AP'] for row in history], 'o-', label=run.name)
    for ax, title in zip(axes, ['Training BCE', 'Validation BCE', 'Validation AP (%)']):
        ax.set_title(title)
        ax.set_xlabel('Completed epoch (1-based)')
        ax.grid(alpha=.2)
    axes[2].legend(fontsize=8)
    fig.suptitle('CubeLearn adapted to HuPR: fixed versus learned Fourier layers')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)


if __name__ == '__main__':
    main()
