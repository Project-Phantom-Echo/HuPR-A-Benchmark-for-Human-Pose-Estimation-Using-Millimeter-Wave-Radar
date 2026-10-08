# HuPR reproduction

Updated 2026-10-08. This reproduction is complete for our reporting scope: one 10-epoch, seed-0 run with the released CSAM + PRGCN model, evaluated once on the test set using the best-validation checkpoint. We are stopping here rather than extending to 200 epochs. This is a close short-run reproduction, not a multi-seed or full-duration reproduction.

## Results

| Source | Test AP | AP50 | AP75 |
| --- | ---: | ---: | ---: |
| Our 10-epoch run | 62.6 | 96.7 | 73.4 |
| Published paper, CSAM + PRGCN | 63.4 | 97.0 | 74.0 |

Paper values were verified against Table 3 of the [official WACV 2023 paper](https://openaccess.thecvf.com/content/WACV2023/papers/Lee_HuPR_A_Benchmark_for_Human_Pose_Estimation_Using_Millimeter_Wave_WACV_2023_paper.pdf). The paper does not specify an epoch count; 200 is the released configuration's default.

Best validation AP was 71.0 at epoch 9 (stored as zero-based epoch 8). Test AP is 0.8 points below the paper. Training job 307146 finished in 6:41:07; test job 307147 finished in 4:39. Both completed successfully. The latest checkpoint is after epoch 10; the best checkpoint is from epoch 9.

![Training loss and validation AP](reproduction/evidence/training_curves.png)

[Epoch metrics](reproduction/evidence/epoch_metrics.csv), [training log](reproduction/evidence/train.log), [test log](reproduction/evidence/test.log).

## Protocol and changes

- Released train/validation/test sequence lists, sampling ratio 1, eight input frames, 14 keypoints, 64×64 heatmaps and 256×256 image coordinates. Predictions are scaled to image coordinates for evaluation as in the released code.
- Adam, batch size 20, weight decay 0.0001, initial LR 0.0001, no warmup. The unchanged scheduler multiplies LR by 0.999 at batch indices 0, 2000, 4000, etc. within each epoch. Ten epochs end at approximately 0.0000970; the schedule was not compressed.
- Model, loss, evaluator sources, optimizer settings and checkpoint selection remain the released implementation. The author's modified COCO evaluator must be installed; the fast launcher verifies its hashes.
- The fast pipeline reads pre-normalized float32 tensors, preserving the released normalization and frame selection. It uses 16 workers, pinned memory, prefetching and nonblocking transfers. `main.py` retains the original loader when `HUPR_NORMALIZED_CACHE` is unset.
- `resume.pth` adds model, optimizer, best validation AP and Python/NumPy/Torch/CUDA RNG state at epoch boundaries. CPU Adam save/reload equivalence was tested; bit-identical resumed GPU training is not claimed.
- No reduced-elevation-cache or rewritten-loss optimization from the later studies is enabled.

The original versus cached short benchmark measured 16.24 versus 73.65 samples/sec (4.53×). This is not an end-to-end run speed comparison. Later warm-cache profiling measured about 114 ms backward, 72 ms forward, 19 ms data wait, 15 ms loss and 12 ms transfer per batch. No individual-layer bottleneck was established. Experimental further optimizations reached ~106 versus ~86 samples/sec in a one-sequence benchmark without optimizer steps, but gradient differences remained unresolved; those experiments are archived only.

## Use the fast pipeline

Python 3.10, PyTorch 2.1.0 and torchvision 0.16.0 were used on an H100. [Requirements](reproduction/requirements.txt) and the [recorded environment](reproduction/evidence/environment.txt) are included. In a dedicated environment:

```bash
python -m pip install -r reproduction/requirements.txt
python reproduction/install_evaluator.py
```

The evaluator installer backs up and replaces `coco.py` and `cocoeval.py` in that Python environment, as required by the authors. Do not use a shared environment for another project's evaluator.

For an existing normalized cache, supply a data directory containing `hrnet_annot_train.json`, `hrnet_annot_val.json`, and `hrnet_annot_test.json`, and a cache directory containing `single_N/{hori.npy,vert.npy,complete.json}` for all configured sequences. Each cached view has shape `(600,8,2,64,64,8)` and dtype float32. Ground-truth JSON files are generated in the data directory, which must be writable by the launching account.

```bash
python run_fast.py --data /path/to/annotations --cache /path/to/cache --check
python run_fast.py --data /path/to/annotations --cache /path/to/cache --run new-seed0 --epochs 10
python run_fast.py --data /path/to/annotations --cache /path/to/cache --run new-seed0 --eval
```

A new run refuses to overwrite existing results. `--resume --epochs N` continues to a total of N epochs using the latest resumable checkpoint. Evaluation loads the best-validation model. The fast launcher requires at least one worker. `--seed` defaults to 0.

To regenerate cached inputs from the released preprocessed radar maps, run one sequence at a time (or independent jobs, without a concurrency cap):

```bash
python reproduction/build_cache.py --sequence 2 --data /path/to/preprocessed/HuPR --cache /path/to/new-cache
python reproduction/check_cached_loader.py --data /path/to/preprocessed/HuPR --cache /path/to/cache
```

The builder preserves normalization arithmetic and checks exact equality against the released transform at frames 0, 300 and 599 of each view. It refuses an existing sequence directory. Raw ADC conversion remains under the upstream `preprocessing/` instructions; building the normalized cache is a separate step after that preprocessing.

## Existing cluster assets and commands

Only `hupr/` remains as an active top-level HuPR folder. Historical campaigns, checkpoints, environments, preprocessing and data are under `../old/hupr-experiments/`. The joint HuPR/RF-CRATE attempt was moved intact, retaining its RF-CRATE assets too. Nothing was deleted. Historical scripts contain their old absolute paths and are provenance records; use this fork's launcher now.

Ignored local links keep the current setup usable:

- `local/data`: pilot annotations and generated ground truth.
- `local/normalized-cache`: existing normalized cache.
- `local/env-hupr`: preserved Python environment. Invoke its interpreter directly; historical activation scripts contain old prefixes.
- `logs`: original pilot checkpoints and predictions, including `release-seed0/model_best.pth` and `resume.pth`.

Check everything without starting training:

```bash
/mnt/weka/fgeikyan/rf-perception-papers/hupr/local/env-hupr/bin/python /mnt/weka/fgeikyan/rf-perception-papers/hupr/run_fast.py --check
```

Optional fresh 10-epoch job from any current directory, under the account invoking `sbatch`:

```bash
HUPR_ROOT=/mnt/weka/fgeikyan/rf-perception-papers/hupr sbatch --export=ALL /mnt/weka/fgeikyan/rf-perception-papers/hupr/reproduction/run.sbatch --run new-seed0 --epochs 10
```

Evaluate the saved pilot checkpoint with the same job script and `--run release-seed0 --eval`, or continue with `--run release-seed0 --resume --epochs 20` (adjust job time for the intended work). No new GPU jobs were submitted during consolidation.

Large data, environments, predictions and checkpoints are intentionally local, not committed to Git. The fork contains the fast source, portable launcher/cache builder, setup instructions, compact results and verification evidence.
