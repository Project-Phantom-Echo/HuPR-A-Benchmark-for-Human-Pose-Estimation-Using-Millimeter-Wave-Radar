# CubeLearn on HuPR: paired pose experiment

Prepared 2026-10-08. This is a new cross-task adaptation, not a reproduction of a published CubeLearn pose result. Results and measured runtimes are recorded in [the project notes](../../notes.md).

## Question and primary comparison

Does learning CubeLearn's complex Fourier layers improve HuPR pose estimation compared with fixing the same layers at DFT initialization? Both arms use exactly the same two-branch architecture, initialization for each seed, batches, optimizer for the real network, pose targets and evaluation. The earlier HuPR CSAM+PRGCN run (62.6 test AP after ten epochs; paper 63.4) is a contextual reference, not an isolated preprocessing comparison.

## Architecture

Each of the two radars has a separate copy of the released CubeLearn D-A-T CNN–LSTM, imported from the sibling `cubelearn/` fork (`network.py`, `network_har.py`, source hashes recorded per run). We keep its three complex linear layers, magnitude, range sum, three CNN blocks, 512-unit LSTM and 128-unit hidden layer. The six-class output is removed. Each branch processes eight frames instead of HAR's twenty, to match HuPR's temporal context.

Concatenate the two 128-dimensional features, project to 32×8×8, then apply three bilinear upsampling/convolution stages to produce 14×64×64 sigmoid heatmaps. This decoder is new. There is no CSAM or PRGCN in this adaptation, no HAR checkpoint transfer, and no pretrained pose weights.

**Known limitation:** D-A-T marginalizes range, which could remove useful pose information. The head must reconstruct spatial joint heatmaps from global features. A negative result therefore would not show that learnable radar preprocessing is inherently unsuitable for pose. If the pilot cannot learn useful localization, inspect predictions before considering a range-preserving variant; do not silently relabel such a variant as the original HAR model.

## Inputs and alignment

Read the original HuPR int16 ADC recordings, not its normalized FFT cache. Decode the released LVDS real/imaginary layout; take TX0 RX0–3 followed by TX2 RX0–3 as the eight-element array, preserving the released geometry. Use all 64 chirps per TX and the first 128 ADC samples, matching the successful HAR implementation's dimensions. TX1's extra elevation elements and the second half of ADC samples are not used. Each radar yields `(8 frames, 64 chirps, 8 antennas, 128 samples)` complex64 input.

No extra FFT, clutter removal, normalization, inverse FFT, amplitude scaling or phase randomization is applied before CubeLearn. Thus this is a different input representation from HuPR's published preprocessing. Preprocessing remains identical between the fixed and learned arms.

For target frame t, use t−4 through t+3; repeat boundary frames within the same recording. This exactly matches the released HuPR indexing, including noncausal future context. Supervise the pose at t, not at the last LSTM frame. Keep the original HuPR train/validation/test sequence lists, with no recording crossing a split. Counts: 115,800 training frames, 12,600 validation frames, 12,600 test frames.

Targets use HuPR's original Gaussian generation at 64×64, including its integer joint conversion, sigma and 256-to-64 coordinate mapping. Use a single mean BCE heatmap loss. This differs from HuPR's two-stage summed loss, because our decoder has one output. Decode by the same heatmap argmax, multiply coordinates by four, and evaluate with the author's modified COCO evaluator. Report AP, AP50 and AP75.

## Staged budget

1. CPU checks: exact ADC decoding against the original parser; exact antenna mapping; all 600 temporal windows; three DFT operations; identical paired predictions; finite nonzero complex-layer gradients. Passed.
2. Thirty-batch GPU benchmarks for fixed and learned preprocessing. Passed, <2 GB allocated model memory in the tested batch-8 setup. Short throughput measurements varied substantially with node/I/O conditions and are not a whole-run estimate.
3. Planned three uncapped seed-0, two-epoch pilots: DFT complex LR 0, CubeLearn 0.0001, CubeLearn 0.001. Real-network LR 0.0003; Adam defaults, no weight decay, constant rates, batch 8, FP32/complex64. Author HAR rates are a starting hypothesis, not a claimed HuPR optimum.
4. Inspect training curves, full validation AP, runtime and predicted poses. If viable, continue the same runs to ten epochs; do not restart or compress the schedule. Compare DFT to the best learned setting selected on validation. Expand that fixed pair to seeds 1 and 2 only if performance and cost justify it. Final test evaluation follows validation-based selection; never pick an epoch or learning rate using test AP.

The learned pilots reached validation AP 62.94 (complex LR 0.0001) and 66.66 (0.001) after two epochs. We selected 0.001 using validation alone and continued that checkpoint to ten total epochs. DFT was moved after its first saved epoch and continued to ten; its original two-epoch allocation was interrupted to address I/O. No extra seeds were launched. The lower-rate learned pilot was not tested.

Checkpoint selection: greatest full-validation AP, then lower validation BCE on ties. Save latest and best model, Adam state, completed epoch, best metric, histories and Python/NumPy/Torch/CUDA random states. Shuffling has an explicit epoch seed shared by all arms. Resuming checks source hashes and key arguments. Same-seed GPU arithmetic is not guaranteed bit-exact.

## Current campaign and commands

Local campaign: `hupr/local/cubelearn-pose-20261008/` (ignored by Git). Benchmark jobs 318118/318119; two-epoch pilots 318121/318122/318123. Training uses the HuPR Python environment plus an isolated copy of cplxmodule 2022.6 under `hupr/local/cubelearn-pose-deps/`; existing environment packages and the completed HuPR pipeline are unchanged.

To recreate the dependency in another checkout, install cplxmodule 2022.6 in a dedicated environment alongside HuPR's requirements and author evaluator. The scripts import the sibling [Project-Phantom-Echo CubeLearn fork](https://github.com/Project-Phantom-Echo/cubelearn), including its `network_har.py` adaptation. The tested commit is `c2873596f0441ee76bd5223319d0492b4d3c8572`; the upstream release alone does not contain that file. `--cubelearn-root` can override the sibling path. The pose campaign records source hashes for both imported model files and an additional environment record for the isolated cplxmodule copy.

Example fresh run (requires a GPU allocation):

```bash
python -m pip install -r reproduction/requirements.txt cplxmodule==2022.6
python reproduction/install_evaluator.py
python reproduction/prepare_pose_ground_truth.py --annotations /path/to/HuPR/annotations --output /path/to/pose-gt
python experiments/cubelearn_pose/run.py --mode train --raw-root /path/to/HuPR --gt-root /path/to/pose-gt --lpp-lr 0.001 --epochs 2 --output /path/to/fresh-run
python experiments/cubelearn_pose/run.py --mode train --raw-root /path/to/HuPR --gt-root /path/to/pose-gt --lpp-lr 0.001 --epochs 10 --output /path/to/fresh-run --resume
python experiments/cubelearn_pose/run.py --mode test --raw-root /path/to/HuPR --gt-root /path/to/pose-gt --lpp-lr 0.001 --output /path/to/fresh-run
```

The first three commands are environment/CPU preparation; only the final three require a GPU. The ground-truth helper uses the unchanged author generator and was verified byte-for-byte against the validation/test files used in this campaign. Original normalized radar caches are not required for this raw-ADC experiment.

Use `--lpp-lr 0` for DFT and a different output directory. Raw-root and ground-truth-root paths are configurable. Resume/evaluation require the original source and path configuration. The test command checks that configuration, loads `best.pth`, and scores the held-out set. It is not run automatically by the training command.

The cluster batch script defaults to 384 GiB RAM after 64 GiB allocations were observed repeatedly evicting raw-data file-cache pages. This changes resource allocation only. Eight data-loader workers and all training parameters remain fixed. The campaign records both discarded partial-epoch allocation time and completed-epoch timings; lower-memory machines can run the same model with more I/O.

Original method sources: [CubeLearn release](https://github.com/zhaoymn/cubelearn), [HuPR paper](https://openaccess.thecvf.com/content/WACV2023/html/Lee_HuPR_A_Benchmark_for_Human_Pose_Estimation_Using_Millimeter_Wave_WACV_2023_paper.html).
