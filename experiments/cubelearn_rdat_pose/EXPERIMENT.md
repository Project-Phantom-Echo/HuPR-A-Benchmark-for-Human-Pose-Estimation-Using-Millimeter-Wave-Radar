# Range-preserving CubeLearn on HuPR

This experiment adapts the released `RDAT_3DCNNLSTM` to HuPR pose estimation.
It is a new cross-task adaptation, not an original published CubeLearn pose model.
The separate D-A-T experiment and its running jobs remain unchanged.

## Architecture and comparison

Each radar uses the released range, Doppler and angle complex layers, modulus,
three 3D convolution/BatchNorm/ReLU/pooling blocks, 512-unit LSTM and 128-unit
hidden feature layer. Range is retained through the CNN instead of being summed.
Only the temporal reshape changes from ten frames to eight, and the final
classification layer is removed. The pose decoder is identical to the previous
D-A-T experiment: concatenate two 128-dimensional features and decode 14 heatmaps.

The model still retains only 128 ADC samples and the eight TX0/TX2 virtual
channels. It uses all 64 chirps per transmitter. This is the released R-D-A-T
input size, not a claim to use every raw HuPR measurement. A full-256-sample or
geometry-matched HuPR frontend would be a different experiment.

`check.py` compares the adapted encoder with the released forward pass at its
original ten-frame length, using identical weights. It also checks identical
fixed/learned initial predictions, FFT initialization, finite nonzero gradients
through every complex layer, and a real-data optimizer step. `data.py` is an
unchanged copy of the verified D-A-T loader.

## Protocol

Official HuPR sequence splits, eight-frame context, center-frame labels, 14
Gaussian heatmaps, mean BCE and author COCO evaluator match the D-A-T experiment.
Training uses Adam, batch 8, network LR 0.0003, complex LR 0.001, constant rates,
no weight decay, no mixed precision and seed 0. Fixed DFT is available with
`--lpp-lr 0`. Training shuffles with seed plus epoch; validation/test do not shuffle.

Select checkpoints by full-validation AP, then lower validation BCE on ties.
Keep latest and best checkpoints, Adam state and RNG state at every completed
epoch. Resume checks source hashes and configuration. Testing always loads the
best-validation checkpoint. Benchmark runs make optimizer updates only for timing;
their weights are discarded and never used to initialize the experiment.

The initial request is for a learned R-D-A-T run lasting approximately six hours,
with 5–10 epochs chosen from measured throughput. Both fixed and learned variants
are benchmarked, but a full fixed-DFT training run is not automatically submitted.
A future paired fixed-DFT run is needed to isolate learning the Fourier layers
from the effect of the different encoder.

## Runtime measurement and environment

Use the existing dedicated HuPR environment, author evaluator and cplxmodule
dependency described in `../cubelearn_pose/EXPERIMENT.md`. One H100, 12 CPU cores,
384 GiB host memory and eight loader workers are used for timing. The benchmark
includes 60 training batches and 30 validation batches, discards five warmup
batches in each phase, and extrapolates full-epoch time. The estimate includes
15 seconds per epoch for metric/checkpoint overhead; actual full-epoch I/O and
checkpoint time can differ. Queue wait is excluded.

Raw data defaults to `/mnt/weka/fgeikyan/rf-datas/hupr`; labels and the CubeLearn
checkout are configurable through the same arguments as the D-A-T runner.
Campaign outputs live under ignored `local/cubelearn-rdat-pose-20261009/`.
Benchmarks and checks run separately from the requested full training.


## Measured choice (2026-10-09)

Correctness checks and both benchmarks passed in job **319297**. The model has
**51,040,862 parameters**, compared with 6,077,262 for the D-A-T adaptation.
Learned R-D-A-T measured 0.1891 seconds per training batch, extrapolating to
**48.0 minutes per epoch including validation**; peak allocated GPU memory was
**5.02 GiB** at batch 8. Fixed DFT measured 39.9 minutes per epoch. These are short
benchmark estimates, not completed-epoch measurements.

Choose **7 epochs**: approximately **5h36m training/validation plus 2–3 minutes
for testing**, rounded to about **5h40m**, excluding queue wait. The training
allocation allows 6h30m for variation; checkpoints remain resumable if the limit
is reached. Only the learned model is selected for the requested run. Benchmark
weights are discarded. Evidence: `../../reproduction/evidence/cubelearn-rdat-pose-20261009/`.

From any account with access to this checkout, environment and datasets:

```bash
bash /mnt/weka/fgeikyan/rf-perception-papers/hupr/experiments/cubelearn_rdat_pose/submit.sh 7
```

The launcher uses an account-specific output name, refuses a duplicate run,
submits one training job and one dependent test job, and records both job IDs
and commands. No job-concurrency cap is applied. `RDAT_CAMPAIGN` can override the
output parent directory. The second argument optionally overrides the run name.
The prepared seven-epoch training has not been submitted during benchmarking.
