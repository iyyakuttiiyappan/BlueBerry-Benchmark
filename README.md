# Silal Blueberry Benchmark

This repository snapshot contains the benchmark code, result tables, figures, and LaTeX manuscript assets for the Silal blueberry multi-task benchmark.

It is intentionally **benchmark-only**:

- no raw RGB images
- no mask files
- no trained checkpoints
- no dataset download links
- no dataset links

Data access details are omitted and can be added after manuscript acceptance.

## Contents

```text
configs/                 Benchmark configuration
figures/                 Paper-ready benchmark figures
results/                 Key benchmark CSV tables
scripts/                 Benchmark, visualization, and bundle-building scripts
src/blueberry_multitask/ Source package used by benchmark scripts
requirements.txt         Python dependencies
```

## Main Benchmark Claim

The benchmark evaluates four tasks:

- detection
- segmentation
- counting
- multiclass ripeness classification

It compares specialist baselines with unified BerryMTL variants that produce all four outputs from one shared model.

## Reproducibility

After placing the dataset locally, edit `configs/fresh_benchmark_514.yaml` and run:

```bash
python scripts/fresh_prepare_annotations.py --config configs/fresh_benchmark_514.yaml --rebuild
python scripts/fresh_run_all.py --config configs/fresh_benchmark_514.yaml
python scripts/fresh_summarize.py --config configs/fresh_benchmark_514.yaml
python scripts/build_latex_bundle_514.py --config configs/fresh_benchmark_514.yaml
```


