# LandBench / GPTCast Pipeline (English Guide)

This repository bundles the official LandBench preprocessing code and the GPTCast training scripts. You can process ERA5-Land soil‑moisture inputs, train baselines (LSTM/ConvLSTM/CNN), and then run GPTCast on the generated `.npy` artifacts. This document replaces the old Chinese README.

## Repository Layout

```
landbench/            # official preprocessing + baselines (main.py, train.py, plot.py, plot_test.py, etc.)
gptcast/              # GPTCast Lightning models, baselines, and launch scripts
config/settings.yaml  # shared configuration (paths, years, hyperparameters)
requirements.txt      # Python dependencies
install_landgpt.sh    # optional bootstrap script
```

## Environment Setup

```bash
conda create -n landgpt python=3.10
conda activate landgpt
pip install -r requirements.txt
conda install -c conda-forge netcdf4 h5py  # needed for NetCDF/HDF backends
```

Place raw LandBench NetCDF archives under `/home/xiezhenang/data/LandBench/<resolution>/` and keep derived artifacts under `/home/xiezhenang/landgpt/outputs/`. Adjust `config/settings.yaml` if your paths differ.

## LandBench Pipeline

1. **Configure paths/years/hyperparameters**
   ```bash
   cd /home/xiezhenang/landgpt
   vim config/settings.yaml   # edit paths.raw_data, paths.preprocessed, landbench.* options
   ```

2. **Preprocess + train LSTM (baseline)**
   ```bash
   cd /home/xiezhenang/landgpt/landbench/src
   python main.py
   ```
   - `selected_year` controls the inclusive [begin, end] span used for training/validation; `test_year` specifies evaluation years.
   - `seq_len` is the length (in days) of each input window, `forcast_time` is the prediction offset (days ahead). Adjust them in `settings.yaml` or via CLI flags.
   - `batch_size`, `niter`, `epochs` reside in the same YAML. Override from CLI when needed (e.g., `python main.py --batch_size 64 --niter 300`).

3. **Postprocess + visualization**
   ```bash
   cd /home/xiezhenang/landgpt/landbench/src
   python postprocess.py --inputs_path /home/xiezhenang/landgpt/outputs --workname LandBench --modelname LSTM
   python plot.py        --inputs_path /home/xiezhenang/landgpt/outputs --workname LandBench --spatial_resolution 1
   python plot_test.py   --inputs_path /home/xiezhenang/landgpt/outputs --workname LandBench --modelname LSTM
   ```
   Important behavior:
   - `--spatial_resolution` automatically collapses `1` and `1.0` into the `/outputs/1/…` directory.
   - You **do not** pass `--plot_models` or legacy flags; both scripts read everything from `config/settings.yaml`.
   - `postprocess.py` filters low-quality pixels using `min_valid_obs` / `min_valid_std` (defaults 30 / 1e‑4). Raise/lower them in YAML to change coverage.
   - `plot.py` always looks for LSTM/CNN/ConvLSTM folders and creates model‑comparison figures (box/metrics/spatial). It does not accept `--plot_models`.
   - `plot_test.py` focuses on a single model. It saves every figure to `/outputs/<resolution>/<workname>/<modelname>/focast_time X/figures/` before displaying it. Use `--plot_figures box spatial metrics timeseries` to render a subset (`box`=Fig.1, `spatial`=Figs.2‑5/7/8, `metrics` overlaps with spatial stats, `timeseries`=Fig.6).

## GPTCast Usage (Optional)

After LandBench preprocessing finishes, GPTCast can consume the `.npy` artifacts:

```bash
cd /home/xiezhenang/landgpt

# 训练
python -m gptcast.run.train_soil_moisture \
  --nc-root /home/xiezhenang/landgpt/outputs \
  --resolution 1 \
  --subset train \
  --input-steps 6 \
  --forecast-steps 3 \
  --batch-size 4 \
  --max-epochs 50 \
  --output-dir outputs/soil_moisture

# 评估
python -m gptcast.run.eval_soil_moisture \
  --nc-root /home/xiezhenang/landgpt/outputs \
  --resolution 1 \
  --subset test \
  --checkpoint outputs/soil_moisture/soil-moisture-XX.ckpt \
  --save-predictions
```

Add `--use-area-weighted-loss` if you want the training loss to respect cos(latitude) weighting, and pass `--lambda-hyd <value>` (for example `0.1`) to activate the new ΔSM≈P–ET–R hydrology regularizer.

To enable probabilistic quantile forecasts, append `--uncertainty quantile --quantiles 0.05 0.5 0.95` while training. Afterwards, run:

```bash
cd /home/xiezhenang/landgpt
python -m gptcast.run.eval_calibration \
  --nc-root /home/xiezhenang/landgpt/outputs \
  --resolution 1 \
  --subset test \
  --checkpoint outputs/soil_moisture/soil-moisture-quantile.ckpt \
  --output-dir outputs/calibration
```

This generates PIT/reliability plots plus `metrics_calibration.json` under `outputs/calibration/calibration/`.

The GPTCast loader automatically applies the LandBench land mask (`Mask with <resolution> spatial resolution.npy`) so that ocean pixels are zeroed out before training. Make sure this file is present under `/home/xiezhenang/landgpt/outputs/<resolution>/`.

### Suggested follow-up steps

After `eval_soil_moisture` finishes, consider the following depending on what features you enabled:

1. **Area-weighted metrics**
   ```bash
   cd /home/xiezhenang/landgpt
   python -m gptcast.run.eval_soil_moisture \
     --nc-root /home/xiezhenang/landgpt/outputs \
     --resolution 1 \
     --subset test \
     --checkpoint outputs/soil_moisture/soil-moisture-XX.ckpt \
     --use-area-weighted-metrics \
     --save-predictions
   ```
2. **Hydrology penalty** – `--lambda-hyd` 开启时，留意训练/验证日志里的 `*_hydro_penalty`；若想把 GPTCast 预测再次参与 LandBench 后处理，可运行：
   ```bash
   cd /home/xiezhenang/landgpt/landbench/src
   python postprocess.py --inputs_path /home/xiezhenang/landgpt/outputs --workname LandBench --modelname LSTM
   ```
3. **Quantile calibration** – 启用了 `--uncertainty quantile` 必须执行：
   ```bash
   cd /home/xiezhenang/landgpt
   python -m gptcast.run.eval_calibration \
     --nc-root /home/xiezhenang/landgpt/outputs \
     --resolution 1 \
     --subset test \
     --checkpoint outputs/soil_moisture/soil-moisture-quantile.ckpt \
     --output-dir outputs/calibration
   ```
4. **LandBench 对比** – 把 GPTCast 预测拷贝或链接到 `outputs/1/LandBench/...` 后，可继续执行：
   ```bash
   cd /home/xiezhenang/landgpt/landbench/src
   python plot.py      --inputs_path /home/xiezhenang/landgpt/outputs --workname LandBench --spatial_resolution 1
   python plot_test.py --inputs_path /home/xiezhenang/landgpt/outputs --workname LandBench --modelname LSTM
   ```

### Advanced CLI options summary

| Flag | Purpose |
|------|---------|
| `--use-area-weighted-loss` | Apply cos(latitude) weights inside the training loss/logged metrics. |
| `--lambda-hyd <float>` | Enable ΔSM≈P−ET−R 正则化（依赖 LandBench 的 P/ET/R 通道）。 |
| `--uncertainty quantile --quantiles ...` | 输出分位数 (默认 q05/q50/q95) 并采用 pinball loss，便于后续校准评估。 |
| `--use-area-weighted-metrics` (eval) | 在 `eval_soil_moisture` 中打印/保存 `_area_weighted` 指标。 |
| `python -m gptcast.run.eval_calibration` | 针对分位数模型生成 CRPS、PIT 直方图、可靠度曲线 (`calibration/*.png` + `metrics_calibration.json`)。 |

Set `gptcast.years`, `batch_size`, etc., inside `config/settings.yaml`. CLI flags override YAML defaults if needed.

## Common Options

| YAML key             | Description                                                                 | Default |
|----------------------|-----------------------------------------------------------------------------|---------|
| `landbench.seq_len`  | Length (days) of each training window                                       | 365     |
| `landbench.forcast_time` | Prediction lead time (days)                                             | 1       |
| `landbench.min_valid_obs` | Minimum valid samples per pixel when computing metrics (postprocess)  | 30      |
| `landbench.min_valid_std` | Minimum variance threshold (postprocess)                              | 1e‑4    |
| `gptcast.years`      | Years passed to GPTCast train/eval scripts                                  | `[2015..2020]` (example) |

Modify these in YAML for persistent changes; use CLI flags (e.g., `--seq_len 540 --forcast_time 7`) for ad‑hoc experiments.

## Troubleshooting

| Symptom                                      | Fix                                                                 |
|----------------------------------------------|---------------------------------------------------------------------|
| `pyproj unable to set PROJ database path`    | Install `conda install -c conda-forge basemap pyproj`; warning is otherwise harmless. |
| `IndexError` or NaNs in `plot_test.py`       | Ensure `selected_year` spans enough days and update `sites_lon_index/lat_index` to valid land pixels. |
| `postprocess.py` prints huge negative r²     | Increase `min_valid_obs`/`min_valid_std` so that low‑variance or missing grids are skipped. |
| Want longer history per sample               | Raise `seq_len` in YAML; the random sampler will use the new window length. |

## FAQ: plot.py vs plot_test.py

- `plot.py` expects LSTM/CNN/ConvLSTM subfolders under `/outputs/<resolution>/<workname>/` and generates multi-model comparison panels (box plots, spatial metrics). It does **not** accept `--plot_models`.
- `plot_test.py` works on a single model, emits eight detailed figures (box, prediction vs observation, spatial r²/r/&c, time series) and saves them under `.../figures/`; `--plot_figures` lets you select subsets and will also pop up each figure after saving.

Use `plot.py` when you need model-to-model comparisons; use `plot_test.py` when analyzing a specific model in depth.

---

With the above commands and configuration, you can preprocess LandBench, train baselines, evaluate metrics, and run GPTCast end‑to‑end. Let me know if additional sections (e.g., multi-GPU GPTCast, custom datasets) are needed and I can extend this README.***
