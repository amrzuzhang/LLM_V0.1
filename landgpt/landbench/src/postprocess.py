import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from utils import unbiased_rmse,_rmse,_bias
from config import get_args
from path_utils import resolve_root, resolve_mask_path, lat_filename

def _resolve_base_path(cfg):
    return resolve_root(cfg['inputs_path'], cfg['product'], cfg['spatial_resolution'])


def _load_latitudes(base_path: Path, spatial_resolution) -> np.ndarray | None:
    lat_path = base_path / lat_filename(spatial_resolution)
    if lat_path.exists():
        return np.load(lat_path)
    print(f"[postprocess] latitude file {lat_path.name} not found; using synthetic grid.")
    return None


def _build_area_weights(mask: np.ndarray, lat_grid: np.ndarray | None) -> np.ndarray:
    height, width = mask.shape
    if lat_grid is None:
        delta = 180.0 / height
        centers = np.linspace(90.0 - delta / 2.0, -90.0 + delta / 2.0, height, dtype=np.float64)
        latitudes = np.repeat(centers[:, None], width, axis=1)
    else:
        latitudes = np.asarray(lat_grid, dtype=np.float64)
        if latitudes.ndim == 1:
            if latitudes.shape[0] != height:
                raise ValueError(f"Latitude vector length {latitudes.shape[0]} != grid height {height}")
            latitudes = np.repeat(latitudes[:, None], width, axis=1)
        elif latitudes.shape != mask.shape:
            raise ValueError(f"Latitude grid shape {latitudes.shape} != mask shape {mask.shape}")
    weights = np.cos(np.deg2rad(latitudes))
    weights = np.clip(weights, 0.0, None)
    weights = np.where(mask, weights, 0.0)
    total = float(np.sum(weights))
    if total <= 0:
        weights = np.where(mask, 1.0, 0.0)
        total = float(np.sum(weights))
    return (weights / total).astype(np.float32)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values)
    masked_weights = weights * valid
    total = float(np.sum(masked_weights))
    if total <= 0:
        return float("nan")
    return float(np.nansum(values * masked_weights) / total)


def _summarize_area_metrics(metric_arrays: dict[str, np.ndarray], weights: np.ndarray) -> dict[str, float]:
    summary: dict[str, float] = {}
    for name, array in metric_arrays.items():
        summary[f"{name}_area_weighted"] = _weighted_mean(array, weights)
    return summary


def _write_metrics_json(out_dir: str | Path, metrics: dict[str, float]) -> None:
    path = Path(out_dir) / "metrics.json"
    existing: dict[str, float] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(metrics)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[postprocess] metrics saved to {path}")

def _point_stats(obs_series, pred_series, min_valid_obs, min_valid_std):
    valid = np.isfinite(obs_series) & np.isfinite(pred_series)
    if np.count_nonzero(valid) < min_valid_obs:
        return None
    obs = obs_series[valid]
    pred = pred_series[valid]
    obs_std = np.std(obs)
    pred_std = np.std(pred)
    if obs_std < min_valid_std or pred_std < min_valid_std:
        return None
    stats = {}
    stats['urmse'] = unbiased_rmse(obs, pred)
    stats['rmse'] = _rmse(obs, pred)
    stats['bias'] = _bias(obs, pred)
    obs_anom = obs - np.mean(obs)
    pred_anom = pred - np.mean(pred)
    denom = np.sqrt(np.sum(obs_anom ** 2) * np.sum(pred_anom ** 2))
    if denom <= 0:
        return None
    stats['r'] = float(np.dot(obs_anom, pred_anom) / denom)
    stats['r2'] = r2_score(obs, pred)
    stats['count'] = int(np.count_nonzero(valid))
    return stats

def lon_transform(x):
  x_new = np.zeros(x.shape)
  x_new[:,:,:int(x.shape[2]/2)] = x[:,:,int(x.shape[2]/2):] 
  x_new[:,:,int(x.shape[2]/2):] = x[:,:,:int(x.shape[2]/2)] 
  return x_new


def _save_with_report(file_path, array, label=None):
    """Persist array to disk and emit quick stats so users can see the values."""
    np.save(file_path, array)
    total = array.size
    valid_mask = np.isfinite(array)
    valid = int(np.count_nonzero(valid_mask))
    if valid:
        mean = float(np.nanmean(array))
        median = float(np.nanmedian(array))
    else:
        mean = float("nan")
        median = float("nan")
    name = label or os.path.basename(file_path)
    print(
        "[postprocess] saved {name}: {path} | shape {shape}, valid {valid}/{total}, mean {mean:.4f}, median {median:.4f}".format(
            name=name,
            path=file_path,
            shape=array.shape,
            valid=valid,
            total=total,
            mean=mean,
            median=median,
        )
    )
def postprocess(cfg):
    base_path = _resolve_base_path(cfg)
    PATH = base_path.as_posix().rstrip('/') + '/'
    mask = np.load(resolve_mask_path(base_path, cfg['spatial_resolution'])).astype(bool)
    lat_grid = _load_latitudes(base_path, cfg['spatial_resolution'])
    area_weights = _build_area_weights(mask, lat_grid)
    work_root = base_path / cfg['workname']
    focast_dir = 'focast_time ' + str(cfg['forcast_time'])
    min_valid_obs = int(cfg.get('min_valid_obs', 30))
    min_valid_std = float(cfg.get('min_valid_std', 1e-4))

    def _model_path(name: str) -> str:
        return (work_root / name / focast_dir).as_posix().rstrip('/') + '/'
    if cfg['modelname'] in ['ConvLSTM']:
        out_path_convlstm = _model_path(cfg['modelname'])
        y_pred_convlstm = np.load(out_path_convlstm+'_predictions.npy')
        y_test_convlstm = np.load(out_path_convlstm+'observations.npy')
        print(y_pred_convlstm.shape, y_test_convlstm.shape)
        # get shape
        nt, nlat, nlon = y_test_convlstm.shape    
        # cal perf
        r2_convlstm = np.full(( nlat, nlon), np.nan)
        urmse_convlstm = np.full(( nlat, nlon), np.nan)
        r_convlstm = np.full(( nlat, nlon), np.nan)
        rmse_convlstm = np.full(( nlat, nlon), np.nan)
        bias_convlstm = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not mask[i, j]:
                    continue
                stats = _point_stats(y_test_convlstm[:, i, j], y_pred_convlstm[:, i, j], min_valid_obs, min_valid_std)
                if not stats:
                    continue
                urmse_convlstm[i, j] = stats['urmse']
                r_convlstm[i, j] = stats['r']
                rmse_convlstm[i, j] = stats['rmse']
                bias_convlstm[i, j] = stats['bias']
                r2_convlstm[i, j] = stats['r2']
        _save_with_report(out_path_convlstm + 'r2_'+cfg['modelname']+'.npy', r2_convlstm, f"{cfg['modelname']} r2")
        _save_with_report(out_path_convlstm + 'r_'+cfg['modelname']+'.npy', r_convlstm, f"{cfg['modelname']} r")
        _save_with_report(out_path_convlstm + 'rmse_'+cfg['modelname']+'.npy', rmse_convlstm, f"{cfg['modelname']} rmse")
        _save_with_report(out_path_convlstm + 'bias_'+cfg['modelname']+'.npy', bias_convlstm, f"{cfg['modelname']} bias")
        _save_with_report(out_path_convlstm + 'urmse_'+cfg['modelname']+'.npy', urmse_convlstm, f"{cfg['modelname']} urmse")
        metrics = _summarize_area_metrics(
            {
                "r2": r2_convlstm,
                "r": r_convlstm,
                "rmse": rmse_convlstm,
                "bias": bias_convlstm,
                "urmse": urmse_convlstm,
            },
            area_weights,
        )
        _write_metrics_json(out_path_convlstm, metrics)
        print(f"[postprocess] {cfg['modelname']} valid pixels: {int(np.isfinite(r2_convlstm).sum())}/{int(mask.sum())} (min_obs={min_valid_obs})")
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
    if not cfg['label'] == ['volumetric_soil_water_layer_20'] and cfg['modelname'] in ['LSTM']:
        out_path_lstm = _model_path(cfg['modelname'])
        y_pred_lstm = np.load(out_path_lstm+'_predictions.npy')
        y_test_lstm = np.load(out_path_lstm+'observations.npy')


        print(y_pred_lstm.shape, y_test_lstm.shape)
        # get shape
        nt, nlat, nlon = y_test_lstm.shape 
        # cal perf
        r2_lstm = np.full(( nlat, nlon), np.nan)
        urmse_lstm = np.full(( nlat, nlon), np.nan)
        r_lstm = np.full(( nlat, nlon), np.nan)
        rmse_lstm = np.full(( nlat, nlon), np.nan)
        bias_lstm = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not mask[i, j]:
                    continue
                stats = _point_stats(y_test_lstm[:, i, j], y_pred_lstm[:, i, j], min_valid_obs, min_valid_std)
                if not stats:
                    continue
                urmse_lstm[i, j] = stats['urmse']
                r_lstm[i, j] = stats['r']
                rmse_lstm[i, j] = stats['rmse']
                bias_lstm[i, j] = stats['bias']
                r2_lstm[i, j] = stats['r2']
        _save_with_report(out_path_lstm + 'r2_'+'LSTM'+'.npy', r2_lstm, 'LSTM r2')
        _save_with_report(out_path_lstm + 'r_'+'LSTM'+'.npy', r_lstm, 'LSTM r')
        _save_with_report(out_path_lstm + 'rmse_'+cfg['modelname']+'.npy', rmse_lstm, 'LSTM rmse')
        _save_with_report(out_path_lstm + 'bias_'+cfg['modelname']+'.npy', bias_lstm, 'LSTM bias')
        _save_with_report(out_path_lstm + 'urmse_'+'LSTM'+'.npy', urmse_lstm, 'LSTM urmse')
        metrics = _summarize_area_metrics(
            {
                "r2": r2_lstm,
                "r": r_lstm,
                "rmse": rmse_lstm,
                "bias": bias_lstm,
                "urmse": urmse_lstm,
            },
            area_weights,
        )
        _write_metrics_json(out_path_lstm, metrics)
        print(f"[postprocess] LSTM valid pixels: {int(np.isfinite(r2_lstm).sum())}/{int(mask.sum())} (min_obs={min_valid_obs})")
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['CNN']:
        out_path_cnn = _model_path(cfg['modelname'])
        y_pred_cnn = np.load(out_path_cnn+'_predictions.npy')
        y_test_cnn = np.load(out_path_cnn+'observations.npy')
        y_pred_cnn = y_pred_cnn[cfg["seq_len"]:]
        y_test_cnn = y_test_cnn[cfg["seq_len"]:]
        print(y_pred_cnn.shape, y_test_cnn.shape)
        # get shape
        nt, nlat, nlon = y_test_cnn.shape 
        # cal perf
        r2_cnn = np.full(( nlat, nlon), np.nan)    
        urmse_cnn = np.full(( nlat, nlon), np.nan)
        r_cnn = np.full(( nlat, nlon), np.nan)
        rmse_cnn = np.full(( nlat, nlon), np.nan)
        bias_cnn = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not mask[i, j]:
                    continue
                stats = _point_stats(y_test_cnn[:, i, j], y_pred_cnn[:, i, j], min_valid_obs, min_valid_std)
                if not stats:
                    continue
                urmse_cnn[i, j] = stats['urmse']
                r_cnn[i, j] = stats['r']
                rmse_cnn[i, j] = stats['rmse']
                bias_cnn[i, j] = stats['bias']
                r2_cnn[i, j] = stats['r2']
        _save_with_report(out_path_cnn + 'r2_'+'CNN'+'.npy', r2_cnn, 'CNN r2')
        _save_with_report(out_path_cnn + 'r_'+'CNN'+'.npy', r_cnn, 'CNN r')
        _save_with_report(out_path_cnn + 'rmse_'+cfg['modelname']+'.npy', rmse_cnn, 'CNN rmse')
        _save_with_report(out_path_cnn + 'bias_'+cfg['modelname']+'.npy', bias_cnn, 'CNN bias')
        _save_with_report(out_path_cnn + 'urmse_'+'CNN'+'.npy', urmse_cnn, 'CNN urmse')
        metrics = _summarize_area_metrics(
            {
                "r2": r2_cnn,
                "r": r_cnn,
                "rmse": rmse_cnn,
                "bias": bias_cnn,
                "urmse": urmse_cnn,
            },
            area_weights,
        )
        _write_metrics_json(out_path_cnn, metrics)
        print(f"[postprocess] CNN valid pixels: {int(np.isfinite(r2_cnn).sum())}/{int(mask.sum())} (min_obs={min_valid_obs})")
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['Process']:
        print('start Process')
        out_path_cnn = _model_path(cfg['modelname'])
        y_pred_cnn = np.load(out_path_cnn+'_predictions.npy')
        if cfg['label'] == ["volumetric_soil_water_layer_20"]:
                y_test_cnn_layer1 = np.load(out_path_cnn+'observations_layer1.npy')
                y_test_cnn_layer2 = np.load(out_path_cnn+'observations_layer2.npy')
                y_test_cnn = (y_test_cnn_layer1*7+13*y_test_cnn_layer2)/20
                np.save(out_path_cnn + 'observations.npy', y_test_cnn)
                y_pred_cnn = y_pred_cnn/1000
             
        else:	
                y_test = np.load(out_path_cnn+'observations.npy')
        if cfg['label'] == ["surface_sensible_heat_flux"]:
                y_pred_cnn = -(y_pred_cnn)/(86400*cfg['forcast_time'])
        y_pred_cnn = y_pred_cnn[1:]
        #y_pred_cnn = lon_transform(y_pred_cnn)
        y_test_cnn = np.load(out_path_cnn+'observations.npy')
        print(y_pred_cnn.shape, y_test_cnn.shape)
        # get shape
        nt, nlat, nlon = y_test_cnn.shape 
        # cal perf
        r2_cnn = np.full(( nlat, nlon), np.nan)    
        urmse_cnn = np.full(( nlat, nlon), np.nan)
        r_cnn = np.full(( nlat, nlon), np.nan)
        rmse_cnn = np.full(( nlat, nlon), np.nan)
        bias_cnn = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not mask[i, j]:
                    continue
                stats = _point_stats(y_test_cnn[:, i, j], y_pred_cnn[:, i, j], min_valid_obs, min_valid_std)
                if not stats:
                    continue
                urmse_cnn[i, j] = stats['urmse']
                r_cnn[i, j] = stats['r']
                rmse_cnn[i, j] = stats['rmse']
                bias_cnn[i, j] = stats['bias']
                r2_cnn[i, j] = stats['r2']
        _save_with_report(out_path_cnn + 'r2_'+'Process'+'.npy', r2_cnn, 'Process r2')
        _save_with_report(out_path_cnn + 'r_'+'Process'+'.npy', r_cnn, 'Process r')
        _save_with_report(out_path_cnn + 'rmse_'+cfg['modelname']+'.npy', rmse_cnn, 'Process rmse')
        _save_with_report(out_path_cnn + 'bias_'+cfg['modelname']+'.npy', bias_cnn, 'Process bias')
        _save_with_report(out_path_cnn + 'urmse_'+'Process'+'.npy', urmse_cnn, 'Process urmse')
        metrics = _summarize_area_metrics(
            {
                "r2": r2_cnn,
                "r": r_cnn,
                "rmse": rmse_cnn,
                "bias": bias_cnn,
                "urmse": urmse_cnn,
            },
            area_weights,
        )
        _write_metrics_json(out_path_cnn, metrics)
        print(f"[postprocess] Process valid pixels: {int(np.isfinite(r2_cnn).sum())}/{int(mask.sum())} (min_obs={min_valid_obs})")
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
    if cfg['label'] == ['volumetric_soil_water_layer_20'] and not cfg['modelname'] in ['Process'] and not cfg['modelname'] in ['Persistence'] and not cfg['modelname'] in ['w_climatology']:
        print('LSTM ---> volumetric_soil_water_layer_20')
        out_path_cnn = _model_path(cfg['modelname'])
        y_pred_cnn_layer1 = np.load(out_path_cnn+'_predictions_layer1.npy')
        y_pred_cnn_layer2 = np.load(out_path_cnn+'_predictions_layer2.npy')
        y_pred_cnn = (y_pred_cnn_layer1*7+13*y_pred_cnn_layer2)/20
        #y_pred_cnn = lon_transform(y_pred_cnn)
        y_test_cnn_layer1 = np.load(out_path_cnn+'observations_layer1.npy')
        y_test_cnn_layer2 = np.load(out_path_cnn+'observations_layer2.npy')
        y_test_cnn = (y_test_cnn_layer1*7+13*y_test_cnn_layer2)/20
        np.save(out_path_cnn + 'observations.npy', y_test_cnn)
        np.save(out_path_cnn + '_predictions.npy', y_pred_cnn)
        print(y_pred_cnn.shape, y_test_cnn.shape)
        # get shape
        nt, nlat, nlon = y_test_cnn.shape 
        # cal perf
        r2_cnn = np.full(( nlat, nlon), np.nan)    
        urmse_cnn = np.full(( nlat, nlon), np.nan)
        r_cnn = np.full(( nlat, nlon), np.nan)
        rmse_cnn = np.full(( nlat, nlon), np.nan)
        bias_cnn = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not mask[i, j]:
                    continue
                stats = _point_stats(y_test_cnn[:, i, j], y_pred_cnn[:, i, j], min_valid_obs, min_valid_std)
                if not stats:
                    continue
                urmse_cnn[i, j] = stats['urmse']
                r_cnn[i, j] = stats['r']
                rmse_cnn[i, j] = stats['rmse']
                bias_cnn[i, j] = stats['bias']
                r2_cnn[i, j] = stats['r2']
        _save_with_report(out_path_cnn + 'r2_'+'Process'+'.npy', r2_cnn, 'Process r2')
        _save_with_report(out_path_cnn + 'r_'+'Process'+'.npy', r_cnn, 'Process r')
        _save_with_report(out_path_cnn + 'rmse_'+cfg['modelname']+'.npy', rmse_cnn, 'Process rmse')
        _save_with_report(out_path_cnn + 'bias_'+cfg['modelname']+'.npy', bias_cnn, 'Process bias')
        _save_with_report(out_path_cnn + 'urmse_'+'Process'+'.npy', urmse_cnn, 'Process urmse')
        metrics = _summarize_area_metrics(
            {
                "r2": r2_cnn,
                "r": r_cnn,
                "rmse": rmse_cnn,
                "bias": bias_cnn,
                "urmse": urmse_cnn,
            },
            area_weights,
        )
        _write_metrics_json(out_path_cnn, metrics)
        print(f"[postprocess] Process valid pixels: {int(np.isfinite(r2_cnn).sum())}/{int(mask.sum())} (min_obs={min_valid_obs})")
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['Persistence']:
        print('Persistence ---> volumetric_soil_water_layer_20')
        out_path = _model_path(cfg['modelname'])
        if not os.path.isdir (out_path):
            os.makedirs(out_path)
        path = PATH
        if cfg['label'] == ['volumetric_soil_water_layer_20']:
            y_test_1 = np.load(path+ cfg['workname'] + '/' + 'y_test_norm_SM1.npy',mmap_mode='r')
            y_test_2 = np.load(path+ cfg['workname'] + '/' + 'y_test_norm_SM2.npy',mmap_mode='r')
            y_test = (y_test_1*7+13*y_test_2)/20
            np.save(out_path + 'observations.npy', y_test)
        else:
            y_test = np.load(path+'y_test_norm.npy',mmap_mode='r')
        print('y_test shape is',y_test.shape)
        y_test_cnn = y_test[cfg['seq_len']+cfg['forcast_time']:,:,:,0]
        np.save(out_path + 'observations.npy', y_test_cnn)
        print(y_test_cnn.shape)
        y_pred_cnn = y_test[cfg['seq_len']+cfg['forcast_time']-cfg['forcast_time']:y_test.shape[0]-cfg['forcast_time'],:,:,0]
        np.save(out_path + '_predictions.npy', y_pred_cnn)

        print(y_pred_cnn.shape, y_test_cnn.shape)
        # get shape
        nt, nlat, nlon = y_test_cnn.shape 
        # cal perf
        r2_cnn = np.full(( nlat, nlon), np.nan)    
        urmse_cnn = np.full(( nlat, nlon), np.nan)
        r_cnn = np.full(( nlat, nlon), np.nan)
        rmse_cnn = np.full(( nlat, nlon), np.nan)
        bias_cnn = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not mask[i, j]:
                    continue
                stats = _point_stats(y_test_cnn[:, i, j], y_pred_cnn[:, i, j], min_valid_obs, min_valid_std)
                if not stats:
                    continue
                urmse_cnn[i, j] = stats['urmse']
                r_cnn[i, j] = stats['r']
                rmse_cnn[i, j] = stats['rmse']
                bias_cnn[i, j] = stats['bias']
                r2_cnn[i, j] = stats['r2']
        _save_with_report(out_path + 'r2_'+'Persistence'+'.npy', r2_cnn, 'Persistence r2')
        _save_with_report(out_path + 'r_'+'Persistence'+'.npy', r_cnn, 'Persistence r')
        _save_with_report(out_path + 'rmse_'+cfg['modelname']+'.npy', rmse_cnn, 'Persistence rmse')
        _save_with_report(out_path + 'bias_'+cfg['modelname']+'.npy', bias_cnn, 'Persistence bias')
        _save_with_report(out_path + 'urmse_'+'Persistence'+'.npy', urmse_cnn, 'Persistence urmse')
        metrics = _summarize_area_metrics(
            {
                "r2": r2_cnn,
                "r": r_cnn,
                "rmse": rmse_cnn,
                "bias": bias_cnn,
                "urmse": urmse_cnn,
            },
            area_weights,
        )
        _write_metrics_json(out_path, metrics)
        print(f"[postprocess] Persistence valid pixels: {int(np.isfinite(r2_cnn).sum())}/{int(mask.sum())} (min_obs={min_valid_obs})")
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
    if cfg['modelname'] in ['w_climatology']:
        out_path = _model_path(cfg['modelname'])
        if not os.path.isdir (out_path):
            os.makedirs(out_path)
        path = PATH

        if cfg['label'] == ['volumetric_soil_water_layer_20']:
            y_pre_1 = np.load(path + cfg['workname'] + '/'+ 'y_train_SM1.npy')
            y_pre_2 = np.load(path + cfg['workname'] + '/'+ 'y_train_SM2.npy')
            y_pre = (y_pre_1*7+13*y_pre_2)/20
        else:
            y_pre = np.load(path + 'y_train.npy')

        if cfg['label'] == ['volumetric_soil_water_layer_20']:
            y_test_1 = np.load(path+ cfg['workname'] + '/' 'y_test_norm_SM1.npy',mmap_mode='r')
            y_test_2 = np.load(path+ cfg['workname'] + '/' 'y_test_norm_SM2.npy',mmap_mode='r')
            y_test = (y_test_1*7+13*y_test_2)/20
            np.save(out_path + 'observations.npy', y_test)
        else:
            y_test = np.load(path+'y_test_norm.npy',mmap_mode='r')

        y_test_cnn = y_test[cfg['seq_len']+cfg['forcast_time']:,:,:,0]
        print('y_test shape is',y_test_cnn.shape)
        np.save(out_path + 'observations.npy', y_test_cnn)
        y_pred_cnn = np.zeros((y_pre.shape))*np.nan
        print('y_pred_cnn shape is',y_pred_cnn.shape)
        data = y_pre
        num_years = data.shape[0]//365
        weekly_climat = np.zeros((num_years,52,data.shape[1],data.shape[2]))
        for year in range (num_years-1):
            year_data = data[year*365:(year+1)*365]
            weekly_climat_per_year = year_data[:-1,:,:].reshape((52,7,year_data.shape[1],year_data.shape[2]))
            weekly_climat[year] = np.nanmean(weekly_climat_per_year,axis=1)
        weekly_mean =np.nanmean(weekly_climat,axis=0)
        weekly_results_ = np.repeat(weekly_mean,7,axis=0)   
        weekly_results = np.concatenate((weekly_results_,np.expand_dims(weekly_results_[-1,:,:],axis=0)),axis=0)
        np.save(out_path + '_predictions.npy', weekly_results)
        y_pred_cnn = weekly_results
        print(weekly_results.shape, y_test_cnn.shape)
        # get shape
        nt, nlat, nlon = y_test_cnn.shape 
        # cal perf
        r2_cnn = np.full(( nlat, nlon), np.nan)    
        urmse_cnn = np.full(( nlat, nlon), np.nan)
        r_cnn = np.full(( nlat, nlon), np.nan)
        rmse_cnn = np.full(( nlat, nlon), np.nan)
        bias_cnn = np.full(( nlat, nlon), np.nan)
        for i in range(nlat):
            for j in range(nlon):
                if not mask[i, j]:
                    continue
                stats = _point_stats(y_test_cnn[:, i, j], y_pred_cnn[:, i, j], min_valid_obs, min_valid_std)
                if not stats:
                    continue
                urmse_cnn[i, j] = stats['urmse']
                r_cnn[i, j] = stats['r']
                rmse_cnn[i, j] = stats['rmse']
                bias_cnn[i, j] = stats['bias']
                r2_cnn[i, j] = stats['r2']
        _save_with_report(out_path + 'r2_'+'w_climatology'+'.npy', r2_cnn, 'w_climatology r2')
        _save_with_report(out_path + 'r_'+'w_climatology'+'.npy', r_cnn, 'w_climatology r')
        _save_with_report(out_path + 'rmse_'+cfg['modelname']+'.npy', rmse_cnn, 'w_climatology rmse')
        _save_with_report(out_path + 'bias_'+cfg['modelname']+'.npy', bias_cnn, 'w_climatology bias')
        _save_with_report(out_path + 'urmse_'+'w_climatology'+'.npy', urmse_cnn, 'w_climatology urmse')
        metrics = _summarize_area_metrics(
            {
                "r2": r2_cnn,
                "r": r_cnn,
                "rmse": rmse_cnn,
                "bias": bias_cnn,
                "urmse": urmse_cnn,
            },
            area_weights,
        )
        _write_metrics_json(out_path, metrics)
        print(f"[postprocess] w_climatology valid pixels: {int(np.isfinite(r2_cnn).sum())}/{int(mask.sum())} (min_obs={min_valid_obs})")
        print('postprocess ove, please go on')
# ------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    cfg = get_args()
    postprocess(cfg)
