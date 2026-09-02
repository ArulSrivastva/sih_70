"""P4 model-improvement shared module (model_improvement area).

Self-contained, non-destructive: loads authoritative on-disk P4 data
(canonical_chrono raw + feature_dataset), builds causal features on top of the
existing 16, provides train-only normalization, track-error metrics, and the
persistence / movement-vector / kinematic baselines.

No protected source is modified. All metrics computed against the authoritative
Y targets.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

PROJ = Path(__file__).resolve().parents[4]           # cyclone-project root
RAW = PROJ / "p4_forecasting" / "phase2" / "results" / "canonical_chronological_clean"
FEAT = PROJ / "p4_forecasting" / "phase4" / "results" / "feature_dataset"

RAW_NAMES = ["lat","lon","wind_speed","pressure","sst","wind_u","wind_v"]
DERIVED_16 = ["delta_lat","delta_lon","movement_speed","movement_direction",
              "wind_change","pressure_change","sst_change",
              "environmental_wind_speed","environmental_wind_direction"]
HORIZONS = [6, 12, 24]
MULT = {6:1.0, 12:2.0, 24:4.0}
EARTH_R = 6371.0088

def wrap_lon_delta(lon_prev, lon_cur):
    return (lon_cur - lon_prev + 180.0) % 360.0 - 180.0

def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*EARTH_R*np.arcsin(np.sqrt(a))

def bearing_degrees(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    dlon = np.radians(wrap_lon_delta(lon1, lon2))
    y = np.sin(dlon)*np.cos(lat2)
    x = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) % 360.0).astype(np.float64)

def load_raw(split):
    d = np.load(RAW / f"{split}.npz", allow_pickle=True)
    return d["X"], d["Y"]                       # (N,5,7), (N,3,3)

def load_feature(split):
    d = np.load(FEAT / f"{split}.npz", allow_pickle=True)
    return d["X"], d["Y"]

def load_meta(split):
    return pd.read_csv(FEAT / f"{split}_metadata.csv")

def engineer_extra(X_raw):
    """Append causal derived features to raw (N,5,7) -> (N,5,7+K).
    Uses only history <= each timestep. Never touches Y."""
    X = np.asarray(X_raw, dtype=np.float32)
    N = X.shape[0]
    lat, lon, wind, pres, sst = X[:,:,0], X[:,:,1], X[:,:,2], X[:,:,3], X[:,:,4]
    # per-step velocity components (deg per 6h) and heading
    dlat = np.zeros_like(lat); dlon = np.zeros_like(lon)
    heading = np.zeros_like(lat); speed = np.zeros_like(lat)
    for i in range(1, 5):
        dlat[:,i] = lat[:,i]-lat[:,i-1]
        dlon[:,i] = wrap_lon_delta(lon[:,i-1], lon[:,i])
        speed[:,i] = haversine_km(lat[:,i-1], lon[:,i-1], lat[:,i], lon[:,i])
        heading[:,i] = bearing_degrees(lat[:,i-1], lon[:,i-1], lat[:,i], lon[:,i])
    # acceleration: velocity(dlat,dlon) change between steps
    acc_lat = np.zeros_like(lat); acc_lon = np.zeros_like(lon)
    turn_rate = np.zeros_like(lat)              # heading change (deg)
    for i in range(2, 5):
        acc_lat[:,i] = (dlat[:,i]-dlat[:,i-1])/(6.0**2)
        acc_lon[:,i] = (dlon[:,i]-dlon[:,i-1])/(6.0**2)
        turn_rate[:,i] = ((heading[:,i]-heading[:,i-1]+180.0)%360.0-180.0)
    # cumulative displacement t-24h->t (latest step only meaningful)
    dis24_lat = np.zeros_like(lat)
    dis24_lon = np.zeros_like(lon)
    dis24_lat[:,4] = lat[:,4]-lat[:,0]
    dis24_lon[:,4] = wrap_lon_delta(lon[:,0], lon[:,4])
    speed24 = np.zeros_like(lat)
    speed24[:,4] = haversine_km(lat[:,0], lon[:,0], lat[:,4], lon[:,4]) / 24.0
    # mean speed across last two 6h steps (recent trend)
    speed_mean2 = np.zeros_like(lat)
    speed_mean2[:,4] = 0.5*(speed[:,3]+speed[:,4])
    extra_names = ["acc_lat","acc_lon","turn_rate","dis24_lat","dis24_lon",
                   "speed24","speed_mean2"]
    stack = np.stack([acc_lat, acc_lon, turn_rate, dis24_lat, dis24_lon,
                      speed24, speed_mean2], axis=2).astype(np.float32)
    return np.concatenate([X, stack], axis=2), extra_names

def normalize_fit_train(Xtr, Ytr):
    """z-score using TRAIN only; return (norm_Xtr, norm_Ytr, stats)."""
    Xstats = (Xtr.reshape(Xtr.shape[0], -1).mean(axis=0).reshape(1,1,-1),
              Xtr.reshape(Xtr.shape[0], -1).std(axis=0).reshape(1,1,-1))
    Xmean, Xstd = Xstats
    Xstd[Xstd == 0] = 1.0
    Ymean = Ytr.reshape(Ytr.shape[0], -1).mean(axis=0)
    Ystd = Ytr.reshape(Ytr.shape[0], -1).std(axis=0)
    Ystd[Ystd == 0] = 1.0
    return {"Xmean":Xmean, "Xstd":Xstd, "Ymean":Ymean, "Ystd":Ystd}

def norm_X(X, s): return (X - s["Xmean"]) / s["Xstd"]
def norm_Y(Y, s): return (Y.reshape(Y.shape[0], -1) - s["Ymean"]) / s["Ystd"]
def denorm_Y(Yn, s): return (Yn * s["Ystd"]) + s["Ymean"]

def track_error_per_sample(pred_latlon, true_latlon):
    """pred_latlon/true_latlon: (N,3,2) -> (N,3) km per [6,12,24]."""
    return haversine_km(true_latlon[:,:,0], true_latlon[:,:,1],
                        pred_latlon[:,:,0], pred_latlon[:,:,1])

def summarize(pred_latlon, true_latlon, pred_wind=None, true_wind=None):
    """Return per-horizon {mean,median,std} track error + wind."""
    e = track_error_per_sample(pred_latlon, true_latlon)
    out = {}
    for hi, h in enumerate(HORIZONS):
        out[str(h)] = {
            "track_error_km_mean": float(np.mean(e[:,hi])),
            "track_error_km_median": float(np.median(e[:,hi])),
            "track_error_km_std": float(np.std(e[:,hi])),
        }
        if pred_wind is not None and true_wind is not None:
            out[str(h)]["wind_mae"] = float(np.mean(np.abs(pred_wind[:,hi]-true_wind[:,hi])))
    return out

def persistence_pred(X_raw):
    """(N,3,3) persistence: lat/lon/wind = t=0 for all horizons."""
    N = X_raw.shape[0]
    lat0 = X_raw[:,4,0]; lon0 = X_raw[:,4,1]; w0 = X_raw[:,4,2]
    out = np.stack([np.stack([np.full(N,0.0),]*3,axis=1),
                    np.full((N,3,3), np.nan)], axis=0) if False else None
    p = np.empty((N,3,3))
    p[:,:,0] = lat0[:,None]; p[:,:,1] = lon0[:,None]; p[:,:,2] = w0[:,None]
    return p

def movement_vector_pred(X_raw):
    """Movement-vector: linear extrapolation of last velocity (t-6h -> t)."""
    N = X_raw.shape[0]
    lat_t = X_raw[:,4,0]; lon_t = X_raw[:,4,1]
    lat_m = X_raw[:,3,0]; lon_m = X_raw[:,3,1]
    dlat = lat_t - lat_m
    dlon = wrap_lon_delta(lon_m, lon_t)
    w = X_raw[:,4,2]
    p = np.empty((N,3,3))
    for hi, h in enumerate(HORIZONS):
        m = MULT[h]
        p[:,hi,0] = lat_t + m*dlat
        p[:,hi,1] = (lon_t + m*dlon) % 360.0
        p[:,hi,2] = w
    return p

def kinematic_pred(X_raw):
    """Kinematic: linear + quadratic (constant acceleration), causal."""
    N = X_raw.shape[0]
    lat_t=X_raw[:,4,0]; lon_t=X_raw[:,4,1]
    lat_m=X_raw[:,3,0]; lon_m=X_raw[:,3,1]
    lat_2=X_raw[:,2,0]; lon_2=X_raw[:,2,1]
    v_lat=(lat_t-lat_m); v_lon=wrap_lon_delta(lon_m,lon_t)      # per 6h
    v2_lat=(lat_m-lat_2); v2_lon=wrap_lon_delta(lon_2,lon_m)
    a_lat=(v_lat-v2_lat); a_lon=(v_lon-v2_lon)                  # accel per 6h^2
    w=X_raw[:,4,2]
    p=np.empty((N,3,3))
    for hi,h in enumerate(HORIZONS):
        m=MULT[h]
        p[:,hi,0]=lat_t + m*v_lat + 0.5*(m**2)*a_lat
        p[:,hi,1]=(lon_t + m*v_lon + 0.5*(m**2)*a_lon) % 360.0
        p[:,hi,2]=w
    return p

def load_split_data(split, use_extra=True):
    """Return (X_engineered[(N,5,F)], Y[(N,3,3)], raw_x[(N,5,7)])."""
    X_raw, Y = load_raw(split)
    if use_extra:
        X_eng, _ = engineer_extra(X_raw)
        return X_eng, Y, X_raw
    return X_raw, Y, X_raw
