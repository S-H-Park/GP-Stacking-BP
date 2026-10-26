"""Multi-Kernel Gaussian Process Stacking for Boiling Point Prediction with Calibrated Uncertainty.
No data are included: pass your own CSV with SMILES and BP (K) columns via --data.
Usage: python gp_stacking_bp.py --stage gpu_all|baseline|tables|figures --data BP.csv [--out DIR] [--dry-run]"""
import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import glob
import multiprocessing as mp
import platform
import random
import signal
import time
import traceback
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
import gpytorch

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors

from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

PREFIX = "0930_"

FULL_KERNELS = ["RQ", "Matern2.5", "RBF+Linear"]
ABLATION_KERNELS = ["RQ", "RBF+Linear"]

_COMMON = {
    "target": "BP",
    "iqr_k": 1.5,
    "test_size": 0.15,
    "split_seed_rule": "seed",
    "vt_threshold": 0.0,
    "yj_standardize": True,
    "stacking_folds": 5,
    "stacking_kfold_seed_rule": "seed",
    "lr": 0.1,
    "max_iter": 500,
    "patience": 50,
    "min_delta": 0.001,
    "weight_optimizer": "SLSQP",
    "weight_maxiter": 1000,
    "weight_ftol": 1e-12,
}

PRESETS = {
    "main": {
        **_COMMON,
        "stage": "main",
        "kernels": FULL_KERNELS,
        "stacking_label": "Stacking-BMA",
    },
    "ablation": {
        **_COMMON,
        "stage": "ablation",
        "kernels": ABLATION_KERNELS,
        "stacking_label": "Stacking-BMA",
    },
    "baseline": {
        **_COMMON,
        "stage": "baseline",
        "baselines": {
            "LinearRegression": {},
            "RandomForest": {"n_estimators": 100, "n_jobs": 1},
            "GradientBoosting": {"n_estimators": 100},
        },
    },
    "cp": {
        **_COMMON,
        "stage": "cp",
        "kernels": FULL_KERNELS,
        "stacking_label": "Stacking",
        "outer_folds": 10,
        "outer_kfold_seed_rule": "seed",
        "fold_seed_rule": "seed*1000+fold_id",
        "inner_kfold_seed_rule": "seed",
        "alpha": 0.05,
        "quantile_rule": "k-th smallest score, k = ceil((n+1)(1-alpha)); inf if k > n",
        "final_model": "main job of the same seed (full 85 % training split)",
    },
}


def seed_everything(seed, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calc_r2_pred(y_true_test, y_pred_test, y_train_ref):
    ss_res = np.sum((y_true_test - y_pred_test) ** 2)
    ss_tot = np.sum((y_true_test - np.mean(y_train_ref)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def calc_ccc(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom < eps:
        return np.nan
    return float(2.0 * cov / denom)


def evaluate_prediction(y_true, y_pred, y_train_ref):
    return {
        "RMSE": rmse(y_true, y_pred),
        "R^2": float(r2_score(y_true, y_pred)),
        "R^2_pred": calc_r2_pred(y_true, y_pred, y_train_ref),
        "CCC": calc_ccc(y_true, y_pred),
    }


def coverage(y, lower, upper):
    y = np.asarray(y)
    return float(np.mean((y >= lower) & (y <= upper)))


def interval_stats(y, lower, upper):
    length = np.asarray(upper) - np.asarray(lower)
    return {
        "Coverage": coverage(y, lower, upper),
        "Mean_Interval_Length": float(np.mean(length)),
        "Median_Interval_Length": float(np.median(length)),
    }


def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.sort(scores[np.isfinite(scores)])
    n = len(scores)
    if n == 0:
        raise ValueError("No finite conformal scores are available.")

    # k-th smallest score (1-based), k = ceil((n+1)(1-alpha)); k > n gives an infinite quantile
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    qhat = np.inf if k > n else scores[k - 1]
    return float(qhat), float(k / n)


def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    res = []
    for _, func in Descriptors.descList:
        try:
            val = func(mol)
            res.append(val if np.isfinite(val) else np.nan)
        except Exception:
            res.append(np.nan)
    return res


def load_and_preprocess(csv_path, target_col, iqr_k=1.5):
    print(f"Data loading and cleaning: {csv_path}", flush=True)
    df_raw = pd.read_csv(csv_path)
    df = df_raw[["SMILES", target_col]].dropna()
    n_raw = len(df)

    y0 = df[target_col].values
    q1, q3 = np.percentile(y0, [25, 75])
    iqr = q3 - q1
    df = df[
        (df[target_col] >= q1 - iqr_k * iqr)
        & (df[target_col] <= q3 + iqr_k * iqr)
    ].reset_index(drop=True)
    n_iqr = len(df)

    print("RDKit descriptor calculation...", flush=True)
    desc_data, valid_idx = [], []
    for i, row in df.iterrows():
        desc = get_descriptors(row["SMILES"])
        if desc is not None:
            desc_data.append(desc)
            valid_idx.append(i)

    df_final = df.loc[valid_idx].reset_index(drop=True)
    desc_names = [n for n, _ in Descriptors.descList]
    desc_df = pd.DataFrame(desc_data, columns=desc_names)
    final_data = pd.concat([df_final, desc_df], axis=1).dropna().reset_index(drop=True)

    feature_names = [c for c in final_data.columns if c not in ("SMILES", target_col)]
    X_raw = final_data.drop(["SMILES", target_col], axis=1).values.astype(np.float64)
    y_raw = final_data[target_col].values.astype(np.float64)
    smiles = final_data["SMILES"].astype(str).to_numpy()

    info = {
        "n_raw_after_dropna": int(n_raw),
        "n_after_iqr": int(n_iqr),
        "n_final": int(len(y_raw)),
        "n_rdkit_descriptors": len(desc_names),
        "n_features_before_vt": int(X_raw.shape[1]),
        "rdkit_version": rdBase.rdkitVersion,
    }
    print(
        f"Preprocessing complete | samples={len(y_raw)} | "
        f"features before VT={X_raw.shape[1]}",
        flush=True,
    )
    return {
        "X_raw": X_raw,
        "y_raw": y_raw,
        "smiles": smiles,
        "feature_names": feature_names,
        "info": info,
    }


def load_data_cached(data_path, target, cache_dir):
    import hashlib

    with open(data_path, "rb") as fh:
        digest = hashlib.md5(fh.read()).hexdigest()[:10]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        f"{PREFIX}descriptors_{digest}_rdkit{rdBase.rdkitVersion}_desc{len(Descriptors.descList)}.npz",
    )
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        print(f"Loaded descriptor cache: {cache_path}", flush=True)
        return {
            "X_raw": z["X_raw"],
            "y_raw": z["y_raw"],
            "smiles": z["smiles"],
            "feature_names": list(z["feature_names"]),
            "info": json.loads(str(z["info"])),
        }

    data = load_and_preprocess(data_path, target)
    atomic_npz(
        cache_path,
        X_raw=data["X_raw"],
        y_raw=data["y_raw"],
        smiles=data["smiles"],
        feature_names=np.array(data["feature_names"]),
        info=json.dumps(data["info"]),
    )
    return data


def fit_feature_transform(X_fit_raw, X_pred_raw, feature_names, threshold=0.0):
    selector = VarianceThreshold(threshold=threshold)
    X_fit_v = selector.fit_transform(X_fit_raw)
    X_pred_v = selector.transform(X_pred_raw)

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit_v)
    X_pred = scaler.transform(X_pred_v)
    kept = [n for n, keep in zip(feature_names, selector.get_support()) if keep]
    return X_fit, X_pred, kept


def create_kernel(kernel_name):
    if kernel_name == "RQ":
        covar = gpytorch.kernels.RQKernel()
    elif kernel_name == "Matern2.5":
        covar = gpytorch.kernels.MaternKernel(nu=2.5)
    elif kernel_name == "RBF+Linear":
        covar = gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel()
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")
    return gpytorch.kernels.ScaleKernel(covar)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_name):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = create_kernel(kernel_name)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


def inverse_mean_and_sd(pt_y, mean_t, variance_t):
    mean_t = np.asarray(mean_t, dtype=np.float64)
    sd_t = np.sqrt(np.maximum(np.asarray(variance_t, dtype=np.float64), 1e-14))

    mean_n = pt_y.inverse_transform(mean_t.reshape(-1, 1)).ravel()
    # Kelvin SD = |T^-1(m+s) - T^-1(m-s)| / 2 (symmetric finite difference; interval ends are not back-transformed)
    low_n = pt_y.inverse_transform((mean_t - sd_t).reshape(-1, 1)).ravel()
    high_n = pt_y.inverse_transform((mean_t + sd_t).reshape(-1, 1)).ravel()
    sd_n = np.abs(high_n - low_n) / 2.0
    sd_n = np.maximum(sd_n, 1e-8)
    return mean_n, sd_n, sd_t


def yj_params(pt_y):
    scaler = getattr(pt_y, "_scaler", None)
    return {
        "lambda": float(pt_y.lambdas_[0]),
        "standardize": bool(pt_y.standardize),
        "scaler_mean": float(scaler.mean_[0]) if scaler is not None else None,
        "scaler_scale": float(scaler.scale_[0]) if scaler is not None else None,
    }


def train_eval_gp(X_train, y_train, X_pred, kernel_name, device, cfg):
    pt_y = PowerTransformer(method="yeo-johnson", standardize=cfg["yj_standardize"])
    y_tr_t = pt_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    tx = torch.tensor(X_train, dtype=torch.float64, device=device)
    ty = torch.tensor(y_tr_t, dtype=torch.float64, device=device)
    px = torch.tensor(X_pred, dtype=torch.float64, device=device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device).double()
    model = ExactGPModel(tx, ty, likelihood, kernel_name).to(device).double()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = float("inf")
    counter = 0
    n_iter = 0
    for _ in range(cfg["max_iter"]):
        n_iter += 1
        optimizer.zero_grad()
        loss = -mll(model(tx), ty)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        if loss_val < best_loss - cfg["min_delta"]:
            best_loss = loss_val
            counter = 0
        else:
            counter += 1
        if counter >= cfg["patience"]:
            break

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        dist = likelihood(model(px))
        mean_t = dist.mean.detach().cpu().numpy()
        variance_t = dist.variance.detach().cpu().numpy()

    mean_n, sd_n, sd_t = inverse_mean_and_sd(pt_y, mean_t, variance_t)

    del model, likelihood, tx, ty, px
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mean_t": mean_t,
        "sd_t": sd_t,
        "mean_k": mean_n,
        "sd_k": sd_n,
        "yj": yj_params(pt_y),
        "best_loss": best_loss,
        "n_iter": n_iter,
    }


def learn_stacking_weights(oof_pred, y_train, kernels, cfg):
    Z = np.column_stack([oof_pred[k] for k in kernels]).astype(np.float64)
    y = np.asarray(y_train, dtype=np.float64)

    mask = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    Z_fit = Z[mask]
    y_fit = y[mask]

    n_models = len(kernels)
    init_w = np.ones(n_models, dtype=np.float64) / n_models

    def objective(w):
        pred = Z_fit @ w
        return np.mean((y_fit - pred) ** 2)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    result = minimize(
        objective,
        init_w,
        method=cfg["weight_optimizer"],
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": cfg["weight_maxiter"], "ftol": cfg["weight_ftol"]},
    )

    if not result.success:
        print(f"Stacking weight optimisation failed: {result.message}. Equal weights used.", flush=True)
        return init_w, False

    weights = np.clip(result.x, 0.0, 1.0)
    if weights.sum() <= 0:
        return init_w, False
    weights /= weights.sum()
    return weights, True


def stacked_mean_sd(mean_dict, sd_dict, kernels, weights):
    means = np.vstack([mean_dict[k] for k in kernels])
    variances = np.vstack([sd_dict[k] ** 2 for k in kernels])
    mean_stack = np.sum(weights[:, None] * means, axis=0)
    variance_stack = np.sum(
        weights[:, None] * (variances + (means - mean_stack[None, :]) ** 2),
        axis=0,
    )
    return mean_stack, np.sqrt(np.maximum(variance_stack, 1e-14))


def fit_component_ensemble(X_fit_raw, y_fit, X_pred_raw, kernels, device, stacking_seed,
                           cfg, feature_names, log_prefix=""):
    X_fit, X_pred, kept = fit_feature_transform(
        X_fit_raw, X_pred_raw, feature_names, cfg["vt_threshold"]
    )

    n = len(y_fit)
    kf = KFold(n_splits=cfg["stacking_folds"], shuffle=True, random_state=stacking_seed)
    oof = {k: {q: np.full(n, np.nan) for q in ("mean_t", "sd_t", "mean_k", "sd_k")} for k in kernels}
    oof_fold = np.full(n, -1, dtype=int)
    fold_yj, fit_log = [], []

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_fit), start=1):
        print(f"{log_prefix} stacking fold {fold_i}/{cfg['stacking_folds']}", flush=True)
        oof_fold[val_idx] = fold_i
        for k in kernels:
            r = train_eval_gp(X_fit[tr_idx], y_fit[tr_idx], X_fit[val_idx], k, device, cfg)
            for q in ("mean_t", "sd_t", "mean_k", "sd_k"):
                oof[k][q][val_idx] = r[q]
            fit_log.append({"fold": fold_i, "kernel": k, "n_iter": r["n_iter"], "best_loss": r["best_loss"]})
        fold_yj.append({"fold": fold_i, **r["yj"]})

    final = {}
    final_yj = None
    for k in kernels:
        print(f"{log_prefix} final GP: {k}", flush=True)
        r = train_eval_gp(X_fit, y_fit, X_pred, k, device, cfg)
        final[k] = {q: r[q] for q in ("mean_t", "sd_t", "mean_k", "sd_k")}
        final_yj = r["yj"]
        fit_log.append({"fold": "final", "kernel": k, "n_iter": r["n_iter"], "best_loss": r["best_loss"]})

    return {
        "oof": oof,
        "oof_fold": oof_fold,
        "final": final,
        "kept_features": kept,
        "fold_yj": fold_yj,
        "final_yj": final_yj,
        "fit_log": fit_log,
    }


def stack_rows(seed, gpu_id, y_train, y_test, ens, kernels, label, cfg):
    oof_means = {k: ens["oof"][k]["mean_k"] for k in kernels}
    weights, ok = learn_stacking_weights(oof_means, y_train, kernels, cfg)
    mean_dict = {k: ens["final"][k]["mean_k"] for k in kernels}
    sd_dict = {k: ens["final"][k]["sd_k"] for k in kernels}
    stack_mean, stack_sd = stacked_mean_sd(mean_dict, sd_dict, kernels, weights)

    rows = [{
        "Model": label,
        "Seed": seed,
        "GPU": gpu_id,
        **evaluate_prediction(y_test, stack_mean, y_train),
        "Stacking_Weight": np.nan,
    }]
    for k, w in zip(kernels, weights):
        rows.append({
            "Model": k,
            "Seed": seed,
            "GPU": gpu_id,
            **evaluate_prediction(y_test, mean_dict[k], y_train),
            "Stacking_Weight": float(w),
        })
    return rows, weights, ok, stack_mean, stack_sd


def fit_baseline_models(X_train_raw, y_train, X_test_raw, y_test, seed, feature_names, cfg):
    X_train, X_test, _ = fit_feature_transform(X_train_raw, X_test_raw, feature_names, cfg["vt_threshold"])
    b = cfg["baselines"]
    models = {
        "LinearRegression": LinearRegression(**b["LinearRegression"]),
        "RandomForest": RandomForestRegressor(random_state=seed, **b["RandomForest"]),
        "GradientBoosting": GradientBoostingRegressor(random_state=seed, **b["GradientBoosting"]),
    }
    rows, preds = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        preds[name] = pred
        rows.append({
            "Model": name,
            "Seed": seed,
            **evaluate_prediction(y_test, pred, y_train),
            "Stacking_Weight": np.nan,
        })
    return rows, preds


def appendix_bp_summary(data_path, target_col="BP", smiles_col="SMILES"):
    bp = pd.read_csv(data_path)
    bp_raw = bp[[smiles_col, target_col]].copy()
    bp_raw = bp_raw.dropna(subset=[smiles_col, target_col]).copy()

    q1 = bp_raw[target_col].quantile(0.25)
    q3 = bp_raw[target_col].quantile(0.75)
    iqr = q3 - q1
    bp_iqr = bp_raw[(bp_raw[target_col] >= q1 - 1.5 * iqr) & (bp_raw[target_col] <= q3 + 1.5 * iqr)].copy()

    def is_valid_descriptor_row(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        try:
            vals = np.array([func(mol) for _, func in Descriptors.descList], dtype=np.float64)
            return bool(np.all(np.isfinite(vals)))
        except Exception:
            return False

    valid_mask = bp_iqr[smiles_col].apply(is_valid_descriptor_row)
    bp_cleaned = bp_iqr[valid_mask].copy().reset_index(drop=True)

    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    bp_cleaned_yj = bp_cleaned.copy()
    bp_cleaned_yj["BP_YJ"] = pt.fit_transform(bp_cleaned[[target_col]])

    def summarize_series(series):
        s = pd.Series(series).dropna()
        return pd.Series({
            "count": s.count(), "min": s.min(), "q1": s.quantile(0.25), "median": s.median(),
            "q3": s.quantile(0.75), "mean": s.mean(), "std": s.std(), "max": s.max(),
        })

    table = pd.DataFrame({
        "Raw": summarize_series(bp_raw[target_col]),
        "Cleaned": summarize_series(bp_cleaned[target_col]),
        "Cleaned + YJ": summarize_series(bp_cleaned_yj["BP_YJ"]),
    }).T[["count", "min", "q1", "median", "q3", "mean", "std", "max"]]
    return table, bp_raw, bp_cleaned, bp_cleaned_yj


class Paths:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.seeds = os.path.join(self.root, "seeds")
        self.tables = os.path.join(self.root, "tables")
        self.figures = os.path.join(self.root, "figures")
        self.logs = os.path.join(self.root, "logs")
        self.job_logs = os.path.join(self.logs, "jobs")
        self.cache = os.path.join(self.root, "cache")
        self.stop_file = os.path.join(self.root, f"{PREFIX}STOP")
        self.manifest = os.path.join(self.root, f"{PREFIX}manifest.csv")
        self.report = os.path.join(self.root, f"{PREFIX}revision_results.md")

    def makedirs(self):
        for d in (self.root, self.seeds, self.tables, self.figures, self.logs, self.job_logs, self.cache):
            os.makedirs(d, exist_ok=True)

    def seed(self, seed, suffix):
        return os.path.join(self.seeds, f"{PREFIX}seed{int(seed):03d}_{suffix}")

    def table(self, name, kind="values", ext="csv"):
        return os.path.join(self.tables, f"{PREFIX}{name}_{kind}.{ext}")

    def figure(self, name):
        return os.path.join(self.figures, f"{PREFIX}{name}")


def _tmp_path(path):
    return f"{path}.tmp{os.getpid()}"


def atomic_npz(path, **arrays):
    tmp = _tmp_path(path)
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def _json_default(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def atomic_json(path, obj):
    tmp = _tmp_path(path)
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    os.replace(tmp, path)


def atomic_csv(df, path, index=False):
    tmp = _tmp_path(path)
    df.to_csv(tmp, index=index, encoding="utf-8-sig")
    os.replace(tmp, path)


def atomic_text(path, text):
    tmp = _tmp_path(path)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def safe(name):
    return name.replace("+", "_plus_").replace(".", "p")


def get_split(n, seed, test_size):
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
    return train_idx, test_idx


def save_ensemble_arrays(arrays, prefix, ens, kernels):
    for k in kernels:
        for q in ("mean_t", "sd_t", "mean_k", "sd_k"):
            arrays[f"{prefix}oof_{safe(k)}_{q}"] = ens["oof"][k][q]
            arrays[f"{prefix}test_{safe(k)}_{q}"] = ens["final"][k][q]
    arrays[f"{prefix}oof_fold"] = ens["oof_fold"]


def run_point_seed(stage, seed, gpu_id, data, paths, opts):
    cfg = PRESETS[stage]
    t0 = time.time()
    seed_everything(seed, opts["deterministic"])

    use_gp = stage in ("main", "ablation")
    if gpu_id is not None and gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    else:
        device = torch.device("cpu")

    X_raw, y_raw, feats = data["X_raw"], data["y_raw"], data["feature_names"]
    train_idx, test_idx = get_split(len(y_raw), seed, cfg["test_size"])
    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test = y_raw[train_idx], y_raw[test_idx]

    split_path = paths.seed(seed, "split.npz")
    if not os.path.exists(split_path):
        atomic_npz(split_path, train_idx=train_idx, test_idx=test_idx)

    prefix = f"[{stage} | GPU {gpu_id} | seed {seed}]"
    print(f"{prefix} start | train={len(y_train)} | test={len(y_test)} | device={device}", flush=True)

    arrays = {"train_idx": train_idx, "test_idx": test_idx, "y_train": y_train, "y_test": y_test}
    meta = {
        "stage": stage,
        "preset": stage,
        "seed": seed,
        "gpu": gpu_id,
        "deterministic": opts["deterministic"],
    }
    rows = []

    if use_gp:
        ens = fit_component_ensemble(
            X_train_raw, y_train, X_test_raw, cfg["kernels"], device,
            stacking_seed=seed, cfg=cfg, feature_names=feats, log_prefix=prefix,
        )
        main_rows, weights, ok, stack_mean, stack_sd = stack_rows(
            seed, gpu_id, y_train, y_test, ens, cfg["kernels"], cfg["stacking_label"], cfg
        )
        for r in main_rows:
            r["Experiment"] = "main"
        rows.extend(main_rows)
        save_ensemble_arrays(arrays, "", ens, cfg["kernels"])
        arrays["weights"] = weights
        arrays["stack_test_mean_k"] = stack_mean
        arrays["stack_test_sd_k"] = stack_sd
        arrays["n_features_after_vt"] = np.array(len(ens["kept_features"]))
        meta.update({
            "kernels": cfg["kernels"],
            "weights": dict(zip(cfg["kernels"], map(float, weights))),
            "weight_optimisation_success": ok,
            "n_features_after_vt": len(ens["kept_features"]),
            "kept_features": ens["kept_features"],
            "yj_oof_folds": ens["fold_yj"],
            "yj_final": ens["final_yj"],
            "gp_fit_log": ens["fit_log"],
            "note_transformed_space": (
                "*_mean_t/*_sd_t are in the Yeo-Johnson(+standardised) space of the "
                "transformer fitted on that fold's training part; OOF values from "
                "different folds are in different transformed spaces (see oof_fold)."
            ),
        })

    if stage == "baseline":
        b_rows, b_preds = fit_baseline_models(X_train_raw, y_train, X_test_raw, y_test, seed, feats, cfg)
        for r in b_rows:
            r["Experiment"] = "ml_baseline"
            r["GPU"] = gpu_id
        rows.extend(b_rows)
        for name, pred in b_preds.items():
            arrays[f"baseline_test_{name}"] = pred

    meta["seconds"] = time.time() - t0
    atomic_npz(paths.seed(seed, f"{stage}.npz"), **arrays)
    atomic_json(paths.seed(seed, f"{stage}_meta.json"), meta)
    atomic_csv(pd.DataFrame(rows), paths.seed(seed, f"{stage}_metrics.csv"))
    print(f"{prefix} done in {meta['seconds']:.1f}s", flush=True)
    return meta["seconds"]


def run_cp_fold(seed, fold_id, gpu_id, data, paths, opts):
    cfg = PRESETS["cp"]
    X_raw, y_raw, feats = data["X_raw"], data["y_raw"], data["feature_names"]
    train85_idx, test15_idx = get_split(len(y_raw), seed, cfg["test_size"])
    X_train85_raw, y_train85 = X_raw[train85_idx], y_raw[train85_idx]
    X_test15_raw = X_raw[test15_idx]
    outer = list(KFold(n_splits=cfg["outer_folds"], shuffle=True, random_state=seed).split(X_train85_raw))
    train_idx, cal_idx = outer[fold_id - 1]

    fold_seed = seed * 1000 + fold_id
    seed_everything(fold_seed, opts["deterministic"])
    if gpu_id is not None and gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    prefix = f"[cp | seed {seed} | outer fold {fold_id} | GPU {gpu_id}]"
    print(f"{prefix} start | fit={len(train_idx)} | calibration={len(cal_idx)}", flush=True)
    t0 = time.time()
    n_cal = len(cal_idx)
    X_pred_raw = np.vstack([X_train85_raw[cal_idx], X_test15_raw])
    ens = fit_component_ensemble(
        X_train85_raw[train_idx], y_train85[train_idx], X_pred_raw, cfg["kernels"], device,
        stacking_seed=seed, cfg=cfg, feature_names=feats, log_prefix=prefix,
    )
    oof_means = {k: ens["oof"][k]["mean_k"] for k in cfg["kernels"]}
    weights, _ = learn_stacking_weights(oof_means, y_train85[train_idx], cfg["kernels"], cfg)
    mean_dict = {k: ens["final"][k]["mean_k"] for k in cfg["kernels"]}
    sd_dict = {k: ens["final"][k]["sd_k"] for k in cfg["kernels"]}
    pred_mean, pred_sd = stacked_mean_sd(mean_dict, sd_dict, cfg["kernels"], weights)

    out = {
        "fold_id": fold_id,
        "gpu_id": gpu_id,
        "train_idx": train_idx,
        "cal_idx": cal_idx,
        "weights": weights,
        "cal_mean": pred_mean[:n_cal],
        "cal_sd": pred_sd[:n_cal],
        "test_mean": pred_mean[n_cal:],
        "test_sd": pred_sd[n_cal:],
        "seconds": time.time() - t0,
    }
    for k in cfg["kernels"]:
        for q in ("mean_t", "sd_t", "mean_k", "sd_k"):
            out[f"cal_{safe(k)}_{q}"] = ens["final"][k][q][:n_cal]
            out[f"test_{safe(k)}_{q}"] = ens["final"][k][q][n_cal:]
    atomic_json(paths.seed(seed, f"cp_fold{fold_id:02d}_meta.json"), {
        "fold_seed": fold_seed,
        "kept_features": ens["kept_features"],
        "yj_inner_folds": ens["fold_yj"],
        "yj_final": ens["final_yj"],
        "gp_fit_log": ens["fit_log"],
        "seconds": out["seconds"],
    })
    atomic_npz(paths.seed(seed, f"cp_fold{fold_id:02d}.npz"), **out)
    print(f"{prefix} done in {out['seconds']:.1f}s", flush=True)


def cp_ready(paths, seed):
    cfg = PRESETS["cp"]
    folds = all(os.path.exists(paths.seed(seed, f"cp_fold{f:02d}.npz")) for f in range(1, cfg["outer_folds"] + 1))
    return folds and os.path.exists(paths.seed(seed, "main_metrics.csv"))


def assemble_cp(seed, data, paths):
    cfg = PRESETS["cp"]
    y_raw, smiles = data["y_raw"], data["smiles"]
    train_idx, test_idx = get_split(len(y_raw), seed, cfg["test_size"])
    y_train85, y_test15 = y_raw[train_idx], y_raw[test_idx]
    n_tr, n_te = len(y_train85), len(y_test15)

    oof_mean = np.full(n_tr, np.nan)
    oof_sd = np.full(n_tr, np.nan)
    oof_fold = np.full(n_tr, -1, dtype=int)
    fold_test_mean = np.zeros((cfg["outer_folds"], n_te))
    fold_test_sd = np.zeros((cfg["outer_folds"], n_te))
    weight_rows, fold_seconds = [], {}
    for fid in range(1, cfg["outer_folds"] + 1):
        r = np.load(paths.seed(seed, f"cp_fold{fid:02d}.npz"))
        idx = r["cal_idx"]
        oof_mean[idx] = r["cal_mean"]
        oof_sd[idx] = r["cal_sd"]
        oof_fold[idx] = fid
        fold_test_mean[fid - 1] = r["test_mean"]
        fold_test_sd[fid - 1] = r["test_sd"]
        fold_seconds[fid] = float(r["seconds"])
        weight_rows.append({"CVCP_Fold": fid, "GPU": int(r["gpu_id"]),
                            **{f"Weight_{k}": float(w) for k, w in zip(cfg["kernels"], r["weights"])}})

    if not (np.all(np.isfinite(oof_mean)) and np.all(np.isfinite(oof_sd))):
        raise RuntimeError("Non-finite OOF predictions or SDs found after CV-CP folds.")

    abs_scores = np.abs(y_train85 - oof_mean)
    normalized_scores = abs_scores / np.maximum(oof_sd, 1e-8)
    q_abs, abs_level = conformal_quantile(abs_scores, cfg["alpha"])
    q_norm, norm_level = conformal_quantile(normalized_scores, cfg["alpha"])

    main = np.load(paths.seed(seed, "main.npz"))
    main_meta = json.load(open(paths.seed(seed, "main_meta.json")))
    assert np.array_equal(main["test_idx"], test_idx)
    test_mean = main["stack_test_mean_k"]
    test_sd = main["stack_test_sd_k"]
    final_weights = main["weights"]

    z = float(norm.ppf(1.0 - cfg["alpha"] / 2.0))
    native_lower, native_upper = test_mean - z * test_sd, test_mean + z * test_sd
    absolute_lower, absolute_upper = test_mean - q_abs, test_mean + q_abs
    normalized_lower, normalized_upper = test_mean - q_norm * test_sd, test_mean + q_norm * test_sd

    test_df = pd.DataFrame({
        "Test_Index": np.arange(n_te),
        "Dataset_Index": test_idx,
        "SMILES": smiles[test_idx],
        "Y_True": y_test15,
        "Stacking_Mean": test_mean,
        "Stacking_SD": test_sd,
        "Native_Lower": native_lower,
        "Native_Upper": native_upper,
        "CVCP_Absolute_Lower": absolute_lower,
        "CVCP_Absolute_Upper": absolute_upper,
        "CVCP_Normalized_Lower": normalized_lower,
        "CVCP_Normalized_Upper": normalized_upper,
        "Native_Covered": (y_test15 >= native_lower) & (y_test15 <= native_upper),
        "CVCP_Absolute_Covered": (y_test15 >= absolute_lower) & (y_test15 <= absolute_upper),
        "CVCP_Normalized_Covered": (y_test15 >= normalized_lower) & (y_test15 <= normalized_upper),
    })
    for k, w in zip(cfg["kernels"], final_weights):
        test_df[f"Final_Weight_{k}"] = w
        test_df[f"Mean_{k}"] = main[f"test_{safe(k)}_mean_k"]
        test_df[f"SD_{k}"] = main[f"test_{safe(k)}_sd_k"]
    atomic_csv(test_df, paths.seed(seed, "cp_test_predictions.csv"))

    atomic_csv(pd.DataFrame({
        "Train85_Index": np.arange(n_tr),
        "Dataset_Index": train_idx,
        "CVCP_Fold": oof_fold,
        "Y_True": y_train85,
        "OOF_Stacking_Mean": oof_mean,
        "OOF_Stacking_SD": oof_sd,
        "Absolute_Residual": abs_scores,
        "Normalized_Score": normalized_scores,
    }), paths.seed(seed, "cp_calibration_scores.csv"))
    atomic_csv(pd.DataFrame(weight_rows), paths.seed(seed, "cp_fold_weights.csv"))
    atomic_npz(
        paths.seed(seed, "cp.npz"),
        oof_mean=oof_mean, oof_sd=oof_sd, oof_fold=oof_fold,
        abs_scores=abs_scores, normalized_scores=normalized_scores,
        fold_test_mean=fold_test_mean, fold_test_sd=fold_test_sd,
        q_abs=q_abs, q_norm=q_norm, abs_level=abs_level, norm_level=norm_level,
    )

    point = evaluate_prediction(y_test15, test_mean, y_train85)
    point["MAE"] = float(np.mean(np.abs(y_test15 - test_mean)))
    rows = []
    for method, lo, hi, qh, ql in [
        ("Native_95", native_lower, native_upper, np.nan, np.nan),
        ("CVCP_Absolute_95", absolute_lower, absolute_upper, q_abs, abs_level),
        ("CVCP_Normalized_95", normalized_lower, normalized_upper, q_norm, norm_level),
    ]:
        row = {"Seed": seed, "Method": method, "Nominal_Coverage": 1.0 - cfg["alpha"],
               "Qhat": qh, "Quantile_Level": ql, **point, **interval_stats(y_test15, lo, hi)}
        for k, w in zip(cfg["kernels"], final_weights):
            row[f"Final_Weight_{k}"] = float(w)
        rows.append(row)
    atomic_json(paths.seed(seed, "cp_meta.json"), {
        "seed": seed,
        "outer_folds": cfg["outer_folds"],
        "fold_seconds": fold_seconds,
        "final_model_from": "main job of this seed (seed_everything(seed) + stacking on train85)",
        "main_weights": main_meta["weights"],
    })
    atomic_csv(pd.DataFrame(rows), paths.seed(seed, "cp_metrics.csv"))
    print(f"[cp] seed {seed} assembled", flush=True)


EST_SECONDS = {"main": 430, "ablation": 320, "cp_fold": 460, "baseline": 8}


def job_name(job):
    name = f"{job['kind']}_seed{job['seed']:03d}"
    if job["kind"] == "cp_fold":
        name += f"_fold{job['fold']:02d}"
    return name


def job_done(paths, job):
    if job["kind"] == "cp_fold":
        return os.path.exists(paths.seed(job["seed"], f"cp_fold{job['fold']:02d}.npz"))
    return os.path.exists(paths.seed(job["seed"], f"{job['kind']}_metrics.csv"))


def job_est(job, opts):
    return EST_SECONDS[job["kind"]]


def build_jobs(stage, seeds, cp_seeds, figure_seed, order):
    jobs = {}

    def add(kind, seed, fold=None):
        job = {"kind": kind, "seed": int(seed)}
        if fold is not None:
            job["fold"] = int(fold)
        jobs.setdefault(job_name(job), job)

    if stage in ("main", "gpu_all"):
        for s in seeds:
            add("main", s)
    if stage in ("ablation", "gpu_all"):
        for s in seeds:
            add("ablation", s)
    if stage in ("cp", "gpu_all"):
        for s in sorted(cp_seeds, key=lambda x: (x != figure_seed, x)):
            add("main", s)
            for f in range(1, PRESETS["cp"]["outer_folds"] + 1):
                add("cp_fold", s, f)
    if stage == "baseline":
        for s in seeds:
            add("baseline", s)

    job_list = list(jobs.values())
    rank = {"main": 0, "ablation": 1, "cp_fold": 2, "baseline": 3}
    if order == "lpt":
        job_list.sort(key=lambda j: -EST_SECONDS.get(j["kind"], 0))
    else:
        job_list.sort(key=lambda j: rank[j["kind"]])
    return job_list


def job_entry(job, root, data_path, gpu_id, opts, log_path):
    fh = open(log_path, "a", buffering=1)
    os.dup2(fh.fileno(), 1)
    os.dup2(fh.fileno(), 2)
    sys.stdout = fh
    sys.stderr = fh
    paths = Paths(root)
    err_path = paths.seed(job["seed"], f"error_{job_name(job)}.json")
    try:
        data = load_data_cached(data_path, "BP", paths.cache)
        if job["kind"] in ("main", "ablation", "baseline"):
            run_point_seed(job["kind"], job["seed"], gpu_id, data, paths, opts)
        elif job["kind"] == "cp_fold":
            run_cp_fold(job["seed"], job["fold"], gpu_id, data, paths, opts)
        else:
            raise ValueError(f"unknown job kind {job['kind']}")
        if os.path.exists(err_path):
            os.remove(err_path)
    except Exception as exc:
        err = traceback.format_exc()
        print(err, flush=True)
        atomic_json(err_path, {"job": job_name(job), "gpu": gpu_id, "error": str(exc), "traceback": err})
        fh.flush()
        sys.exit(1)
    fh.flush()


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stale_tmp_files(paths):
    stale = []
    for p in glob.glob(os.path.join(paths.seeds, f"{PREFIX}*.tmp*")):
        tail = p.rsplit(".tmp", 1)[-1]
        if not tail.isdigit() or not _pid_alive(int(tail)):
            stale.append(p)
    return stale


def clean_stale_tmp(paths):
    stale = stale_tmp_files(paths)
    for p in stale:
        os.remove(p)
    return len(stale)


def make_lanes(stage, gpus, tasks_per_gpu, opts):
    if stage == "baseline":
        return [-1] * max(1, opts["cpu_workers"])
    if not gpus:
        return [-1]
    return [g for _ in range(tasks_per_gpu) for g in gpus]


def simulate_plan(jobs, lanes, tasks_per_gpu, opts):
    slow = tasks_per_gpu if lanes and lanes[0] >= 0 else 1
    free_at = [0.0] * len(lanes)
    plan, finish = [], {}
    for job in jobs:
        li = int(np.argmin(free_at))
        start = free_at[li]
        free_at[li] += job_est(job, opts) * slow
        plan.append((job_name(job), lanes[li], start, free_at[li]))
        finish[job["kind"]] = max(finish.get(job["kind"], 0.0), free_at[li])
    return plan, (max(free_at) if free_at else 0.0), finish


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class _Terminate(Exception):
    pass


def run_scheduler(stage, jobs_all, lanes, paths, data_path, opts, cp_seeds, dry_run=False):
    done_before = [j for j in jobs_all if job_done(paths, j)]
    todo = [j for j in jobs_all if not job_done(paths, j)]
    kinds = Counter(j["kind"] for j in jobs_all)
    kinds_todo = Counter(j["kind"] for j in todo)
    stale = len(stale_tmp_files(paths))
    per_gpu = Counter(lanes)
    tpg = max(per_gpu.values()) if per_gpu else 1

    print(f"{ts()} | PLAN | stage={stage} | lanes={dict(per_gpu)} (device: slots; -1 = CPU)", flush=True)
    for k in kinds:
        print(f"{ts()} | PLAN | {k:9s}: total {kinds[k]:5d} | already done {kinds[k] - kinds_todo[k]:5d} | "
              f"to run {kinds_todo[k]:5d}", flush=True)
    print(f"{ts()} | PLAN | jobs total {len(jobs_all)} | skip (done) {len(done_before)} | remaining {len(todo)} | "
          f"stale temp files {stale}", flush=True)

    plan, makespan, finish = simulate_plan(todo, lanes, tpg, opts)
    by_lane = Counter(p[1] for p in plan)
    print(f"{ts()} | PLAN | jobs per device: {dict(sorted(by_lane.items()))}", flush=True)
    for k, t in finish.items():
        print(f"{ts()} | PLAN | est. last {k} job finishes after {t / 3600:.1f} h", flush=True)
    share = (f"x{tpg} when {tpg} jobs share a GPU" if lanes and lanes[0] >= 0
             else f"{len(lanes)} parallel CPU workers")
    print(f"{ts()} | PLAN | est. total wall time {makespan / 3600:.1f} h "
          f"(single-job times {EST_SECONDS}, {share})", flush=True)
    for name, lane, start, end in plan[:12]:
        print(f"{ts()} | PLAN |   {name:28s} -> device {lane:2d} | est. start {start / 3600:5.2f} h", flush=True)
    if len(plan) > 12:
        print(f"{ts()} | PLAN |   ... {len(plan) - 12} more", flush=True)
    if dry_run:
        print(f"{ts()} | DRY-RUN | no job started", flush=True)
        return []

    paths.makedirs()
    removed = clean_stale_tmp(paths)
    if removed:
        print(f"{ts()} | removed {removed} stale temp files from interrupted jobs", flush=True)
    if os.path.exists(paths.stop_file):
        os.remove(paths.stop_file)
        print(f"{ts()} | removed old STOP file", flush=True)
    pid_path = os.path.join(paths.logs, f"{PREFIX}{stage}.pid")
    atomic_text(pid_path, f"{os.getpid()}\n")

    data = load_data_cached(data_path, "BP", paths.cache)
    assembled = set()

    def try_assemble(seed):
        if seed in assembled or seed not in cp_seeds:
            return
        if os.path.exists(paths.seed(seed, "cp_metrics.csv")):
            assembled.add(seed)
            return
        if cp_ready(paths, seed):
            try:
                assemble_cp(seed, data, paths)
                assembled.add(seed)
                print(f"{ts()} | CP-ASSEMBLED | seed {seed:03d}", flush=True)
            except Exception as exc:
                print(f"{ts()} | CP-ASSEMBLY-FAILED | seed {seed:03d} | {exc}", flush=True)

    for s in cp_seeds:
        try_assemble(s)

    ctx = mp.get_context("spawn")
    pending = list(todo)
    running = {}
    n_total, n_done = len(todo), 0
    failures = []

    def on_term(signum, frame):
        raise _Terminate()

    signal.signal(signal.SIGTERM, on_term)
    stopping = False
    try:
        while pending or running:
            if not stopping and os.path.exists(paths.stop_file):
                stopping = True
                print(f"{ts()} | STOP file found: no new jobs; waiting for {len(running)} running job(s)", flush=True)
            if not stopping:
                for li, gpu in enumerate(lanes):
                    if li in running or not pending:
                        continue
                    job = pending.pop(0)
                    if job_done(paths, job):
                        n_done += 1
                        continue
                    log_path = os.path.join(paths.job_logs, f"{PREFIX}{job_name(job)}.log")
                    p = ctx.Process(target=job_entry, args=(job, paths.root, data_path, gpu, opts, log_path),
                                    name=job_name(job))
                    p.start()
                    running[li] = (p, job, time.time())
                    print(f"{ts()} | START | device {gpu:2d} | {job_name(job):28s} | "
                          f"done {n_done}/{n_total} | running {len(running)}", flush=True)
            for li, (p, job, t0) in list(running.items()):
                if p.is_alive():
                    continue
                p.join()
                del running[li]
                n_done += 1
                dt = time.time() - t0
                ok = p.exitcode == 0 and job_done(paths, job)
                status = "END-OK" if ok else f"FAILED(exit={p.exitcode})"
                print(f"{ts()} | {status} | device {lanes[li]:2d} | {job_name(job):28s} | {dt:7.1f}s | "
                      f"done {n_done}/{n_total} | running {len(running)}", flush=True)
                if not ok:
                    failures.append({"job": job_name(job), "device": lanes[li], "exitcode": p.exitcode,
                                     "log": os.path.join(paths.job_logs, f"{PREFIX}{job_name(job)}.log")})
                if job["kind"] in ("main", "cp_fold"):
                    try_assemble(job["seed"])
            if stopping and not running:
                break
            time.sleep(opts["poll"])
    except (_Terminate, KeyboardInterrupt):
        print(f"{ts()} | TERMINATED: stopping {len(running)} running job(s); their partial files are "
              f"temporary and will be removed on restart", flush=True)
        for p, job, _ in running.values():
            p.terminate()
        for p, job, _ in running.values():
            p.join(30)
        raise SystemExit(1)

    for s in cp_seeds:
        try_assemble(s)
    left = len(pending)
    print(f"{ts()} | FINISHED | stage={stage} | completed {n_done}/{n_total} | failed {len(failures)} | "
          f"not started {left}", flush=True)
    if failures:
        fail_path = os.path.join(paths.logs, f"{PREFIX}{stage}_failures.csv")
        atomic_csv(pd.DataFrame(failures), fail_path)
        print(f"{ts()} | FAILED JOBS ({len(failures)}), see {fail_path}:", flush=True)
        for f in failures:
            print(f"    {f['job']} (device {f['device']}, exit {f['exitcode']}) log: {f['log']}", flush=True)
    return failures


def load_stage_metrics(paths, stage, seeds):
    frames = [pd.read_csv(p) for p in (paths.seed(s, f"{stage}_metrics.csv") for s in seeds) if os.path.exists(p)]
    return pd.concat(frames, ignore_index=True) if frames else None


def describe(values):
    v = np.asarray([x for x in values if pd.notna(x) and np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"n_seeds": 0, "mean": np.nan, "sd": np.nan, "se": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
    sd = float(np.std(v, ddof=1)) if n > 1 else np.nan
    return {
        "n_seeds": n,
        "mean": float(np.mean(v)),
        "sd": sd,
        "se": sd / np.sqrt(n) if n > 1 else np.nan,
        "median": float(np.median(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def fmt_mean_se(mean, se, digits, scale=1.0):
    if pd.isna(mean):
        return "-"
    se_txt = "nan" if pd.isna(se) else f"{se * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f} ± {se_txt}"


def df_to_md(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:.6g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_table(paths, name, values_df, formatted_df=None, registry=None, title=None):
    atomic_csv(values_df, paths.table(name, "values"))
    files = [paths.table(name, "values")]
    if formatted_df is not None:
        atomic_csv(formatted_df, paths.table(name, "formatted"))
        md = (f"### {title}\n\n" if title else "") + df_to_md(formatted_df) + "\n"
        atomic_text(paths.table(name, "formatted", "md"), md)
        files += [paths.table(name, "formatted"), paths.table(name, "formatted", "md")]
    if registry is not None:
        registry[name] = {"files": files, "formatted": formatted_df, "title": title or name}
    return files


METRIC_LABELS = {"RMSE": "RMSE (K)", "R^2": "R²", "R^2_pred": "R²pred", "CCC": "CCC"}
MODEL_LABELS = {
    "Stacking-BMA": "Stacking (RQ + Matérn + RBF+Linear)",
    "Stacking": "Stacking (RQ + Matérn + RBF+Linear)",
    "RQ": "RQ",
    "Matern2.5": "Matérn (ν = 2.5)",
    "RBF+Linear": "RBF+Linear",
    "GradientBoosting": "Gradient boosting",
    "RandomForest": "Random forest",
    "LinearRegression": "Linear regression",
}


def metric_table(df, models, metrics, digits=4, source="", label_map=None):
    label_map = label_map or MODEL_LABELS
    vals, fmt_rows = [], []
    for m in models:
        sub = df[df["Model"] == m]
        frow = {"Model": label_map.get(m, m)}
        for met in metrics:
            d = describe(sub[met].tolist())
            vals.append({"model": m, "metric": met, **d, "source": source})
            frow[f"{METRIC_LABELS.get(met, met)} (mean ± SE, n = {d['n_seeds']})"] = fmt_mean_se(d["mean"], d["se"], digits)
        fmt_rows.append(frow)
    return pd.DataFrame(vals), pd.DataFrame(fmt_rows)


def t_ci(values, conf=0.95):
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return np.nan, np.nan
    m, s = v.mean(), v.std(ddof=1)
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * s / np.sqrt(n)
    return m - h, m + h


def nadeau_bengio(delta, test_train_ratio=0.15 / 0.85):
    d = np.asarray(delta, dtype=float)
    n = len(d)
    if n < 2 or np.std(d, ddof=1) == 0:
        return np.nan, np.nan, (np.nan, np.nan)
    var = np.var(d, ddof=1)
    se = np.sqrt((1.0 / n + test_train_ratio) * var)
    t = d.mean() / se
    p = 2 * stats.t.sf(abs(t), n - 1)
    h = stats.t.ppf(0.975, n - 1) * se
    return float(t), float(p), (d.mean() - h, d.mean() + h)


def wilcoxon_p(delta):
    d = np.asarray(delta, dtype=float)
    if np.count_nonzero(d) < 1:
        return np.nan
    try:
        return float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        return np.nan


def paper_tables(paths, reg, data_path, seeds):
    main = load_stage_metrics(paths, "main", seeds)
    abl = load_stage_metrics(paths, "ablation", seeds)
    base = load_stage_metrics(paths, "baseline", seeds)
    metrics = ["RMSE", "R^2", "R^2_pred", "CCC"]

    if main is not None:
        v, f = metric_table(main, ["Stacking-BMA"] + FULL_KERNELS, metrics, source="main")
        save_table(paths, "table3_stacking_vs_kernels", v, f, reg,
                   "Table 3. Stacking vs single kernels (± = SE over seeds)")
        vals, fmt = [], []
        for k in FULL_KERNELS:
            d = describe(main.loc[main["Model"] == k, "Stacking_Weight"].tolist())
            vals.append({"model": k, "metric": "stacking_weight", **d, "source": "main"})
            fmt.append({"Kernel": MODEL_LABELS[k],
                        f"Stacking weight (mean ± SE, n = {d['n_seeds']})": fmt_mean_se(d["mean"], d["se"], 4),
                        "Seeds with weight < 1e-6": int((main.loc[main["Model"] == k, "Stacking_Weight"] < 1e-6).sum())})
        save_table(paths, "table5_stacking_weights", pd.DataFrame(vals), pd.DataFrame(fmt), reg,
                   "Table 5. Stacking weights (± = SE over seeds)")

    if base is not None:
        ml = base[base["Experiment"] == "ml_baseline"].copy()
        frames = [ml]
        if main is not None:
            st = main[main["Model"] == "Stacking-BMA"].copy()
            frames.insert(0, st)
        comb = pd.concat(frames, ignore_index=True)
        v, f = metric_table(comb, ["Stacking-BMA", "GradientBoosting", "RandomForest", "LinearRegression"], metrics,
                            source="Stacking: main; baselines: baseline")
        save_table(paths, "table4_stacking_vs_baselines", v, f, reg,
                   "Table 4. Stacking vs ML baselines (± = SE over seeds)")
        per_seed = ml.pivot_table(index="Seed", columns="Model", values=metrics)
        per_seed.columns = [f"{m}_{met}" for met, m in per_seed.columns]
        per_seed = per_seed.reset_index()
        atomic_csv(per_seed, os.path.join(paths.tables, f"{PREFIX}table4_baselines_per_seed.csv"))
        reg["table4_stacking_vs_baselines"]["files"].append(os.path.join(paths.tables, f"{PREFIX}table4_baselines_per_seed.csv"))

    if abl is not None:
        v, f = metric_table(abl, ["Stacking-BMA"] + ABLATION_KERNELS, metrics, source="ablation",
                            label_map={**MODEL_LABELS, "Stacking-BMA": "Stacking (RQ + RBF+Linear)"})
        save_table(paths, "table6_matern_ablation", v, f, reg,
                   "Table 6. Matérn ablation, GPs refitted without Matérn (± = SE over seeds)")

    table, *_ = appendix_bp_summary(data_path)
    vals = table.reset_index().rename(columns={"index": "dataset"})
    c = table.round(2).loc["Cleaned"]
    fmt = pd.DataFrame([{
        "Min (K)": f"{c['min']:.2f}", "Q1 (K)": f"{c['q1']:.2f}", "Median (K)": f"{c['median']:.2f}",
        "Q3 (K)": f"{c['q3']:.2f}", "Max (K)": f"{c['max']:.2f}", "Mean (K)": f"{c['mean']:.2f}",
        "SD (K)": f"{c['std']:.2f}", "N": int(c["count"]),
    }])
    save_table(paths, "tableA1_bp_summary", vals, fmt, reg, "Table A.1. Boiling point summary of the curated data set")
    return main, abl, base


CP_METHODS = [("Native_95", "Native (±1.96σ)"),
              ("CVCP_Normalized_95", "CP-normalized (|y−μ|/σ, paper)"),
              ("CVCP_Absolute_95", "CP-absolute (|y−μ|)")]


def r2_cp(paths, reg, seeds):
    frames = [pd.read_csv(p) for p in (paths.seed(s, "cp_metrics.csv") for s in seeds) if os.path.exists(p)]
    if not frames:
        print("R2: no CP results yet, skipped", flush=True)
        return None
    cp = pd.concat(frames, ignore_index=True)
    per_seed = cp[["Seed", "Method", "Coverage", "Mean_Interval_Length", "Median_Interval_Length", "Qhat",
                   "Quantile_Level", "RMSE"]].rename(columns=str.lower)
    save_table(paths, "rev_cp_per_seed", per_seed, None, reg, "R2. CP results per seed")
    vals, fmt = [], []
    for m, label in CP_METHODS:
        sub = cp[cp["Method"] == m]
        row = {"Method": label}
        for q, qlab, scale, dig in [("Coverage", "Coverage (%)", 100, 2),
                                    ("Mean_Interval_Length", "Mean interval length (K)", 1, 1),
                                    ("Median_Interval_Length", "Median interval length (K)", 1, 1)]:
            d = describe(sub[q].tolist())
            extra = {}
            if q == "Coverage":
                extra["frac_seeds_coverage_ge_0.95"] = float(np.mean(sub[q] >= 0.95)) if len(sub) else np.nan
            vals.append({"method": m, "quantity": q, **d, **extra})
            row[f"{qlab} (mean ± SE, n = {d['n_seeds']})"] = fmt_mean_se(d["mean"], d["se"], dig, scale)
            if q == "Coverage":
                row["Coverage SD (%)"] = f"{d['sd'] * 100:.2f}" if pd.notna(d["sd"]) else "-"
                row["Coverage range (%)"] = f"{d['min'] * 100:.2f}–{d['max'] * 100:.2f}"
                row["Seeds with coverage ≥ 95%"] = f"{extra['frac_seeds_coverage_ge_0.95'] * 100:.0f}%"
        s42 = sub[sub["Seed"] == 42]
        if len(s42):
            row["Seed 42 (coverage %, mean length K)"] = f"{s42['Coverage'].iloc[0] * 100:.2f}, {s42['Mean_Interval_Length'].iloc[0]:.1f}"
        fmt.append(row)
    save_table(paths, "rev_cp_100seed_summary", pd.DataFrame(vals), pd.DataFrame(fmt), reg,
               "R2. Native vs CP intervals over seeds (stacking predictions, ± = SE)")
    return cp


def r3_ablation(paths, reg, main, abl):
    if main is None:
        return None
    metrics = ["RMSE", "R^2", "R^2_pred", "CCC"]
    full = main[main["Model"] == "Stacking-BMA"].set_index("Seed")[metrics]
    per_seed = []

    if abl is not None:
        a = abl[abl["Model"] == "Stacking-BMA"].set_index("Seed")[metrics]
        common = full.index.intersection(a.index)
        for s in common:
            for met in metrics:
                per_seed.append({"version": "a_separate_refit", "seed": int(s), "metric": met,
                                 "full": full.loc[s, met], "ablated": a.loc[s, met]})

    cfg = PRESETS["main"]
    for s in full.index:
        p = paths.seed(s, "main.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        oof = {k: z[f"oof_{safe(k)}_mean_k"] for k in ABLATION_KERNELS}
        w, _ = learn_stacking_weights(oof, z["y_train"], ABLATION_KERNELS, cfg)
        pred = sum(wi * z[f"test_{safe(k)}_mean_k"] for wi, k in zip(w, ABLATION_KERNELS))
        ev = evaluate_prediction(z["y_test"], pred, z["y_train"])
        for met in metrics:
            per_seed.append({"version": "b_no_refit_from_main", "seed": int(s), "metric": met,
                             "full": full.loc[s, met], "ablated": ev[met],
                             "weight_RQ": float(w[0]), "weight_RBF+Linear": float(w[1])})

    ps = pd.DataFrame(per_seed)
    if ps.empty:
        return None
    ps["delta_ablated_minus_full"] = ps["ablated"] - ps["full"]
    tol = 1e-8
    ps["full_better"] = np.where(ps["metric"] == "RMSE", ps["delta_ablated_minus_full"] > tol,
                                 ps["delta_ablated_minus_full"] < -tol)
    ps["tie"] = ps["delta_ablated_minus_full"].abs() <= tol
    save_table(paths, "rev_matern_ablation_per_seed", ps, None, reg, "R3. Matérn ablation per seed")

    vals, fmt = [], []
    for ver, g in ps.groupby("version"):
        for met in metrics:
            d = g[g["metric"] == met]
            delta = d["delta_ablated_minus_full"].to_numpy(float)
            delta_w = np.where(np.abs(delta) <= tol, 0.0, delta)
            n = len(delta)
            lo, hi = t_ci(delta)
            t_nb, p_nb, (nb_lo, nb_hi) = nadeau_bengio(delta)
            wins, ties = int(d["full_better"].sum()), int(d["tie"].sum())
            row = {"version": ver, "metric": met, "n_seeds": n, "full_wins": wins, "ties": ties,
                   "ablated_wins": n - wins - ties, "full_win_rate": wins / n if n else np.nan,
                   "delta_mean": float(np.mean(delta)), "delta_sd": float(np.std(delta, ddof=1)) if n > 1 else np.nan,
                   "delta_ci95_low": lo, "delta_ci95_high": hi, "delta_median": float(np.median(delta)),
                   "wilcoxon_p": wilcoxon_p(delta_w), "nb_t": t_nb, "nb_p": p_nb,
                   "nb_ci95_low": nb_lo, "nb_ci95_high": nb_hi}
            vals.append(row)
            dig = 4
            fmt.append({
                "Version": "(a) separate refit (ablation stage)" if ver.startswith("a") else "(b) no refit (weights re-solved)",
                "Metric": METRIC_LABELS[met],
                "n": n,
                "Full stacking better": f"{wins}/{n} ({wins / n * 100:.0f}%)" if n else "-",
                "Ties": ties,
                "Δ mean (ablated − full)": f"{row['delta_mean']:.{dig}f}",
                "Δ SD": f"{row['delta_sd']:.{dig}f}" if n > 1 else "-",
                "Δ 95% CI": f"[{lo:.{dig}f}, {hi:.{dig}f}]" if n > 1 else "-",
                "Δ median": f"{row['delta_median']:.{dig}f}",
                "Wilcoxon p": f"{row['wilcoxon_p']:.3g}" if pd.notna(row["wilcoxon_p"]) else "-",
                "Nadeau–Bengio p": f"{p_nb:.3g}" if pd.notna(p_nb) else "-",
            })
    save_table(paths, "rev_matern_ablation_paired", pd.DataFrame(vals), pd.DataFrame(fmt), reg,
               "R3. Paired Matérn ablation (Δ = without Matérn − full; ties: |Δ| ≤ 1e-8; "
               "Nadeau–Bengio uses n_test/n_train = 0.15/0.85)")
    return ps


def r4_win_rates(paths, reg, main, base):
    if main is None:
        return None
    st = main[main["Model"] == "Stacking-BMA"].set_index("Seed")["RMSE"]
    others = {k: main[main["Model"] == k].set_index("Seed")["RMSE"] for k in FULL_KERNELS}
    if base is not None:
        ml = base[base["Experiment"] == "ml_baseline"]
        for k in ["GradientBoosting", "RandomForest", "LinearRegression"]:
            others[k] = ml[ml["Model"] == k].set_index("Seed")["RMSE"]
    vals, fmt = [], []
    for k, o in others.items():
        common = st.index.intersection(o.index)
        if not len(common):
            continue
        diff = (o.loc[common] - st.loc[common]).to_numpy(float)
        wins = int(np.sum(diff > 0))
        lo, hi = t_ci(diff)
        vals.append({"competitor": k, "n_seeds": len(common), "stacking_wins": wins,
                     "win_rate": wins / len(common), "rmse_diff_mean_competitor_minus_stacking": float(diff.mean()),
                     "ci95_low": lo, "ci95_high": hi})
        fmt.append({"Competitor": MODEL_LABELS.get(k, k), "n": len(common),
                    "Stacking lower RMSE": f"{wins}/{len(common)} ({wins / len(common) * 100:.0f}%)",
                    "RMSE difference, competitor − stacking (K), mean [95% CI]":
                        f"{diff.mean():.4f} [{lo:.4f}, {hi:.4f}]" if len(common) > 1 else f"{diff.mean():.4f}"})
    save_table(paths, "rev_stacking_win_rates", pd.DataFrame(vals), pd.DataFrame(fmt), reg,
               "R4. Per-seed RMSE win rates of the full stacking model")


def r5_data(paths, reg, data_path, seeds):
    cfg = PRESETS["main"]
    df_raw = pd.read_csv(data_path)
    n0 = len(df_raw)
    df = df_raw[["SMILES", cfg["target"]]].dropna()
    n1 = len(df)
    y0 = df[cfg["target"]].values
    q1, q3 = np.percentile(y0, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - cfg["iqr_k"] * iqr, q3 + cfg["iqr_k"] * iqr
    df = df[(df[cfg["target"]] >= lo) & (df[cfg["target"]] <= hi)].reset_index(drop=True)
    n2 = len(df)
    n_below, n_above = int((y0 < lo).sum()), int((y0 > hi).sum())
    parse_ok, nan_rows = [], 0
    for smi in df["SMILES"]:
        desc = get_descriptors(smi)
        if desc is None:
            parse_ok.append(False)
            continue
        parse_ok.append(True)
        if not np.all(np.isfinite(np.asarray(desc, dtype=float))):
            nan_rows += 1
    n_parse_fail = int(np.sum(~np.array(parse_ok)))
    n3 = n2 - n_parse_fail - nan_rows

    data = load_data_cached(data_path, cfg["target"], paths.cache)
    flow = pd.DataFrame([
        {"step": "1. BP.csv rows", "n_compounds": n0, "removed": 0, "note": ""},
        {"step": "2. drop missing SMILES/BP", "n_compounds": n1, "removed": n0 - n1, "note": ""},
        {"step": "3. BP 1.5·IQR filter (full data, before split)", "n_compounds": n2, "removed": n1 - n2,
         "note": f"keep {lo:.2f} K ≤ BP ≤ {hi:.2f} K (Q1 {q1:.2f}, Q3 {q3:.2f}); below {n_below}, above {n_above}"},
        {"step": "4. SMILES parse failure (RDKit)", "n_compounds": n2 - n_parse_fail, "removed": n_parse_fail, "note": ""},
        {"step": "5. rows with NaN/inf descriptor", "n_compounds": n3, "removed": nan_rows, "note": ""},
        {"step": "6. final data set", "n_compounds": len(data["y_raw"]), "removed": np.nan,
         "note": f"pipeline cache check: {'OK' if len(data['y_raw']) == n3 else 'MISMATCH'}"},
    ])
    save_table(paths, "rev_data_flow", flow, flow, reg, "R5. Data processing flow")

    n_desc = len(Descriptors.descList)
    vt = []
    for s in seeds:
        tr, _ = get_split(len(data["y_raw"]), s, cfg["test_size"])
        vt.append(int(VarianceThreshold(threshold=cfg["vt_threshold"]).fit(data["X_raw"][tr]).get_support().sum()))
    vt = np.array(vt)
    meta_n = {}
    for s in seeds:
        p = paths.seed(s, "main_meta.json")
        if os.path.exists(p):
            meta_n[s] = json.load(open(p)).get("n_features_after_vt")
    mism = [s for s, v in meta_n.items() if v is not None and v != vt[seeds.index(s)]]
    mode = Counter(vt.tolist()).most_common(1)[0][0] if len(vt) else np.nan
    counts = pd.DataFrame([
        {"quantity": "RDKit version", "value": rdBase.rdkitVersion},
        {"quantity": "len(Descriptors.descList)", "value": n_desc},
        {"quantity": "descriptors used (before VarianceThreshold; none removed by hand)", "value": len(data["feature_names"])},
        {"quantity": f"after VarianceThreshold(0) per seed: min (n = {len(vt)} seeds)", "value": int(vt.min())},
        {"quantity": "after VarianceThreshold(0) per seed: max", "value": int(vt.max())},
        {"quantity": "after VarianceThreshold(0) per seed: mode", "value": int(mode)},
        {"quantity": "after VarianceThreshold(0): seed 42", "value": int(vt[seeds.index(42)]) if 42 in seeds else "-"},
        {"quantity": "distribution (count: seeds)", "value": "; ".join(f"{k}: {v}" for k, v in sorted(Counter(vt.tolist()).items()))},
        {"quantity": "check vs main-job metadata", "value": f"{len(meta_n)} seeds compared, {len(mism)} mismatches"},
    ])
    save_table(paths, "rev_descriptor_counts", counts, counts, reg, "R5. Descriptor counts")

    smiles, y = data["smiles"].astype(str), data["y_raw"]
    canon = np.array([Chem.CanonSmiles(s) for s in smiles])
    dup_rows, leak_rows = [], []
    for level, keys in [("exact SMILES string", smiles), ("RDKit canonical SMILES", canon)]:
        g = pd.DataFrame({"key": keys, "y": y}).groupby("key")["y"].agg(["count", "min", "max"])
        g = g[g["count"] > 1]
        spread = (g["max"] - g["min"])
        conflict = spread[spread > 0]
        dup_rows += [
            {"level": level, "quantity": "duplicate groups", "value": len(g)},
            {"level": level, "quantity": "compounds in duplicate groups", "value": int(g["count"].sum())},
            {"level": level, "quantity": "groups with different BP values", "value": len(conflict)},
            {"level": level, "quantity": "BP spread in conflicting groups: median (K)", "value": float(conflict.median()) if len(conflict) else 0.0},
            {"level": level, "quantity": "BP spread in conflicting groups: max (K)", "value": float(conflict.max()) if len(conflict) else 0.0},
        ]
        key_series = pd.Series(keys)
        dup_keys = set(g.index)
        per_seed_pairs = []
        for s in seeds:
            tr, te = get_split(len(y), s, cfg["test_size"])
            trc = Counter(key_series.iloc[tr])
            pairs = sum(trc.get(k, 0) for k in key_series.iloc[te] if k in dup_keys)
            n_test_with_dup = sum(1 for k in key_series.iloc[te] if k in dup_keys and trc.get(k, 0) > 0)
            per_seed_pairs.append(pairs)
            leak_rows.append({"level": level, "seed": s, "train_test_pairs_same_smiles": pairs,
                              "test_compounds_with_duplicate_in_train": n_test_with_dup})
        pp = np.array(per_seed_pairs)
        dup_rows += [
            {"level": level, "quantity": f"seeds with ≥1 train–test duplicate pair (of {len(seeds)})", "value": int((pp > 0).sum())},
            {"level": level, "quantity": "train–test duplicate pairs per seed: mean", "value": float(pp.mean())},
            {"level": level, "quantity": "train–test duplicate pairs per seed: min–max", "value": f"{pp.min()}–{pp.max()}"},
        ]
    dups = pd.DataFrame(dup_rows)
    save_table(paths, "rev_duplicates", dups, dups, reg, "R5. Duplicate SMILES and train–test leakage")
    atomic_csv(pd.DataFrame(leak_rows), os.path.join(paths.tables, f"{PREFIX}rev_duplicates_per_seed.csv"))
    reg["rev_duplicates"]["files"].append(os.path.join(paths.tables, f"{PREFIX}rev_duplicates_per_seed.csv"))
    return flow, counts, dups


def _read_setting(fn):
    try:
        return fn()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def r6_hyperparameters(paths, reg):
    import linear_operator
    from linear_operator import settings as los

    cfg = PRESETS["main"]
    cp = PRESETS["cp"]
    rows = []

    def add(group, param, value, source, status):
        rows.append({"group": group, "parameter": param, "value": str(value), "source": source,
                     "fixed_or_estimated": status})

    add("GP model", "model class", "gpytorch.models.ExactGP (ExactGPModel)", "code", "fixed")
    add("GP model", "mean function (main, ablation, CP)", "gpytorch.means.ZeroMean (prior mean 0 in the Yeo-Johnson "
        "transformed, standardised target space)", "code", "fixed")
    zm = gpytorch.means.ZeroMean()
    add("GP model", "ZeroMean trainable parameters", sum(p.numel() for p in zm.parameters()), "library (object)", "fixed")
    lik = gpytorch.likelihoods.GaussianLikelihood()
    add("Likelihood", "class", "gpytorch.likelihoods.GaussianLikelihood (homoscedastic)", "code", "fixed")
    add("Likelihood", "noise initial value", f"{float(lik.noise.item()):.6g}", "library default (object)",
        "estimated per GP fit (MLL)")
    add("Likelihood", "noise constraint", lik.noise_covar.raw_noise_constraint, "library default (object)", "fixed")
    add("Likelihood", "predictive variance includes noise", "yes: likelihood(model(x)) is used", "code", "fixed")

    for name in FULL_KERNELS:
        k = create_kernel(name)
        g = f"Kernel {name}"
        add(g, "structure", k.__class__.__name__ + "(" + k.base_kernel.__class__.__name__ + ")", "code", "fixed")
        add(g, "outputscale initial", f"{float(k.outputscale.item()):.6g}", "library default (object)", "estimated per GP fit (MLL)")
        add(g, "outputscale constraint", k.raw_outputscale_constraint, "library default (object)", "fixed")
        base = k.base_kernel
        subs = base.kernels if hasattr(base, "kernels") else [base]
        for sk in subs:
            sn = sk.__class__.__name__
            if getattr(sk, "has_lengthscale", False):
                add(g, f"{sn} lengthscale form", f"isotropic (ard_num_dims={sk.ard_num_dims}), shape {tuple(sk.lengthscale.shape)}",
                    "library default (object)", "fixed")
                add(g, f"{sn} lengthscale initial", f"{float(sk.lengthscale.reshape(-1)[0]):.6g}", "library default (object)",
                    "estimated per GP fit (MLL)")
                add(g, f"{sn} lengthscale constraint", sk.raw_lengthscale_constraint, "library default (object)", "fixed")
            if sn == "MaternKernel":
                add(g, "Matern nu", sk.nu, "code", "fixed")
            if sn == "RQKernel":
                add(g, "RQ alpha initial", f"{float(sk.alpha.reshape(-1)[0]):.6g}", "library default (object)", "estimated per GP fit (MLL)")
                add(g, "RQ alpha constraint", sk.raw_alpha_constraint, "library default (object)", "fixed")
            if sn == "LinearKernel":
                add(g, "Linear variance initial", f"{float(sk.variance.reshape(-1)[0]):.6g}", "library default (object)",
                    "estimated per GP fit (MLL)")
                add(g, "Linear variance constraint", sk.raw_variance_constraint, "library default (object)", "fixed")
                add(g, "Linear offset", "none (LinearKernel has no offset parameter)", "library", "fixed")

    params = [torch.nn.Parameter(torch.zeros(1))]
    adam = torch.optim.Adam(params, lr=cfg["lr"])
    add("Optimisation", "optimizer", f"torch.optim.Adam {dict((k, v) for k, v in adam.defaults.items() if k in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad'))}",
        "code (lr) + library defaults", "fixed")
    add("Optimisation", "objective", "−ExactMarginalLogLikelihood (mean over n)", "code", "fixed")
    add("Optimisation", "max iterations", cfg["max_iter"], "code", "fixed")
    add("Optimisation", "early stopping", f"stop after {cfg['patience']} consecutive iterations without loss improvement "
        f"> {cfg['min_delta']} (on the stochastic loss)", "code", "fixed")
    add("Optimisation", "restarts", "none (single run from default initial values)", "code", "fixed")
    add("Optimisation", "hyperparameters", "re-estimated in every GP fit (each inner fold, each final fit, each kernel)",
        "code", "estimated per GP fit (MLL)")

    for sname in ["max_cholesky_size", "cg_tolerance", "eval_cg_tolerance", "max_cg_iterations", "num_trace_samples",
                  "max_preconditioner_size", "min_preconditioning_size", "max_lanczos_quadrature_iterations",
                  "max_root_decomposition_size", "preconditioner_tolerance"]:
        obj = getattr(los, sname, None)
        if obj is not None:
            add("GPyTorch numerics", sname, _read_setting(obj.value), f"linear_operator {linear_operator.__version__} default", "fixed")
    for sname in ["deterministic_probes", "skip_logdet_forward", "fast_computations"]:
        obj = getattr(los, sname, None)
        if obj is None:
            continue
        if sname == "fast_computations":
            add("GPyTorch numerics", "fast_computations.covar_root_decomposition", _read_setting(los.fast_computations.covar_root_decomposition.on),
                "default", "fixed")
            add("GPyTorch numerics", "fast_computations.log_prob", _read_setting(los.fast_computations.log_prob.on), "default", "fixed")
            add("GPyTorch numerics", "fast_computations.solves", _read_setting(los.fast_computations.solves.on), "default", "fixed")
        else:
            add("GPyTorch numerics", sname, _read_setting(obj.on), "default", "fixed")
    add("GPyTorch numerics", "cholesky_jitter (float64)", _read_setting(lambda: los.cholesky_jitter.value(torch.float64)), "default", "fixed")
    add("GPyTorch numerics", "prediction", "gpytorch.settings.fast_pred_var() (LOVE)", "code", "fixed")
    add("GPyTorch numerics", "consequence", "n_train > max_cholesky_size → CG + stochastic Lanczos log-det with random "
        "probe vectors: results depend on the torch RNG sequence", "linear_operator behaviour", "fixed")
    add("GPyTorch numerics", "dtype", "float64 (torch.set_default_dtype, .double())", "code", "fixed")

    add("Stacking", "weight optimizer", f"scipy.optimize.minimize(method='{cfg['weight_optimizer']}', maxiter={cfg['weight_maxiter']}, ftol={cfg['weight_ftol']})",
        "code", "fixed")
    add("Stacking", "objective", "mean squared error of Kelvin-scale OOF point predictions", "code", "fixed")
    add("Stacking", "constraints", "0 ≤ w ≤ 1, Σw = 1; initial w = 1/M; equal weights if optimisation fails", "code", "fixed")
    add("Stacking", "inner folds", f"KFold({cfg['stacking_folds']}, shuffle=True, random_state=seed)", "code", "fixed")
    add("Stacking", "weights", "re-estimated for every split (and every CP outer fold)", "code", "estimated per split")
    add("Stacking", "predictive variance", "Eq. 9 mixture: Σ w (σ_m² + (μ_m − μ̄)²), in Kelvin", "code", "fixed")

    add("CP", "outer folds", f"KFold({cp['outer_folds']}, shuffle=True, random_state=seed) on the 85 % training part", "code", "fixed")
    add("CP", "inner folds", "KFold(5, random_state=seed) — same seed in every outer fold", "code", "fixed")
    add("CP", "fold seeds", "seed_everything(seed*1000 + fold_id)", "code", "fixed")
    add("CP", "alpha", cp["alpha"], "code", "fixed")
    add("CP", "score (paper)", "|y − μ| / σ (Kelvin, stacked μ and mixture σ)", "code", "fixed")
    add("CP", "score (baseline)", "|y − μ|", "code", "fixed")
    add("CP", "quantile rule", "k-th smallest score, k = ⌈(n+1)(1−α)⌉ (1-based); +∞ if k > n", "code", "estimated per split")
    add("CP", "test interval model", "full-training (85 %) stacking model, Eq. 15", "code", "fixed")
    add("CP", "σ to Kelvin", "|T⁻¹(μ_t+σ_t) − T⁻¹(μ_t−σ_t)| / 2 (symmetric one-σ finite difference)", "code", "fixed")
    add("CP", "native interval", "μ ± 1.95996 σ in Kelvin", "code", "fixed")

    add("Preprocessing", "target filter", f"1.5·IQR on full data before split", "code", "fixed")
    add("Preprocessing", "descriptor set", f"all RDKit descriptors (len(Descriptors.descList) = {len(Descriptors.descList)}); "
        "none removed by hand", "code", "fixed")
    add("Preprocessing", "VarianceThreshold", f"threshold={cfg['vt_threshold']}, fitted on each training part", "code", "estimated per split")
    add("Preprocessing", "StandardScaler", "fitted on each training part", "code", "estimated per split")
    add("Preprocessing", "Yeo-Johnson", "PowerTransformer(method='yeo-johnson', standardize=True), refitted on the training "
        "part of every GP fit", "code", "estimated per GP fit")

    for name, est in [("LinearRegression", LinearRegression()),
                      ("RandomForest", RandomForestRegressor(random_state=0, **PRESETS["baseline"]["baselines"]["RandomForest"])),
                      ("GradientBoosting", GradientBoostingRegressor(random_state=0, **PRESETS["baseline"]["baselines"]["GradientBoosting"]))]:
        for p, v in sorted(est.get_params().items()):
            if p == "random_state":
                v = "seed (1–100)"
            add(f"Baseline {name}", p, v, "code" if p in ("n_estimators", "n_jobs", "random_state") else "library default (object)",
                "fixed")
        add(f"Baseline {name}", "input / target", "VT + StandardScaler features; BP in Kelvin (no Yeo-Johnson)", "code", "fixed")

    hp = pd.DataFrame(rows)
    save_table(paths, "rev_hyperparameters", hp, hp, reg, "R6. Hyperparameters and settings (main, ablation and CP presets)")
    env = environment_info()
    ver = pd.DataFrame([{"software": k, "version": v if not isinstance(v, list) else "; ".join(v)} for k, v in env.items()])
    ver = pd.concat([ver, pd.DataFrame([{"software": "linear_operator", "version": linear_operator.__version__}])])
    save_table(paths, "rev_software_versions", ver, ver, reg, "R6. Software versions")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def configure_plot_style(plt):
    plt.rcdefaults()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, path_no_ext):
    fig.savefig(path_no_ext + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight", facecolor="white")
    return [path_no_ext + ".png", path_no_ext + ".pdf"]


def prepare_cp_plot_df(test_csv):
    df = pd.read_csv(test_csv)
    df["CP_Lower"] = df["CVCP_Normalized_Lower"]
    df["CP_Upper"] = df["CVCP_Normalized_Upper"]
    df["Native_Length"] = df["Native_Upper"] - df["Native_Lower"]
    df["CP_Length"] = df["CP_Upper"] - df["CP_Lower"]
    df["Length_Reduction"] = df["Native_Length"] - df["CP_Length"]
    df["Absolute_Error"] = np.abs(df["Y_True"] - df["Stacking_Mean"])
    df["Native_Covered_Plot"] = (df["Y_True"] >= df["Native_Lower"]) & (df["Y_True"] <= df["Native_Upper"])
    df["CP_Covered_Plot"] = (df["Y_True"] >= df["CP_Lower"]) & (df["Y_True"] <= df["CP_Upper"])
    return df


def figure2(df, path_no_ext, plt):
    order = np.argsort(df["Y_True"].to_numpy())
    plot_df = df.iloc[order].reset_index(drop=True)
    x = np.arange(len(plot_df))

    native_cov = plot_df["Native_Covered_Plot"].mean()
    cp_cov = plot_df["CP_Covered_Plot"].mean()
    native_len = plot_df["Native_Length"].mean()
    cp_len = plot_df["CP_Length"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    axes[0].fill_between(x, plot_df["Native_Lower"], plot_df["Native_Upper"],
                         color="#6F78FF", alpha=0.30, label="Native interval", zorder=1)
    axes[0].plot(x, plot_df["Stacking_Mean"], color="#3B4CFF", linewidth=1.5, label="Pred mean", zorder=3)
    axes[0].scatter(x, plot_df["Y_True"], color="red", s=10, label="True value", zorder=4)
    axes[0].set_title(f"Stacking (Native)\nCov: {native_cov * 100:.2f}% | Avg Len: {native_len:.1f} K")
    axes[0].set_xlabel("Test compound index (sorted by observed BP)")
    axes[0].set_ylabel("Boiling point (K)")
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].fill_between(x, plot_df["CP_Lower"], plot_df["CP_Upper"],
                         color="#FF9A4A", alpha=0.30, label="CP interval", zorder=1)
    axes[1].plot(x, plot_df["Stacking_Mean"], color="#FF4D00", linewidth=1.5, label="Pred mean", zorder=3)
    axes[1].scatter(x, plot_df["Y_True"], color="red", s=10, label="True value", zorder=4)
    axes[1].set_title(f"Stacking (CP)\nCov: {cp_cov * 100:.2f}% | Avg Len: {cp_len:.1f} K")
    axes[1].set_xlabel("Test compound index (sorted by observed BP)")
    axes[1].legend(loc="upper left", frameon=True)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.margins(x=0)

    all_lower = np.minimum(plot_df["Native_Lower"], plot_df["CP_Lower"])
    all_upper = np.maximum(plot_df["Native_Upper"], plot_df["CP_Upper"])
    y_min = min(float(all_lower.min()), float(plot_df["Y_True"].min()))
    y_max = max(float(all_upper.max()), float(plot_df["Y_True"].max()))
    pad = max(5.0, 0.025 * (y_max - y_min))
    axes[0].set_ylim(y_min - pad, y_max + pad)

    fig.tight_layout()
    files = _save(fig, path_no_ext)
    plt.close(fig)
    return (native_cov, native_len, cp_cov, cp_len), files


def shorten_smiles(value, max_len=24):
    text = str(value)
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


def select_best_fit(data, n):
    ranked = data.sort_values("Absolute_Error", ascending=True)
    offset = min(100, max(0, len(ranked) - n))
    return ranked.iloc[offset:offset + n].copy().reset_index(drop=True)


def figure3(df, path_no_ext, plt, n_samples=5):
    selected = select_best_fit(df, n_samples)
    labels = [shorten_smiles(v) for v in selected["SMILES"]]

    y_native = np.arange(len(selected), dtype=float)
    y_cp = y_native + 0.18
    fig_h = max(5.2, 0.72 * len(selected) + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    ax.hlines(y_native, selected["Native_Lower"], selected["Native_Upper"],
              linewidth=7, alpha=0.45, color="#5DAAD1", label="Native interval", zorder=1)
    ax.hlines(y_cp, selected["CP_Lower"], selected["CP_Upper"],
              linewidth=7, alpha=0.85, color="#F4A33B", label="CP interval", zorder=2)
    ax.scatter(selected["Stacking_Mean"], y_cp, s=70, color="red", edgecolor="white",
               linewidth=1.0, label="Pred mean", zorder=5)
    ax.scatter(selected["Y_True"], y_cp, s=70, color="black", edgecolor="white",
               linewidth=0.6, label="True value", zorder=6)

    combined_left = np.minimum(selected["Native_Lower"], selected["CP_Lower"])
    combined_right = np.maximum(selected["Native_Upper"], selected["CP_Upper"])
    total_range = max(1.0, float(combined_right.max() - combined_left.min()))
    text_gap = max(2.0, total_range * 0.012)

    for i, (_, row) in enumerate(selected.iterrows()):
        ax.text(max(row["Native_Upper"], row["CP_Upper"]) + text_gap, y_cp[i],
                f"Err={row['Absolute_Error']:.2f}, Δlen={row['Length_Reduction']:.1f} K",
                fontsize=9, fontweight="bold", va="center", zorder=10)

    ax.set_yticks(y_cp)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Boiling point (K)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.grid(axis="y", visible=False)

    annotation_space = max(35.0, total_range * 0.20)
    ax.set_xlim(float(combined_left.min()) - max(10.0, total_range * 0.04),
                float(combined_right.max()) + annotation_space)
    ax.set_ylim(-0.35, len(selected) - 1 + 0.55)

    fig.tight_layout()
    files = _save(fig, path_no_ext)
    plt.close(fig)
    return selected, files


def figureA1(data_path, path_no_ext, plt):
    _, bp_raw, bp_cleaned, bp_cleaned_yj = appendix_bp_summary(data_path)
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    target_col = "BP"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(bp_raw[target_col], bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_title(f"Raw (N={len(bp_raw)})", fontsize=16, fontweight="bold")
    axes[0].set_xlabel("Boiling point (K)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[1].hist(bp_cleaned[target_col], bins=30, edgecolor="black", alpha=0.7)
    axes[1].set_title(f"Cleaned (N={len(bp_cleaned)})", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("Boiling point (K)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[2].hist(bp_cleaned_yj["BP_YJ"], bins=30, edgecolor="black", alpha=0.7)
    axes[2].set_title(f"Cleaned + Yeo-Johnson (N={len(bp_cleaned_yj)})", fontsize=16, fontweight="bold")
    axes[2].set_xlabel("Yeo-Johnson transformed BP (standardized, dimensionless)", fontsize=12)
    axes[2].set_ylabel("Count", fontsize=12)
    plt.tight_layout()
    files = _save(fig, path_no_ext)
    plt.close(fig)
    return files


def _box_with_points(ax, groups, labels, colors):
    rng = np.random.default_rng(0)
    kw = dict(widths=0.5, patch_artist=True, showfliers=False, medianprops={"color": "black", "linewidth": 1.5})
    try:
        bp = ax.boxplot(groups, tick_labels=labels, **kw)
    except TypeError:
        bp = ax.boxplot(groups, labels=labels, **kw)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    for i, (g, c) in enumerate(zip(groups, colors), start=1):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(g)), g, s=10, color=c, alpha=0.7, zorder=3, edgecolor="none")


def figureR2(cp, path_no_ext, plt):
    configure_plot_style(plt)
    colors = ["#3B4CFF", "#FF7F0E", "#2CA02C"]
    labels = ["Native\n(±1.96σ)", "CP-normalized\n(|y−μ|/σ)", "CP-absolute\n(|y−μ|)"]
    cov = [cp.loc[cp["Method"] == m, "Coverage"].to_numpy() * 100 for m, _ in CP_METHODS]
    length = [cp.loc[cp["Method"] == m, "Mean_Interval_Length"].to_numpy() for m, _ in CP_METHODS]
    n = cp["Seed"].nunique()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _box_with_points(axes[0], cov, labels, colors)
    axes[0].axhline(95, color="black", linestyle="--", linewidth=1, label="Nominal 95%")
    axes[0].set_ylabel("Test-set coverage (%)")
    axes[0].set_title(f"Coverage over {n} splits")
    axes[0].legend(loc="lower left", frameon=True)
    _box_with_points(axes[1], length, labels, colors)
    axes[1].set_ylabel("Mean interval length (K)")
    axes[1].set_title(f"Mean interval length over {n} splits")
    fig.tight_layout()
    files = _save(fig, path_no_ext)
    plt.close(fig)
    return files


def figureR3(ps, path_no_ext, plt):
    configure_plot_style(plt)
    d = ps[ps["metric"] == "RMSE"]
    groups, labels = [], []
    for ver, lab in [("a_separate_refit", "(a) separate refit\n(ablation stage, Table 6)"),
                     ("b_no_refit_from_main", "(b) no refit\n(weights re-solved)")]:
        g = d.loc[d["version"] == ver, "delta_ablated_minus_full"].to_numpy()
        if len(g):
            groups.append(g)
            labels.append(f"{lab}\nn = {len(g)}")
    if not groups:
        return []
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    _box_with_points(ax, groups, labels, ["#9467BD", "#17BECF"][:len(groups)])
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("ΔRMSE (K) = without Matérn − full stacking")
    ax.set_title("Matérn ablation: paired per-split differences")
    fig.tight_layout()
    files = _save(fig, path_no_ext)
    plt.close(fig)
    return files


MANIFEST_ITEMS = [
    ("Table 3", "Stacking vs single kernels", "tables", "table3_stacking_vs_kernels"),
    ("Table 4", "Stacking vs ML baselines (+ per-seed baselines)", "tables", "table4_stacking_vs_baselines"),
    ("Table 5", "Stacking weights", "tables", "table5_stacking_weights"),
    ("Table 6", "Matérn ablation (separate refit)", "tables", "table6_matern_ablation"),
    ("Table A.1", "BP summary of curated data", "tables", "tableA1_bp_summary"),
    ("Figure 2", "Native vs CP intervals, seed 42", "figures", "fig2_native_vs_cp_seed042"),
    ("Figure 3", "Sample intervals, seed 42", "figures", "fig3_sample_intervals_seed042"),
    ("Figure A.1", "BP histograms", "figures", "figA1_bp_histograms"),
    ("R2", "CP over 100 seeds (summary)", "tables", "rev_cp_100seed_summary"),
    ("R2", "CP over 100 seeds (per seed)", "tables", "rev_cp_per_seed"),
    ("R2", "CP coverage / length figure", "figures", "figR2_cp_coverage_length_100seed"),
    ("R3", "Paired Matérn ablation", "tables", "rev_matern_ablation_paired"),
    ("R3", "Paired Matérn ablation (per seed)", "tables", "rev_matern_ablation_per_seed"),
    ("R3", "Paired Matérn ablation figure", "figures", "figR3_matern_ablation_paired"),
    ("R4", "Stacking win rates", "tables", "rev_stacking_win_rates"),
    ("R5", "Data flow", "tables", "rev_data_flow"),
    ("R5", "Descriptor counts", "tables", "rev_descriptor_counts"),
    ("R5", "Duplicates and leakage", "tables", "rev_duplicates"),
    ("R6", "Hyperparameters", "tables", "rev_hyperparameters"),
    ("R6", "Software versions", "tables", "rev_software_versions"),
]


def write_manifest(paths):
    rows = []
    for item, desc, kind, name in MANIFEST_ITEMS:
        folder = paths.tables if kind == "tables" else paths.figures
        files = sorted(glob.glob(os.path.join(folder, f"{PREFIX}{name}*")))
        if name == "table4_stacking_vs_baselines":
            files += sorted(glob.glob(os.path.join(folder, f"{PREFIX}table4_baselines_per_seed.csv")))
        stage = "tables" if kind == "tables" else "figures"
        rows.append({
            "item": item,
            "description": desc,
            "files": "; ".join(os.path.relpath(f, paths.root) for f in files) if files else "(not generated yet)",
            "stage": stage,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(max(os.path.getmtime(f) for f in files)))
            if files else "",
        })
    df = pd.DataFrame(rows)
    atomic_csv(df, paths.manifest)
    return df


def write_revision_report(paths):
    man = write_manifest(paths)

    def read_md(name):
        p = paths.table(name, "formatted", "md")
        return open(p, encoding="utf-8").read() if os.path.exists(p) else f"_{name}: not generated yet_\n"

    def read_vals(name):
        p = paths.table(name, "values")
        return pd.read_csv(p) if os.path.exists(p) else None

    lines = ["# 0930 revision results", "", f"Generated {ts()} by `gp_stacking_bp.py`. File paths are relative to the output folder.", ""]
    lines += ["## Summary", "", "### Paper items → files", ""]
    lines += [df_to_md(man[["item", "description", "files"]]), ""]

    r2 = read_vals("rev_cp_100seed_summary")
    lines += ["### R2 CP over seeds", ""]
    if r2 is not None:
        for m, label in CP_METHODS:
            c = r2[(r2["method"] == m) & (r2["quantity"] == "Coverage")]
            l = r2[(r2["method"] == m) & (r2["quantity"] == "Mean_Interval_Length")]
            if len(c) and len(l):
                lines.append(f"- {label}: mean coverage {c['mean'].iloc[0] * 100:.2f}% (SD {c['sd'].iloc[0] * 100:.2f}, "
                             f"n = {int(c['n_seeds'].iloc[0])}), mean length {l['mean'].iloc[0]:.1f} K, "
                             f"seeds with coverage ≥ 95%: {c['frac_seeds_coverage_ge_0.95'].iloc[0] * 100:.0f}%")
        lines.append("")
    else:
        lines += ["- CP results not available yet", ""]

    r3 = read_vals("rev_matern_ablation_paired")
    lines += ["### R3 Matérn ablation (paired)", ""]
    if r3 is not None:
        for _, r in r3[r3["metric"] == "RMSE"].iterrows():
            lines.append(f"- {r['version']}: full stacking better in {int(r['full_wins'])}/{int(r['n_seeds'])} seeds "
                         f"(ties {int(r['ties'])}), ΔRMSE mean {r['delta_mean']:.4f} K "
                         f"[95% CI {r['delta_ci95_low']:.4f}, {r['delta_ci95_high']:.4f}], "
                         f"Wilcoxon p = {r['wilcoxon_p']:.3g}, Nadeau–Bengio p = {r['nb_p']:.3g}")
        lines.append("")

    r5 = read_vals("rev_data_flow")
    lines += ["### R5 data flow", ""]
    if r5 is not None:
        lines += [" → ".join(f"{r['step'].split('. ', 1)[-1]} {int(r['n_compounds'])}" for _, r in r5.iterrows()), ""]

    lines += ["## All tables", ""]
    for item, desc, kind, name in MANIFEST_ITEMS:
        if kind == "tables":
            lines += [f"#### {item}: {desc}", "", read_md(name), ""]
    lines += ["## Figures", ""]
    for item, desc, kind, name in MANIFEST_ITEMS:
        if kind == "figures":
            files = sorted(glob.glob(os.path.join(paths.figures, f"{PREFIX}{name}.*")))
            lines.append(f"- {item} ({desc}): " + (", ".join(f"`{os.path.relpath(f, paths.root)}`" for f in files)
                                                     if files else "not generated yet"))
    atomic_text(paths.report, "\n".join(lines) + "\n")
    print(f"Report written: {paths.report}", flush=True)


def run_tables(paths, data_path, seeds):
    paths.makedirs()
    data = load_data_cached(data_path, "BP", paths.cache)
    for s in sorted(set(seeds)):
        if not os.path.exists(paths.seed(s, "cp_metrics.csv")) and cp_ready(paths, s):
            assemble_cp(s, data, paths)
    reg = {}
    main, abl, base = paper_tables(paths, reg, data_path, seeds)
    r2_cp(paths, reg, seeds)
    r3_ablation(paths, reg, main, abl)
    r4_win_rates(paths, reg, main, base)
    r5_data(paths, reg, data_path, seeds)
    r6_hyperparameters(paths, reg)
    write_revision_report(paths)
    print(f"Tables written to {paths.tables}", flush=True)


def run_figures(paths, data_path, figure_seed, seeds):
    paths.makedirs()
    plt = _plt()
    test_csv = paths.seed(figure_seed, "cp_test_predictions.csv")
    if os.path.exists(test_csv):
        configure_plot_style(plt)
        df = prepare_cp_plot_df(test_csv)
        (nc, nl, cc, cl), _ = figure2(df, paths.figure(f"fig2_native_vs_cp_seed{figure_seed:03d}"), plt)
        configure_plot_style(plt)
        sel, _ = figure3(df, paths.figure(f"fig3_sample_intervals_seed{figure_seed:03d}"), plt)
        atomic_csv(sel[["SMILES", "Y_True", "Stacking_Mean", "Native_Lower", "Native_Upper", "CP_Lower",
                        "CP_Upper", "Absolute_Error", "Length_Reduction"]],
                   paths.figure(f"fig3_sample_intervals_seed{figure_seed:03d}_selected.csv"))
        print(f"[figures] seed {figure_seed}: native {nc * 100:.2f}% / {nl:.1f} K | CP {cc * 100:.2f}% / {cl:.1f} K", flush=True)
    else:
        print(f"[figures] no CP result for seed {figure_seed}; Figures 2-3 skipped", flush=True)
    figureA1(data_path, paths.figure("figA1_bp_histograms"), plt)

    frames = [pd.read_csv(p) for p in (paths.seed(s, "cp_metrics.csv") for s in seeds) if os.path.exists(p)]
    if frames:
        figureR2(pd.concat(frames, ignore_index=True), paths.figure("figR2_cp_coverage_length_100seed"), plt)
    else:
        print("[figures] no CP results; Figure R2 skipped", flush=True)
    ps_path = paths.table("rev_matern_ablation_per_seed", "values")
    if os.path.exists(ps_path):
        figureR3(pd.read_csv(ps_path), paths.figure("figR3_matern_ablation_paired"), plt)
    else:
        print("[figures] run --stage tables first for Figure R3", flush=True)
    write_revision_report(paths)
    print(f"Figures written to {paths.figures}", flush=True)


def parse_seeds(text):
    seeds = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def parse_gpus(text):
    if text is None or str(text).lower() in ("", "cpu", "none"):
        return []
    return [int(g) for g in str(text).split(",") if g.strip() != ""]


RUN_CONFIG = "run_config.json"
TRAIN_STAGES = ("main", "ablation", "cp", "gpu_all", "baseline")
REUSE_MSG = "This output folder holds results made with different data or settings. Use a new --out."


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_settings():
    gp = ExactGPModel(torch.zeros(2, 1), torch.zeros(2), gpytorch.likelihoods.GaussianLikelihood(), FULL_KERNELS[0])
    return json.loads(json.dumps({
        "descriptors": {"set": "all of Descriptors.descList", "n": len(Descriptors.descList), "rdkit": rdBase.rdkitVersion},
        "mean_function": f"{type(gp.mean_module).__module__}.{type(gp.mean_module).__name__}",
        "presets": PRESETS,
    }))


def _diff_keys(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [k for key in sorted(set(a) | set(b)) for k in _diff_keys(a.get(key), b.get(key), f"{prefix}{key}.")]
    return [] if a == b else [prefix.rstrip(".")]


def check_run_config(paths, data_path, training, dry_run):
    path = os.path.join(paths.root, RUN_CONFIG)
    tag = "DRY-RUN | " if dry_run else ""
    current = {
        "data_file": os.path.basename(data_path),
        "data_sha256": file_sha256(data_path),
        "code_file": os.path.basename(__file__),
        "code_sha256": file_sha256(os.path.abspath(__file__)),
        "settings": run_settings(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not os.path.exists(path):
        if any(os.path.isdir(d) and os.listdir(d) for d in (paths.seeds, paths.tables, paths.figures)):
            raise SystemExit(f"{ts()} | STOP | {tag}result files but no {RUN_CONFIG} in {paths.root}\n{REUSE_MSG}")
        if not training:
            raise SystemExit(f"{ts()} | STOP | {tag}no {RUN_CONFIG} in {paths.root}; run the training stages with this --out first")
        if not dry_run:
            os.makedirs(paths.root, exist_ok=True)
            atomic_json(path, current)
        print(f"{ts()} | {tag}new output folder: {RUN_CONFIG} {'would be ' if dry_run else ''}created", flush=True)
        return
    with open(path) as fh:
        saved = json.load(fh)
    diff = _diff_keys({k: saved.get(k) for k in ("data_sha256", "settings")},
                      {k: current[k] for k in ("data_sha256", "settings")})
    if diff:
        raise SystemExit(f"{ts()} | STOP | {tag}{RUN_CONFIG} differs in: {', '.join(diff)}\n{REUSE_MSG}")
    if saved.get("code_sha256") != current["code_sha256"]:
        print(f"{ts()} | WARNING | {tag}code file differs from {RUN_CONFIG} (created {saved.get('created')}); "
              "data and settings match, continuing", flush=True)
    print(f"{ts()} | {tag}{RUN_CONFIG} matches (created {saved.get('created')}): finished jobs are reused", flush=True)


def environment_info():
    import scipy
    import sklearn
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpytorch": gpytorch.__version__,
        "rdkit": rdBase.rdkitVersion,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available() else [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True,
                        choices=["main", "ablation", "cp", "gpu_all", "baseline", "tables", "figures", "all"])
    parser.add_argument("--seeds", default="1-100", help="random-split seeds (paper: 1-100)")
    parser.add_argument("--cp-seeds", default="1-100", help="seeds for CP jobs (paper figure: 42)")
    parser.add_argument("--figure-seed", type=int, default=42, help="seed shown in Figures 2-3")
    parser.add_argument("--gpus", default="0,1,2,3", help="comma-separated GPU ids, or 'cpu'")
    parser.add_argument("--tasks-per-gpu", type=int, default=2)
    parser.add_argument("--cpu-workers", type=int, default=4, help="parallel CPU jobs for the baseline stage")
    parser.add_argument("--order", choices=["interim", "lpt"], default="interim",
                        help="interim: main -> ablation -> CP (Tables 3-6 early); lpt: longest jobs first")
    parser.add_argument("--no-deterministic", action="store_true",
                        help="do not call torch.use_deterministic_algorithms(True)")
    parser.add_argument("--dry-run", action="store_true", help="print the job plan only; nothing is trained")
    parser.add_argument("--poll", type=float, default=5.0, help="scheduler polling interval (s)")
    parser.add_argument("--data", default="BP.csv", help="input CSV with SMILES and BP (K) columns")
    parser.add_argument("--out", default="0930_results", help="output folder")
    args = parser.parse_args()

    paths = Paths(args.out)
    seeds = parse_seeds(args.seeds)
    cp_seeds = parse_seeds(args.cp_seeds)
    gpus = parse_gpus(args.gpus)
    opts = {
        "deterministic": not args.no_deterministic,
        "cpu_workers": args.cpu_workers,
        "poll": args.poll,
    }
    print(f"{ts()} | stage={args.stage} | seeds n={len(seeds)} ({seeds[0] if seeds else '-'}..{seeds[-1] if seeds else '-'}) | "
          f"cp_seeds n={len(cp_seeds)} | GPUs={gpus} x {args.tasks_per_gpu} | order={args.order} | "
          f"deterministic={opts['deterministic']} | out={paths.root}", flush=True)

    stages = ["gpu_all", "baseline", "tables", "figures"] if args.stage == "all" else [args.stage]
    check_run_config(paths, args.data, any(st in TRAIN_STAGES for st in stages), args.dry_run)
    if not args.dry_run:
        paths.makedirs()
        atomic_json(os.path.join(paths.logs, f"{PREFIX}environment.json"), environment_info())
        atomic_json(os.path.join(paths.logs, f"{PREFIX}presets.json"), PRESETS)

    failures = []
    for stage in stages:
        t0 = time.time()
        if stage in TRAIN_STAGES:
            jobs = build_jobs(stage, seeds, cp_seeds if stage in ("cp", "gpu_all") else [], args.figure_seed, args.order)
            lanes = make_lanes(stage, gpus, args.tasks_per_gpu, opts)
            failures += run_scheduler(stage, jobs, lanes, paths, args.data, opts,
                                      cp_seeds if stage in ("cp", "gpu_all") else [], dry_run=args.dry_run)
        elif args.dry_run:
            print(f"{ts()} | DRY-RUN | stage {stage}: reads result files only (no training)", flush=True)
        elif stage == "tables":
            run_tables(paths, args.data, seeds)
        elif stage == "figures":
            run_figures(paths, args.data, args.figure_seed, seeds)
        print(f"{ts()} | stage {stage} finished in {time.time() - t0:.1f}s", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
