import sys
from __future__ import annotations

import os
from os import path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import xarray as xr
import requests



# ============================================================
# GitHub Release URLs (edit only if your tag/filenames differ)
# ============================================================
RELEASE_TAG = "v0.1.0"

BASE_REPO = "M-Uzair-Rahil/rahil-clm"

DEFAULT_XLSX_NAME = "finalized_params_for_run.xlsx"
DEFAULT_NC_NAME   = "clm50_params.c240207b.nc"

GITHUB_XLSX_URL = f"https://github.com/{BASE_REPO}/releases/download/{RELEASE_TAG}/{DEFAULT_XLSX_NAME}"
GITHUB_NC_URL   = f"https://github.com/{BASE_REPO}/releases/download/{RELEASE_TAG}/{DEFAULT_NC_NAME}"


# ============================================================
# OUTPUT CONTAINER
# ============================================================
@dataclass
class GenerateLHSResult:
    psets_df: pd.DataFrame
    param_output_dir: str
    main_run_file: str
    param_list_file: str
    case_ids: List[str]
    used_excel: str
    used_base_nc: str
    cache_dir: str


# ============================================================
# CACHE + DOWNLOAD HELPERS
# ============================================================
def _default_cache_dir(appname: str = "rahil-clm") -> str:
    """
    Cross-platform cache directory.
    Windows: %LOCALAPPDATA%\\<appname>\\cache
    macOS:   ~/Library/Caches/<appname>
    Linux:   ~/.cache/<appname>
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or path.expanduser(r"~\AppData\Local")
        return path.join(base, appname, "cache")

    # macOS
    if sys.platform == "darwin":  # type: ignore[name-defined]
        return path.join(path.expanduser("~/Library/Caches"), appname)

    # linux/unix
    base = os.environ.get("XDG_CACHE_HOME", path.expanduser("~/.cache"))
    return path.join(base, appname)


def _download_file(url: str, out_path: str, timeout: int = 120) -> None:
    os.makedirs(path.dirname(out_path), exist_ok=True)

    # already downloaded
    if path.exists(out_path) and path.getsize(out_path) > 0:
        return

    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()

    tmp_path = out_path + ".part"
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    os.replace(tmp_path, out_path)


def _ensure_inputs(cache_dir: Optional[str] = None) -> tuple[str, str, str]:
    """
    Ensure Excel and NetCDF exist locally; download if missing.
    Returns (excel_path, nc_path, cache_dir_used).
    """
    if cache_dir is None:
        cache_dir = _default_cache_dir()

    os.makedirs(cache_dir, exist_ok=True)

    xlsx_path = path.join(cache_dir, DEFAULT_XLSX_NAME)
    nc_path   = path.join(cache_dir, DEFAULT_NC_NAME)

    _download_file(GITHUB_XLSX_URL, xlsx_path)
    _download_file(GITHUB_NC_URL, nc_path)

    return xlsx_path, nc_path, cache_dir


# ============================================================
# LHS SAMPLER
# ============================================================
def lhs(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sampling in [0,1]."""
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if n_dim <= 0:
        raise ValueError("n_dim must be > 0")

    cut = np.linspace(0.0, 1.0, n_samples + 1)
    u = rng.random((n_samples, n_dim))
    a = cut[:-1]
    b = cut[1:]
    H = u * (b - a)[:, None] + a[:, None]
    for j in range(n_dim):
        rng.shuffle(H[:, j])
    return H


# ============================================================
# HELPERS
# ============================================================
def _normalize_param_colname(pf: pd.DataFrame) -> pd.DataFrame:
    pf = pf.copy()
    pf.columns = pf.columns.str.strip()

    rename_map = {}
    for cand in ("parameter", "Parameter", "PARAMETER"):
        if cand in pf.columns:
            rename_map[cand] = "Parameters"
    if rename_map:
        pf = pf.rename(columns=rename_map)

    if "Parameters" not in pf.columns:
        raise KeyError(
            "Input file must contain a 'Parameters' column "
            "(or 'parameter' / 'Parameter')."
        )

    pf = pf.dropna(subset=["Parameters"])
    pf["Parameters"] = pf["Parameters"].astype(str).str.strip()
    return pf


def _find_pft_dim(da: xr.DataArray) -> str:
    for d in da.dims:
        if "pft" in d.lower():
            return d
    raise ValueError(f"No PFT dimension found in {da.dims}")


def _cast_like_base(param: str, base_dtype, new_val: float):
    """
    Cast sampled value to base NetCDF dtype.
    Special handling for mxmat (integer DAYS).
    """
    if param == "mxmat":
        v = float(new_val)
        # nanoseconds -> days (if huge)
        if abs(v) > 1.0e6:
            v = v / (86400.0 * 1.0e9)
        return int(np.rint(v))

    if np.issubdtype(base_dtype, np.integer):
        return int(np.rint(new_val))

    return float(new_val)


# ============================================================
# PUBLIC API (no user file paths needed)
# ============================================================
def generate_lhs(
    Ninit: int = 150,
    seed: int = 42,
    output_dir: str = ".",
    *,
    location: str = "pe_crops",
    iteration: int = 0,
    PFT_CORN: int = 17,    # rainfed_temperate_corn
    PFT_SOY: int = 23,     # rainfed_temperate_soybean
    PFT_WHEAT: int = 19,   # rainfed_spring_wheat
    bounds_cols: Optional[Dict[str, Tuple[str, str]]] = None,
    cache_dir: Optional[str] = None,
) -> GenerateLHSResult:
    """
    Minimal user call:
        import rahil
        out = rahil.generate_lhs(Ninit=150, seed=42, output_dir="outputs")

    Downloads (once) and caches:
      - Excel bounds table
      - Base CLM NetCDF params file
    from GitHub Releases.
    """
    if bounds_cols is None:
        bounds_cols = {
            "corn": ("corn min", "corn max"),
            "soybean": ("soybean min", "soybean max"),
            "wheat": ("wheat min", "wheat max"),
        }

    # 0) ensure input files exist locally (download/cache)
    excel_path, base_nc_path, cache_dir_used = _ensure_inputs(cache_dir=cache_dir)

    output_dir = path.abspath(output_dir)
    workflow_dir = path.join(output_dir, "workflow")
    param_output_dir = path.join(output_dir, "paramfile", location)
    os.makedirs(workflow_dir, exist_ok=True)
    os.makedirs(param_output_dir, exist_ok=True)

    # 1) read Excel bounds
    pf = pd.read_excel(excel_path)
    pf = _normalize_param_colname(pf)

    need_cols = [
        "Parameters",
        bounds_cols["corn"][0], bounds_cols["corn"][1],
        bounds_cols["soybean"][0], bounds_cols["soybean"][1],
        bounds_cols["wheat"][0], bounds_cols["wheat"][1],
    ]
    missing = [c for c in need_cols if c not in pf.columns]
    if missing:
        raise KeyError(f"Missing columns in Excel: {missing}")

    pf = pf[need_cols].copy()
    pf = pf.dropna(subset=["Parameters"])
    pf["Parameters"] = pf["Parameters"].astype(str).str.strip()
    param_list = pf["Parameters"].values

    # 2) build bounds vectors and map for 3 crops
    xlb: List[float] = []
    xub: List[float] = []
    var_map: List[Tuple[str, str, int]] = []

    for _, row in pf.iterrows():
        param = row["Parameters"]

        mn, mx = bounds_cols["corn"]
        xlb.append(float(row[mn])); xub.append(float(row[mx]))
        var_map.append((param, "corn", PFT_CORN))

        mn, mx = bounds_cols["soybean"]
        xlb.append(float(row[mn])); xub.append(float(row[mx]))
        var_map.append((param, "soybean", PFT_SOY))

        mn, mx = bounds_cols["wheat"]
        xlb.append(float(row[mn])); xub.append(float(row[mx]))
        var_map.append((param, "wheat", PFT_WHEAT))

    xlb_arr = np.array(xlb, dtype=float)
    xub_arr = np.array(xub, dtype=float)

    # 3) LHS and scale
    rng = np.random.default_rng(seed)
    X01 = lhs(Ninit, len(xlb_arr), rng)
    perturbed = X01 * (xub_arr - xlb_arr) + xlb_arr

    # 4) build case IDs + workflow outputs
    case_ids = [f"{location}_{iteration}_{i:04d}" for i in range(Ninit)]
    colnames = [f"{p}__pft{pid}_{tag}" for (p, tag, pid) in var_map]
    psets_df = pd.DataFrame(perturbed, columns=colnames, index=case_ids)

    # mxmat: ensure integer days
    for c in psets_df.columns:
        if c.startswith("mxmat__"):
            v = psets_df[c].astype(float).to_numpy()
            v = np.where(np.abs(v) > 1.0e6, v / (86400.0 * 1.0e9), v)
            psets_df[c] = np.rint(v).astype(int)

    param_list_file = path.join(workflow_dir, f"{location}_{iteration}.param_list.txt")
    main_run_file = path.join(workflow_dir, f"{location}_{iteration}.main_run.txt")

    psets_df.to_csv(param_list_file)
    with open(main_run_file, "w") as f:
        f.write("\n".join(case_ids) + "\n")

    # 5) write NetCDFs
    base = xr.open_dataset(base_nc_path, decode_times=False)

    missing_params = [p for p in param_list if p not in base.variables]
    if missing_params:
        base.close()
        raise KeyError(f"These parameters are not in the base NetCDF: {missing_params}")

    param_meta = {}
    for p in param_list:
        da = base[p]
        pft_dim = _find_pft_dim(da)
        other_dims = [d for d in da.dims if d != pft_dim]
        param_meta[p] = {"pft_dim": pft_dim, "dtype": da.dtype, "other_dims": other_dims}
    base.close()

    for case_id, row in psets_df.iterrows():
        ds = xr.open_dataset(base_nc_path, decode_times=False)
        encoding = {}

        for (param, tag, pid) in var_map:
            col = f"{param}__pft{pid}_{tag}"
            new_val = float(row[col])

            meta = param_meta[param]
            pft_dim = meta["pft_dim"]
            base_dtype = meta["dtype"]
            other_dims = meta["other_dims"]

            casted = _cast_like_base(param, base_dtype, new_val)

            if len(other_dims) == 0:
                ds[param].loc[{pft_dim: pid}] = casted
            else:
                idx = {pft_dim: pid}
                for d in other_dims:
                    idx[d] = slice(None)
                ds[param].loc[idx] = casted

            if param == "mxmat":
                mx = ds["mxmat"]
                mxv = np.array(mx, dtype="float64")
                mask_ns = np.isfinite(mxv) & (np.abs(mxv) > 1.0e6)
                mxv[mask_ns] = mxv[mask_ns] / (86400.0 * 1.0e9)
                mxv = np.rint(mxv)
                mxv = np.where(np.isfinite(mxv), mxv, 0.0)

                ds["mxmat"] = xr.DataArray(
                    mxv.astype("int32"),
                    dims=mx.dims,
                    coords=mx.coords
                )
                encoding["mxmat"] = {"dtype": "i4", "_FillValue": 0}

        out_nc = path.join(param_output_dir, f"{case_id}.nc")
        ds.to_netcdf(out_nc, mode="w", encoding=encoding)
        ds.close()

    return GenerateLHSResult(
        psets_df=psets_df,
        param_output_dir=param_output_dir,
        main_run_file=main_run_file,
        param_list_file=param_list_file,
        case_ids=case_ids,
        used_excel=excel_path,
        used_base_nc=base_nc_path,
        cache_dir=cache_dir_used,
    )
