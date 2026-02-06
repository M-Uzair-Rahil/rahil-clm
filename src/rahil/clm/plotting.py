from __future__ import annotations

import os
import math
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse your core download/cache logic + constants
from .core import _ensure_inputs  # must exist in your core.py


# ============================================================
# Defaults (must match your LHS generator)
# ============================================================
PFTS = {"corn": 17, "soybean": 23, "wheat": 19}

BOUND_COLS = {
    "corn":    ("corn min", "corn max"),
    "soybean": ("soybean min", "soybean max"),
    "wheat":   ("wheat min", "wheat max"),
}

NS_PER_DAY = 86400.0 * 1e9


# ============================================================
# Helpers
# ============================================================
def _find_latest_param_list(workflow_dir: str) -> str:
    candidates = glob.glob(os.path.join(workflow_dir, "*.param_list.txt"))
    if not candidates:
        raise FileNotFoundError(
            f"No *.param_list.txt found in workflow dir: {workflow_dir}\n"
            "Make sure you already ran rahil.generate_lhs(output_dir=...)."
        )
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def _normalize_parameter_index(bounds_df: pd.DataFrame) -> pd.DataFrame:
    bounds_df = _normalize_columns(bounds_df)

    # normalize parameter column name -> "Parameters"
    rename_map = {}
    for c in bounds_df.columns:
        if str(c).strip().lower() in ["parameter", "parameters", "param", "par"]:
            rename_map[c] = "Parameters"
    bounds_df = bounds_df.rename(columns=rename_map)

    if "Parameters" not in bounds_df.columns:
        raise KeyError(
            "No parameter column found in bounds file.\n"
            f"Available columns: {list(bounds_df.columns)}"
        )

    bounds_df["Parameters"] = bounds_df["Parameters"].astype(str).str.strip()
    bounds_df = bounds_df.dropna(subset=["Parameters"]).set_index("Parameters")
    bounds_df = bounds_df[~bounds_df.index.duplicated(keep="first")]
    return bounds_df


def _infer_param_names_from_sampled_columns(cols) -> list[str]:
    params = set()
    for c in cols:
        c = str(c)
        if "__pft" in c:
            params.add(c.split("__pft")[0].strip())
    return sorted(params)


def _load_bounds_excel(bounds_xlsx_path: str, sheet_name=None) -> pd.DataFrame:
    """
    Load bounds from Excel. If sheet_name is None, try to find a sheet that contains
    required columns; otherwise use the provided sheet.
    """
    xls = pd.ExcelFile(bounds_xlsx_path)

    if sheet_name is not None:
        b = pd.read_excel(bounds_xlsx_path, sheet_name=sheet_name)
        return _normalize_parameter_index(b)

    required = {
        "Parameters",
        "corn min", "corn max",
        "soybean min", "soybean max",
        "wheat min", "wheat max",
    }

    for sh in xls.sheet_names:
        b = pd.read_excel(bounds_xlsx_path, sheet_name=sh)
        b.columns = b.columns.str.strip()
        b = b.rename(columns={"parameter": "Parameters", "Parameter": "Parameters", "PARAMETER": "Parameters"})
        if required.issubset(set(b.columns)):
            return _normalize_parameter_index(b)

    # fallback: first sheet, but will raise meaningful error if missing columns
    b = pd.read_excel(bounds_xlsx_path, sheet_name=xls.sheet_names[0])
    return _normalize_parameter_index(b)


# ============================================================
# Public API
# ============================================================
def check_distribution(
    output_dir: str = "outputs",
    *,
    sampled_file: str | None = None,
    bounds_sheet=None,
    outdir: str | None = None,
    n_cols: int = 4,
    panel_w: float = 3.2,
    panel_h: float = 2.4,
    show: bool = False,
    dpi: int = 300,
    cache_dir: str | None = None,
) -> str:
    """
    Create distribution plots (one figure per crop) for whichever samples exist
    in output_dir/workflow/*.param_list.txt.

    Parameters
    ----------
    output_dir : str
        Folder used in rahil.generate_lhs(output_dir=...).
    sampled_file : str | None
        Optional explicit path to *.param_list.txt. If None, automatically picks newest.
    bounds_sheet : str|int|None
        Optional bounds Excel sheet name (or index). If None, auto-detect.
    outdir : str | None
        Where to save plots. Default: output_dir/figs_distributions_by_crop
    n_cols : int
        Number of columns in subplot grid (article layout).
    panel_w, panel_h : float
        Size per subplot (inches).
    show : bool
        If True, plt.show() each figure. Default False (faster, cleaner).
    dpi : int
        Saved figure dpi.
    cache_dir : str | None
        Optional cache dir for bounds/base files. If None, core decides.

    Returns
    -------
    str : directory path where figures were saved
    """
    # 1) locate sampled param list
    workflow_dir = os.path.join(output_dir, "workflow")
    if sampled_file is None:
        sampled_file = _find_latest_param_list(workflow_dir)

    # 2) ensure bounds file exists (download/cache from GitHub release)
    bounds_xlsx_path, _, _ = _ensure_inputs(cache_dir=cache_dir)

    # 3) read sampled + bounds
    df = pd.read_csv(sampled_file, index_col=0)
    df = _normalize_columns(df)

    bounds = _load_bounds_excel(bounds_xlsx_path, sheet_name=bounds_sheet)
    bounds.index = bounds.index.astype(str).str.strip()

    sampled_params = _infer_param_names_from_sampled_columns(df.columns)
    params_to_plot = [p for p in bounds.index if p in sampled_params]

    # 4) output plot directory
    if outdir is None:
        outdir = os.path.join(output_dir, "figs_distributions_by_crop")
    os.makedirs(outdir, exist_ok=True)

    plt.rcParams.update({"axes.grid": True, "font.size": 9})

    # 5) one figure per crop
    for crop, pid in PFTS.items():
        min_col, max_col = BOUND_COLS[crop]

        n = len(params_to_plot)
        if n == 0:
            raise ValueError(
                "No parameters found to plot.\n"
                "Check that your sampled param_list columns match the bounds table parameter names."
            )

        ncols = min(n_cols, max(1, n))
        nrows = int(math.ceil(n / ncols))
        fig_w = panel_w * ncols
        fig_h = panel_h * nrows

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))
        axes = np.array(axes).reshape(-1)

        fig.suptitle(
            f"{crop.capitalize()} (PFT {pid}) — LHS sampled distributions",
            fontsize=14, y=0.995
        )

        for i, param in enumerate(params_to_plot):
            ax = axes[i]
            col = f"{param}__pft{pid}_{crop}"

            if col not in df.columns:
                ax.set_title(f"{param} (missing)", fontsize=9)
                ax.axis("off")
                continue

            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()

            vmin = float(bounds.loc[param, min_col])
            vmax = float(bounds.loc[param, max_col])

            # mxmat bounds may be ns in excel; values are days in sampled list
            if param == "mxmat":
                if abs(vmin) > 1e6 or abs(vmax) > 1e6:
                    vmin /= NS_PER_DAY
                    vmax /= NS_PER_DAY
                vals = np.rint(vals).astype(int)

            ax.hist(vals, bins=20, edgecolor="k", alpha=0.75)
            ax.axvline(vmin, color="red", linestyle="--", linewidth=1)
            ax.axvline(vmax, color="red", linestyle="--", linewidth=1)

            ax.set_title(param, fontsize=10)
            ax.set_ylabel("count", fontsize=8)
            ax.set_xlabel("value" + (" (days)" if param == "mxmat" else ""), fontsize=8)
            ax.tick_params(axis="both", labelsize=8)

        for j in range(n, len(axes)):
            axes[j].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(outdir, f"LHS_distributions_{crop}.png"),
                    dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(fig)

    return outdir
