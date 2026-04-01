"""
baseline.py
-----------
Computes the expected baseline for each camera column using:

    1. Day-of-week (DOW) segmentation       — each weekday gets its own baseline
    2. Recency-weighted median               — recent weeks contribute more
    3. Exclusion of event ISO weeks          — entire week excluded if any day has event
    4. Exclusion of zero rows                — rows where camera value is 0 are dropped
    5. Automatic outlier exclusion (new)     — weeks with residual > 3-sigma excluded
    6. Monthly seasonality factor (new)      — multiplicative adjustment per calendar month
    7. Configurable lookback window          — 3, 6, or 12 months

Monthly seasonality
-------------------
For each (camera, month) pair the system computes a multiplicative factor:

    factor_m = median(DOW-medians in month m) / median(DOW-medians across all months)

The baseline becomes: baseline_DOW * factor_month

Requires MIN_MONTHS_FOR_SEASONALITY (24) months of history to activate.
Falls back to factor = 1.0 if insufficient history.

Automatic outlier exclusion
---------------------------
After computing a first-pass baseline, weeks whose median absolute residual
exceeds OUTLIER_SIGMA_THRESHOLD * sigma are flagged and removed from the pool.
This captures rain, construction and other unregistered atypical days without
requiring manual event tagging.

Output
------
For every (date, camera) pair:
    {cam}_baseline_median   : recency-weighted DOW median * monthly factor
    {cam}_baseline_mean     : weighted mean (reference only)
    {cam}_residual          : observed - baseline_median
    {cam}_auto_outlier      : bool — ISO week flagged as auto-detected outlier
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal

from loader import DATE_COL


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LookbackMonths = Literal[3, 6, 12]

MIN_MONTHS_FOR_SEASONALITY = 24  # require 2 full years to activate monthly factor
OUTLIER_SIGMA_THRESHOLD = 3.0  # weeks beyond this sigma are auto-excluded


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_baseline(
    df: pd.DataFrame,
    camera_cols: list[str],
    lookback_months: LookbackMonths = 12,
    decay: float = 0.3,
) -> pd.DataFrame:
    """
    Compute baseline values for every camera column.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe from loader.load_csv.
    camera_cols : list[str]
        Camera column names to process.
    lookback_months : {3, 6, 12}
        Historical window used for baseline computation.
    decay : float in [0.0, 1.0]
        Recency decay factor. 0 = flat weights, 1 = heavy recency preference.

    Returns
    -------
    pd.DataFrame
        Original dataframe enriched with per-camera baseline columns.
    """
    cutoff = _cutoff_date(df, lookback_months)
    result = df.copy()

    # Pre-compute monthly seasonality factors using FULL history (not just lookback)
    monthly_factors = {cam: _compute_monthly_factors(df, cam) for cam in camera_cols}

    for cam in camera_cols:
        factors = monthly_factors[cam]

        # ------------------------------------------------------------------
        # Pass 1 — raw baseline without outlier exclusion
        # ------------------------------------------------------------------
        raw_medians = _compute_dow_medians(
            df,
            cam,
            cutoff,
            decay,
            exclude_outlier_weeks=set(),
            factors=factors,
        )

        # ------------------------------------------------------------------
        # Flag outlier ISO weeks from pass-1 residuals
        # ------------------------------------------------------------------
        raw_residuals = df[cam].values.astype(float) - np.array(raw_medians)
        outlier_weeks = _flag_outlier_weeks(df, raw_residuals)

        # ------------------------------------------------------------------
        # Pass 2 — recompute with outlier weeks excluded
        # ------------------------------------------------------------------
        final_medians = _compute_dow_medians(
            df,
            cam,
            cutoff,
            decay,
            exclude_outlier_weeks=outlier_weeks,
            factors=factors,
        )
        final_means = _compute_dow_means(
            df,
            cam,
            cutoff,
            decay,
            exclude_outlier_weeks=outlier_weeks,
            factors=factors,
        )

        result[f"{cam}_baseline_median"] = final_medians
        result[f"{cam}_baseline_mean"] = final_means
        result[f"{cam}_residual"] = result[cam] - result[f"{cam}_baseline_median"]
        result[f"{cam}_auto_outlier"] = result["_iso_week_id"].isin(outlier_weeks)

    return result


# ---------------------------------------------------------------------------
# Monthly seasonality
# ---------------------------------------------------------------------------


def _compute_monthly_factors(df: pd.DataFrame, cam: str) -> dict[int, float]:
    """
    Compute multiplicative monthly seasonality factors for a camera.

    Returns dict mapping month (1-12) → multiplicative factor.
    Returns {m: 1.0} for all months if < MIN_MONTHS_FOR_SEASONALITY of history.
    """
    flat = {m: 1.0 for m in range(1, 13)}

    history_months = (df[DATE_COL].max() - df[DATE_COL].min()).days / 30.4
    if history_months < MIN_MONTHS_FOR_SEASONALITY:
        return flat

    clean_mask = (~df["_iso_week_has_event"]) & (df[cam] > 0)
    clean = df.loc[clean_mask, [DATE_COL, cam, "_dow"]].copy()
    clean["_month"] = clean[DATE_COL].dt.month

    if clean.empty:
        return flat

    # DOW-median per (month, dow) to control for weekday composition bias
    dow_month_med = clean.groupby(["_month", "_dow"])[cam].median().reset_index()

    # Monthly median across DOW medians (avoids weekend-heavy months skewing)
    monthly_med = dow_month_med.groupby("_month")[cam].median()
    global_med = float(monthly_med.median())

    if global_med == 0 or np.isnan(global_med):
        return flat

    factors = {}
    for m in range(1, 13):
        if (
            m in monthly_med.index
            and not np.isnan(monthly_med[m])
            and monthly_med[m] > 0
        ):
            factors[m] = float(monthly_med[m]) / global_med
        else:
            factors[m] = 1.0  # no data for this month → no adjustment

    return factors


# ---------------------------------------------------------------------------
# Outlier week detection
# ---------------------------------------------------------------------------


def _flag_outlier_weeks(
    df: pd.DataFrame,
    residuals: np.ndarray,
) -> set[str]:
    """
    Flag ISO weeks whose median absolute residual exceeds OUTLIER_SIGMA_THRESHOLD
    standard deviations. Uses IQR-based sigma estimate for robustness.

    Only flags non-event weeks — event weeks are already excluded.
    """
    tmp = df[["_iso_week_id", "_iso_week_has_event"]].copy()
    tmp["_residual"] = residuals

    non_event = tmp[~tmp["_iso_week_has_event"]].copy()
    if non_event.empty:
        return set()

    # IQR-based sigma (robust to existing outliers contaminating the estimate)
    q75, q25 = np.nanpercentile(non_event["_residual"], [75, 25])
    sigma = max((q75 - q25) / 1.349, 1.0)

    # Median absolute residual per ISO week
    week_resid = (
        non_event.groupby("_iso_week_id")["_residual"]
        .apply(lambda x: float(np.nanmedian(np.abs(x))) if x.notna().any() else np.nan)
        .dropna()
    )

    return set(week_resid[week_resid > OUTLIER_SIGMA_THRESHOLD * sigma].index.tolist())


# ---------------------------------------------------------------------------
# Core DOW baseline computation
# ---------------------------------------------------------------------------


def _get_pool(
    df: pd.DataFrame,
    cam: str,
    target_dow: int,
    target_date: pd.Timestamp,
    cutoff: pd.Timestamp,
    exclude_outlier_weeks: set[str],
) -> pd.DataFrame:
    """Return the clean observation pool for a given (camera, DOW, date)."""
    mask = (
        (df["_dow"] == target_dow)
        & (df[DATE_COL] >= cutoff)
        & (df[DATE_COL] < target_date)
        & (~df["_iso_week_has_event"])
        & (~df["_iso_week_id"].isin(exclude_outlier_weeks))
        & (df[cam] > 0)
    )
    return df.loc[mask, [DATE_COL, cam, "_iso_week_id"]].dropna(subset=[cam])


def _get_pool_fallback(
    df: pd.DataFrame,
    cam: str,
    target_dow: int,
    target_date: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Last-resort pool: relax event and outlier exclusion, keep zero exclusion."""
    mask = (
        (df["_dow"] == target_dow)
        & (df[DATE_COL] >= cutoff)
        & (df[DATE_COL] < target_date)
        & (df[cam] > 0)
    )
    return df.loc[mask, [DATE_COL, cam, "_iso_week_id"]].dropna(subset=[cam])


def _compute_dow_medians(
    df: pd.DataFrame,
    cam: str,
    cutoff: pd.Timestamp,
    decay: float,
    exclude_outlier_weeks: set[str],
    factors: dict[int, float],
) -> list[float]:
    medians: list[float] = []

    for _, row in df.iterrows():
        target_dow = row["_dow"]
        target_date = row[DATE_COL]
        target_month = target_date.month

        pool = _get_pool(
            df, cam, target_dow, target_date, cutoff, exclude_outlier_weeks
        )

        if pool.empty:
            # Relax outlier exclusion only
            pool = _get_pool(df, cam, target_dow, target_date, cutoff, set())

        if pool.empty:
            # Last resort: relax everything except zeros
            pool = _get_pool_fallback(df, cam, target_dow, target_date, cutoff)

        if pool.empty:
            medians.append(np.nan)
            continue

        values = pool[cam].values.astype(float)
        weights = _recency_weights(pool[DATE_COL], decay)
        med = _weighted_median(values, weights) * factors.get(target_month, 1.0)
        medians.append(med)

    return medians


def _compute_dow_means(
    df: pd.DataFrame,
    cam: str,
    cutoff: pd.Timestamp,
    decay: float,
    exclude_outlier_weeks: set[str],
    factors: dict[int, float],
) -> list[float]:
    means: list[float] = []

    for _, row in df.iterrows():
        target_dow = row["_dow"]
        target_date = row[DATE_COL]
        target_month = target_date.month

        pool = _get_pool(
            df, cam, target_dow, target_date, cutoff, exclude_outlier_weeks
        )

        if pool.empty:
            means.append(np.nan)
            continue

        values = pool[cam].values.astype(float)
        weights = _recency_weights(pool[DATE_COL], decay)
        mean = float(np.average(values, weights=weights)) * factors.get(
            target_month, 1.0
        )
        means.append(mean)

    return means


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
# MSX


def _cutoff_date(df: pd.DataFrame, months: LookbackMonths) -> pd.Timestamp:
    return df[DATE_COL].max() - pd.DateOffset(months=months)


def _recency_weights(dates: pd.Series, decay: float) -> np.ndarray:
    if decay == 0.0 or len(dates) <= 1:
        return np.ones(len(dates)) / len(dates)

    max_date = dates.max()
    days_back = (max_date - dates).dt.days.values.astype(float)
    rate = decay * (np.log(100) / 365)
    weights = np.exp(-rate * days_back)
    return weights / weights.sum()


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan

    sort_idx = np.argsort(values)
    sorted_values = values[sort_idx]
    sorted_weights = weights[sort_idx]
    cumulative = np.cumsum(sorted_weights)
    median_idx = min(np.searchsorted(cumulative, 0.5), len(sorted_values) - 1)
    return float(sorted_values[median_idx])
