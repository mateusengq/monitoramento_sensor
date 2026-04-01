"""
correlation.py
--------------
Computes the intra-week correlation index between cameras.

The goal is to distinguish two failure modes:
    (A) Camera problem   → one camera drops while others stay normal
    (B) Real flow drop   → all cameras drop together (correlated)

Isolation score — group-mask fix
---------------------------------
Previous implementation used the current-week group median as the anchor for
each camera's isolation score. This causes a group-mask problem: if most cameras
drop together (e.g. a real flow crisis), the group median also drops, making each
camera's delta appear "normal" relative to the group — and isolation scores come
out low for all cameras, masking a potential widespread equipment failure.

Fix: each camera is now anchored to its OWN historical delta distribution over
the last 12 clean ISO weeks. Isolation measures deviation from the camera's own
historical behaviour, not from the current group.

    historical_anchor_i  = median(weekly_delta_i over last 12 clean weeks)
    historical_mad_i     = MAD(weekly_delta_i over last 12 clean weeks)
    isolation_i          = |delta_i_current - historical_anchor_i| / historical_mad_i

Fallback: if < 4 weeks of history exist, reverts to current group median.

Output columns
--------------
    _corr_index        : mean pairwise Pearson correlation in the analysis week
    {cam}_isolation    : isolation score anchored to camera's own history
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations

from loader import DATE_COL
from utils import last_complete_iso_week, iso_week_mask


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_correlation(
    df: pd.DataFrame,
    camera_cols: list[str],
) -> pd.DataFrame:
    """
    Compute intra-week correlation index and per-camera isolation scores.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with baseline columns attached.
    camera_cols : list[str]
        Camera column names.

    Returns
    -------
    pd.DataFrame
        Input dataframe with _corr_index and {cam}_isolation columns appended.
    """
    result = df.copy()

    week_start, week_end = last_complete_iso_week(result)
    mask_week = iso_week_mask(result, week_start, week_end)
    week_df = result.loc[mask_week, [DATE_COL] + camera_cols].copy()

    # ------------------------------------------------------------------
    # 1. Correlation index — mean pairwise Pearson over daily values
    # ------------------------------------------------------------------
    corr_index = _mean_pairwise_correlation(week_df, camera_cols)
    result["_corr_index"] = np.nan
    result.loc[mask_week, "_corr_index"] = corr_index

    # ------------------------------------------------------------------
    # 2. Current-week relative delta per camera
    # ------------------------------------------------------------------
    current_deltas: dict[str, float] = {}

    for cam in camera_cols:
        baseline_col = f"{cam}_baseline_median"
        if baseline_col not in result.columns:
            current_deltas[cam] = np.nan
            continue

        obs_med = week_df[cam].median()
        base_med = result.loc[mask_week, baseline_col].median()

        if pd.isna(base_med) or base_med == 0:
            current_deltas[cam] = np.nan
        else:
            current_deltas[cam] = (obs_med - base_med) / abs(base_med)

    # ------------------------------------------------------------------
    # 3. Isolation score anchored to each camera's own historical delta
    # ------------------------------------------------------------------
    for cam in camera_cols:
        result[f"{cam}_isolation"] = np.nan

        delta_curr = current_deltas.get(cam, np.nan)
        if pd.isna(delta_curr):
            continue

        anchor, mad = _historical_delta_anchor(result, cam, week_start, n_weeks=12)

        if pd.isna(anchor):
            # Fallback: group median of current week (original behaviour)
            valid = np.array([v for v in current_deltas.values() if not np.isnan(v)])
            if len(valid) >= 2:
                anchor = float(np.median(valid))
                mad = float(np.median(np.abs(valid - anchor))) + 1e-9
            else:
                continue

        isolation = min(abs(delta_curr - anchor) / mad, 10.0)
        result.loc[mask_week, f"{cam}_isolation"] = float(isolation)

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _historical_delta_anchor(
    df: pd.DataFrame,
    cam: str,
    before_date: pd.Timestamp,
    n_weeks: int = 12,
) -> tuple[float, float]:
    """
    Compute the median and MAD of the weekly relative delta for a camera
    over the last n_weeks clean ISO weeks preceding before_date.

    Returns (median_delta, mad_delta). Both NaN if < 4 clean weeks available.
    """
    baseline_col = f"{cam}_baseline_median"
    if baseline_col not in df.columns:
        return np.nan, 1.0

    past = df[(df[DATE_COL] < before_date) & (~df["_iso_week_has_event"])][
        [DATE_COL, "_iso_week_id", cam, baseline_col]
    ].copy()

    if past.empty:
        return np.nan, 1.0

    week_deltas: list[float] = []
    for _, grp in past.groupby("_iso_week_id"):
        obs_med = grp[cam].median()
        base_med = grp[baseline_col].median()
        if pd.isna(base_med) or base_med == 0 or pd.isna(obs_med):
            continue
        week_deltas.append((obs_med - base_med) / abs(base_med))

    # Keep only the last n_weeks
    week_deltas = week_deltas[-n_weeks:]

    if len(week_deltas) < 4:
        return np.nan, 1.0

    arr = np.array(week_deltas)
    anchor = float(np.median(arr))
    mad = float(np.median(np.abs(arr - anchor))) + 1e-9
    return anchor, mad


def _mean_pairwise_correlation(
    week_df: pd.DataFrame,
    camera_cols: list[str],
) -> float:
    """Mean Pearson correlation across all camera pairs in the analysis week."""
    valid_cols = [
        c for c in camera_cols if week_df[c].notna().sum() >= 3 and week_df[c].std() > 0
    ]
    if len(valid_cols) < 2:
        return np.nan
    # MSX
    correlations = []
    for c1, c2 in combinations(valid_cols, 2):
        pair_df = week_df[[c1, c2]].dropna()
        if len(pair_df) < 3:
            continue
        r = float(pair_df[c1].corr(pair_df[c2]))
        if not np.isnan(r):
            correlations.append(r)

    return float(np.mean(correlations)) if correlations else np.nan
