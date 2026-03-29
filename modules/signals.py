"""
signals.py
----------
Computes all analytical signals per camera column over the full time series.

Signals produced
----------------
For each camera:

    EWMA
    ----
    {cam}_ewma          : exponentially weighted moving average of residuals
    {cam}_ewma_alarm    : bool — EWMA crossed its control limit

    CUSUM
    -----
    {cam}_cusum_pos     : cumulative sum tracking upward deviations
    {cam}_cusum_neg     : cumulative sum tracking downward deviations (stored as positive)
    {cam}_cusum_alarm   : bool — either CUSUM arm crossed threshold h

    Slope
    -----
    {cam}_slope_90d     : OLS slope of observed values over the last 90 days

    Zeros
    -----
    {cam}_zeros_pct_12m : fraction of non-event rows where camera = 0, last 12 months
    {cam}_zeros_week    : count of zero readings in the current analysis week

    Weekly drop
    -----------
    {cam}_week_pct_chg  : % change vs. the immediately preceding week (by DOW median)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from loader import DATE_COL, TOTAL_COL
from utils import last_complete_iso_week, week_scalar

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
