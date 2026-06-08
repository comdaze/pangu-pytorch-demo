"""
Real results and configuration extracted from the paper:
"Bridging the Weather Forecasting Gap: Region-Aware and Variable-Specific
 Adaptation of Weather Foundation Models" (VAAWM).

All RMSE numbers below are transcribed directly from the paper's Tables 1-4.
Lower RMSE is better. Lead times T+1 ... T+10 correspond to 24H-interval
forecasts over a 10-day horizon.
"""

# Lead-time labels for the 10-day, 24H tables
LEAD_TIMES = [f"T+{i}" for i in range(1, 11)]

# -----------------------------------------------------------------------------
# Table 1: 10-day 10-meter wind speed forecast RMSE in Xinjiang (Post-2018, 24H)
# -----------------------------------------------------------------------------
TABLE1_XINJIANG_10M_POST2018 = {
    "Pangu-Weather (zero-shot)": [0.859, 0.942, 1.002, 1.079, 1.159, 1.273, 1.376, 1.442, 1.502, 1.554],
    "Pangu-Weather (VAAWM, Ours)": [0.724, 0.778, 0.839, 0.912, 0.999, 1.121, 1.231, 1.314, 1.376, 1.433],
    "AutoGluon (Ensemble)": [1.202, 1.234, 1.239, 1.241, 1.242, 1.245, 1.246, 1.242, 1.242, 1.245],
}

# -----------------------------------------------------------------------------
# Table 2: 10-day 10-meter wind speed forecast RMSE in Xinjiang (Pre-2019, 24H)
# -----------------------------------------------------------------------------
TABLE2_XINJIANG_10M_PRE2019 = {
    "Pangu-Weather (zero-shot)": [0.822, 0.912, 0.984, 1.063, 1.161, 1.287, 1.399, 1.469, 1.553, 1.609],
    "Pangu-Weather (VAAWM, Ours)": [0.729, 0.789, 0.847, 0.919, 1.021, 1.148, 1.261, 1.337, 1.418, 1.477],
    "AutoGluon (Ensemble)": [1.234, 1.257, 1.258, 1.257, 1.256, 1.256, 1.256, 1.258, 1.260, 1.261],
}

# -----------------------------------------------------------------------------
# Table 3: Ablation - 10-day 10-meter wind speed RMSE in Xinjiang (Post-2018, 24H)
# -----------------------------------------------------------------------------
TABLE3_ABLATION_XINJIANG_10M_POST2018 = {
    "Zero-shot": [0.859, 0.942, 1.002, 1.079, 1.159, 1.273, 1.376, 1.442, 1.502, 1.554],
    "Standard Fine-tuning (uniform MSE)": [0.900, 0.973, 1.028, 1.096, 1.171, 1.274, 1.371, 1.429, 1.486, 1.532],
    "VAAWM (Ours)": [0.724, 0.778, 0.839, 0.912, 0.999, 1.121, 1.231, 1.314, 1.376, 1.433],
}

# -----------------------------------------------------------------------------
# Table 4: 10-day 850hPa wind speed forecast RMSE in Xinjiang (Post-2018, 2024 test)
# -----------------------------------------------------------------------------
TABLE4_XINJIANG_850_POST2018 = {
    "Pangu-Weather (zero-shot)": [1.575, 1.715, 1.822, 1.986, 2.178, 2.419, 2.694, 2.861, 2.946, 3.026],
    "Pangu-Weather (VAAWM, Ours)": [1.390, 1.505, 1.633, 1.807, 2.024, 2.273, 2.574, 2.769, 2.854, 2.947],
}

# Lookup used by the interactive results section.
# key -> (title, data dict)
RESULTS = {
    ("Xinjiang", "10m wind speed", "Post-2018"): (
        "Table 1: 10-m wind speed RMSE, Xinjiang (Post-2018, 24H)",
        TABLE1_XINJIANG_10M_POST2018,
    ),
    ("Xinjiang", "10m wind speed", "Pre-2019"): (
        "Table 2: 10-m wind speed RMSE, Xinjiang (Pre-2019, 24H)",
        TABLE2_XINJIANG_10M_PRE2019,
    ),
    ("Xinjiang", "850hPa wind speed", "Post-2018"): (
        "Table 4: 850hPa wind speed RMSE, Xinjiang (Post-2018, 2024 test)",
        TABLE4_XINJIANG_850_POST2018,
    ),
}

# Headline numbers reported in the paper (Abstract / Sec 4.2 / 4.3 / Appendix A).
HIGHLIGHTS = {
    "post2018_T1": 15.7,   # % RMSE reduction at T+1 vs zero-shot (10m, Post-2018)
    "post2018_T2": 17.4,   # % RMSE reduction at T+2 vs zero-shot (10m, Post-2018)
    "ablation_vs_zeroshot_T1": 15.7,
    "ablation_vs_standard_T1": 19.6,
    "standard_ft_degrade_T1": 4.8,   # standard FT is WORSE than zero-shot at T+1
    "upper850_T1": 11.7,   # % reduction at T+1 for 850hPa wind speed
    "headline_max": 17.4,  # up to 17.4% in short-term forecasts
}

# Region definitions, consistent with custom_mask.ipynb (lat/lon bounding boxes
# on the 0.25 deg, 721 x 1440 ERA5 grid).
# Xinjiang values are taken verbatim from custom_mask.ipynb.
# Zhejiang is an approximate bounding box matching Figure 2b of the paper.
REGIONS = {
    "新疆 (Xinjiang)": {
        "ascii": "Xinjiang",
        "lat_min": 34, "lat_max": 49,
        "lon_min": 73, "lon_max": 96,
        "desc": "干旱大陆性气候，地形平坦，昼夜温差大。",
        "exact": True,
    },
    "浙江 (Zhejiang)": {
        "ascii": "Zhejiang",
        "lat_min": 27, "lat_max": 31,
        "lon_min": 118, "lon_max": 123,
        "desc": "湿润沿海气候，地形复杂，天气变率更高。",
        "exact": False,
    },
}

# Chinese display names for the variables (for the UI).
UPPER_VARS_CN = {"z": "位势 z", "q": "比湿 q", "t": "温度 t", "u": "纬向风 u", "v": "经向风 v"}
SURFACE_VARS_CN = {"msl": "海平面气压 msl", "u10": "10m纬向风 u10",
                   "v10": "10m经向风 v10", "t2m": "2m温度 t2m"}

# Variable-specific weights actually used in the codebase (era5_data/config.py).
# Order matches ERA5_UPPER_VARIABLES = [z, q, t, u, v] and
# ERA5_SURFACE_VARIABLES = [msl, u10, v10, t2m].
UPPER_VARS = ["z", "q", "t", "u", "v"]
SURFACE_VARS = ["msl", "u10", "v10", "t2m"]
UPPER_WEIGHTS = [3.00, 0.60, 1.50, 0.77, 0.54]
SURFACE_WEIGHTS = [1.50, 0.77, 0.66, 3.00]


def reduction_pct(baseline, ours):
    """Percentage RMSE reduction of `ours` relative to `baseline` (positive = better)."""
    return [(b - o) / b * 100.0 for b, o in zip(baseline, ours)]
