#!/usr/bin/env python3
"""
Evaluate how nonparametric GP features evolve as alerts accumulate.

Simulates real-time classification by truncating each light curve at
increasing numbers of alerts (5, 10, 15, 20, 30, 50, all) and tracking
feature quality, stability, and availability.
"""

import sys, os, csv, json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import lightcurve_fitting as lcf

DATA_DIR = "/fred/oz480/mcoughli/AppleCider/photo_events/train"
MANIFEST = "/fred/oz480/mcoughli/AppleCider/photo_events/manifest_train.csv"

SUBCLASS_NAMES = [
    "SN Ia", "SN Ib", "SN Ic", "SN II", "SN IIP", "SN IIn", "SN IIb",
    "Cataclysmic", "AGN", "TDE"
]
SHORT_NAMES = ["Ia", "Ib", "Ic", "II", "IIP", "IIn", "IIb", "CV", "AGN", "TDE"]

BAND_MAP = {0: "ztfg", 1: "ztfr", 2: "ztfi"}
ZP = 23.9

# Alert horizons to evaluate
HORIZONS = [5, 10, 15, 20, 30, 50, None]  # None = all alerts

# Key features to track
KEY_FEATURES = [
    'peak_mag', 'dm15', 'decay_efold', 'decay_halfmax',
    'near_peak_rise_rate', 'near_peak_decay_rate',
    'gp_dfdt_now', 'gp_predicted_mag_1d',
    'von_neumann_ratio', 'post_peak_monotonicity',
    'gp_peak_to_peak', 'gp_fit_lengthscale',
    'fwhm', 'rise_time',
]

N_PER_CLASS = 30


def load_source(obj_id):
    """Load AppleCider .npz, return sorted arrays."""
    path = os.path.join(DATA_DIR, f"{obj_id}.npz")
    d = np.load(path)
    data = d['data']
    label = int(d['label'])

    dt = data[:, 0]
    band_id = data[:, 2].astype(int)
    logflux = data[:, 3]
    logflux_err = data[:, 4]

    valid = (np.isfinite(logflux) & np.isfinite(logflux_err)
             & (logflux_err < 2.0) & (logflux_err > 0))
    dt = dt[valid]
    band_id = band_id[valid]
    logflux = logflux[valid]
    logflux_err = logflux_err[valid]

    mags = ZP - 2.5 * logflux
    mag_errs = 2.5 * logflux_err
    bands = [BAND_MAP.get(b, f"band{b}") for b in band_id]

    # Sort by time
    order = np.argsort(dt)
    dt = dt[order]
    mags = mags[order]
    mag_errs = mag_errs[order]
    bands = [bands[i] for i in order]

    return dt, mags, mag_errs, bands, label


def fit_at_horizon(dt, mags, mag_errs, bands, n_alerts):
    """Fit using only the first n_alerts observations."""
    if n_alerts is None:
        n = len(dt)
    else:
        n = min(n_alerts, len(dt))

    if n < 5:
        return None, None

    t = dt[:n].tolist()
    m = mags[:n].tolist()
    e = mag_errs[:n].tolist()
    b = bands[:n]

    try:
        band_data = lcf.build_mag_bands(t, m, e, b)
        results = lcf.fit_nonparametric(band_data)
    except Exception:
        return None, None

    # Aggregate: for each feature, collect values across bands
    feat_vals = {}
    for r in results:
        for key in KEY_FEATURES:
            val = r.get(key)
            if val is not None and np.isfinite(val):
                if key not in feat_vals:
                    feat_vals[key] = []
                feat_vals[key].append(val)

    # Also do thermal
    thermal_vals = {}
    try:
        band_data = lcf.build_mag_bands(t, m, e, b)
        thermal = lcf.fit_thermal(band_data)
        if thermal is not None:
            for key in ['log_temp_peak', 'log_temp_latest', 'cooling_rate']:
                val = thermal.get(key)
                if val is not None and np.isfinite(val):
                    thermal_vals[f"thermal_{key}"] = val
    except Exception:
        pass

    return feat_vals, thermal_vals


def main():
    # Read manifest
    rows = []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            row['label'] = int(row['label'])
            row['n_events'] = int(row['n_events'])
            rows.append(row)

    rng = np.random.RandomState(42)

    # Sample per class — only sources with enough alerts to be interesting
    sampled = []
    for label_id in range(10):
        subset = [r for r in rows
                  if r['label'] == label_id and r['n_events'] >= 20]
        if len(subset) > N_PER_CLASS:
            idxs = rng.choice(len(subset), N_PER_CLASS, replace=False)
            subset = [subset[i] for i in idxs]
        sampled.extend(subset)

    print(f"Evaluating {len(sampled)} sources across horizons {HORIZONS}")
    print()

    # Structure: features_by_horizon[horizon][class_name][feature] = [values...]
    features_by_horizon = {h: defaultdict(lambda: defaultdict(list))
                           for h in HORIZONS}
    fit_success = {h: defaultdict(lambda: [0, 0]) for h in HORIZONS}  # [success, total]

    # Also track feature stability: for each source, how much does the feature
    # change between consecutive horizons?
    # stability[class_name][feature] = list of (h1, h2, relative_change)
    stability = defaultdict(lambda: defaultdict(list))

    for row in sampled:
        obj_id = row['obj_id']
        label = row['label']
        class_name = SUBCLASS_NAMES[label]

        try:
            dt, mags, mag_errs, bands, _ = load_source(obj_id)
        except Exception:
            continue

        if len(dt) < 5:
            continue

        prev_feats = {}
        for h in HORIZONS:
            feat_vals, thermal_vals = fit_at_horizon(dt, mags, mag_errs, bands, h)

            actual_n = len(dt) if h is None else min(h, len(dt))
            fit_success[h][class_name][1] += 1

            if feat_vals is None:
                continue

            fit_success[h][class_name][0] += 1

            # Store per-band median for each feature
            for key, vals in feat_vals.items():
                med = np.median(vals)
                features_by_horizon[h][class_name][key].append(med)

            if thermal_vals:
                for key, val in thermal_vals.items():
                    features_by_horizon[h][class_name][key].append(val)

            # Compute stability vs previous horizon
            curr_feats = {k: np.median(v) for k, v in feat_vals.items()}
            for key in curr_feats:
                if key in prev_feats:
                    old = prev_feats[key]
                    new = curr_feats[key]
                    if abs(old) > 1e-10:
                        rel_change = abs(new - old) / abs(old)
                        stability[class_name][key].append(rel_change)
            prev_feats = curr_feats

    # =========================================================================
    # Print results
    # =========================================================================

    # 1. Feature availability (non-NaN rate) vs horizon
    print("=" * 100)
    print("FEATURE AVAILABILITY (% of sources with finite value) vs number of alerts")
    print("=" * 100)
    for feat in KEY_FEATURES + ['thermal_log_temp_peak', 'thermal_cooling_rate']:
        print(f"\n  {feat}:")
        print(f"  {'Horizon':<10}", end="")
        for s in SHORT_NAMES:
            print(f" {s:>6}", end="")
        print()
        print(f"  {'':<10}", end="")
        for _ in SHORT_NAMES:
            print(f" {'----':>6}", end="")
        print()

        for h in HORIZONS:
            h_label = "all" if h is None else str(h)
            print(f"  {h_label:<10}", end="")
            for label_id in range(10):
                class_name = SUBCLASS_NAMES[label_id]
                vals = features_by_horizon[h][class_name].get(feat, [])
                total = fit_success[h][class_name][0]
                if total > 0:
                    pct = len(vals) / total * 100
                    print(f" {pct:5.0f}%", end="")
                else:
                    print(f" {'---':>6}", end="")
            print()

    # 2. Median feature value vs horizon (focus on most discriminating features)
    print()
    print("=" * 100)
    print("MEDIAN FEATURE VALUES vs number of alerts")
    print("=" * 100)
    focus_features = ['dm15', 'near_peak_rise_rate', 'near_peak_decay_rate',
                      'gp_dfdt_now', 'peak_mag', 'von_neumann_ratio',
                      'gp_peak_to_peak', 'thermal_log_temp_peak']

    for feat in focus_features:
        print(f"\n  {feat}:")
        print(f"  {'Horizon':<10}", end="")
        for s in SHORT_NAMES:
            print(f" {s:>8}", end="")
        print()
        print(f"  {'':<10}", end="")
        for _ in SHORT_NAMES:
            print(f" {'------':>8}", end="")
        print()

        for h in HORIZONS:
            h_label = "all" if h is None else str(h)
            print(f"  {h_label:<10}", end="")
            for label_id in range(10):
                class_name = SUBCLASS_NAMES[label_id]
                vals = features_by_horizon[h][class_name].get(feat, [])
                if len(vals) >= 2:
                    print(f" {np.median(vals):8.3f}", end="")
                elif len(vals) == 1:
                    print(f" {vals[0]:8.3f}", end="")
                else:
                    print(f" {'---':>8}", end="")
            print()

    # 3. Feature stability: median relative change between consecutive horizons
    print()
    print("=" * 100)
    print("FEATURE STABILITY (median relative change between consecutive horizons)")
    print("=" * 100)
    stab_features = ['peak_mag', 'dm15', 'near_peak_rise_rate',
                     'near_peak_decay_rate', 'gp_dfdt_now', 'von_neumann_ratio']
    print(f"  {'Feature':<28}", end="")
    for s in SHORT_NAMES:
        print(f" {s:>8}", end="")
    print()
    print("  " + "-" * 26 + ("-" * 10) * 10)

    for feat in stab_features:
        print(f"  {feat:<28}", end="")
        for label_id in range(10):
            class_name = SUBCLASS_NAMES[label_id]
            changes = stability[class_name].get(feat, [])
            if len(changes) >= 3:
                print(f" {np.median(changes):8.1%}", end="")
            else:
                print(f" {'---':>8}", end="")
        print()

    # 4. Fit success rate vs horizon
    print()
    print("=" * 100)
    print("FIT SUCCESS RATE vs number of alerts")
    print("=" * 100)
    print(f"  {'Horizon':<10}", end="")
    for s in SHORT_NAMES:
        print(f" {s:>6}", end="")
    print()
    for h in HORIZONS:
        h_label = "all" if h is None else str(h)
        print(f"  {h_label:<10}", end="")
        for label_id in range(10):
            class_name = SUBCLASS_NAMES[label_id]
            succ, total = fit_success[h][class_name]
            if total > 0:
                print(f" {succ/total*100:5.0f}%", end="")
            else:
                print(f" {'---':>6}", end="")
        print()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
