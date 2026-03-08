#!/usr/bin/env python3
"""
Evaluate how parametric fit features evolve as alerts accumulate.

Simulates real-time classification by truncating each light curve at
increasing numbers of alerts and tracking model selection, chi2, and
parameter stability.
"""

import sys, os, csv
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
HORIZONS = [5, 10, 15, 20, 30, 50, None]

# Models we track
ALL_MODELS = ["Bazin", "Villar", "MetzgerKN", "Tde", "Arnett", "Magnetar",
              "ShockCooling", "Afterglow"]

N_PER_CLASS = 30


def load_source_flux(obj_id):
    """Load AppleCider .npz, return flux-space data."""
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

    # Convert to linear flux
    flux = 10.0 ** logflux
    # Error propagation: d(10^x) = 10^x * ln(10) * dx
    flux_err = flux * np.log(10.0) * logflux_err

    bands = [BAND_MAP.get(b, f"band{b}") for b in band_id]

    # Sort by time
    order = np.argsort(dt)
    dt = dt[order]
    flux = flux[order]
    flux_err = flux_err[order]
    bands = [bands[i] for i in order]

    return dt, flux, flux_err, bands, label


def fit_at_horizon(dt, flux, flux_err, bands, n_alerts):
    """Parametric fit using only the first n_alerts observations."""
    if n_alerts is None:
        n = len(dt)
    else:
        n = min(n_alerts, len(dt))

    if n < 5:
        return None

    t = dt[:n].tolist()
    f = flux[:n].tolist()
    e = flux_err[:n].tolist()
    b = bands[:n]

    try:
        band_data = lcf.build_raw_flux_bands(t, f, e, b)
        results = lcf.fit_parametric(band_data, fit_all_models=True, method="svi")
        return results
    except Exception as ex:
        return None


def main():
    # Read manifest
    rows = []
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            row['label'] = int(row['label'])
            row['n_events'] = int(row['n_events'])
            rows.append(row)

    rng = np.random.RandomState(42)

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

    # Track: model selection, chi2, parameter uncertainty, fit success
    # model_counts[horizon][class][model_name] = count
    model_counts = {h: defaultdict(lambda: defaultdict(int)) for h in HORIZONS}
    # chi2_vals[horizon][class] = [chi2 values]
    chi2_vals = {h: defaultdict(list) for h in HORIZONS}
    # mag_chi2_vals[horizon][class] = [mag_chi2 values]
    mag_chi2_vals = {h: defaultdict(list) for h in HORIZONS}
    # per_model_chi2[horizon][class][model] = [chi2 values]
    per_model_chi2 = {h: defaultdict(lambda: defaultdict(list)) for h in HORIZONS}
    # param_uncertainty[horizon][class] = [mean log_sigma]
    param_unc = {h: defaultdict(list) for h in HORIZONS}
    # fit_success[horizon][class] = [success, total]
    fit_success = {h: defaultdict(lambda: [0, 0]) for h in HORIZONS}
    # n_bands_fit[horizon][class] = [n_bands per source]
    n_bands_fit = {h: defaultdict(list) for h in HORIZONS}

    # Model stability: does the selected model change between horizons?
    model_changes = defaultdict(lambda: [0, 0])  # [changes, comparisons]

    for row in sampled:
        obj_id = row['obj_id']
        label = row['label']
        class_name = SUBCLASS_NAMES[label]

        try:
            dt, flux, flux_err, bands, _ = load_source_flux(obj_id)
        except Exception:
            for h in HORIZONS:
                fit_success[h][class_name][1] += 1
            continue

        if len(dt) < 5:
            for h in HORIZONS:
                fit_success[h][class_name][1] += 1
            continue

        prev_models = {}  # band -> model name
        for h in HORIZONS:
            fit_success[h][class_name][1] += 1
            results = fit_at_horizon(dt, flux, flux_err, bands, h)

            if results is None or len(results) == 0:
                continue

            fit_success[h][class_name][0] += 1
            n_bands_fit[h][class_name].append(len(results))

            for r in results:
                band = r.get('band', '')
                model = r.get('model', '')
                chi2 = r.get('pso_chi2')
                mchi2 = r.get('mag_chi2')
                log_sigma = r.get('svi_log_sigma', [])

                model_counts[h][class_name][model] += 1

                if chi2 is not None and np.isfinite(chi2):
                    chi2_vals[h][class_name].append(chi2)
                if mchi2 is not None and np.isfinite(mchi2):
                    mag_chi2_vals[h][class_name].append(mchi2)

                if log_sigma:
                    finite_ls = [s for s in log_sigma if np.isfinite(s)]
                    if finite_ls:
                        param_unc[h][class_name].append(np.mean(finite_ls))

                # Per-model chi2
                pmc = r.get('per_model_chi2', {})
                for mname, mchi2_val in pmc.items():
                    if mchi2_val is not None and np.isfinite(mchi2_val):
                        per_model_chi2[h][class_name][mname].append(mchi2_val)

                # Track model stability
                if band in prev_models:
                    model_changes[class_name][1] += 1
                    if prev_models[band] != model:
                        model_changes[class_name][0] += 1
                prev_models[band] = model

    # =========================================================================
    # Print results
    # =========================================================================

    # 1. Fit success rate
    print("=" * 100)
    print("FIT SUCCESS RATE (% of sources producing at least one band result)")
    print("=" * 100)
    print(f"  {'Horizon':<10}", end="")
    for s in SHORT_NAMES:
        print(f" {s:>6}", end="")
    print()
    for h in HORIZONS:
        h_label = "all" if h is None else str(h)
        print(f"  {h_label:<10}", end="")
        for label_id in range(10):
            cn = SUBCLASS_NAMES[label_id]
            s, t = fit_success[h][cn]
            if t > 0:
                print(f" {s/t*100:5.0f}%", end="")
            else:
                print(f" {'---':>6}", end="")
        print()

    # 2. Mean number of bands fit per source
    print()
    print("=" * 100)
    print("MEAN BANDS FIT PER SOURCE")
    print("=" * 100)
    print(f"  {'Horizon':<10}", end="")
    for s in SHORT_NAMES:
        print(f" {s:>6}", end="")
    print()
    for h in HORIZONS:
        h_label = "all" if h is None else str(h)
        print(f"  {h_label:<10}", end="")
        for label_id in range(10):
            cn = SUBCLASS_NAMES[label_id]
            vals = n_bands_fit[h][cn]
            if vals:
                print(f" {np.mean(vals):5.1f}", end="")
            else:
                print(f" {'---':>6}", end="")
        print()

    # 3. Best model selection distribution
    print()
    print("=" * 100)
    print("BEST MODEL SELECTION (% of band-fits selecting each model)")
    print("=" * 100)
    for model in ALL_MODELS:
        print(f"\n  {model}:")
        print(f"  {'Horizon':<10}", end="")
        for s in SHORT_NAMES:
            print(f" {s:>6}", end="")
        print()
        for h in HORIZONS:
            h_label = "all" if h is None else str(h)
            print(f"  {h_label:<10}", end="")
            for label_id in range(10):
                cn = SUBCLASS_NAMES[label_id]
                total = sum(model_counts[h][cn].values())
                count = model_counts[h][cn].get(model, 0)
                if total > 0:
                    print(f" {count/total*100:5.0f}%", end="")
                else:
                    print(f" {'---':>6}", end="")
            print()

    # 4. Median PSO chi2
    print()
    print("=" * 100)
    print("MEDIAN PSO CHI2 (best model, flux-space)")
    print("=" * 100)
    print(f"  {'Horizon':<10}", end="")
    for s in SHORT_NAMES:
        print(f" {s:>8}", end="")
    print()
    for h in HORIZONS:
        h_label = "all" if h is None else str(h)
        print(f"  {h_label:<10}", end="")
        for label_id in range(10):
            cn = SUBCLASS_NAMES[label_id]
            vals = chi2_vals[h][cn]
            if len(vals) >= 2:
                print(f" {np.median(vals):8.2f}", end="")
            elif len(vals) == 1:
                print(f" {vals[0]:8.2f}", end="")
            else:
                print(f" {'---':>8}", end="")
        print()

    # 5. Median mag chi2
    print()
    print("=" * 100)
    print("MEDIAN MAGNITUDE CHI2 (best model)")
    print("=" * 100)
    print(f"  {'Horizon':<10}", end="")
    for s in SHORT_NAMES:
        print(f" {s:>8}", end="")
    print()
    for h in HORIZONS:
        h_label = "all" if h is None else str(h)
        print(f"  {h_label:<10}", end="")
        for label_id in range(10):
            cn = SUBCLASS_NAMES[label_id]
            vals = mag_chi2_vals[h][cn]
            if len(vals) >= 2:
                print(f" {np.median(vals):8.2f}", end="")
            elif len(vals) == 1:
                print(f" {vals[0]:8.2f}", end="")
            else:
                print(f" {'---':>8}", end="")
        print()

    # 6. Parameter uncertainty (mean log_sigma)
    print()
    print("=" * 100)
    print("MEDIAN PARAMETER UNCERTAINTY (mean log_sigma across params)")
    print("=" * 100)
    print(f"  {'Horizon':<10}", end="")
    for s in SHORT_NAMES:
        print(f" {s:>8}", end="")
    print()
    for h in HORIZONS:
        h_label = "all" if h is None else str(h)
        print(f"  {h_label:<10}", end="")
        for label_id in range(10):
            cn = SUBCLASS_NAMES[label_id]
            vals = param_unc[h][cn]
            if len(vals) >= 2:
                print(f" {np.median(vals):8.2f}", end="")
            elif len(vals) == 1:
                print(f" {vals[0]:8.2f}", end="")
            else:
                print(f" {'---':>8}", end="")
        print()

    # 7. Per-model median chi2 at full data
    print()
    print("=" * 100)
    print("PER-MODEL MEDIAN CHI2 (at full data, all classes combined)")
    print("=" * 100)
    h_full = None
    all_pmc = defaultdict(list)
    for label_id in range(10):
        cn = SUBCLASS_NAMES[label_id]
        for model, vals in per_model_chi2[h_full][cn].items():
            all_pmc[model].extend(vals)
    print(f"  {'Model':<20} {'median chi2':>12} {'p25':>8} {'p75':>8} {'n':>6}")
    print("  " + "-" * 56)
    for model in ALL_MODELS:
        vals = all_pmc.get(model, [])
        if vals:
            print(f"  {model:<20} {np.median(vals):12.2f} {np.percentile(vals, 25):8.2f} {np.percentile(vals, 75):8.2f} {len(vals):>6}")

    # 8. Model stability
    print()
    print("=" * 100)
    print("MODEL STABILITY (% of band-fits where model changes between horizons)")
    print("=" * 100)
    print(f"  {'Class':<12} {'changes':>8} {'comparisons':>12} {'change rate':>12}")
    print("  " + "-" * 46)
    for label_id in range(10):
        cn = SUBCLASS_NAMES[label_id]
        changes, comps = model_changes[cn]
        if comps > 0:
            print(f"  {SHORT_NAMES[label_id]:<12} {changes:>8} {comps:>12} {changes/comps*100:11.1f}%")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
