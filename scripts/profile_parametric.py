#!/usr/bin/env python3
"""Profile parametric fitter throughput and fit quality on AppleCider sources.

Usage:
    python scripts/profile_parametric.py /path/to/AppleCider/photo_events/train
    python scripts/profile_parametric.py /path/to/data --n-sources 500 --fit-all-models

Requires lightcurve_fitting to be installed (e.g. via `maturin develop --release`
from the python/ subdirectory).
"""

import argparse
import glob
import os
import time
from collections import Counter

import numpy as np

import lightcurve_fitting as lcf

BAND_MAP = {0: "ztfg", 1: "ztfr", 2: "ztfi"}


def load_applecider_source(path):
    """Load an AppleCider .npz source file.

    Returns (times, mags, mag_errs, band_names) or None if the file is
    invalid or has fewer than 10 valid observations.
    """
    try:
        d = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if "data" not in d or "columns" not in d:
        return None

    data = d["data"]
    cols = list(d["columns"])
    if "dt" not in cols or "logflux" not in cols:
        return None

    dt = data[:, cols.index("dt")].astype(float)
    logflux = data[:, cols.index("logflux")].astype(float)
    logflux_err = data[:, cols.index("logflux_err")].astype(float)
    band_id = data[:, cols.index("band_id")].astype(int)

    mag = -2.5 * logflux + 23.9
    mag_err = 2.5 * logflux_err
    bands = [BAND_MAP.get(b, f"band{b}") for b in band_id]

    good = np.isfinite(dt) & np.isfinite(mag) & np.isfinite(mag_err) & (mag_err > 0)
    if good.sum() < 10:
        return None

    dt = dt[good]
    mag = mag[good]
    mag_err = mag_err[good]
    bands = [bands[i] for i in range(len(bands)) if good[i]]

    return dt.tolist(), mag.tolist(), mag_err.tolist(), bands


def load_sources(data_dir, n_sources, seed=42):
    """Load up to n_sources from data_dir, shuffled deterministically."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    sources = []
    for f in files:
        if len(sources) >= n_sources:
            break
        src = load_applecider_source(f)
        if src is not None:
            sources.append((os.path.basename(f), src))

    return sources


def profile_throughput(sources):
    """Time individual fits and report statistics."""
    times_list = []
    n_pts_list = []

    for _name, (t, m, e, b) in sources:
        bands = lcf.build_flux_bands(t, m, e, b)
        t0 = time.perf_counter()
        lcf.fit_parametric(bands, False)
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)
        n_pts_list.append(len(t))

    times_arr = np.array(times_list)
    pts_arr = np.array(n_pts_list)
    total_obs = pts_arr.sum()

    print(f"\n=== Per-source timing (N={len(sources)}) ===")
    print(f"Total:  {times_arr.sum():.2f} s")
    print(f"Mean:   {times_arr.mean()*1000:.1f} ms")
    print(f"Median: {np.median(times_arr)*1000:.1f} ms")
    print(f"p90:    {np.percentile(times_arr, 90)*1000:.1f} ms")
    print(f"p99:    {np.percentile(times_arr, 99)*1000:.1f} ms")
    print(f"Max:    {times_arr.max()*1000:.1f} ms")
    print(f"Min:    {times_arr.min()*1000:.1f} ms")
    print(f"\nThroughput: {len(sources)/times_arr.sum():.0f} src/sec, "
          f"{total_obs/times_arr.sum():.0f} obs/sec")

    print(f"\n=== Scaling with point count ===")
    for lo, hi in [(0, 30), (30, 60), (60, 100), (100, 200), (200, 500), (500, 10000)]:
        mask = (pts_arr >= lo) & (pts_arr < hi)
        if mask.sum() > 0:
            print(f"  [{lo:>4d}, {hi:>4d}): n={mask.sum():>3d}, "
                  f"mean={times_arr[mask].mean()*1000:>7.1f} ms, "
                  f"median={np.median(times_arr[mask])*1000:>7.1f} ms")

    print(f"\n=== Top 10 slowest ===")
    order = np.argsort(times_arr)[::-1]
    for i in order[:10]:
        name = sources[i][0]
        print(f"  {name:30s}  {times_arr[i]*1000:>8.1f} ms  ({pts_arr[i]:>4d} pts)")


def profile_quality(sources, fit_all_models):
    """Report chi2 distribution and model selection breakdown."""
    chi2_values = []
    model_names = []

    t0 = time.perf_counter()
    for _name, (t, m, e, b) in sources:
        bands = lcf.build_flux_bands(t, m, e, b)
        results = lcf.fit_parametric(bands, fit_all_models)
        for r in results:
            chi2_values.append(r["pso_chi2"])
            model_names.append(r["model"])
    elapsed = time.perf_counter() - t0

    chi2_arr = np.array(chi2_values)

    print(f"\n=== Fit quality (N={len(sources)}, fit_all_models={fit_all_models}) ===")
    print(f"Fit in {elapsed:.1f}s ({len(sources)/elapsed:.0f} src/sec)")
    print(f"Total band-fits: {len(chi2_arr)}")
    print(f"\nPSO cost distribution:")
    print(f"  Mean:   {np.mean(chi2_arr):.3f}")
    print(f"  Median: {np.median(chi2_arr):.3f}")
    print(f"  p10:    {np.percentile(chi2_arr, 10):.3f}")
    print(f"  p90:    {np.percentile(chi2_arr, 90):.3f}")
    print(f"  p99:    {np.percentile(chi2_arr, 99):.3f}")

    model_counts = Counter(model_names)
    print(f"\nModel selection:")
    for model, count in model_counts.most_common():
        print(f"  {model:20s}: {count:>4d} ({100*count/len(model_names):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", help="Directory containing AppleCider .npz files")
    parser.add_argument("--n-sources", type=int, default=200,
                        help="Number of sources to profile (default: 200)")
    parser.add_argument("--fit-all-models", action="store_true",
                        help="Fit all 8 models (slow) instead of early-stopping at Bazin")
    parser.add_argument("--quality-only", action="store_true",
                        help="Skip throughput profiling, only report fit quality")
    args = parser.parse_args()

    sources = load_sources(args.data_dir, args.n_sources)
    pts_per_src = [len(s[1][0]) for s in sources]
    print(f"Loaded {len(sources)} sources")
    print(f"Points per source: min={min(pts_per_src)}, median={np.median(pts_per_src):.0f}, "
          f"max={max(pts_per_src)}, mean={np.mean(pts_per_src):.0f}")

    if not args.quality_only:
        profile_throughput(sources)

    profile_quality(sources, args.fit_all_models)


if __name__ == "__main__":
    main()
