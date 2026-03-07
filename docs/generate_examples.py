#!/usr/bin/env python3
"""Generate example fit plots for documentation.

Reads tests/fixtures/real_sources.json and produces example plots showing
nonparametric GP fits and parametric model fits for representative sources.

Requires: lightcurve_fitting (maturin develop --release), matplotlib
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lightcurve_fitting as lcf

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "real_sources.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "img")

BAND_COLORS = {"g": "#4daf4a", "r": "#e41a1c", "i": "#984ea3",
               "ztfg": "#4daf4a", "ztfr": "#e41a1c", "ztfi": "#984ea3"}


def load_sources():
    with open(FIXTURES) as f:
        return json.load(f)


def source_to_mag_bands(src):
    """Convert fixture source to BandDataMap (magnitude space, strip ztf prefix)."""
    d = {}
    for band_name, bdata in src["bands"].items():
        short = band_name.replace("ztf", "")
        d[short] = (bdata["times"], bdata["values"], bdata["errors"])
    return lcf.BandDataMap.from_dict(d)


def source_to_flux_bands(src):
    """Convert fixture source to BandDataMap (flux space, strip ztf prefix)."""
    times, mags, errs, bands_list = [], [], [], []
    for band_name, bdata in src["bands"].items():
        short = band_name.replace("ztf", "")
        for t, v, e in zip(bdata["times"], bdata["values"], bdata["errors"]):
            times.append(t)
            mags.append(v)
            errs.append(e)
            bands_list.append(short)
    return lcf.build_flux_bands(times, mags, errs, bands_list)


def plot_nonparametric(src, results, out_path):
    """Plot GP fit results for a source."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4),
                             squeeze=False, sharey=True)

    for i, r in enumerate(results):
        ax = axes[0, i]
        band = r["band"]
        color = BAND_COLORS.get(band, "#333333")

        # Get original data
        orig_band = f"ztf{band}" if f"ztf{band}" in src["bands"] else band
        if orig_band in src["bands"]:
            bdata = src["bands"][orig_band]
            ax.errorbar(bdata["times"], bdata["values"],
                       yerr=bdata["errors"], fmt="o", color=color,
                       markersize=3, alpha=0.6, label=f"{band} data")

        # Mark features
        if r.get("peak_mag") is not None and r.get("t0") is not None:
            ax.axhline(r["peak_mag"], color=color, ls=":", alpha=0.5)
            ax.axvline(r["t0"], color="gray", ls="--", alpha=0.3)

        if r.get("fwhm") is not None and r.get("t0") is not None:
            t0 = r["t0"]
            hw = r["fwhm"] / 2
            ax.axvspan(t0 - hw, t0 + hw, alpha=0.08, color=color)

        ax.set_xlabel("Time (days)")
        if i == 0:
            ax.set_ylabel("Magnitude")
        ax.set_title(f"{band}-band")
        ax.invert_yaxis()
        ax.legend(fontsize=8)

        # Add feature text
        features = []
        if r.get("peak_mag") is not None:
            features.append(f"peak={r['peak_mag']:.1f}")
        if r.get("rise_time") is not None:
            features.append(f"rise={r['rise_time']:.1f}d")
        if r.get("decay_time") is not None:
            features.append(f"decay={r['decay_time']:.1f}d")
        if r.get("fwhm") is not None:
            features.append(f"FWHM={r['fwhm']:.1f}d")
        if features:
            ax.text(0.02, 0.02, "\n".join(features),
                   transform=ax.transAxes, fontsize=7,
                   verticalalignment="bottom", fontfamily="monospace",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.suptitle(f"{src['obj_id']} ({src['label_name']})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_parametric(src, results, out_path):
    """Plot parametric model fit results for a source."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4),
                             squeeze=False, sharey=True)

    for i, r in enumerate(results):
        ax = axes[0, i]
        band = r["band"]
        color = BAND_COLORS.get(band, "#333333")

        # Get original data
        orig_band = f"ztf{band}" if f"ztf{band}" in src["bands"] else band
        if orig_band in src["bands"]:
            bdata = src["bands"][orig_band]
            # Convert mag to flux for plotting
            zp = 23.9
            flux = [10**((zp - v) / 2.5) for v in bdata["values"]]
            flux_err = [f * e * math.log(10) / 2.5 for f, e in zip(flux, bdata["errors"])]
            ax.errorbar(bdata["times"], flux, yerr=flux_err,
                       fmt="o", color=color, markersize=3, alpha=0.6,
                       label=f"{band} data")

        # Plot model prediction
        model = r.get("model")
        params = r.get("svi_mu") or r.get("pso_params")
        if model and params and orig_band in src["bands"]:
            bdata = src["bands"][orig_band]
            zp = 23.9
            obs_flux = [10**((zp - v) / 2.5) for v in bdata["values"]]
            t_min = min(bdata["times"]) - 5
            t_max = max(bdata["times"]) + 20
            t_pred = [t_min + (t_max - t_min) * j / 199 for j in range(200)]
            try:
                f_pred = lcf.eval_model(model, params, t_pred)
                # eval_model returns normalized flux; scale to match observed
                f_data_at_obs = lcf.eval_model(model, params, list(bdata["times"]))
                if f_data_at_obs and max(abs(v) for v in f_data_at_obs) > 1e-30:
                    scale = max(obs_flux) / max(abs(v) for v in f_data_at_obs)
                else:
                    scale = 1.0
                f_scaled = [f * scale for f in f_pred]
                ax.plot(t_pred, f_scaled, "-", color=color, linewidth=1.5,
                       alpha=0.8, label=f"{model}")
            except Exception:
                pass

        ax.set_xlabel("Time (days)")
        if i == 0:
            ax.set_ylabel("Flux")
        ax.set_title(f"{band}-band")
        ax.legend(fontsize=8)

        # Add model info
        info = []
        if model:
            info.append(f"model: {model}")
        if r.get("pso_chi2") is not None:
            info.append(f"chi2={r['pso_chi2']:.2f}")
        if info:
            ax.text(0.02, 0.98, "\n".join(info),
                   transform=ax.transAxes, fontsize=7,
                   verticalalignment="top", fontfamily="monospace",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.suptitle(f"{src['obj_id']} ({src['label_name']}) — Parametric Fit", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sources = load_sources()

    # Pick one source per class for examples
    by_class = {}
    for src in sources:
        label = src["label_name"]
        if label not in by_class:
            by_class[label] = src

    # Generate nonparametric plots
    print("Nonparametric GP fits:")
    for label, src in sorted(by_class.items()):
        bands = source_to_mag_bands(src)
        results = lcf.fit_nonparametric(bands)
        if results:
            safe_label = label.replace(" ", "_").lower()
            plot_nonparametric(src, results,
                             os.path.join(OUT_DIR, f"np_{safe_label}.png"))

    # Generate parametric plots
    print("\nParametric model fits:")
    for label, src in sorted(by_class.items()):
        bands = source_to_flux_bands(src)
        results = lcf.fit_parametric(bands, method="laplace")
        if results:
            safe_label = label.replace(" ", "_").lower()
            plot_parametric(src, results,
                          os.path.join(OUT_DIR, f"param_{safe_label}.png"))

    # Generate thermal example
    print("\nThermal fits:")
    for label in ["SN Ia", "TDE"]:
        if label not in by_class:
            continue
        src = by_class[label]
        bands = source_to_mag_bands(src)
        result = lcf.fit_fast(bands)
        thermal = result.get("thermal")
        if thermal and thermal.get("log_temp_peak") is not None:
            safe_label = label.replace(" ", "_").lower()
            temp = 10 ** thermal["log_temp_peak"]
            print(f"  {src['obj_id']} ({label}): T_peak = {temp:.0f} K")

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
