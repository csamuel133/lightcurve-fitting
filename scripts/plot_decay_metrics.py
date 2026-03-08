#!/usr/bin/env python3
"""Plot light curves with decay metric crossings overlaid for visual verification.

Usage:
    python scripts/plot_decay_metrics.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lightcurve_fitting as lcf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "docs", "img")

# Mag offsets matching the Rust code
EFOLD_MAG = 2.5 * np.log10(np.e)   # ~1.086
HALFMAX_MAG = 2.5 * np.log10(2.0)  # ~0.753

# Representative sources (one per type)
SOURCES_TO_PLOT = {
    "sn_ia":  "ZTF21aaabwzx",
    "sn_ib":  "ZTF18abktmfz",
    "sn_ic":  "ZTF18abfzhct",
    "sn_ii":  "ZTF21abhyqlv",
    "sn_iip": "ZTF20aatqesi",
    "sn_iin": "ZTF19aacjbsj",
    "sn_iib": "ZTF20abgbuly",
    "tde":    "ZTF22aadesap",
    "cv":     "ZTF20acjpwas",
}

BAND_COLORS = {"ztfg": "green", "ztfr": "red", "ztfi": "purple"}


def main():
    with open(os.path.join(REPO, "tests", "fixtures", "real_sources.json")) as f:
        all_sources = json.load(f)

    by_id = {s["obj_id"]: s for s in all_sources}

    for slug, obj_id in SOURCES_TO_PLOT.items():
        src = by_id.get(obj_id)
        if src is None:
            print(f"Skipping {slug} ({obj_id}): not found")
            continue

        # Build bands and fit
        times, mags, errs, bands_list = [], [], [], []
        for bname, bdata in src["bands"].items():
            times.extend(bdata["times"])
            mags.extend(bdata["values"])
            errs.extend(bdata["errors"])
            bands_list.extend([bname] * len(bdata["times"]))

        mag_bands = lcf.build_mag_bands(times, mags, errs, bands_list)
        results = lcf.fit_nonparametric(mag_bands)

        n_bands = len(results)
        fig, axes = plt.subplots(1, n_bands, figsize=(3.5 * n_bands, 3.0))
        if n_bands == 1:
            axes = [axes]

        label_name = src.get("label_name", slug)
        fig.suptitle(f"{obj_id} ({label_name})", fontsize=11)

        for ax, r in zip(axes, results):
            band = r["band"]
            color = BAND_COLORS.get(band, "black")

            # Plot data
            bd = src["bands"][band]
            t_data = np.array(bd["times"])
            m_data = np.array(bd["values"])
            e_data = np.array(bd["errors"])
            ax.errorbar(t_data, m_data, yerr=e_data, fmt=".", color=color,
                        alpha=0.4, markersize=3, zorder=1)

            peak_mag = r["peak_mag"]
            peak_t = r["t0"]
            if peak_mag is None or peak_t is None:
                ax.set_title(f"{band} (no peak)", fontsize=9)
                continue

            # Mark peak
            ax.plot(peak_t, peak_mag, "k*", markersize=10, zorder=5)

            # --- e-fold threshold ---
            efold_threshold = peak_mag + EFOLD_MAG
            efold = r.get("decay_efold")
            t_plot_max = t_data.max()
            ax.axhline(efold_threshold, color="blue", linestyle=":", alpha=0.4, linewidth=0.8)
            if efold is not None:
                ax.plot(peak_t + efold, efold_threshold, "D", color="blue",
                        markersize=6, zorder=5)
                ax.annotate(f"e-fold\n{efold:.0f}d", (peak_t + efold, efold_threshold),
                            fontsize=6, color="blue", ha="left", va="bottom",
                            xytext=(3, 2), textcoords="offset points")

            # --- half-max threshold ---
            halfmax_threshold = peak_mag + HALFMAX_MAG
            halfmax = r.get("decay_halfmax")
            ax.axhline(halfmax_threshold, color="orange", linestyle=":", alpha=0.4, linewidth=0.8)
            if halfmax is not None:
                ax.plot(peak_t + halfmax, halfmax_threshold, "o", color="orange",
                        markersize=6, zorder=5)
                ax.annotate(f"t_1/2\n{halfmax:.0f}d", (peak_t + halfmax, halfmax_threshold),
                            fontsize=6, color="orange", ha="left", va="bottom",
                            xytext=(3, 2), textcoords="offset points")

            # --- dm15 ---
            dm15_val = r.get("dm15")
            if dm15_val is not None and dm15_val > 0.05:
                dm15_t = peak_t + 15.0
                dm15_mag = peak_mag + dm15_val
                ax.plot([dm15_t, dm15_t], [peak_mag, dm15_mag], "-",
                        color="red", linewidth=2, alpha=0.7, zorder=4)
                ax.plot(dm15_t, dm15_mag, "s", color="red", markersize=5, zorder=5)
                ax.annotate(f"dm15\n{dm15_val:.2f}",
                            (dm15_t, dm15_mag), fontsize=6, color="red",
                            ha="left", va="top", xytext=(3, -2),
                            textcoords="offset points")

            ax.set_title(f"{band} ({len(t_data)} pts)", fontsize=9)
            ax.set_xlabel("MJD", fontsize=8)
            ax.invert_yaxis()
            ax.tick_params(labelsize=7)

        axes[0].set_ylabel("Magnitude", fontsize=8)
        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f"decay_{slug}.png")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
