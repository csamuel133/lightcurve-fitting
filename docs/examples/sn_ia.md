# SN Ia Example

Type Ia supernovae are thermonuclear explosions of white dwarfs and serve as
standardizable candles in cosmology. Their lightcurves exhibit a fast rise to
peak (~15-20 days), followed by a smooth exponential decay powered by
radioactive nickel. In magnitude space this produces a short FWHM, high
post-peak monotonicity, and a steep power-law decay index.

## Loading ZTF-like photometry

```python
import lightcurve_fitting as lcf

# Flat arrays from a ZTF-like survey
times     = [0.0, 6.0, 10.0, 13.0, 16.0, 20.0, 26.0, 32.0, 39.0, 45.0]
mags      = [19.3, 18.7, 18.7, 19.0, 19.2, 19.6, 20.0, 20.4, 20.9, 21.2]
mag_errs  = [0.17, 0.10, 0.10, 0.27, 0.15, 0.12, 0.18, 0.22, 0.25, 0.30]
bands_str = ["ztfr"] * 5 + ["ztfg"] * 5

bands = lcf.build_mag_bands(times, mags, mag_errs, bands_str)
```

Alternatively, construct a `BandDataMap` directly from a dictionary:

```python
bands = lcf.BandDataMap.from_dict({
    "r": ([0.0, 6.0, 10.0, 13.0, 32.0],
          [19.3, 18.7, 18.7, 19.0, 20.4],
          [0.17, 0.10, 0.10, 0.27, 0.22]),
    "g": ([6.0, 10.0, 13.0, 26.0, 39.0],
          [18.7, 18.7, 19.1, 20.3, 20.5],
          [0.08, 0.10, 0.19, 0.23, 0.30]),
})
```

## Nonparametric + thermal fit (`fit_fast`)

`fit_fast` runs the GP-based nonparametric fitter and the blackbody thermal
fitter in a single call, reusing the GP internally for the thermal step:

```python
result = lcf.fit_fast(bands)

np_results = result["nonparametric"]   # list of dicts, one per band
thermal    = result["thermal"]         # dict or None
```

Each nonparametric dict contains GP-derived features. For a typical SN Ia band:

```python
for r in np_results:
    print(f"Band {r['band']}:")
    print(f"  FWHM            = {r['fwhm']:.1f} days")
    print(f"  Post-peak mono  = {r['post_peak_monotonicity']:.3f}")
    print(f"  PL decay index  = {r['decay_power_law_index']:.2f}")
    print(f"  Von Neumann     = {r['von_neumann_ratio']:.3f}")
```

### Discriminating features for SN Ia

| Feature                  | Typical SN Ia range | Why                                      |
|--------------------------|---------------------|------------------------------------------|
| `fwhm`                   | ~20-40 days         | Fastest FWHM of all SN types             |
| `post_peak_monotonicity` | ~0.85-1.0           | Smooth, monotonic post-peak decline       |
| `decay_power_law_index`  | ~5-8+               | Steepest power-law index (exponential decay mapped onto power-law fit) |
| `von_neumann_ratio`      | ~0.2-0.6            | Low, reflecting smooth evolution          |

These features are validated by the class-discriminating tests in
`tests/test_real_data.rs`: SN Ia has the shortest FWHM among SN types
(`snia_fastest_fwhm`), the highest post-peak monotonicity of all classes
(`snia_highest_monotonicity`), and the steepest power-law decay index
(`snia_steepest_power_law`).

## Parametric fit

For parametric model fitting, convert to flux space first:

```python
flux_bands = lcf.build_flux_bands(times, mags, mag_errs, bands_str)
param_results = lcf.fit_parametric(flux_bands)

for r in param_results:
    print(f"Band {r['band']}: best model = {r['model']}, chi2 = {r['pso_chi2']:.4f}")
```

SN Ia lightcurves are typically best fit by the **Bazin** model, which captures
the fast-rise/exponential-decay shape. You can evaluate the best-fit model on a
dense time grid:

```python
import numpy as np

dense_t = list(np.linspace(0, 60, 200))
flux_pred = lcf.eval_model(r["model"], r["pso_params"], dense_t)
```

![Parametric fit](../img/param_sn_ia.png)
