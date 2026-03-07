# TDE Example

Tidal disruption events (TDEs) occur when a star passes close enough to a
supermassive black hole to be torn apart by tidal forces. The resulting flare
shows a characteristic slow power-law decay (flux proportional to t^{-5/3}),
high blackbody temperatures (>10,000 K), and smooth monotonic evolution in
magnitude space.

## Loading photometry

```python
import lightcurve_fitting as lcf

# Example TDE-like photometry (two ZTF bands)
bands = lcf.BandDataMap.from_dict({
    "g": ([0, 5, 12, 20, 35, 55, 80, 110, 150],
          [18.5, 18.2, 18.0, 18.3, 18.8, 19.4, 19.9, 20.3, 20.7],
          [0.05, 0.04, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13, 0.16]),
    "r": ([0, 5, 12, 20, 35, 55, 80, 110, 150],
          [18.8, 18.5, 18.3, 18.6, 19.0, 19.5, 20.0, 20.4, 20.8],
          [0.06, 0.05, 0.04, 0.05, 0.07, 0.09, 0.11, 0.14, 0.17]),
})
```

## Nonparametric + thermal fit

```python
result = lcf.fit_fast(bands)

np_results = result["nonparametric"]
thermal    = result["thermal"]
```

### Discriminating features for TDEs

| Feature                  | Typical TDE range | Why                                            |
|--------------------------|-------------------|------------------------------------------------|
| `decay_power_law_index`  | ~4.2              | Flux ~ t^{-5/3} maps to 2.5 * 5/3 ~ 4.2 in magnitudes |
| `decay_power_law_chi2`   | Low               | Power-law is an excellent fit (unlike SN Ia)   |
| `von_neumann_ratio`      | ~0.1-0.5          | Very smooth, monotonic evolution               |
| `post_peak_monotonicity` | ~0.8-1.0          | Nearly monotonic fading after peak             |
| `fwhm`                   | ~50-150 days      | Longer than SNe due to slow t^{-5/3} decay    |

```python
for r in np_results:
    print(f"Band {r['band']}:")
    print(f"  PL decay index   = {r['decay_power_law_index']:.2f}")
    print(f"  PL decay chi2    = {r['decay_power_law_chi2']:.4f}")
    print(f"  Von Neumann      = {r['von_neumann_ratio']:.3f}")
    print(f"  Post-peak mono   = {r['post_peak_monotonicity']:.3f}")
```

### Thermal features

TDEs are among the hottest transients. The thermal fitter measures blackbody
temperature from cross-band color differences:

```python
if thermal is not None:
    T_peak = 10 ** thermal["log_temp_peak"]
    print(f"Peak temperature   = {T_peak:.0f} K")
    print(f"Cooling rate       = {thermal['cooling_rate']:.4f}")
    print(f"Thermal chi2       = {thermal['chi2']:.4f}")
```

The `thermal_tde_hottest` test in `tests/test_real_data.rs` validates that TDEs
have higher median peak temperatures than SN Ia, SN II, SN IIP, SN Ic, and
SN Ib.

| Thermal feature    | Typical TDE range    | Why                                  |
|--------------------|----------------------|--------------------------------------|
| `log_temp_peak`    | >4.0 (>10,000 K)    | Accretion-powered, very blue         |
| `cooling_rate`     | Moderate             | Gradual cooling over months          |

## Parametric fit

For parametric fitting, the **Tde** model (flux ~ t^{-5/3} rise and decay) is
typically selected as the best fit:

```python
times_flat = [0, 5, 12, 20, 35, 55, 80, 110, 150] * 2
mags_flat  = [18.5, 18.2, 18.0, 18.3, 18.8, 19.4, 19.9, 20.3, 20.7,
              18.8, 18.5, 18.3, 18.6, 19.0, 19.5, 20.0, 20.4, 20.8]
errs_flat  = [0.05, 0.04, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13, 0.16,
              0.06, 0.05, 0.04, 0.05, 0.07, 0.09, 0.11, 0.14, 0.17]
bands_flat = ["g"] * 9 + ["r"] * 9

flux_bands = lcf.build_flux_bands(times_flat, mags_flat, errs_flat, bands_flat)
param_results = lcf.fit_parametric(flux_bands)

for r in param_results:
    print(f"Band {r['band']}: best model = {r['model']}, chi2 = {r['pso_chi2']:.4f}")
```

![NP fit](../img/np_tde.png)
![Parametric fit](../img/param_tde.png)
