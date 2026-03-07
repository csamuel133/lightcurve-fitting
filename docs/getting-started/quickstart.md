# Quickstart

This guide walks through the core Python API using ZTF-like photometry.

## 1. Prepare band data

All fitters take a `BandDataMap` object that groups photometry by band.
Two constructors are provided depending on whether you need magnitude-space
or flux-space data.

```python
import lightcurve_fitting as lcf

# Flat arrays from your photometry table (e.g. ZTF forced photometry)
times    = [59000.1, 59000.5, 59001.2, 59002.0, 59003.5, ...]  # MJD
mags     = [20.1, 19.8, 19.3, 18.9, 19.1, ...]                 # AB mag
mag_errs = [0.12, 0.10, 0.08, 0.07, 0.09, ...]
bands    = ["ztfg", "ztfr", "ztfg", "ztfr", "ztfg", ...]

# Magnitude-space (for GP / nonparametric / thermal fitters)
mag_bands = lcf.build_mag_bands(times, mags, mag_errs, bands)

# Flux-space (for parametric fitter -- converts mags to flux with ZP=23.9)
flux_bands = lcf.build_flux_bands(times, mags, mag_errs, bands)
```

You can also build a `BandDataMap` from a pre-grouped dict:

```python
band_map = lcf.BandDataMap.from_dict({
    "ztfg": (g_times, g_mags, g_errs),
    "ztfr": (r_times, r_mags, r_errs),
})
```

## 2. Nonparametric fitting (`fit_nonparametric`)

Fits a Gaussian Process per band and extracts model-independent features.

```python
results = lcf.fit_nonparametric(mag_bands)
```

`results` is a list of dicts, one per band. Each dict contains:

```python
{
    "band": "ztfg",
    "peak_mag": 18.42,
    "t0": 3.7,                       # time of peak (relative days)
    "rise_time": 4.2,                # days from first obs to peak
    "decay_time": 12.8,              # days from peak to 1-mag fade
    "fwhm": 16.3,                    # full-width at half-maximum (days)
    "rise_rate": 0.35,               # mag/day on the rise
    "decay_rate": -0.08,             # mag/day on the decay
    "chi2": 1.02,                    # GP goodness-of-fit
    "baseline_chi2": 48.7,           # chi2 of a flat-line model
    "n_obs": 47,
    "gp_dfdt_now": -0.05,            # GP 1st derivative at last obs
    "gp_dfdt_next": -0.04,           # GP 1st derivative 1 day ahead
    "gp_d2fdt2_now": 0.002,          # GP 2nd derivative at last obs
    "gp_predicted_mag_1d": 19.1,     # predicted mag 1 day ahead
    "gp_predicted_mag_2d": 19.15,    # predicted mag 2 days ahead
    "gp_time_to_peak": -8.3,         # days until GP peak (negative = past)
    "gp_extrap_slope": -0.04,        # extrapolation slope
    "gp_sigma_f": 0.62,              # GP amplitude hyperparameter
    "gp_peak_to_peak": 1.8,          # peak-to-peak GP range
    "gp_snr_max": 24.1,              # max signal-to-noise on GP curve
    "gp_dfdt_max": 0.41,             # max 1st derivative
    "gp_dfdt_min": -0.12,            # min 1st derivative
    "gp_frac_of_peak": 0.87,         # current mag as fraction of peak
    "gp_post_var_mean": 0.003,       # mean GP posterior variance
    "gp_post_var_max": 0.01,         # max GP posterior variance
    "gp_skewness": -0.3,             # skewness of GP curve
    "gp_kurtosis": 2.1,              # kurtosis of GP curve
    "gp_n_inflections": 2.0,         # number of inflection points
    "decay_power_law_index": 1.8,    # power-law decay slope
    "decay_power_law_chi2": 0.4,     # chi2 of power-law fit
    "mag_at_30d": 20.1,              # GP mag at peak + 30 days
    "mag_at_60d": 20.9,              # GP mag at peak + 60 days
    "mag_at_90d": 21.3,              # GP mag at peak + 90 days
    "von_neumann_ratio": 0.35,       # smoothness statistic
    "pre_peak_rms": 0.15,            # mag scatter before peak
    "rise_amplitude_over_noise": 12.4,  # rise significance
    "post_peak_monotonicity": 0.95,  # fraction of monotonic decay
}
```

All feature values are `None` when they cannot be computed (e.g. too few
observations).

## 3. Parametric fitting (`fit_parametric`)

Performs model selection across 8 lightcurve models using PSO, then
estimates posterior uncertainties via Laplace approximation or SVI.

```python
# method can be "laplace" (fast) or "svi" (more accurate posteriors)
results = lcf.fit_parametric(flux_bands, fit_all_models=False, method="laplace")
```

The 8 candidate models are: **Bazin**, **Villar**, **MetzgerKN**, **Tde**,
**Arnett**, **Magnetar**, **ShockCooling**, **Afterglow**.

Each band result dict contains:

```python
{
    "band": "ztfg",
    "model": "Bazin",                # best-fit model name
    "pso_params": [0.12, ...],       # PSO best-fit parameters (internal/transformed)
    "pso_chi2": 1.05,                # PSO chi2
    "svi_mu": [0.11, ...],           # posterior mean (Laplace or SVI)
    "svi_log_sigma": [-2.3, ...],    # log posterior std dev
    "svi_elbo": -42.1,               # ELBO (SVI) or log-evidence (Laplace)
    "n_obs": 47,
    "mag_chi2": 0.98,                # chi2 in magnitude space
    "per_model_chi2": {              # chi2 for every model tried
        "Bazin": 1.05,
        "Villar": 1.12,
        ...
    },
    "uncertainty_method": "Laplace",
}
```

Set `fit_all_models=True` to also populate `per_model_params` with PSO
parameters for every model (useful for model comparison studies).

## 4. Thermal fitting (`fit_thermal`)

Fits a blackbody temperature model to cross-band color differences.

```python
result = lcf.fit_thermal(mag_bands)
```

Returns a single dict (or `None` if fewer than 2 bands are present):

```python
{
    "log_temp_peak": 4.12,       # log10(T/K) at peak
    "cooling_rate": -0.015,      # d(log T)/dt
    "log_temp_peak_err": 0.08,   # uncertainty on log_temp_peak
    "cooling_rate_err": 0.003,   # uncertainty on cooling_rate
    "chi2": 0.9,
    "n_color_obs": 32,
    "n_bands_used": 2,
    "ref_band": "ztfg",
}
```

## 5. Combined fast fitting (`fit_fast`)

Runs nonparametric + thermal fitting in a single call, reusing the GP fits
from the nonparametric step for the thermal step (avoids redundant work).

```python
result = lcf.fit_fast(mag_bands)

np_results = result["nonparametric"]   # list of dicts (same as fit_nonparametric)
thermal    = result["thermal"]         # dict or None (same as fit_thermal)
```

## 6. Batch processing

Process many sources in parallel with Rayon-based multithreading.

```python
# Build band data for each source
sources_mag  = [lcf.build_mag_bands(t, m, e, b) for t, m, e, b in source_table]
sources_flux = [lcf.build_flux_bands(t, m, e, b) for t, m, e, b in source_table]

# Nonparametric + thermal (combined)
batch_results = lcf.fit_batch_fast(sources_mag)
# -> list of {"nonparametric": [...], "thermal": {...}} dicts

# Nonparametric only
batch_np = lcf.fit_batch_nonparametric(sources_mag)
# -> list of [band_result, ...] per source

# Parametric only
batch_par = lcf.fit_batch_parametric(sources_flux, fit_all_models=False, method="svi")
# -> list of [band_result, ...] per source
```

## 7. Forward model evaluation (`eval_model`)

Evaluate a parametric model at arbitrary times, useful for plotting
best-fit curves.

```python
import numpy as np

# Use the best-fit model and parameters from a parametric fit
par = results[0]  # first band result from fit_parametric
t_plot = np.linspace(0, 60, 200).tolist()

flux_pred = lcf.eval_model(par["model"], par["pso_params"], t_plot)
```

Valid model names: `"Bazin"`, `"Villar"`, `"MetzgerKN"`, `"Tde"`,
`"Arnett"`, `"Magnetar"`, `"ShockCooling"`, `"Afterglow"`.

## 8. Raw GP fitting and prediction (`fit_gp_predict`)

Low-level interface: fit a GP to training data and predict at query points.
Useful when you want GP interpolation without the full nonparametric
feature extraction.

```python
import numpy as np

train_t = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
train_m = [20.1, 19.5, 18.9, 18.4, 19.0, 19.8, 20.3]
train_e = [0.1, 0.08, 0.07, 0.06, 0.08, 0.1, 0.12]

query_t = np.linspace(0, 25, 100).tolist()

amp_candidates = [0.1, 0.5, 1.0, 2.0, 5.0]
ls_candidates  = [1.0, 3.0, 5.0, 10.0, 20.0]

result = lcf.fit_gp_predict(
    train_t, train_m, train_e,
    query_t,
    amp_candidates, ls_candidates,
)

if result is not None:
    predictions, std_devs = result
    # predictions: GP mean at each query time
    # std_devs:    GP standard deviation at each query time
```

## 9. Physical kilonova magnitudes (`metzger_kn_mags`)

Evaluate the Metzger kilonova model in physical AB magnitude space across
multiple bands. Unlike `eval_model("MetzgerKN", ...)` which returns
normalized flux, this produces absolute magnitudes given physical
parameters and a luminosity distance.

```python
params = [
    -2.0,   # log10(M_ej / M_sun)
    -0.8,   # log10(v_ej / c)
     0.5,   # log10(kappa / cm^2 g^-1)
     0.0,   # t0 offset (days)
]

times = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0]

band_freqs = [
    ("g", 6.3e14),   # ZTF g-band central frequency (Hz)
    ("r", 4.7e14),   # ZTF r-band
    ("i", 3.9e14),   # ZTF i-band
]

d_l_cm = 1.26e26  # 40 Mpc in cm

mags = lcf.metzger_kn_mags(params, times, band_freqs, d_l_cm)
# mags = {"g": [mag_values], "r": [mag_values], "i": [mag_values]}
```
