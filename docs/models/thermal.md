# Thermal / Temperature Fitting

## Overview

The thermal fitter estimates blackbody temperature evolution from multi-band
photometric color differences, using GP-interpolated magnitudes to align
observations across bands to a common time grid.

The entry point is `fit_thermal`, which returns a `ThermalResult` containing
the peak temperature, cooling rate, uncertainties, and fit quality.

## How It Works

### Step 1: Reference Band GP

A reference band is selected (preferring `g` as the bluest and most
temperature-sensitive, falling back to `r`). A DenseGP is fit to the reference
band using the same grid-search strategy as the nonparametric fitter
(amplitude candidates `[0.1, 0.3]`, lengthscale factors `[6, 12, 24]`,
selected by train RMS).

If `fit_thermal` is called with pre-fitted GPs from `fit_nonparametric`, the
reference band GP is reused directly, avoiding redundant computation. This is
what `fit_fast` does internally.

### Step 2: Color Observations

For every observation in every non-reference band, the reference-band magnitude
at that time is predicted via the GP. The color (magnitude difference) is
computed as:

```
color = ref_mag_gp(t) - obs_mag(t)
```

Color errors are propagated as:

```
color_err = sqrt(obs_err^2 + 0.02^2)
```

where the 0.02 mag floor accounts for GP interpolation uncertainty.

### Step 3: Temperature Model

The temperature is modeled as an exponential in log-space:

```
T(t) = 10^(log_T0 + cooling_rate * t)
```

The predicted color between two bands at wavelengths lambda_ref and lambda is
the Planck blackbody color difference:

```
delta_m = -2.5 * log10( (lambda/lambda_ref)^3 * (exp(hc/lambda*kT) - 1) / (exp(hc/lambda_ref*kT) - 1) )
```

with a Wien-regime approximation used when exponents exceed 500 to avoid
overflow.

Supported bands and their effective wavelengths:

| Band | Wavelength |
|------|-----------|
| g | 4770 A |
| r | 6231 A |
| i | 7625 A |

### Step 4: PSO Fitting

The temperature model parameters (`log_T0`, `cooling_rate`) are fit via PSO
minimizing the reduced chi2 of predicted vs. observed colors:

| Setting | Value |
|---------|-------|
| Particles | 10 |
| Max iterations | 30 |
| Restarts | 2 (for uncertainty estimation) |
| Bounds: log_T0 | [3.0, 6.0] (1,000 to 1,000,000 K) |
| Bounds: cooling_rate | [-0.05, 0.01] |

Uncertainties are estimated from the standard deviation of best-fit parameters
across the 2 PSO restarts.

## Output

The `ThermalResult` struct contains:

| Field | Description |
|-------|-------------|
| `log_temp_peak` | log10 of the best-fit peak temperature (K) |
| `cooling_rate` | Temperature cooling rate (log10(K) per day, typically negative) |
| `log_temp_peak_err` | Uncertainty on log_temp_peak (std across restarts) |
| `cooling_rate_err` | Uncertainty on cooling_rate (std across restarts) |
| `chi2` | Reduced chi2 of the best fit |
| `n_color_obs` | Number of color observations used |
| `n_bands_used` | Number of non-reference bands contributing color data |
| `ref_band` | Name of the reference band ("g" or "r") |

A minimum of 3 color observations is required; otherwise the result is returned
with `None` for all fitted values.

## Python Usage

```python
import lightcurve_fitting as lcf

# Build multi-band data
bands = lcf.BandDataMap.from_dict({
    "g": ([0, 1, 2, 5, 10, 15, 20],
          [20.1, 19.5, 18.9, 19.2, 19.8, 20.3, 20.7],
          [0.05, 0.04, 0.03, 0.04, 0.05, 0.06, 0.07]),
    "r": ([0, 1, 2, 5, 10, 15, 20],
          [19.8, 19.2, 18.7, 19.0, 19.5, 20.0, 20.4],
          [0.05, 0.04, 0.03, 0.04, 0.05, 0.06, 0.07]),
    "i": ([0, 1, 2, 5, 10, 15, 20],
          [19.5, 18.9, 18.5, 18.8, 19.3, 19.8, 20.2],
          [0.06, 0.05, 0.04, 0.05, 0.06, 0.07, 0.08]),
})

# Standalone thermal fit
result = lcf.fit_thermal(bands)
if result is not None:
    print(f"Peak temperature: 10^{result['log_temp_peak']:.2f} K")
    print(f"Cooling rate: {result['cooling_rate']:.4f} log10(K)/day")
    print(f"Chi2: {result['chi2']:.2f}")
    print(f"Reference band: {result['ref_band']}")
    print(f"Color observations: {result['n_color_obs']}")

# Combined nonparametric + thermal (recommended, reuses GP)
combined = lcf.fit_fast(bands)
np_results = combined["nonparametric"]
thermal = combined["thermal"]
```
