# lightcurve-fitting

High-performance Rust library with Python bindings for transient light curve feature extraction. Implements nonparametric Gaussian process fitting, parametric model selection (PSO + Laplace/SVI uncertainty), and blackbody temperature estimation.

## Key Features

- **Nonparametric GP** --- custom DenseGP/SparseGP with automatic hyperparameter selection, 30+ extracted features
- **Parametric models** --- 8 models (Bazin, Villar, TDE, Arnett, Magnetar, ShockCooling, Afterglow, MetzgerKN) with PSO optimization
- **Thermal fitting** --- cross-band blackbody temperature and cooling rate estimation
- **GPU acceleration** --- CUDA batch fitting for both parametric (PSO) and nonparametric (GP) pipelines
- **Parallel batch processing** --- Rayon-based CPU parallelism across sources
- **Python bindings** --- full API via PyO3/maturin

## Quick Example

```python
import lightcurve_fitting as lcf

bands = lcf.build_mag_bands(times, mags, errors, band_names)
result = lcf.fit_fast(bands)
# result["nonparametric"] - GP features per band
# result["thermal"] - temperature estimates
```

## Getting Started

See the [Installation](getting-started/installation.md) and [Quick Start](getting-started/quickstart.md) guides.
