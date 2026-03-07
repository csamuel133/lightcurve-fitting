# Installation

## Python

### From PyPI

```bash
pip install lightcurve-fitting
```

### From source (requires Rust toolchain)

Clone the repository and build with [maturin](https://www.maturin.rs/):

```bash
git clone https://github.com/boom-astro/lightcurve-fitting.git
cd lightcurve-fitting/python
pip install maturin
maturin develop --release
```

The `--release` flag enables compiler optimizations and is strongly
recommended for any real workload.

## Rust

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
lightcurve-fitting = { git = "https://github.com/boom-astro/lightcurve-fitting.git" }
```

Then build as usual:

```bash
cargo build --release
```

## GPU support (optional)

The parametric fitter supports CUDA-accelerated batch PSO. To enable it,
build with the `cuda` feature:

```bash
# Rust
cargo build --release --features cuda

# Python (from source)
cd python
RUSTFLAGS="--cfg feature=\"cuda\"" maturin develop --release
```

The build system auto-detects `nvcc` via the `CUDA_PATH` environment
variable or standard system locations (`/usr/local/cuda`, etc.).

## Dependencies

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Rust** | 1.70+ | Required for building from source or using the Rust crate directly |
| **Python** | 3.8+ | Required for the Python bindings |
| **CUDA toolkit** | 12+ | Optional; only needed when building with `--features cuda` |

No additional Python packages are required at runtime -- the compiled
extension is self-contained. Build-time only: `maturin` (for the Python
package) and `cc` (pulled automatically by Cargo when the `cuda` feature
is active).
