pub mod batch;
pub mod common;
pub mod gp;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod nonparametric;
pub mod parametric;
pub mod sparse_gp;
pub mod thermal;

pub use batch::{FastFitResult, fit_batch_fast, fit_batch_parametric};
pub use common::{BandData, LightcurveFittingResult, build_mag_bands, build_flux_bands};
pub use nonparametric::{fit_nonparametric, fit_nonparametric_with_opts, NonparametricBandResult};
#[cfg(feature = "cuda")]
pub use nonparametric::{fit_nonparametric_batch_gpu, fit_nonparametric_batch_gpu_with_opts};
#[cfg(feature = "cuda")]
pub use gpu::{GpBandInput, GpBandOutput};
pub use parametric::{eval_model_flux, metzger_kn_mags, fit_parametric, ParametricBandResult, SviModelName, UncertaintyMethod};
pub use thermal::{fit_thermal, ThermalResult};
pub use gp::fit_gp_predict;
