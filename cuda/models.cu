// CUDA kernels for parametric lightcurve model evaluation and batch PSO cost.
//
// Two kernel families:
//   1. *_eval: forward model evaluation (draw × time → flux)
//   2. batch_pso_cost: fused model eval + likelihood for batch PSO fitting
//      Thread grid: n_sources × n_particles. Each thread loops over its
//      source's observations.

#include <math.h>

// ===========================================================================
// Device helpers
// ===========================================================================

__device__ inline double softplus(double x) {
    return log(1.0 + exp(x)) + 1e-6;
}

__device__ inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__device__ inline double log_normal_cdf_d(double x) {
    if (x > 8.0) return 0.0;
    if (x < -30.0) return -0.5 * x * x - 0.5 * log(2.0 * M_PI) - log(-x);
    double z = -x * M_SQRT1_2;
    double az = fabs(z);
    double t = 1.0 / (1.0 + 0.3275911 * az);
    double poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027
        + t * 1.061405429))));
    double erfc_z = poly * exp(-z * z);
    double phi = (z >= 0.0) ? 0.5 * erfc_z : 1.0 - 0.5 * erfc_z;
    return log(fmax(phi, 1e-300));
}

// ===========================================================================
// Device model evaluation functions
// ===========================================================================

// Model IDs (must match GpuModelName::model_id() in Rust):
//   0=Bazin, 1=Villar, 2=TDE, 3=Arnett, 4=Magnetar, 5=ShockCooling, 6=Afterglow

__device__ inline double bazin_at(const double* p, double t) {
    double a        = exp(p[0]);
    double b        = p[1];
    double t0       = p[2];
    double tau_rise = exp(p[3]);
    double tau_fall = exp(p[4]);
    double dt       = t - t0;
    return a * exp(-dt / tau_fall) * sigmoid(dt / tau_rise) + b;
}

__device__ inline double villar_at(const double* p, double t) {
    double a        = exp(p[0]);
    double beta     = p[1];
    double gamma    = exp(p[2]);
    double t0       = p[3];
    double tau_rise = exp(p[4]);
    double tau_fall = exp(p[5]);
    double phase    = t - t0;
    double sig_rise = sigmoid(phase / tau_rise);
    double w        = sigmoid(10.0 * (phase - gamma));
    double piece_left  = 1.0 - beta * phase;
    double piece_right = (1.0 - beta * gamma) * exp((gamma - phase) / tau_fall);
    return a * sig_rise * ((1.0 - w) * piece_left + w * piece_right);
}

__device__ inline double tde_at(const double* p, double t) {
    double a        = exp(p[0]);
    double b        = p[1];
    double t0       = p[2];
    double tau_rise = exp(p[3]);
    double tau_fall = exp(p[4]);
    double alpha    = p[5];
    double phase    = t - t0;
    double sig      = sigmoid(phase / tau_rise);
    double ps       = softplus(phase);
    return a * sig * pow(1.0 + ps / tau_fall, -alpha) + b;
}

__device__ inline double arnett_at(const double* p, double t) {
    double a       = exp(p[0]);
    double t0      = p[1];
    double tau_m   = exp(p[2]);
    double logit_f = p[3];
    double ps      = softplus(t - t0);
    double f       = sigmoid(logit_f);
    double e_ni    = exp(-ps / 8.8);
    double e_co    = exp(-ps / 111.3);
    double heat    = f * e_ni + (1.0 - f) * e_co;
    double x       = ps / tau_m;
    return a * heat * (1.0 - exp(-x * x));
}

__device__ inline double magnetar_at(const double* p, double t) {
    double a        = exp(p[0]);
    double t0       = p[1];
    double tau_sd   = exp(p[2]);
    double tau_diff = exp(p[3]);
    double ps       = softplus(t - t0);
    double w        = 1.0 + ps / tau_sd;
    double x        = ps / tau_diff;
    return a * (1.0 / (w * w)) * (1.0 - exp(-x * x));
}

__device__ inline double shock_cooling_at(const double* p, double t) {
    double a      = exp(p[0]);
    double t0     = p[1];
    double n_exp  = p[2];
    double tau_tr = exp(p[3]);
    double phase  = t - t0;
    double ps     = softplus(phase);
    double ratio  = ps / tau_tr;
    return a * sigmoid(phase * 5.0) * pow(ps, -n_exp) * exp(-ratio * ratio);
}

__device__ inline double afterglow_at(const double* p, double t) {
    double a      = exp(p[0]);
    double t0     = p[1];
    double t_b    = exp(p[2]);
    double alpha1 = p[3];
    double alpha2 = p[4];
    double ps     = softplus(t - t0);
    double ln_r   = log(ps / t_b);
    double u1     = exp(2.0 * alpha1 * ln_r);
    double u2     = exp(2.0 * alpha2 * ln_r);
    return a * pow(u1 + u2, -0.5);
}

__device__ inline double eval_model_at(int model_id, const double* p, double t) {
    switch (model_id) {
        case 0: return bazin_at(p, t);
        case 1: return villar_at(p, t);
        case 2: return tde_at(p, t);
        case 3: return arnett_at(p, t);
        case 4: return magnetar_at(p, t);
        case 5: return shock_cooling_at(p, t);
        case 6: return afterglow_at(p, t);
        default: return 0.0;
    }
}

// ===========================================================================
// Forward evaluation kernels (one thread per draw×time)
// ===========================================================================

#define DEFINE_EVAL_KERNEL(name, model_fn)                                    \
extern "C" __global__ void name(                                             \
    const double* __restrict__ params,                                       \
    const double* __restrict__ times,                                        \
    double* __restrict__ out,                                                \
    int n_draws, int n_times, int n_params)                                  \
{                                                                            \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                         \
    if (idx >= n_draws * n_times) return;                                    \
    int draw = idx / n_times;                                                \
    int ti   = idx % n_times;                                                \
    out[idx] = model_fn(params + draw * n_params, times[ti]);                \
}

DEFINE_EVAL_KERNEL(bazin_eval,          bazin_at)
DEFINE_EVAL_KERNEL(villar_eval,         villar_at)
DEFINE_EVAL_KERNEL(tde_eval,            tde_at)
DEFINE_EVAL_KERNEL(arnett_eval,         arnett_at)
DEFINE_EVAL_KERNEL(magnetar_eval,       magnetar_at)
DEFINE_EVAL_KERNEL(shock_cooling_eval,  shock_cooling_at)
DEFINE_EVAL_KERNEL(afterglow_eval,      afterglow_at)

// ===========================================================================
// Batch PSO cost kernel
// ===========================================================================
//
// One thread per (source, particle) pair.  Each thread loops over its
// source's observations and computes the negative log-likelihood (divided
// by n_obs) — matching PsoCost::cost on the CPU side.
//
// Data layout:
//   all_times, all_flux, all_obs_var, all_upper_flux: concatenated across
//       sources.  source_offsets[src] .. source_offsets[src+1] gives the
//       range for source `src`.
//   all_is_upper: int (0/1) array, same layout.
//   positions: [n_sources × n_particles × n_params], row-major.
//   costs (output): [n_sources × n_particles].

extern "C" __global__ void batch_pso_cost(
    const double* __restrict__ all_times,
    const double* __restrict__ all_flux,
    const double* __restrict__ all_obs_var,
    const int*    __restrict__ all_is_upper,
    const double* __restrict__ all_upper_flux,
    const int*    __restrict__ source_offsets,
    const double* __restrict__ positions,
    double* __restrict__ costs,
    int n_sources,
    int n_particles,
    int n_params,
    int model_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sources * n_particles) return;

    int src = idx / n_particles;
    int pid = idx % n_particles;

    int obs_start = source_offsets[src];
    int obs_end   = source_offsets[src + 1];
    int n_obs     = obs_end - obs_start;
    if (n_obs <= 0) { costs[idx] = 1e99; return; }

    const double* p = positions + (long long)(src * n_particles + pid) * n_params;
    int se_idx = n_params - 1;  // sigma_extra is always the last parameter
    double sigma_extra = exp(p[se_idx]);
    double sigma_extra_sq = sigma_extra * sigma_extra;

    double neg_ll = 0.0;
    for (int i = obs_start; i < obs_end; i++) {
        double pred = eval_model_at(model_id, p, all_times[i]);
        if (!isfinite(pred)) { costs[idx] = 1e99; return; }

        double total_var = all_obs_var[i] + sigma_extra_sq;

        if (all_is_upper[i]) {
            double z = (all_upper_flux[i] - pred) / sqrt(total_var);
            neg_ll -= log_normal_cdf_d(z);
        } else {
            double diff = pred - all_flux[i];
            neg_ll += diff * diff / total_var + log(total_var);
        }
    }

    costs[idx] = neg_ll / (double)n_obs;
}

// ===========================================================================
// Host-side launch wrappers (callable from Rust via FFI)
// ===========================================================================

#define DEFINE_LAUNCHER(name, kernel)                                         \
extern "C" void name(                                                        \
    const double* params, const double* times, double* out,                  \
    int n_draws, int n_times, int n_params, int grid, int block)             \
{                                                                            \
    kernel<<<grid, block>>>(params, times, out, n_draws, n_times, n_params); \
}

DEFINE_LAUNCHER(launch_bazin,          bazin_eval)
DEFINE_LAUNCHER(launch_villar,         villar_eval)
DEFINE_LAUNCHER(launch_tde,            tde_eval)
DEFINE_LAUNCHER(launch_arnett,         arnett_eval)
DEFINE_LAUNCHER(launch_magnetar,       magnetar_eval)
DEFINE_LAUNCHER(launch_shock_cooling,  shock_cooling_eval)
DEFINE_LAUNCHER(launch_afterglow,      afterglow_eval)

extern "C" void launch_batch_pso_cost(
    const double* all_times,
    const double* all_flux,
    const double* all_obs_var,
    const int*    all_is_upper,
    const double* all_upper_flux,
    const int*    source_offsets,
    const double* positions,
    double* costs,
    int n_sources,
    int n_particles,
    int n_params,
    int model_id,
    int grid,
    int block)
{
    batch_pso_cost<<<grid, block>>>(
        all_times, all_flux, all_obs_var, all_is_upper, all_upper_flux,
        source_offsets, positions, costs,
        n_sources, n_particles, n_params, model_id);
}
