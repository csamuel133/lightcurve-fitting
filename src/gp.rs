use crate::sparse_gp::DenseGP;

/// Subsample data to at most `max_points` by uniform striding.
pub fn subsample_data(
    times: &[f64],
    mags: &[f64],
    errors: &[f64],
    max_points: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if times.len() <= max_points {
        return (times.to_vec(), mags.to_vec(), errors.to_vec());
    }

    let step = times.len() as f64 / max_points as f64;
    let mut indices = Vec::with_capacity(max_points);
    for i in 0..max_points {
        let idx = ((i as f64 + 0.5) * step).floor() as usize;
        indices.push(idx.min(times.len() - 1));
    }

    let times_sub: Vec<f64> = indices.iter().map(|&i| times[i]).collect();
    let mags_sub: Vec<f64> = indices.iter().map(|&i| mags[i]).collect();
    let errors_sub: Vec<f64> = indices.iter().map(|&i| errors[i]).collect();

    (times_sub, mags_sub, errors_sub)
}

/// Fit a GP to training data and predict at query points.
///
/// Does a grid search over amplitude and lengthscale, using measurement
/// error variance as noise. Returns predictions at `query_times`.
///
/// Returns `(predictions, std_devs)` or None if fitting fails.
pub fn fit_gp_predict(
    train_times: &[f64],
    train_values: &[f64],
    train_errors: &[f64],
    query_times: &[f64],
    amp_candidates: &[f64],
    ls_candidates: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let noise_var: Vec<f64> = train_errors.iter().map(|e| (e * e).max(1e-6)).collect();

    let mut best_gp: Option<DenseGP> = None;
    let mut best_score = f64::INFINITY;

    for &amp in amp_candidates {
        for &ls in ls_candidates {
            if let Some(gp) = DenseGP::fit(train_times, train_values, &noise_var, amp, ls) {
                let rms = gp.train_rms(train_values);
                if rms.is_finite() && rms < best_score {
                    best_score = rms;
                    best_gp = Some(gp);
                }
            }
        }
    }

    let gp = best_gp?;
    let (pred, std) = gp.predict_with_std(query_times);
    Some((pred, std))
}
