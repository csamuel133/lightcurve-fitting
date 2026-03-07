//! Tests for custom DenseGP and SparseGP implementations.

use lightcurve_fitting::sparse_gp::{DenseGP, SparseGP};
use std::time::Instant;

/// Generate a synthetic Bazin lightcurve (magnitude space).
fn bazin_lc(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let t0 = 30.0;
    let tau_rise = 3.0;
    let tau_fall = 25.0;

    let mut rng_state = seed.max(1);
    let mut rng_f64 = || -> f64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state >> 11) as f64 / ((1u64 << 53) as f64)
    };

    let mut times = Vec::with_capacity(n);
    let mut mags = Vec::with_capacity(n);
    let mut errs = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64) * 100.0 / n as f64;
        let dt = t - t0;
        let flux = (-dt / tau_fall).exp() / (1.0 + (-dt / tau_rise).exp()) + 0.01;
        let mag = if flux > 0.0 { -2.5 * flux.log10() + 23.9 } else { 25.0 };
        let sigma = 0.05 + 0.03 * rng_f64();
        let u1 = rng_f64().max(1e-15);
        let u2 = rng_f64();
        let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        times.push(t);
        mags.push(mag + sigma * noise);
        errs.push(sigma);
    }

    (times, mags, errs)
}

#[test]
fn dense_gp_fits_and_predicts() {
    let (times, mags, errs) = bazin_lc(30, 42);
    let noise_var: Vec<f64> = errs.iter().map(|e| e * e).collect();

    let gp = DenseGP::fit(&times, &mags, &noise_var, 0.1, 10.0)
        .expect("custom fit should succeed");

    let query: Vec<f64> = (0..20).map(|i| i as f64 * 5.0).collect();
    let pred = gp.predict(&query);
    let (pred_std, std_vec) = gp.predict_with_std(&query);

    assert_eq!(pred.len(), 20);
    assert_eq!(pred_std.len(), 20);
    assert_eq!(std_vec.len(), 20);

    // Predictions should be finite
    for &p in &pred {
        assert!(p.is_finite(), "prediction should be finite");
    }
    for &s in &std_vec {
        assert!(s.is_finite() && s >= 0.0, "std should be non-negative finite");
    }
}

#[test]
fn dense_gp_speed() {
    let sizes = [25, 50, 100, 200];
    let n_query = 10_000;

    eprintln!("\n{:>8} {:>12} {:>12} {:>12}",
        "n_train", "fit_us", "pred_std_us", "total_us");
    eprintln!("{}", "-".repeat(48));

    for &n in &sizes {
        let (times, mags, errs) = bazin_lc(n, 100);
        let noise_var: Vec<f64> = errs.iter().map(|e| e * e).collect();
        let query: Vec<f64> = (0..n_query).map(|i| i as f64 * 100.0 / n_query as f64).collect();

        let t0 = Instant::now();
        let gp = DenseGP::fit(&times, &mags, &noise_var, 0.1, 10.0).unwrap();
        let fit_us = t0.elapsed().as_micros();

        let t0 = Instant::now();
        let _ = gp.predict_with_std(&query);
        let pred_us = t0.elapsed().as_micros();

        eprintln!("{:>8} {:>12} {:>12} {:>12}",
            n, fit_us, pred_us, fit_us + pred_us);
    }
}

#[test]
fn sparse_gp_agrees_with_dense_on_small_data() {
    let (times, mags, errs) = bazin_lc(30, 42);
    let noise_var: Vec<f64> = errs.iter().map(|e| e * e).collect();

    let amp = 0.1;
    let ls = 10.0;

    let dense = DenseGP::fit(&times, &mags, &noise_var, amp, ls).unwrap();
    let sparse = SparseGP::fit(&times, &mags, &noise_var, amp, ls, 30).unwrap();

    let query: Vec<f64> = (0..20).map(|i| i as f64 * 5.0).collect();
    let d_pred = dense.predict(&query);
    let s_pred = sparse.predict(&query);

    eprintln!("\n{:>8} {:>12} {:>12} {:>8}", "t", "dense", "sparse", "diff");
    eprintln!("{}", "-".repeat(44));
    let mut max_diff = 0.0f64;
    for i in 0..query.len() {
        let diff = (d_pred[i] - s_pred[i]).abs();
        max_diff = max_diff.max(diff);
        eprintln!("{:>8.1} {:>12.4} {:>12.4} {:>8.4}", query[i], d_pred[i], s_pred[i], diff);
    }
    eprintln!("Max absolute difference: {:.6}", max_diff);
    assert!(max_diff < 0.5, "sparse diverges from dense: max_diff={max_diff:.4}");
}

#[test]
fn sparse_gp_speed_at_scale() {
    let sizes = [500, 1000, 5000, 10000];
    let n_query = 10_000;

    eprintln!("\n{:>8} {:>12} {:>12} {:>12} {:>12}",
        "n_train", "fit_ms", "pred_ms", "pred_std_ms", "total_ms");
    eprintln!("{}", "-".repeat(60));

    for &n in &sizes {
        let (times, mags, errs) = bazin_lc(n, 100);
        let noise_var: Vec<f64> = errs.iter().map(|e| e * e).collect();
        let query: Vec<f64> = (0..n_query).map(|i| i as f64 * 100.0 / n_query as f64).collect();

        let amp = 0.1;
        let ls = 10.0;
        let m = 30;

        let t0 = Instant::now();
        let gp = SparseGP::fit(&times, &mags, &noise_var, amp, ls, m).unwrap();
        let fit_ms = t0.elapsed().as_millis();

        let t0 = Instant::now();
        let _ = gp.predict(&query);
        let pred_ms = t0.elapsed().as_millis();

        let t0 = Instant::now();
        let _ = gp.predict_with_std(&query);
        let pred_std_ms = t0.elapsed().as_millis();

        eprintln!("{:>8} {:>12} {:>12} {:>12} {:>12}",
            n, fit_ms, pred_ms, pred_std_ms, fit_ms + pred_ms + pred_std_ms);
    }
}
