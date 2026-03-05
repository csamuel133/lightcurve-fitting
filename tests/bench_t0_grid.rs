use std::fs;
use std::io::BufRead;

use lightcurve_fitting::{build_flux_bands, fit_parametric, UncertaintyMethod};

fn load_source_csv(name: &str) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<String>)> {
    let path = format!("/fred/oz480/mcoughli/data_ztf/{}/photometry.csv", name);
    let file = fs::File::open(&path).ok()?;
    let reader = std::io::BufReader::new(file);

    let mut times = Vec::new();
    let mut mags = Vec::new();
    let mut errs = Vec::new();
    let mut bands = Vec::new();

    let mut lines = reader.lines();
    let header = lines.next()?.ok()?;
    let cols: Vec<&str> = header.split(',').collect();
    let jd_idx = cols.iter().position(|c| *c == "jd")?;
    let fid_idx = cols.iter().position(|c| *c == "fid")?;
    let mag_idx = cols.iter().position(|c| *c == "magpsf")?;
    let err_idx = cols.iter().position(|c| *c == "sigmapsf")?;

    for line in lines {
        let line = line.ok()?;
        let fields: Vec<&str> = line.split(',').collect();
        let mag_str = fields.get(mag_idx)?.trim();
        let err_str = fields.get(err_idx)?.trim();
        if mag_str.is_empty() || err_str.is_empty() {
            continue;
        }
        let jd: f64 = fields[jd_idx].parse().ok()?;
        let mag: f64 = mag_str.parse().ok()?;
        let err: f64 = err_str.parse().ok()?;
        if !mag.is_finite() || !err.is_finite() || err <= 0.0 {
            continue;
        }
        let band = match fields[fid_idx].trim() {
            "1" => "g",
            "2" => "r",
            "3" => "i",
            _ => continue,
        };
        times.push(jd);
        mags.push(mag);
        errs.push(err);
        bands.push(band.to_string());
    }

    if times.is_empty() {
        return None;
    }
    Some((times, mags, errs, bands))
}

#[test]
#[ignore]
fn bench_t0_grid() {
    let targets = [
        // Original dense set
        "ZTF17aaaacjo",   // 445
        "ZTF17aaabgkn",   // 292
        "ZTF17aaabmmr",   // 640
        "ZTF17aaacdos",   //2071
        "ZTF17aaadyei",   //1813
        // Recent sparse
        "ZTF25aaftbug",   //   5
        "ZTF25aaivcgm",   //   7
        "ZTF25aaovvcg",   //   8
        "ZTF25aaxdccu",   //   9
        "ZTF25aahmbod",   //  14
        // Recent moderate
        "ZTF25aabylkr",   //  25
        "ZTF25aaizxrf",   //  37
        "ZTF25aafkxdu",   //  40
        "ZTF25aapwhnu",   //  48
        "ZTF25aacaxre",   //  75
        // Recent dense
        "ZTF25aahccao",   //  91
        "ZTF25aadevqv",   // 115
        "ZTF25aadnogd",   // 147
        "ZTF25aairhqk",   // 308
        "ZTF25aajqtfg",   // 371
    ];

    eprintln!("\n{:<16} {:>6} {:>8} {:>8} {:>14} {:>14}",
        "source", "n_obs", "lap_ms", "svi_ms", "lap_chi2", "svi_chi2");
    eprintln!("{}", "-".repeat(74));

    let mut total_lap = 0u128;
    let mut total_svi = 0u128;

    for name in &targets {
        let Some((times, mags, errs, bands)) = load_source_csv(name) else {
            eprintln!("{name:<16} skipped");
            continue;
        };
        let flux_bands = build_flux_bands(&times, &mags, &errs, &bands);
        let n_obs: usize = flux_bands.values().map(|b| b.values.len()).sum();

        let t0 = std::time::Instant::now();
        let lap_results = fit_parametric(&flux_bands, false, UncertaintyMethod::Laplace);
        let lap_ms = t0.elapsed().as_millis();

        let t0 = std::time::Instant::now();
        let svi_results = fit_parametric(&flux_bands, false, UncertaintyMethod::Svi);
        let svi_ms = t0.elapsed().as_millis();

        total_lap += lap_ms;
        total_svi += svi_ms;

        let lap_chi2 = lap_results.first().and_then(|r| r.mag_chi2).unwrap_or(f64::NAN);
        let svi_chi2 = svi_results.first().and_then(|r| r.mag_chi2).unwrap_or(f64::NAN);

        eprintln!("{:<16} {:>6} {:>8} {:>8} {:>14.2} {:>14.2}",
            name, n_obs, lap_ms, svi_ms, lap_chi2, svi_chi2);
    }

    eprintln!("{}", "-".repeat(74));
    eprintln!("{:<16} {:>6} {:>8} {:>8}", "TOTAL", "", total_lap, total_svi);
    eprintln!();
}
