use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

#[derive(Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    fn identity(n: usize) -> Self {
        let mut a = Self::zeros(n, n);
        for i in 0..n {
            a.set(i, i, 1.0);
        }
        a
    }

    #[inline]
    fn idx(&self, r: usize, c: usize) -> usize {
        r * self.cols + c
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> f64 {
        self.data[self.idx(r, c)]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, value: f64) {
        let k = self.idx(r, c);
        self.data[k] = value;
    }

    fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, x.len());
        let mut y = vec![0.0; self.rows];
        for r in 0..self.rows {
            let mut sum = 0.0;
            for c in 0..self.cols {
                sum += self.get(r, c) * x[c];
            }
            y[r] = sum;
        }
        y
    }

    fn mul_matrix(&self, b: &Matrix) -> Matrix {
        assert_eq!(self.cols, b.rows);
        let mut out = Matrix::zeros(self.rows, b.cols);
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(r, k);
                if a == 0.0 {
                    continue;
                }
                for c in 0..b.cols {
                    let idx = out.idx(r, c);
                    out.data[idx] += a * b.get(k, c);
                }
            }
        }
        out
    }

    fn dot_columns(&self, p: usize, q: usize) -> f64 {
        let mut sum = 0.0;
        for r in 0..self.rows {
            sum += self.get(r, p) * self.get(r, q);
        }
        sum
    }

    fn column_norm2(&self, c: usize) -> f64 {
        self.dot_columns(c, c)
    }

    fn rotate_columns(&mut self, p: usize, q: usize, c: f64, s: f64) {
        for r in 0..self.rows {
            let xp = self.get(r, p);
            let xq = self.get(r, q);
            self.set(r, p, c * xp - s * xq);
            self.set(r, q, s * xp + c * xq);
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn normalize(a: &mut [f64]) {
    let n = norm(a);
    assert!(n > 0.0);
    for x in a {
        *x /= n;
    }
}

fn periodic_gaussian_matrix(grid: &[f64], sigma: f64) -> Matrix {
    let n = grid.len();
    let mut a = Matrix::zeros(n, n);
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            let raw = (grid[i] - grid[j]).abs();
            let d = raw.min(1.0 - raw);
            let value = (-(d * d) / (2.0 * sigma * sigma)).exp();
            a.set(i, j, value);
            row_sum += value;
        }
        for j in 0..n {
            a.set(i, j, a.get(i, j) / row_sum);
        }
    }
    a
}

fn relu_spline_basis(grid: &[f64], width: usize) -> Matrix {
    let n = grid.len();
    let d = width + 2;
    let mut raw = Matrix::zeros(n, d);
    for r in 0..n {
        raw.set(r, 0, 1.0);
        raw.set(r, 1, grid[r]);
        for j in 0..width {
            let knot = (j + 1) as f64 / (width + 1) as f64;
            raw.set(r, j + 2, (grid[r] - knot).max(0.0));
        }
    }

    // Modified Gram-Schmidt with a second reorthogonalization pass.
    let mut q = Matrix::zeros(n, d);
    for j in 0..d {
        let mut v = (0..n).map(|r| raw.get(r, j)).collect::<Vec<_>>();
        for _ in 0..2 {
            for k in 0..j {
                let mut proj = 0.0;
                for r in 0..n {
                    proj += q.get(r, k) * v[r];
                }
                for r in 0..n {
                    v[r] -= proj * q.get(r, k);
                }
            }
        }
        let nv = norm(&v);
        assert!(nv > 1.0e-13, "rank-deficient spline basis");
        for r in 0..n {
            q.set(r, j, v[r] / nv);
        }
    }
    q
}

struct JacobiSvd {
    singular: Vec<f64>,
    u: Matrix,
    v: Matrix,
}

fn jacobi_svd(a: &Matrix) -> JacobiSvd {
    // One-sided Jacobi SVD: orthogonalize columns of B=A*V.
    let mut b = a.clone();
    let mut v = Matrix::identity(a.cols);
    let n = a.cols;
    let tol = 2.0e-15;
    let max_sweeps = 120;

    for _ in 0..max_sweeps {
        let mut changed = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let alpha = b.column_norm2(p);
                let beta = b.column_norm2(q);
                if alpha == 0.0 || beta == 0.0 {
                    continue;
                }
                let gamma = b.dot_columns(p, q);
                if gamma.abs() <= tol * (alpha * beta).sqrt() {
                    continue;
                }

                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta == 0.0 {
                    1.0
                } else {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                b.rotate_columns(p, q, c, s);
                v.rotate_columns(p, q, c, s);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut order = (0..n)
        .map(|j| (j, b.column_norm2(j).sqrt()))
        .collect::<Vec<_>>();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut singular = vec![0.0; n];
    let mut u = Matrix::zeros(a.rows, n);
    let mut vs = Matrix::zeros(n, n);
    for (new_j, &(old_j, sigma)) in order.iter().enumerate() {
        singular[new_j] = sigma;
        if sigma > 0.0 {
            for r in 0..a.rows {
                u.set(r, new_j, b.get(r, old_j) / sigma);
            }
        }
        for r in 0..n {
            vs.set(r, new_j, v.get(r, old_j));
        }
    }

    JacobiSvd { singular, u, v: vs }
}

impl JacobiSvd {
    fn solve_least_squares(&self, y: &[f64]) -> Vec<f64> {
        assert_eq!(self.u.rows, y.len());
        let n = self.singular.len();
        let smax = self.singular[0];
        let cutoff = (self.u.rows.max(n) as f64) * f64::EPSILON * smax;
        let mut weights = vec![0.0; n];
        for j in 0..n {
            let s = self.singular[j];
            if s <= cutoff {
                continue;
            }
            let mut uty = 0.0;
            for r in 0..self.u.rows {
                uty += self.u.get(r, j) * y[r];
            }
            weights[j] = uty / s;
        }

        let mut x = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += self.v.get(i, j) * weights[j];
            }
            x[i] = sum;
        }
        x
    }
}

struct SplitMix64 {
    state: u64,
    spare_normal: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed, spare_normal: None }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn uniform_open01(&mut self) -> f64 {
        // 53 random bits, shifted away from the exact endpoints.
        let v = self.next_u64() >> 11;
        ((v as f64) + 0.5) * (1.0 / 9007199254740992.0)
    }

    fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        let u1 = self.uniform_open01();
        let u2 = self.uniform_open01();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * PI * u2;
        let z0 = radius * angle.cos();
        let z1 = radius * angle.sin();
        self.spare_normal = Some(z1);
        z0
    }
}

#[derive(Clone)]
struct Row {
    width: usize,
    model_dimension: usize,
    approximation_error: f64,
    alpha: f64,
    mean: f64,
    sd: f64,
    theorem_bound: f64,
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_sd(values: &[f64], mu: f64) -> f64 {
    let ss = values.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>();
    (ss / (values.len() as f64 - 1.0)).sqrt()
}

fn write_csv(rows: &[Row]) -> std::io::Result<()> {
    let file = File::create("experiment_results.csv")?;
    let mut w = BufWriter::new(file);
    writeln!(w, "width,model_dimension,approximation_error,alpha_n,reconstruction_error_mean,reconstruction_error_sd,theorem_bound")?;
    for r in rows {
        writeln!(
            w,
            "{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            r.width, r.model_dimension, r.approximation_error, r.alpha, r.mean, r.sd, r.theorem_bound
        )?;
    }
    Ok(())
}

fn svg_polyline(points: &[(f64, f64)], dash: Option<&str>) -> String {
    let pts = points
        .iter()
        .map(|(x, y)| format!("{:.3},{:.3}", x, y))
        .collect::<Vec<_>>()
        .join(" ");
    let dash_attr = dash.map(|d| format!(" stroke-dasharray=\"{}\"", d)).unwrap_or_default();
    format!("<polyline points=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"2\"{} />\n", pts, dash_attr)
}

fn map_log(v: f64, vmin: f64, vmax: f64, out_min: f64, out_max: f64) -> f64 {
    let t = (v.ln() - vmin.ln()) / (vmax.ln() - vmin.ln());
    out_min + t * (out_max - out_min)
}

fn map_lin(v: f64, vmin: f64, vmax: f64, out_min: f64, out_max: f64) -> f64 {
    let t = (v - vmin) / (vmax - vmin);
    out_min + t * (out_max - out_min)
}

fn write_svg(
    rows: &[Row],
    grid: &[f64],
    truth: &[f64],
    selected: &[(usize, Vec<f64>)],
) -> std::io::Result<()> {
    let file = File::create("stability_tradeoff.svg")?;
    let mut w = BufWriter::new(file);
    let width = 1200.0;
    let height = 360.0;
    writeln!(w, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1200\" height=\"360\" viewBox=\"0 0 1200 360\">")?;
    writeln!(w, "<rect width=\"1200\" height=\"360\" fill=\"white\"/>")?;
    writeln!(w, "<g font-family=\"sans-serif\" font-size=\"13\" fill=\"black\">")?;

    let panels = [(55.0, 35.0, 310.0, 255.0), (445.0, 35.0, 310.0, 255.0), (835.0, 35.0, 310.0, 255.0)];
    for &(x, y, pw, ph) in &panels {
        writeln!(w, "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"black\"/>", x, y, pw, ph)?;
    }

    let xmin = rows.first().unwrap().width as f64;
    let xmax = rows.last().unwrap().width as f64;

    // Panel 1: approximation versus stability.
    let ymin1 = rows.iter().map(|r| r.alpha.min(r.approximation_error)).fold(f64::INFINITY, f64::min) * 0.7;
    let ymax1 = rows.iter().map(|r| r.alpha.max(r.approximation_error)).fold(0.0, f64::max) * 1.3;
    let (x0, y0, pw, ph) = panels[0];
    let p_sigma = rows.iter().map(|r| {
        let x = map_log(r.width as f64, xmin, xmax, x0, x0 + pw);
        let y = map_log(r.approximation_error, ymin1, ymax1, y0 + ph, y0);
        (x, y)
    }).collect::<Vec<_>>();
    let p_alpha = rows.iter().map(|r| {
        let x = map_log(r.width as f64, xmin, xmax, x0, x0 + pw);
        let y = map_log(r.alpha, ymin1, ymax1, y0 + ph, y0);
        (x, y)
    }).collect::<Vec<_>>();
    write!(w, "{}", svg_polyline(&p_sigma, None))?;
    write!(w, "{}", svg_polyline(&p_alpha, Some("7,5")))?;
    writeln!(w, "<text x=\"{}\" y=\"315\" text-anchor=\"middle\">hidden width n</text>", x0 + pw / 2.0)?;
    writeln!(w, "<text x=\"{}\" y=\"22\" text-anchor=\"middle\" font-weight=\"bold\">Approximation vs. stability</text>", x0 + pw / 2.0)?;
    writeln!(w, "<text x=\"{}\" y=\"52\">solid: approximation error</text>", x0 + 8.0)?;
    writeln!(w, "<text x=\"{}\" y=\"70\">dashed: alpha_n</text>", x0 + 8.0)?;

    // Panel 2: noisy reconstruction.
    let ymin2 = rows.iter().map(|r| (r.mean - r.sd).max(1.0e-12)).fold(f64::INFINITY, f64::min) * 0.7;
    let ymax2 = rows.iter().map(|r| r.mean + r.sd).fold(0.0, f64::max) * 1.3;
    let (x0, y0, pw, ph) = panels[1];
    let p_mean = rows.iter().map(|r| {
        let x = map_log(r.width as f64, xmin, xmax, x0, x0 + pw);
        let y = map_log(r.mean, ymin2, ymax2, y0 + ph, y0);
        (x, y)
    }).collect::<Vec<_>>();
    write!(w, "{}", svg_polyline(&p_mean, None))?;
    for r in rows {
        let x = map_log(r.width as f64, xmin, xmax, x0, x0 + pw);
        let lo = (r.mean - r.sd).max(ymin2);
        let hi = r.mean + r.sd;
        let ylo = map_log(lo, ymin2, ymax2, y0 + ph, y0);
        let yhi = map_log(hi, ymin2, ymax2, y0 + ph, y0);
        writeln!(w, "<line x1=\"{x:.3}\" y1=\"{ylo:.3}\" x2=\"{x:.3}\" y2=\"{yhi:.3}\" stroke=\"black\" stroke-width=\"1\"/>")?;
    }
    writeln!(w, "<text x=\"{}\" y=\"315\" text-anchor=\"middle\">hidden width n</text>", x0 + pw / 2.0)?;
    writeln!(w, "<text x=\"{}\" y=\"22\" text-anchor=\"middle\" font-weight=\"bold\">Noisy reconstruction</text>", x0 + pw / 2.0)?;

    // Panel 3: truth and representative mean reconstructions.
    let mut ymin3 = truth.iter().copied().fold(f64::INFINITY, f64::min);
    let mut ymax3 = truth.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for (_, curve) in selected {
        ymin3 = ymin3.min(curve.iter().copied().fold(f64::INFINITY, f64::min));
        ymax3 = ymax3.max(curve.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    }
    let pad = 0.08 * (ymax3 - ymin3).max(1.0e-12);
    ymin3 -= pad;
    ymax3 += pad;
    let (x0, y0, pw, ph) = panels[2];
    let truth_pts = grid.iter().zip(truth).map(|(&xv, &yv)| {
        (map_lin(xv, 0.0, 1.0, x0, x0 + pw), map_lin(yv, ymin3, ymax3, y0 + ph, y0))
    }).collect::<Vec<_>>();
    write!(w, "{}", svg_polyline(&truth_pts, None))?;
    let dashes = ["8,4", "5,3,1,3", "2,4"];
    for (idx, (model_width, curve)) in selected.iter().enumerate() {
        let pts = grid.iter().zip(curve).map(|(&xv, &yv)| {
            (map_lin(xv, 0.0, 1.0, x0, x0 + pw), map_lin(yv, ymin3, ymax3, y0 + ph, y0))
        }).collect::<Vec<_>>();
        write!(w, "{}", svg_polyline(&pts, Some(dashes[idx % dashes.len()])))?;
        writeln!(w, "<text x=\"{}\" y=\"{}\">n={}</text>", x0 + 8.0, 52.0 + 18.0 * idx as f64, model_width)?;
    }
    writeln!(w, "<text x=\"{}\" y=\"315\" text-anchor=\"middle\">t</text>", x0 + pw / 2.0)?;
    writeln!(w, "<text x=\"{}\" y=\"22\" text-anchor=\"middle\" font-weight=\"bold\">Mean reconstruction</text>", x0 + pw / 2.0)?;

    writeln!(w, "</g></svg>")?;
    let _ = (width, height); // kept explicit for easy figure resizing.
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sample_count = 256usize;
    let grid = (0..sample_count)
        .map(|i| i as f64 / sample_count as f64)
        .collect::<Vec<_>>();
    let forward = periodic_gaussian_matrix(&grid, 0.028);

    // The periodic Gaussian matrix is symmetric and doubly stochastic on this
    // uniform circular grid, so its spectral norm is exactly one.
    let beta = 1.0f64;

    let mut truth = grid
        .iter()
        .map(|&t| {
            0.65 * (2.0 * PI * t).sin()
                + 0.25 * (6.0 * PI * t + 0.35).sin()
                + 0.35 * (-((t - 0.31) / 0.055).powi(2)).exp()
                - 0.22 * (-((t - 0.72) / 0.035).powi(2)).exp()
        })
        .collect::<Vec<_>>();
    normalize(&mut truth);
    let exact_data = forward.mul_vec(&truth);
    let noise_level = 0.003 * norm(&exact_data);

    let widths = [2usize, 4, 6, 8, 12, 16, 24, 32, 48, 64];
    let trials = 100usize;
    let mut rng = SplitMix64::new(20260916);
    let mut rows = Vec::<Row>::new();
    let mut selected = Vec::<(usize, Vec<f64>)>::new();

    for &width in &widths {
        let q = relu_spline_basis(&grid, width);
        let aq = forward.mul_matrix(&q);
        let svd = jacobi_svd(&aq);
        let alpha = *svd.singular.last().unwrap();

        let mut qt_truth = vec![0.0; q.cols];
        for j in 0..q.cols {
            let mut sum = 0.0;
            for r in 0..q.rows {
                sum += q.get(r, j) * truth[r];
            }
            qt_truth[j] = sum;
        }
        let approximation = q.mul_vec(&qt_truth);
        let approximation_error = norm(
            &truth.iter().zip(&approximation).map(|(a, b)| a - b).collect::<Vec<_>>(),
        );

        let mut errors = Vec::<f64>::with_capacity(trials);
        let mut reconstruction_sum = vec![0.0; sample_count];

        for _ in 0..trials {
            let mut noise = (0..sample_count).map(|_| rng.normal()).collect::<Vec<_>>();
            let nrm = norm(&noise);
            for x in &mut noise {
                *x *= noise_level / nrm;
            }
            let noisy_data = exact_data.iter().zip(&noise).map(|(a, b)| a + b).collect::<Vec<_>>();
            let coefficients = svd.solve_least_squares(&noisy_data);
            let reconstruction = q.mul_vec(&coefficients);
            let err = norm(
                &reconstruction.iter().zip(&truth).map(|(a, b)| a - b).collect::<Vec<_>>(),
            );
            errors.push(err);
            if matches!(width, 8 | 24 | 32) {
                for i in 0..sample_count {
                    reconstruction_sum[i] += reconstruction[i];
                }
            }
        }

        let mu = mean(&errors);
        let sd = sample_sd(&errors, mu);
        let theorem_bound = (1.0 + 2.0 * beta / alpha) * approximation_error + 2.0 * noise_level / alpha;
        rows.push(Row {
            width,
            model_dimension: q.cols,
            approximation_error,
            alpha,
            mean: mu,
            sd,
            theorem_bound,
        });

        if matches!(width, 8 | 24 | 32) {
            for x in &mut reconstruction_sum {
                *x /= trials as f64;
            }
            selected.push((width, reconstruction_sum));
        }
    }

    write_csv(&rows)?;
    write_svg(&rows, &grid, &truth, &selected)?;

    println!("width  dim  approximation_error  alpha_n  reconstruction_mean  reconstruction_sd");
    for r in &rows {
        println!(
            "{:>5} {:>4} {:>19.6e} {:>10.6e} {:>20.6e} {:>17.6e}",
            r.width, r.model_dimension, r.approximation_error, r.alpha, r.mean, r.sd
        );
    }
    println!("\nWrote experiment_results.csv and stability_tradeoff.svg");
    Ok(())
}
