use rustfft::{num_complex::Complex, FftPlanner};

fn fft(mut x: Vec<Complex<f64>>, len: Option<usize>) -> Vec<Complex<f64>> {
    let mut planner = FftPlanner::new();
    let fft = match len {
        Some(n) => {
            if n < x.len() {
                x.truncate(n);
            } else if n > x.len() {
                x.resize(n, Complex { re: 0.0, im: 0.0 });
            }
            planner.plan_fft_forward(n)
        }
        None => planner.plan_fft_forward(x.len()),
    };
    fft.process(&mut x);
    x
}

fn ifft(mut x: Vec<Complex<f64>>, len: Option<usize>) -> Vec<Complex<f64>> {
    let mut planner = FftPlanner::new();
    let fft = match len {
        Some(n) => {
            if n < x.len() {
                x.truncate(n);
            } else if n > x.len() {
                x.resize(n, Complex { re: 0.0, im: 0.0 });
            }
            planner.plan_fft_inverse(n)
        }
        None => planner.plan_fft_inverse(x.len()),
    };
    fft.process(&mut x);
    x
}

/// Find the smallest number at least as large as `n` whose prime factors are 2
/// and 3 only (2^i * 3^j for some whole i, j).
///
/// This is required for optimally fast FFT computation with [`rustfft`].
fn optimal_size_above(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    if n <= 4 {
        return n;
    }
    let mut history = vec![1, 2, 3, 4];
    let (mut i, mut j) = (2, 1);
    let mut x2 = 2 * history[i];
    let mut x3 = 3 * history[j];
    loop {
        history.push(x2.min(x3));
        match history.last() {
            Some(h) => {
                if h >= &n {
                    break *h;
                }
                if h == &x2 {
                    i += 1;
                    x2 = 2 * history[i];
                } else if h == &x3 {
                    j += 1;
                    x3 = 3 * history[j];
                }
            }
            _ => unreachable!("one of the above arms must match by construction"),
        }
    }
}

pub fn auto_covariance_function(x: &[f64]) -> Vec<f64> {
    let xlen = x.len() as f64;
    let mean = x.iter().sum::<f64>() / xlen;
    let xo: Vec<Complex<f64>> = x
        .iter()
        .map(|v| Complex {
            re: *v - mean,
            im: 0.0,
        })
        .collect();
    // Number of terms in a full correlation
    let num_correlate = 2 * x.len() + 1;
    let fft_optimal_size = optimal_size_above(num_correlate);
    let num_correlate_f64 = fft_optimal_size as f64;
    let frf = fft(xo, Some(fft_optimal_size));
    let auto_cov = ifft(
        frf.iter().zip(&frf).map(|(f, i)| f * i.conj()).collect(),
        None,
    );
    // Take the first xlen terms only. They correspond to the last xlen terms
    // (positive lag terms) from a full correlation. Division by
    // num_correlate_f64 because rustfft does not normalize the FFT.
    auto_cov
        .into_iter()
        .take(x.len())
        .map(|v| v.re / num_correlate_f64 / xlen)
        .collect()
}

pub fn auto_correlation_function(x: &[f64]) -> Vec<f64> {
    let acvf = auto_covariance_function(x);
    match acvf.first() {
        Some(el) => acvf.iter().map(|v| v / el).collect(),
        None => acvf,
    }
}

#[cfg(test)]
mod tests {
    use crate::optimal_size_above;

    #[test]
    fn expected_optimal_sizes() {
        let ns = vec![0, 1, 2, 3, 4, 5, 6, 7, 13];
        let exp = vec![1, 1, 2, 3, 4, 6, 6, 8, 16];
        for (n, e) in ns.iter().zip(&exp) {
            assert_eq!(*e, optimal_size_above(*n));
        }
    }
}
