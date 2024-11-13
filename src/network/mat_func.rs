use faer::Mat;
use itertools::Itertools;

pub trait MatFunc {
    fn sigmoid_mat(&mut self);

    fn mult_mat_by_n(&mut self, n: f32);
}

impl MatFunc for Mat<f32> {
    fn sigmoid_mat(&mut self) {
        for col in self.col_iter_mut() {
            for val in col.iter_mut() {
                *val = sigmoid(*val);
            }
        }
    }

    fn mult_mat_by_n(&mut self, n: f32) {
        for col in self.col_iter_mut() {
            for val in col.iter_mut() {
                *val *= n;
            }
        }
    }
}

pub trait VecMatFunc {
    fn mult_v_mat_by_n(&mut self, n: f32);

    fn add_vmat(&mut self, other: &[Mat<f32>]);
}

impl VecMatFunc for Vec<Mat<f32>> {
    fn mult_v_mat_by_n(&mut self, n: f32) {
        for mat in self.iter_mut() {
            mat.mult_mat_by_n(n);
        }
    }

    fn add_vmat(&mut self, delta: &[Mat<f32>]) {
        // for each element in Vec and delta Vec
        for (e, d) in self.iter_mut().zip_eq(delta.iter()) {
            *e += d;
        }
    }
}

pub fn hadamard(m1: &Mat<f32>, m2: &Mat<f32>) -> Mat<f32> {
    assert_eq!(
        m1.nrows(),
        m2.nrows(),
        "Matrices must have same number of rows: {} != {}",
        m1.nrows(),
        m2.nrows(),
    );
    assert_eq!(
        m1.ncols(),
        m2.ncols(),
        "Matrices must have same number of columns: {} != {}",
        m1.ncols(),
        m2.ncols(),
    );

    let mut output = m1.clone();

    for (base_col, mult_col) in output.col_iter_mut().zip_eq(m2.col_iter()) {
        for (base, mult) in base_col.iter_mut().zip_eq(mult_col.iter()) {
            *base *= *mult;
        }
    }

    output
}

pub fn _cost_sum(result: &Mat<f32>, target: &Mat<f32>) -> f32 {
    assert_eq!(
        target.ncols(),
        1,
        "Target Mat has {} columns. It should only have 1 column.",
        target.ncols()
    );
    assert_eq!(
        result.ncols(),
        1,
        "Result Mat has {} columns. It should only have 1 column.",
        target.ncols()
    );
    assert_eq!(
        target.nrows(),
        result.nrows(),
        "Target Mat {} rows do not match Result Mat {} rows.",
        target.nrows(),
        result.nrows(),
    );

    // C = 0.5 * (result - target)^2
    let mut output = 0.0;

    // for each element, result (a) and target (y)
    for (y, a) in result.col(0).iter().zip_eq(target.col(0).iter()) {
        output += 0.5 * (*a - *y).powi(2);
    }

    output
}

pub fn cost_prime(result: &Mat<f32>, target: &Mat<f32>) -> Mat<f32> {
    assert_eq!(
        target.ncols(),
        1,
        "Target Mat has {} columns. It should only have 1 column.",
        target.ncols()
    );
    assert_eq!(
        result.ncols(),
        1,
        "Result Mat has {} columns. It should only have 1 column.",
        target.ncols()
    );
    assert_eq!(
        target.nrows(),
        result.nrows(),
        "Target Mat {} rows do not match Result Mat {} rows.",
        target.nrows(),
        result.nrows(),
    );

    // dC / da = result - target
    let mut target = target.clone();

    // for each element, result (a) and target (y)
    for (y, a) in target.col_mut(0).iter_mut().zip_eq(result.col(0).iter()) {
        *y = *a - *y;
    }

    target
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sig_prime(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn sig_prime_mat(from: &Mat<f32>) -> Mat<f32> {
    assert_eq!(from.ncols(), 1, "Expected single column for sig_prime_mat");

    let mut output = from.clone();
    for element in output.col_mut(0).iter_mut() {
        *element = sig_prime(*element);
    }

    output
}
