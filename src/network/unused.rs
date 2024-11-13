fn cost_function(result: &Mat<f32>, target: &Mat<f32>) -> Mat<f32> {
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

    // C = 0.5 * (target - result)^2
    let mut output = result.clone();

    // for each element, result (a) and target (y)
    for (y, a) in output.col_mut(0).iter_mut().zip_eq(target.col(0).iter()) {
        *y = 0.5 * (*y - *a).powi(2);
    }

    output
}
