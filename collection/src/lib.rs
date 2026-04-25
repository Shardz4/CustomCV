use pyo3::prelude::*;

mod helpers;
mod transforms;
mod histogram;
mod morphological;
mod edge_detection;

#[pymodule]
fn rust_cv_lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // --- Point / Pixel Transforms ---
    m.add_function(wrap_pyfunction!(transforms::apply_negative, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::apply_log, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::apply_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::rgb_to_gray, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::apply_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::rgb_to_cmy, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::apply_frequency_filter, m)?)?;

    // --- Histogram Operations ---
    m.add_function(wrap_pyfunction!(histogram::hist_equalize_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(histogram::hist_equalize_gray, m)?)?;
    m.add_function(wrap_pyfunction!(histogram::hist_spec_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(histogram::hist_spec_gray, m)?)?;
    m.add_function(wrap_pyfunction!(histogram::apply_otsu_threshold, m)?)?;

    // --- Edge Detection & Spatial Filters ---
    m.add_function(wrap_pyfunction!(edge_detection::median_filter, m)?)?;
    m.add_function(wrap_pyfunction!(edge_detection::laplacian_filter, m)?)?;
    m.add_function(wrap_pyfunction!(edge_detection::apply_canny, m)?)?;

    // --- Morphological Operations ---
    m.add_function(wrap_pyfunction!(morphological::apply_erosion, m)?)?;
    m.add_function(wrap_pyfunction!(morphological::apply_dilation, m)?)?;

    Ok(())
}