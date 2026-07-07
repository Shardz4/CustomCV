use pyo3::prelude::*;

mod helpers;
mod transforms;
mod histogram;
mod morphological;
mod edge_detection;
mod arithematic;
mod geometric;
mod filters;
mod smoothing;
mod vid;
mod color_convert;
mod gradient;
mod contours;

#[pymodule]
fn rust_cv_lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // --- Point / Pixel Transforms ---
    m.add_function(wrap_pyfunction!(transforms::apply_negative, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::apply_log, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::apply_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::rgb_to_gray, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::apply_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::apply_threshold_binary_inv, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::apply_threshold_trunc, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::apply_threshold_tozero, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::apply_threshold_tozero_inv, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::apply_threshold_triangle, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::apply_otsu_with_mode, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::rgb_to_cmy, m)?)?;
    m.add_function(wrap_pyfunction!(transforms::apply_frequency_filter, m)?)?;

    // --- Color Space Conversions ---
    m.add_function(wrap_pyfunction!(color_convert::rgb_to_hsv, m)?)?;
    m.add_function(wrap_pyfunction!(color_convert::rgb_to_hls, m)?)?;
    m.add_function(wrap_pyfunction!(color_convert::rgb_to_ycrcb, m)?)?;
    m.add_function(wrap_pyfunction!(color_convert::rgb_to_xyz, m)?)?;
    m.add_function(wrap_pyfunction!(color_convert::rgb_to_lab, m)?)?;
    m.add_function(wrap_pyfunction!(color_convert::rgb_to_luv, m)?)?;
    m.add_function(wrap_pyfunction!(color_convert::bgr_to_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(color_convert::gray_to_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(color_convert::rgb_to_yuv, m)?)?;

    // --- Histogram Operations ---
    m.add_function(wrap_pyfunction!(histogram::hist_equalize_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(histogram::hist_equalize_gray, m)?)?;
    m.add_function(wrap_pyfunction!(histogram::hist_spec_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(histogram::hist_spec_gray, m)?)?;
    m.add_function(wrap_pyfunction!(histogram::apply_otsu_threshold, m)?)?;

    // --- Spatial Filters ---
    m.add_function(wrap_pyfunction!(filters::median_filter, m)?)?;
    m.add_function(wrap_pyfunction!(filters::laplacian_filter, m)?)?;

        // --- Image Pyramids ---
    m.add_function(wrap_pyfunction!(filters::pyr_down, m)?)?;
    m.add_function(wrap_pyfunction!(filters::pyr_up, m)?)?;

    // --- Edge & Feature Detection ---
    m.add_function(wrap_pyfunction!(edge_detection::apply_canny, m)?)?;
    m.add_function(wrap_pyfunction!(edge_detection::harris_corner, m)?)?;
    m.add_function(wrap_pyfunction!(edge_detection::shi_tomasi_corners, m)?)?;
    m.add_function(wrap_pyfunction!(edge_detection::hough_lines, m)?)?;
    m.add_function(wrap_pyfunction!(edge_detection::hough_circles, m)?)?;

    // --- Gradient & Edge Operators ---
    m.add_function(wrap_pyfunction!(gradient::apply_sobel, m)?)?;
    m.add_function(wrap_pyfunction!(gradient::apply_scharr, m)?)?;
    m.add_function(wrap_pyfunction!(gradient::apply_laplacian, m)?)?;
    m.add_function(wrap_pyfunction!(smoothing::apply_filter2d, m)?)?;

    // --- Morphological Operations ---
    m.add_function(wrap_pyfunction!(morphological::apply_erosion, m)?)?;
    m.add_function(wrap_pyfunction!(morphological::apply_dilation, m)?)?;
    m.add_function(wrap_pyfunction!(morphological::opening, m)?)?;
    m.add_function(wrap_pyfunction!(morphological::apply_closing, m)?)?;
    m.add_function(wrap_pyfunction!(morphological::morphological_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(morphological::top_hat, m)?)?;
    m.add_function(wrap_pyfunction!(morphological::black_hat, m)?)?;

    // --- Arithmetic & Bitwise Operations ---
    m.add_function(wrap_pyfunction!(arithematic::add_images, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::sub_images, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::add_weighted, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::bitwise_and, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::bitwise_or, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::bitwise_not, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::bitwise_xor, m)?)?;

    // --- Geometric Transforms ---
    m.add_function(wrap_pyfunction!(geometric::apply_resize, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::apply_translate, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::apply_rotate, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::apply_warp, m)?)?;

    // --- Smoothing Filters ---
    m.add_function(wrap_pyfunction!(smoothing::apply_blur, m)?)?;
    m.add_function(wrap_pyfunction!(smoothing::apply_gaussian_blur, m)?)?;
    m.add_function(wrap_pyfunction!(smoothing::apply_median_blur, m)?)?;
    m.add_function(wrap_pyfunction!(smoothing::apply_bilateral_filter, m)?)?;

    // --- Video Operations ---
    m.add_function(wrap_pyfunction!(vid::video_capture, m)?)?;
    m.add_function(wrap_pyfunction!(vid::extract_images_from_video, m)?)?;
    m.add_function(wrap_pyfunction!(vid::extract_video_from_images, m)?)?;
    m.add_function(wrap_pyfunction!(vid::background_subtract_mog2, m)?)?;

    // --- Contour & Shape Analysis ---
    m.add_function(wrap_pyfunction!(contours::find_contours, m)?)?;
    m.add_function(wrap_pyfunction!(contours::draw_contours, m)?)?;
    m.add_function(wrap_pyfunction!(contours::contour_area, m)?)?;
    m.add_function(wrap_pyfunction!(contours::arc_length, m)?)?;
    Ok(())
}