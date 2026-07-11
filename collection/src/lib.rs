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
mod segmentation;
mod drawing;
mod features2d;

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
    m.add_function(wrap_pyfunction!(histogram::calc_hist, m)?)?;
    m.add_function(wrap_pyfunction!(histogram::compare_hist, m)?)?;
    m.add_function(wrap_pyfunction!(histogram::match_template, m)?)?;
    m.add_function(wrap_pyfunction!(histogram::calc_back_project, m)?)?;
    m.add_function(wrap_pyfunction!(histogram::emd_1d, m)?)?;

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
    m.add_function(wrap_pyfunction!(arithematic::apply_multiply, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::multiply, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::apply_divide, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::divide, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::absdiff, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::apply_min, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::min, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::apply_max, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::max, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::apply_pow, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::apply_sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::apply_exp, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::apply_log_op, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::apply_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::normalize, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::convert_scale_abs, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::in_range, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::in_range_cv, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::apply_lut, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::lut_cv, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::split_channels, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::split, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::merge_channels, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::merge, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::mix_channels, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::mix_channels_cv, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::count_non_zero, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::count_non_zero_cv, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::mean_std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::mean_std_dev_cv, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::min_max_loc, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::min_max_loc_cv, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::calculate_phase, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::calculate_magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::cart_to_polar, m)?)?;
    m.add_function(wrap_pyfunction!(arithematic::polar_to_cart, m)?)?;

    // --- Geometric Transforms ---
    m.add_function(wrap_pyfunction!(geometric::apply_resize, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::resize, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::apply_translate, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::apply_rotate, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::apply_warp, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::apply_warp_affine, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::warp_affine, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::get_rotation_matrix_2d, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::get_rotation_matrix_2d_cv, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::get_affine_transform, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::get_affine_transform_cv, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::get_perspective_transform, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::get_perspective_transform_cv, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::apply_flip, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::flip, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::apply_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::transpose, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::apply_remap, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::remap, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::invert_affine_transform, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::invert_affine_transform_cv, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::apply_linear_polar, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::linear_polar, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::apply_log_polar, m)?)?;
    m.add_function(wrap_pyfunction!(geometric::log_polar, m)?)?;

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
    m.add_function(wrap_pyfunction!(vid::calc_optical_flow_pyr_lk, m)?)?;
    m.add_function(wrap_pyfunction!(vid::calc_optical_flow_farneback, m)?)?;
    m.add_function(wrap_pyfunction!(vid::create_background_subtractor_knn, m)?)?;
    m.add_function(wrap_pyfunction!(vid::mean_shift, m)?)?;
    m.add_function(wrap_pyfunction!(vid::cam_shift, m)?)?;
    m.add_function(wrap_pyfunction!(vid::kalman_filter_constructor, m)?)?;
    m.add_function(wrap_pyfunction!(vid::dis_optical_flow_create, m)?)?;
    m.add_function(wrap_pyfunction!(vid::sparse_pyr_lk_optical_flow_create, m)?)?;
    m.add_function(wrap_pyfunction!(vid::build_optical_flow_pyramid, m)?)?;

    // --- Contour & Shape Analysis ---
    m.add_function(wrap_pyfunction!(contours::find_contours, m)?)?;
    m.add_function(wrap_pyfunction!(contours::draw_contours, m)?)?;
    m.add_function(wrap_pyfunction!(contours::contour_area, m)?)?;
    m.add_function(wrap_pyfunction!(contours::arc_length, m)?)?;
    m.add_function(wrap_pyfunction!(contours::bounding_rect, m)?)?;
    m.add_function(wrap_pyfunction!(contours::min_area_rect, m)?)?;
    m.add_function(wrap_pyfunction!(contours::min_enclosing_circle, m)?)?;
    m.add_function(wrap_pyfunction!(contours::fit_ellipse, m)?)?;
    m.add_function(wrap_pyfunction!(contours::convex_hull, m)?)?;
    m.add_function(wrap_pyfunction!(contours::convexity_defects, m)?)?;
    m.add_function(wrap_pyfunction!(contours::approx_poly_dp, m)?)?;
    m.add_function(wrap_pyfunction!(contours::moments, m)?)?;
    m.add_function(wrap_pyfunction!(contours::hu_moments, m)?)?;
    m.add_function(wrap_pyfunction!(contours::match_shapes, m)?)?;
    m.add_function(wrap_pyfunction!(contours::is_contour_convex, m)?)?;
    m.add_function(wrap_pyfunction!(contours::point_polygon_test, m)?)?;

    // --- Image Segmentation ---
    m.add_function(wrap_pyfunction!(segmentation::connected_components, m)?)?;
    m.add_function(wrap_pyfunction!(segmentation::connected_components_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(segmentation::distance_transform, m)?)?;
    m.add_function(wrap_pyfunction!(segmentation::flood_fill, m)?)?;
    m.add_function(wrap_pyfunction!(segmentation::watershed, m)?)?;
    m.add_function(wrap_pyfunction!(segmentation::grab_cut, m)?)?;
    m.add_function(wrap_pyfunction!(segmentation::grab_cut_cv, m)?)?;

    // --- Drawing & Annotation ---
    m.add_function(wrap_pyfunction!(drawing::line, m)?)?;
    m.add_function(wrap_pyfunction!(drawing::rectangle, m)?)?;
    m.add_function(wrap_pyfunction!(drawing::circle, m)?)?;
    m.add_function(wrap_pyfunction!(drawing::ellipse, m)?)?;
    m.add_function(wrap_pyfunction!(drawing::polylines, m)?)?;
    m.add_function(wrap_pyfunction!(drawing::fill_poly, m)?)?;
    m.add_function(wrap_pyfunction!(drawing::arrowed_line, m)?)?;
    m.add_function(wrap_pyfunction!(drawing::put_text, m)?)?;

    // --- Feature Detection & Matching ---
    m.add_class::<features2d::KeyPoint>()?;
    m.add_class::<features2d::DMatch>()?;
    m.add_function(wrap_pyfunction!(features2d::fast_detect, m)?)?;
    m.add_function(wrap_pyfunction!(features2d::good_features_to_track, m)?)?;
    m.add_function(wrap_pyfunction!(features2d::orb_detect_and_compute, m)?)?;
    m.add_function(wrap_pyfunction!(features2d::sift_detect_and_compute, m)?)?;
    m.add_function(wrap_pyfunction!(features2d::brisk_detect_and_compute, m)?)?;
    m.add_function(wrap_pyfunction!(features2d::akaze_detect_and_compute, m)?)?;
    m.add_function(wrap_pyfunction!(features2d::mser_detect, m)?)?;
    m.add_function(wrap_pyfunction!(features2d::simple_blob_detect, m)?)?;
    m.add_function(wrap_pyfunction!(features2d::bf_match, m)?)?;
    m.add_function(wrap_pyfunction!(features2d::knn_match, m)?)?;
    Ok(())
}