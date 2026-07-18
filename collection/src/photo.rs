use pyo3::prelude::*;

/// inpaint() - Inpaints an image using one of the available methods.
/// @py: Python interpreter token.
/// @src: Source image (u8).
/// @inpaint_mask: Inpainting mask (u8).
/// @inpaint_radius: Radius of a circular neighborhood of each point inpainted.
/// @flags: Inpainting algorithm flag.
///
/// Restores the selected region in an image using the surrounding pixels.
///
/// Return: Inpainted image array.
#[pyfunction]
#[pyo3(signature = (src, inpaint_mask, inpaint_radius, flags))]
pub fn inpaint<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    inpaint_mask: &pyo3::PyAny,
    inpaint_radius: f64,
    flags: i32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let res = cv2.call_method1("inpaint", (src, inpaint_mask, inpaint_radius, flags))?;
    Ok(res.into())
}

/// fast_nl_means_denoising() - Perform non-local means denoising on grayscale images.
/// @py: Python interpreter token.
/// @src: Source image (u8).
/// @h: Parameter deciding filter strength.
/// @template_window_size: Size in pixels of the template patch.
/// @search_window_size: Size in pixels of the search window.
///
/// Denoises the grayscale image using non-local means denoising algorithm.
///
/// Return: Denoised image array.
#[pyfunction(name = "fastNlMeansDenoising")]
#[pyo3(signature = (src, h = 3.0, template_window_size = 7, search_window_size = 21))]
pub fn fast_nl_means_denoising<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    h: f32,
    template_window_size: i32,
    search_window_size: i32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let args = (src,);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("h", h)?;
    kwargs.set_item("templateWindowSize", template_window_size)?;
    kwargs.set_item("searchWindowSize", search_window_size)?;
    let res = cv2.call_method("fastNlMeansDenoising", args, Some(&kwargs))?;
    Ok(res.into())
}

/// fast_nl_means_denoising_colored() - Perform non-local means denoising on colored images.
/// @py: Python interpreter token.
/// @src: Source image (u8).
/// @h: Parameter deciding filter strength for luminance component.
/// @h_color: Parameter deciding filter strength for color components.
/// @template_window_size: Size in pixels of the template patch.
/// @search_window_size: Size in pixels of the search window.
///
/// Denoises the color image using non-local means denoising algorithm.
///
/// Return: Denoised image array.
#[pyfunction(name = "fastNlMeansDenoisingColored")]
#[pyo3(signature = (src, h = 3.0, h_color = 3.0, template_window_size = 7, search_window_size = 21))]
pub fn fast_nl_means_denoising_colored<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    h: f32,
    h_color: f32,
    template_window_size: i32,
    search_window_size: i32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let args = (src,);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("h", h)?;
    kwargs.set_item("hColor", h_color)?;
    kwargs.set_item("templateWindowSize", template_window_size)?;
    kwargs.set_item("searchWindowSize", search_window_size)?;
    let res = cv2.call_method("fastNlMeansDenoisingColored", args, Some(&kwargs))?;
    Ok(res.into())
}

/// seamless_clone() - Seamlessly clone a source image patch into a destination image.
/// @py: Python interpreter token.
/// @src: Source image.
/// @dst: Destination image.
/// @mask: Mask image.
/// @p: Coordinates in dst where the center of src is placed.
/// @flags: Cloning method flags.
///
/// Performs seamless cloning of an image region onto another image.
///
/// Return: Cloned image array.
#[pyfunction(name = "seamlessClone")]
#[pyo3(signature = (src, dst, mask, p, flags))]
pub fn seamless_clone<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    dst: &pyo3::PyAny,
    mask: &pyo3::PyAny,
    p: (i32, i32),
    flags: i32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let res = cv2.call_method1("seamlessClone", (src, dst, mask, p, flags))?;
    Ok(res.into())
}

/// color_change() - Modifies the color of the specified region seamlessly.
/// @py: Python interpreter token.
/// @src: Source image.
/// @mask: Mask image.
/// @red_mul: Red channel multiplier.
/// @green_mul: Green channel multiplier.
/// @blue_mul: Blue channel multiplier.
///
/// Changes the color of a region seamlessly in the image.
///
/// Return: Modified image array.
#[pyfunction(name = "colorChange")]
#[pyo3(signature = (src, mask, red_mul = 1.0, green_mul = 1.0, blue_mul = 1.0))]
pub fn color_change<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    mask: &pyo3::PyAny,
    red_mul: f32,
    green_mul: f32,
    blue_mul: f32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let args = (src, mask);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("red_mul", red_mul)?;
    kwargs.set_item("green_mul", green_mul)?;
    kwargs.set_item("blue_mul", blue_mul)?;
    let res = cv2.call_method("colorChange", args, Some(&kwargs))?;
    Ok(res.into())
}

/// illumination_change() - Modifies the illumination of the specified region seamlessly.
/// @py: Python interpreter token.
/// @src: Source image.
/// @mask: Mask image.
/// @alpha: Value multiplier.
/// @beta: Value multiplier.
///
/// Changes the illumination of a region seamlessly in the image.
///
/// Return: Modified image array.
#[pyfunction(name = "illuminationChange")]
#[pyo3(signature = (src, mask, alpha = 0.2, beta = 0.4))]
pub fn illumination_change<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    mask: &pyo3::PyAny,
    alpha: f32,
    beta: f32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let args = (src, mask);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("alpha", alpha)?;
    kwargs.set_item("beta", beta)?;
    let res = cv2.call_method("illuminationChange", args, Some(&kwargs))?;
    Ok(res.into())
}

/// texture_flattening() - Flattens the texture of the specified region seamlessly.
/// @py: Python interpreter token.
/// @src: Source image.
/// @mask: Mask image.
/// @low_threshold: Lower threshold.
/// @high_threshold: Upper threshold.
/// @kernel_size: Size of the kernel.
///
/// Flattens the texture of a region seamlessly in the image.
///
/// Return: Modified image array.
#[pyfunction(name = "textureFlattening")]
#[pyo3(signature = (src, mask, low_threshold = 30.0, high_threshold = 45.0, kernel_size = 3))]
pub fn texture_flattening<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    mask: &pyo3::PyAny,
    low_threshold: f32,
    high_threshold: f32,
    kernel_size: i32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let args = (src, mask);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("low_threshold", low_threshold)?;
    kwargs.set_item("high_threshold", high_threshold)?;
    kwargs.set_item("kernel_size", kernel_size)?;
    let res = cv2.call_method("textureFlattening", args, Some(&kwargs))?;
    Ok(res.into())
}

/// decolor() - Converts a color image to grayscale with contrast preservation.
/// @py: Python interpreter token.
/// @src: Source image.
///
/// Converts color images to grayscale while preserving local contrast.
///
/// Return: A tuple containing (grayscale image array, contrast boost image array).
#[pyfunction]
pub fn decolor<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
) -> PyResult<(PyObject, PyObject)> {
    let cv2 = py.import_bound("cv2")?;
    let res = cv2.call_method1("decolor", (src,))?;
    let (gray, boost): (PyObject, PyObject) = res.extract()?;
    Ok((gray, boost))
}

/// create_tonemap() - Creates a Tonemap object.
/// @py: Python interpreter token.
/// @gamma: Gamma value.
///
/// Creates a Tonemap object for High Dynamic Range (HDR) processing.
///
/// Return: Tonemap instance.
#[pyfunction(name = "createTonemap")]
#[pyo3(signature = (gamma = 1.0))]
pub fn create_tonemap<'py>(
    py: Python<'py>,
    gamma: f32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let res = cv2.call_method1("createTonemap", (gamma,))?;
    Ok(res.into())
}

/// create_merge_mertens() - Creates a MergeMertens object.
/// @py: Python interpreter token.
/// @contrast_weight: Contrast weight.
/// @saturation_weight: Saturation weight.
/// @exposure_weight: Exposure weight.
///
/// Creates a MergeMertens object for exposure fusion.
///
/// Return: MergeMertens instance.
#[pyfunction(name = "createMergeMertens")]
#[pyo3(signature = (contrast_weight = 1.0, saturation_weight = 1.0, exposure_weight = 0.0))]
pub fn create_merge_mertens<'py>(
    py: Python<'py>,
    contrast_weight: f32,
    saturation_weight: f32,
    exposure_weight: f32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let res = cv2.call_method1("createMergeMertens", (contrast_weight, saturation_weight, exposure_weight))?;
    Ok(res.into())
}

/// create_calibrate_debevec() - Creates a CalibrateDebevec object.
/// @py: Python interpreter token.
/// @samples: Number of pixel samples.
/// @lambda_val: Regularization term weight.
/// @random: Use random samples.
///
/// Creates a CalibrateDebevec object for camera response function calibration.
///
/// Return: CalibrateDebevec instance.
#[pyfunction(name = "createCalibrateDebevec")]
#[pyo3(signature = (samples = 70, lambda_val = 10.0, random = false))]
pub fn create_calibrate_debevec<'py>(
    py: Python<'py>,
    samples: i32,
    lambda_val: f32,
    random: bool,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let res = cv2.call_method1("createCalibrateDebevec", (samples, lambda_val, random))?;
    Ok(res.into())
}

/// create_merge_debevec() - Creates a MergeDebevec object.
/// @py: Python interpreter token.
///
/// Creates a MergeDebevec object to merge exposures.
///
/// Return: MergeDebevec instance.
#[pyfunction(name = "createMergeDebevec")]
pub fn create_merge_debevec<'py>(
    py: Python<'py>,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let res = cv2.call_method0("createMergeDebevec")?;
    Ok(res.into())
}

/// edge_preserving_filter() - Applies an edge-preserving smoothing filter to an image.
/// @py: Python interpreter token.
/// @src: Source image.
/// @flags: Edge-preserving filter flag.
/// @sigma_s: Range between 0 to 200.
/// @sigma_r: Range between 0 to 1.
///
/// Smooths the image while preserving its edges.
///
/// Return: Filtered image array.
#[pyfunction(name = "edgePreservingFilter")]
#[pyo3(signature = (src, flags = 1, sigma_s = 60.0, sigma_r = 0.4))]
pub fn edge_preserving_filter<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    flags: i32,
    sigma_s: f32,
    sigma_r: f32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let args = (src,);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("flags", flags)?;
    kwargs.set_item("sigma_s", sigma_s)?;
    kwargs.set_item("sigma_r", sigma_r)?;
    let res = cv2.call_method("edgePreservingFilter", args, Some(&kwargs))?;
    Ok(res.into())
}

/// detail_enhance() - Applies a detail enhancement filter to an image.
/// @py: Python interpreter token.
/// @src: Source image.
/// @sigma_s: Range between 0 to 200.
/// @sigma_r: Range between 0 to 1.
///
/// Enhances details in the image using an edge-preserving filter.
///
/// Return: Enhanced image array.
#[pyfunction(name = "detailEnhance")]
#[pyo3(signature = (src, sigma_s = 10.0, sigma_r = 0.15))]
pub fn detail_enhance<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    sigma_s: f32,
    sigma_r: f32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let args = (src,);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("sigma_s", sigma_s)?;
    kwargs.set_item("sigma_r", sigma_r)?;
    let res = cv2.call_method("detailEnhance", args, Some(&kwargs))?;
    Ok(res.into())
}

/// pencil_sketch() - Generates pencil sketch images from a color image.
/// @py: Python interpreter token.
/// @src: Source image.
/// @sigma_s: Range between 0 to 200.
/// @sigma_r: Range between 0 to 1.
/// @shade_factor: Range between 0 to 0.1.
///
/// Converts a color image into a pencil sketch.
///
/// Return: A tuple containing (grayscale pencil sketch, color pencil sketch).
#[pyfunction(name = "pencilSketch")]
#[pyo3(signature = (src, sigma_s = 60.0, sigma_r = 0.07, shade_factor = 0.02))]
pub fn pencil_sketch<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    sigma_s: f32,
    sigma_r: f32,
    shade_factor: f32,
) -> PyResult<(PyObject, PyObject)> {
    let cv2 = py.import_bound("cv2")?;
    let args = (src,);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("sigma_s", sigma_s)?;
    kwargs.set_item("sigma_r", sigma_r)?;
    kwargs.set_item("shade_factor", shade_factor)?;
    let res = cv2.call_method("pencilSketch", args, Some(&kwargs))?;
    let (dst1, dst2): (PyObject, PyObject) = res.extract()?;
    Ok((dst1, dst2))
}

/// stylization() - Applies a non-photorealistic stylization filter to an image.
/// @py: Python interpreter token.
/// @src: Source image.
/// @sigma_s: Range between 0 to 200.
/// @sigma_r: Range between 0 to 1.
///
/// Stylizes the image using a non-photorealistic edge-preserving filter.
///
/// Return: Stylized image array.
#[pyfunction(name = "stylization")]
#[pyo3(signature = (src, sigma_s = 60.0, sigma_r = 0.07))]
pub fn stylization<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    sigma_s: f32,
    sigma_r: f32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let args = (src,);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("sigma_s", sigma_s)?;
    kwargs.set_item("sigma_r", sigma_r)?;
    let res = cv2.call_method("stylization", args, Some(&kwargs))?;
    Ok(res.into())
}















