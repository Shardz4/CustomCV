use pyo3::prelude::*;

/// Inpaints an image using one of the available methods.
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

/// Perform non-local means denoising on grayscale images.
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

/// Perform non-local means denoising on colored images.
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

/// Seamlessly clone a source image patch into a destination image.
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

/// Modifies the color of the specified region seamlessly.
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

/// Modifies the illumination of the specified region seamlessly.
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

/// Flattens the texture of the specified region seamlessly.
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

/// Converts a color image to grayscale with contrast preservation.
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







