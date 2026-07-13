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

