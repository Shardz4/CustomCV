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
