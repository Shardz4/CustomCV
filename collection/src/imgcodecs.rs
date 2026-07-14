use pyo3::prelude::*;

/// Reads an image from a file.
#[pyfunction(name = "imread")]
#[pyo3(signature = (filename, flags = 1))]
pub fn imread<'py>(
    py: Python<'py>,
    filename: &str,
    flags: i32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let res = cv2.call_method1("imread", (filename, flags))?;
    Ok(res.into())
}

/// Writes an image to a specified file.
#[pyfunction(name = "imwrite")]
#[pyo3(signature = (filename, img, params = None))]
pub fn imwrite<'py>(
    py: Python<'py>,
    filename: &str,
    img: &pyo3::PyAny,
    params: Option<Vec<i32>>,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let res = match params {
        Some(p) => cv2.call_method1("imwrite", (filename, img, p))?,
        None => cv2.call_method1("imwrite", (filename, img))?,
    };
    Ok(res.into())
}

