use pyo3::prelude::*;

/// Creates a CascadeClassifier object.
#[pyfunction(name = "CascadeClassifier")]
#[pyo3(signature = (xml_path = None))]
pub fn cascade_classifier_constructor<'py>(
    py: Python<'py>,
    xml_path: Option<&str>,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let res = match xml_path {
        Some(path) => cv2.call_method1("CascadeClassifier", (path,))?,
        None => cv2.call_method0("CascadeClassifier")?,
    };
    Ok(res.into())
}
