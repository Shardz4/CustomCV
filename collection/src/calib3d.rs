use pyo3::prelude::*;

/// Estimates intrinsic camera parameters and extrinsic parameters.
#[pyfunction(name = "calibrateCamera")]
#[pyo3(signature = (object_points, image_points, image_size, camera_matrix = None, dist_coeffs = None, flags = 0, criteria = None))]
pub fn calibrate_camera<'py>(
    py: Python<'py>,
    object_points: &pyo3::PyAny,
    image_points: &pyo3::PyAny,
    image_size: (i32, i32),
    camera_matrix: Option<&pyo3::PyAny>,
    dist_coeffs: Option<&pyo3::PyAny>,
    flags: i32,
    criteria: Option<&pyo3::PyAny>,
) -> PyResult<(PyObject, PyObject, PyObject, PyObject, PyObject)> {
    let cv2 = py.import_bound("cv2")?;
    let args = (object_points, image_points, image_size, camera_matrix, dist_coeffs);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("flags", flags)?;
    if let Some(crit) = criteria {
        kwargs.set_item("criteria", crit)?;
    }
    let res = cv2.call_method("calibrateCamera", args, Some(&kwargs))?;
    let (retval, camera_matrix_out, dist_coeffs_out, rvecs, tvecs): (PyObject, PyObject, PyObject, PyObject, PyObject) = res.extract()?;
    Ok((retval, camera_matrix_out, dist_coeffs_out, rvecs, tvecs))
}
