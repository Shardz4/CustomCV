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

/// Finds the positions of internal corners of the chessboard.
#[pyfunction(name = "findChessboardCorners")]
#[pyo3(signature = (image, pattern_size, flags = 0))]
pub fn find_chessboard_corners<'py>(
    py: Python<'py>,
    image: &pyo3::PyAny,
    pattern_size: (i32, i32),
    flags: i32,
) -> PyResult<(PyObject, PyObject)> {
    let cv2 = py.import_bound("cv2")?;
    let args = (image, pattern_size);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("flags", flags)?;
    let res = cv2.call_method("findChessboardCorners", args, Some(&kwargs))?;
    let (retval, corners): (PyObject, PyObject) = res.extract()?;
    Ok((retval, corners))
}

/// Transforms an image to compensate for lens distortion.
#[pyfunction(name = "undistort")]
#[pyo3(signature = (src, camera_matrix, dist_coeffs, new_camera_matrix = None))]
pub fn undistort<'py>(
    py: Python<'py>,
    src: &pyo3::PyAny,
    camera_matrix: &pyo3::PyAny,
    dist_coeffs: &pyo3::PyAny,
    new_camera_matrix: Option<&pyo3::PyAny>,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let args = (src, camera_matrix, dist_coeffs);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    if let Some(new_cam) = new_camera_matrix {
        kwargs.set_item("newCameraMatrix", new_cam)?;
    }
    let res = cv2.call_method("undistort", args, Some(&kwargs))?;
    Ok(res.into())
}


