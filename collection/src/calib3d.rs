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

/// Finds an object pose from 3D-2D correspondences.
#[pyfunction(name = "solvePnP")]
#[pyo3(signature = (object_points, image_points, camera_matrix, dist_coeffs, use_extrinsic_guess = false, flags = 0))]
pub fn solve_pnp<'py>(
    py: Python<'py>,
    object_points: &pyo3::PyAny,
    image_points: &pyo3::PyAny,
    camera_matrix: &pyo3::PyAny,
    dist_coeffs: &pyo3::PyAny,
    use_extrinsic_guess: bool,
    flags: i32,
) -> PyResult<(PyObject, PyObject, PyObject)> {
    let cv2 = py.import_bound("cv2")?;
    let args = (object_points, image_points, camera_matrix, dist_coeffs);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("useExtrinsicGuess", use_extrinsic_guess)?;
    kwargs.set_item("flags", flags)?;
    let res = cv2.call_method("solvePnP", args, Some(&kwargs))?;
    let (retval, rvec, tvec): (PyObject, PyObject, PyObject) = res.extract()?;
    Ok((retval, rvec, tvec))
}

/// Calibrates a stereo camera setup.
#[pyfunction(name = "stereoCalibrate")]
#[pyo3(signature = (object_points, image_points1, image_points2, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, image_size, flags = 0, criteria = None))]
pub fn stereo_calibrate<'py>(
    py: Python<'py>,
    object_points: &pyo3::PyAny,
    image_points1: &pyo3::PyAny,
    image_points2: &pyo3::PyAny,
    camera_matrix1: &pyo3::PyAny,
    dist_coeffs1: &pyo3::PyAny,
    camera_matrix2: &pyo3::PyAny,
    dist_coeffs2: &pyo3::PyAny,
    image_size: (i32, i32),
    flags: i32,
    criteria: Option<&pyo3::PyAny>,
) -> PyResult<(PyObject, PyObject, PyObject, PyObject, PyObject, PyObject, PyObject, PyObject, PyObject)> {
    let cv2 = py.import_bound("cv2")?;
    let args = (object_points, image_points1, image_points2, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, image_size);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("flags", flags)?;
    if let Some(crit) = criteria {
        kwargs.set_item("criteria", crit)?;
    }
    let res = cv2.call_method("stereoCalibrate", args, Some(&kwargs))?;
    let (retval, cm1, dc1, cm2, dc2, r, t, e, f): (PyObject, PyObject, PyObject, PyObject, PyObject, PyObject, PyObject, PyObject, PyObject) = res.extract()?;
    Ok((retval, cm1, dc1, cm2, dc2, r, t, e, f))
}

/// Computes rectification transforms for each head of a calibrated stereo camera.
#[pyfunction(name = "stereoRectify")]
#[pyo3(signature = (camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, image_size, r, t, flags = 1, alpha = -1.0, new_image_size = None))]
pub fn stereo_rectify<'py>(
    py: Python<'py>,
    camera_matrix1: &pyo3::PyAny,
    dist_coeffs1: &pyo3::PyAny,
    camera_matrix2: &pyo3::PyAny,
    dist_coeffs2: &pyo3::PyAny,
    image_size: (i32, i32),
    r: &pyo3::PyAny,
    t: &pyo3::PyAny,
    flags: i32,
    alpha: f64,
    new_image_size: Option<(i32, i32)>,
) -> PyResult<(PyObject, PyObject, PyObject, PyObject, PyObject, PyObject, PyObject)> {
    let cv2 = py.import_bound("cv2")?;
    let args = (camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, image_size, r, t);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("flags", flags)?;
    kwargs.set_item("alpha", alpha)?;
    if let Some(new_sz) = new_image_size {
        kwargs.set_item("newImageSize", new_sz)?;
    }
    let res = cv2.call_method("stereoRectify", args, Some(&kwargs))?;
    let (r1, r2, p1, p2, q, roi1, roi2): (PyObject, PyObject, PyObject, PyObject, PyObject, PyObject, PyObject) = res.extract()?;
    Ok((r1, r2, p1, p2, q, roi1, roi2))
}

/// Reprojects a disparity image to 3D space.
#[pyfunction(name = "reprojectImageTo3D")]
#[pyo3(signature = (disparity, q, handle_missing_values = false, ddepth = -1))]
pub fn reproject_image_to_3d<'py>(
    py: Python<'py>,
    disparity: &pyo3::PyAny,
    q: &pyo3::PyAny,
    handle_missing_values: bool,
    ddepth: i32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let args = (disparity, q);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("handleMissingValues", handle_missing_values)?;
    kwargs.set_item("ddepth", ddepth)?;
    let res = cv2.call_method("reprojectImageTo3D", args, Some(&kwargs))?;
    Ok(res.into())
}






