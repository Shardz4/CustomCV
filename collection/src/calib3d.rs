use pyo3::prelude::*;

/// calibrate_camera() - Estimates intrinsic camera parameters and extrinsic parameters.
/// @py: Python interpreter token.
/// @object_points: Vector of vectors of calibration pattern points in calibration pattern coordinate space.
/// @image_points: Vector of vectors of projections of calibration pattern points.
/// @image_size: Size of the image used only to initialize the intrinsic camera matrix.
/// @camera_matrix: Input/output camera matrix.
/// @dist_coeffs: Input/output vector of distortion coefficients.
/// @flags: Calibration flags.
/// @criteria: Termination criteria for the iterative optimization algorithm.
///
/// Estimates intrinsic camera parameters and extrinsic parameters for each camera view.
///
/// Return: A tuple containing (retval, camera_matrix, dist_coeffs, rvecs, tvecs).
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

/// find_chessboard_corners() - Finds the positions of internal corners of the chessboard.
/// @py: Python interpreter token.
/// @image: Source chessboard view. It must be an 8-bit grayscale or color image.
/// @pattern_size: Number of inner corners per chessboard row and column.
/// @flags: Various operation flags.
///
/// Finds the positions of internal corners of the chessboard calibration pattern.
///
/// Return: A tuple containing (retval, corners).
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

/// undistort() - Transforms an image to compensate for lens distortion.
/// @py: Python interpreter token.
/// @src: Input image.
/// @camera_matrix: Input camera matrix.
/// @dist_coeffs: Input vector of distortion coefficients.
/// @new_camera_matrix: Camera matrix of the undistorted image.
///
/// Transforms an image to compensate for radial and tangential lens distortion.
///
/// Return: The undistorted image.
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

/// solve_pnp() - Finds an object pose from 3D-2D correspondences.
/// @py: Python interpreter token.
/// @object_points: Array of object points in the object coordinate space.
/// @image_points: Array of corresponding image points.
/// @camera_matrix: Input camera matrix.
/// @dist_coeffs: Input vector of distortion coefficients.
/// @use_extrinsic_guess: If true, uses the initial rvec and tvec as guess.
/// @flags: Method for solving the PNP problem.
///
/// Finds an object pose from 3D-2D correspondences.
///
/// Return: A tuple containing (retval, rvec, tvec).
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

/// stereo_calibrate() - Calibrates a stereo camera setup.
/// @py: Python interpreter token.
/// @object_points: Vector of vectors of calibration pattern points.
/// @image_points1: Vector of vectors of projections in the first camera.
/// @image_points2: Vector of vectors of projections in the second camera.
/// @camera_matrix1: Camera matrix of the first camera.
/// @dist_coeffs1: Distortion coefficients of the first camera.
/// @camera_matrix2: Camera matrix of the second camera.
/// @dist_coeffs2: Distortion coefficients of the second camera.
/// @image_size: Size of the image.
/// @flags: Calibration flags.
/// @criteria: Termination criteria.
///
/// Calibrates a stereo camera setup.
///
/// Return: A tuple containing (retval, cm1, dc1, cm2, dc2, r, t, e, f).
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

/// stereo_rectify() - Computes rectification transforms for each head of a calibrated stereo camera.
/// @py: Python interpreter token.
/// @camera_matrix1: First camera matrix.
/// @dist_coeffs1: First camera distortion coefficients.
/// @camera_matrix2: Second camera matrix.
/// @dist_coeffs2: Second camera distortion coefficients.
/// @image_size: Size of the image.
/// @r: Rotation matrix between the first and second camera coordinate systems.
/// @t: Translation vector between coordinate systems.
/// @flags: Operation flags.
/// @alpha: Free scaling parameter.
/// @new_image_size: New image resolution after rectification.
///
/// Computes rectification transforms for a calibrated stereo camera.
///
/// Return: A tuple containing (r1, r2, p1, p2, q, roi1, roi2).
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

/// reproject_image_to_3d() - Reprojects a disparity image to 3D space.
/// @py: Python interpreter token.
/// @disparity: Input single-channel 8-bit or 16-bit signed disparity image.
/// @q: 4x4 perspective transformation matrix.
/// @handle_missing_values: If true, pixels with minimum disparity are set to large values.
/// @ddepth: Output depth.
///
/// Reprojects a disparity image to 3D space.
///
/// Return: The 3-channel 3D point cloud image.
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

/// find_essential_mat() - Calculates an essential matrix from corresponding points in two images.
/// @py: Python interpreter token.
/// @points1: Array of N corresponding points from the first image.
/// @points2: Array of N corresponding points from the second image.
/// @camera_matrix: Camera matrix.
/// @method: Method for robust estimation.
/// @prob: Parameter used for RANSAC or LMedS methods.
/// @threshold: Parameter used for RANSAC.
///
/// Calculates an essential matrix from corresponding points in two images.
///
/// Return: A tuple containing (essential_matrix, mask).
#[pyfunction(name = "findEssentialMat")]
#[pyo3(signature = (points1, points2, camera_matrix = None, method = 8, prob = 0.999, threshold = 1.0))]
pub fn find_essential_mat<'py>(
    py: Python<'py>,
    points1: &pyo3::PyAny,
    points2: &pyo3::PyAny,
    camera_matrix: Option<&pyo3::PyAny>,
    method: i32,
    prob: f64,
    threshold: f64,
) -> PyResult<(PyObject, PyObject)> {
    let cv2 = py.import_bound("cv2")?;
    let args = (points1, points2);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    if let Some(cam) = camera_matrix {
        kwargs.set_item("cameraMatrix", cam)?;
    }
    kwargs.set_item("method", method)?;
    kwargs.set_item("prob", prob)?;
    kwargs.set_item("threshold", threshold)?;
    let res = cv2.call_method("findEssentialMat", args, Some(&kwargs))?;
    let (e, mask): (PyObject, PyObject) = res.extract()?;
    Ok((e, mask))
}

/// find_fundamental_mat() - Calculates a fundamental matrix from corresponding points in two images.
/// @py: Python interpreter token.
/// @points1: Array of N corresponding points from the first image.
/// @points2: Array of N corresponding points from the second image.
/// @method: Method for robust estimation.
/// @ransac_reproj_threshold: Parameter used for RANSAC.
/// @confidence: Parameter specifying confidence level.
/// @max_iters: Maximum number of iterations.
///
/// Calculates a fundamental matrix from corresponding points in two images.
///
/// Return: A tuple containing (fundamental_matrix, mask).
#[pyfunction(name = "findFundamentalMat")]
#[pyo3(signature = (points1, points2, method = 8, ransac_reproj_threshold = 3.0, confidence = 0.99, max_iters = 2000))]
pub fn find_fundamental_mat<'py>(
    py: Python<'py>,
    points1: &pyo3::PyAny,
    points2: &pyo3::PyAny,
    method: i32,
    ransac_reproj_threshold: f64,
    confidence: f64,
    max_iters: i32,
) -> PyResult<(PyObject, PyObject)> {
    let cv2 = py.import_bound("cv2")?;
    let args = (points1, points2);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("method", method)?;
    kwargs.set_item("ransacReprojThreshold", ransac_reproj_threshold)?;
    kwargs.set_item("confidence", confidence)?;
    kwargs.set_item("maxIters", max_iters)?;
    let res = cv2.call_method("findFundamentalMat", args, Some(&kwargs))?;
    let (f, mask): (PyObject, PyObject) = res.extract()?;
    Ok((f, mask))
}

/// decompose_homography_mat() - Decomposes a homography matrix to rotation and translation.
/// @py: Python interpreter token.
/// @h: Homography matrix.
/// @k: Camera intrinsic matrix.
///
/// Decomposes a homography matrix to rotation, translation, and plane normal vectors.
///
/// Return: A tuple containing (retval, rotations, translations, normals).
#[pyfunction(name = "decomposeHomographyMat")]
pub fn decompose_homography_mat<'py>(
    py: Python<'py>,
    h: &pyo3::PyAny,
    k: &pyo3::PyAny,
) -> PyResult<(PyObject, PyObject, PyObject, PyObject)> {
    let cv2 = py.import_bound("cv2")?;
    let res = cv2.call_method1("decomposeHomographyMat", (h, k))?;
    let (retval, rotations, translations, normals): (PyObject, PyObject, PyObject, PyObject) = res.extract()?;
    Ok((retval, rotations, translations, normals))
}

/// triangulate_points() - Reconstructs 3D points from stereo camera observations.
/// @py: Python interpreter token.
/// @proj_matrix1: 3x4 projection matrix of the first camera.
/// @proj_matrix2: 3x4 projection matrix of the second camera.
/// @proj_points1: 2xN array of feature points in the first image.
/// @proj_points2: 2xN array of feature points in the second image.
///
/// Reconstructs 3D points from stereo camera observations.
///
/// Return: A 4xN array of reconstructed points in homogeneous coordinates.
#[pyfunction(name = "triangulatePoints")]
pub fn triangulate_points<'py>(
    py: Python<'py>,
    proj_matrix1: &pyo3::PyAny,
    proj_matrix2: &pyo3::PyAny,
    proj_points1: &pyo3::PyAny,
    proj_points2: &pyo3::PyAny,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let res = cv2.call_method1("triangulatePoints", (proj_matrix1, proj_matrix2, proj_points1, proj_points2))?;
    Ok(res.into())
}










