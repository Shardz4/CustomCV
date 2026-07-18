use pyo3::prelude::*;

/// cascade_classifier_constructor() - Creates a CascadeClassifier object.
/// @py: Python interpreter token.
/// @xml_path: Optional path to the XML classifier file.
///
/// Constructs a cascade classifier from a file (e.g. Haar or LBP cascade).
///
/// Return: CascadeClassifier instance PyObject.
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

/// hog_descriptor_constructor() - Creates a HOGDescriptor object.
/// @py: Python interpreter token.
/// @win_size: Detection window size.
/// @block_size: Block size.
/// @block_stride: Block stride.
/// @cell_size: Cell size.
/// @nbins: Number of bins.
/// @deriv_aperture: Derivative aperture.
/// @win_sigma: Gaussian smoothing parameter.
/// @histogram_norm_type: Histogram normalization type.
/// @l2_hys_threshold: L2-Hys normalization threshold.
/// @gamma_correction: Flag to enable gamma correction preprocessing.
/// @nlevels: Maximum number of detection levels.
/// @signed_gradient: Flag to use signed gradients.
///
/// Constructs a Histogram of Oriented Gradients (HOG) descriptor and detector.
///
/// Return: HOGDescriptor instance PyObject.
#[pyfunction(name = "HOGDescriptor")]
#[pyo3(signature = (win_size = (64, 128), block_size = (16, 16), block_stride = (8, 8), cell_size = (8, 8), nbins = 9, deriv_aperture = 1, win_sigma = 4.0, histogram_norm_type = 0, l2_hys_threshold = 2e-1, gamma_correction = true, nlevels = 64, signed_gradient = false))]
pub fn hog_descriptor_constructor<'py>(
    py: Python<'py>,
    win_size: (i32, i32),
    block_size: (i32, i32),
    block_stride: (i32, i32),
    cell_size: (i32, i32),
    nbins: i32,
    deriv_aperture: i32,
    win_sigma: f64,
    histogram_norm_type: i32,
    l2_hys_threshold: f64,
    gamma_correction: bool,
    nlevels: i32,
    signed_gradient: bool,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let args = (
        win_size,
        block_size,
        block_stride,
        cell_size,
        nbins,
        deriv_aperture,
        win_sigma,
        histogram_norm_type,
        l2_hys_threshold,
        gamma_correction,
        nlevels,
        signed_gradient,
    );
    let res = cv2.call_method1("HOGDescriptor", args)?;
    Ok(res.into())
}

/// qr_code_detector_constructor() - Creates a QRCodeDetector object.
/// @py: Python interpreter token.
///
/// Constructs a QR code detector and decoder.
///
/// Return: QRCodeDetector instance PyObject.
#[pyfunction(name = "QRCodeDetector")]
pub fn qr_code_detector_constructor<'py>(
    py: Python<'py>,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let res = cv2.call_method0("QRCodeDetector")?;
    Ok(res.into())
}

/// group_rectangles() - Groups overlapping rectangles.
/// @py: Python interpreter token.
/// @rect_list: Input list of rectangles to be grouped.
/// @group_threshold: Minimum number of neighbor rectangles minus 1 to retain.
/// @eps: Relative difference between sides of the rectangles to merge them.
///
/// Groups overlapping rectangles using a similarity threshold.
///
/// Return: A tuple containing (grouped rectangles, weights of grouped rectangles).
#[pyfunction(name = "groupRectangles")]
#[pyo3(signature = (rect_list, group_threshold, eps = 0.2))]
pub fn group_rectangles<'py>(
    py: Python<'py>,
    rect_list: &pyo3::PyAny,
    group_threshold: i32,
    eps: f64,
) -> PyResult<(PyObject, PyObject)> {
    let cv2 = py.import_bound("cv2")?;
    let res = cv2.call_method1("groupRectangles", (rect_list, group_threshold, eps))?;
    let (rects_out, weights_out): (PyObject, PyObject) = res.extract()?;
    Ok((rects_out, weights_out))
}



