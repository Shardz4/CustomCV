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

/// Creates a HOGDescriptor object.
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

/// Creates a QRCodeDetector object.
#[pyfunction(name = "QRCodeDetector")]
pub fn qr_code_detector_constructor<'py>(
    py: Python<'py>,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let res = cv2.call_method0("QRCodeDetector")?;
    Ok(res.into())
}


