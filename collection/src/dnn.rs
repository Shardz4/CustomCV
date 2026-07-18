use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct Net {
    pub inner: PyObject,
}

#[pymethods]
impl Net {
    /// Net::new() - Constructor for Net wrapper.
    /// @inner: Underlying OpenCV Net PyObject.
    ///
    /// Wraps an OpenCV dnn Net PyObject into the Net struct.
    ///
    /// Return: A Net instance.
    #[new]
    pub fn new(inner: PyObject) -> Self {
        Net { inner }
    }

    /// Net::setInput() - Set input blob for the network.
    /// @py: Python interpreter token.
    /// @blob: Input blob.
    /// @name: Name of the input layer.
    ///
    /// Sets the new value for the input of the specified layer name.
    ///
    /// Return: PyResult indicating success or failure.
    #[pyo3(signature = (blob, name = ""))]
    pub fn setInput<'py>(
        &self,
        py: Python<'py>,
        blob: &pyo3::PyAny,
        name: &str,
    ) -> PyResult<()> {
        let net = self.inner.bind(py);
        net.call_method1("setInput", (blob, name))?;
        Ok(())
    }

    /// Net::forward() - Run forward pass of the network.
    /// @py: Python interpreter token.
    /// @output_name: Optional name of output layer to compute.
    ///
    /// Runs forward pass to compute output of the specified layer.
    ///
    /// Return: Network output PyObject.
    #[pyo3(signature = (output_name = None))]
    pub fn forward<'py>(
        &self,
        py: Python<'py>,
        output_name: Option<&str>,
    ) -> PyResult<PyObject> {
        let net = self.inner.bind(py);
        let res = match output_name {
            Some(name) => net.call_method1("forward", (name,))?,
            None => net.call_method0("forward")?,
        };
        Ok(res.into())
    }
}

/// read_net() - Loads a neural network from file.
/// @py: Python interpreter token.
/// @model: Path to the model file.
/// @config: Optional path to the configuration file.
/// @framework: Optional name of the framework.
///
/// Loads a neural network using OpenCV's readNet function.
///
/// Return: A Net instance wrapper.
#[pyfunction(name = "readNet")]
#[pyo3(signature = (model, config = None, framework = None))]
pub fn read_net<'py>(
    py: Python<'py>,
    model: &str,
    config: Option<&str>,
    framework: Option<&str>,
) -> PyResult<Net> {
    let cv2 = py.import_bound("cv2")?;
    let dnn = cv2.getattr("dnn")?;
    let res = match (config, framework) {
        (Some(c), Some(f)) => dnn.call_method1("readNet", (model, c, f))?,
        (Some(c), None) => dnn.call_method1("readNet", (model, c))?,
        (None, _) => dnn.call_method1("readNet", (model,))?,
    };
    Ok(Net { inner: res.into() })
}

/// read_net_from_onnx() - Loads a neural network from ONNX format.
/// @py: Python interpreter token.
/// @onnx_file: Path to the ONNX file.
///
/// Loads an ONNX format neural network.
///
/// Return: A Net instance wrapper.
#[pyfunction(name = "readNetFromONNX")]
pub fn read_net_from_onnx<'py>(
    py: Python<'py>,
    onnx_file: &str,
) -> PyResult<Net> {
    let cv2 = py.import_bound("cv2")?;
    let dnn = cv2.getattr("dnn")?;
    let res = dnn.call_method1("readNetFromONNX", (onnx_file,))?;
    Ok(Net { inner: res.into() })
}

/// read_net_from_tensorflow() - Loads a neural network from Tensorflow format.
/// @py: Python interpreter token.
/// @model: Path to the model file.
/// @config: Optional path to the configuration file.
///
/// Loads a Tensorflow format neural network.
///
/// Return: A Net instance wrapper.
#[pyfunction(name = "readNetFromTensorflow")]
#[pyo3(signature = (model, config = None))]
pub fn read_net_from_tensorflow<'py>(
    py: Python<'py>,
    model: &str,
    config: Option<&str>,
) -> PyResult<Net> {
    let cv2 = py.import_bound("cv2")?;
    let dnn = cv2.getattr("dnn")?;
    let res = match config {
        Some(c) => dnn.call_method1("readNetFromTensorflow", (model, c))?,
        None => dnn.call_method1("readNetFromTensorflow", (model,))?,
    };
    Ok(Net { inner: res.into() })
}

/// blob_from_image() - Creates 4-dimensional blob from image.
/// @py: Python interpreter token.
/// @image: Input image.
/// @scalefactor: Multiplier for image values.
/// @size: Spatial size of the output image.
/// @mean: Mean subtraction values.
/// @swap_rb: If true, swaps Red and Blue channels.
/// @crop: If true, crops the image.
/// @ddepth: Depth of the output blob.
///
/// Creates a 4-dimensional blob from an input image.
///
/// Return: The 4D blob PyObject.
#[pyfunction(name = "blobFromImage")]
#[pyo3(signature = (image, scalefactor = 1.0, size = (0, 0), mean = (0.0, 0.0, 0.0), swap_rb = false, crop = false, ddepth = 5))]
pub fn blob_from_image<'py>(
    py: Python<'py>,
    image: &pyo3::PyAny,
    scalefactor: f64,
    size: (i32, i32),
    mean: (f64, f64, f64),
    swap_rb: bool,
    crop: bool,
    ddepth: i32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let dnn = cv2.getattr("dnn")?;
    let args = (image,);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("scalefactor", scalefactor)?;
    kwargs.set_item("size", size)?;
    kwargs.set_item("mean", mean)?;
    kwargs.set_item("swapRB", swap_rb)?;
    kwargs.set_item("crop", crop)?;
    kwargs.set_item("ddepth", ddepth)?;
    let res = dnn.call_method("blobFromImage", args, Some(&kwargs))?;
    Ok(res.into())
}

/// nms_boxes() - Performs non-maximum suppression.
/// @py: Python interpreter token.
/// @bboxes: List of bounding boxes.
/// @scores: List of confidence scores.
/// @score_threshold: Score threshold.
/// @nms_threshold: Non-maximum suppression threshold.
/// @eta: Rate parameter.
/// @top_k: Top K boxes to keep.
///
/// Performs non-maximum suppression given boxes and scores.
///
/// Return: Indices of boxes to keep.
#[pyfunction(name = "NMSBoxes")]
#[pyo3(signature = (bboxes, scores, score_threshold, nms_threshold, eta = 1.0, top_k = 0))]
pub fn nms_boxes<'py>(
    py: Python<'py>,
    bboxes: &pyo3::PyAny,
    scores: &pyo3::PyAny,
    score_threshold: f32,
    nms_threshold: f32,
    eta: f32,
    top_k: i32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let dnn = cv2.getattr("dnn")?;
    let args = (bboxes, scores, score_threshold, nms_threshold);
    let kwargs = pyo3::types::PyDict::new_bound(py);
    kwargs.set_item("eta", eta)?;
    kwargs.set_item("top_k", top_k)?;
    let res = dnn.call_method("NMSBoxes", args, Some(&kwargs))?;
    Ok(res.into())
}


