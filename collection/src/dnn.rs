use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct Net {
    pub inner: PyObject,
}

#[pymethods]
impl Net {
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

/// Loads a neural network from file.
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

/// Loads a neural network from ONNX format.
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

/// Loads a neural network from Tensorflow format.
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

/// Creates 4-dimensional blob from image.
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

