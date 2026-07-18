use pyo3::prelude::*;

/// imread() - Reads an image from a file.
/// @py: Python interpreter token.
/// @filename: Name of file to be loaded.
/// @flags: Flags specifying the color type of a loaded image:
///         >0: Return a 3-channel color image.
///         =0: Return a grayscale image.
///         <0: Return the loaded image as is (with alpha channel).
///
/// Loads an image from the specified file path.
///
/// Return: Image array PyObject.
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

/// imwrite() - Writes an image to a specified file.
/// @py: Python interpreter token.
/// @filename: Name of the file.
/// @img: Image to be saved.
/// @params: Format-specific parameters encoded as pairs (paramId, paramValue).
///
/// Saves an image to the specified file.
///
/// Return: Boolean success PyObject.
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

/// imdecode() - Decodes an image from a memory buffer.
/// @py: Python interpreter token.
/// @buf: Input array or vector of bytes.
/// @flags: Same flags as in imread.
///
/// Reads an image from a specified buffer in memory.
///
/// Return: Decoded image array PyObject.
#[pyfunction(name = "imdecode")]
#[pyo3(signature = (buf, flags))]
pub fn imdecode<'py>(
    py: Python<'py>,
    buf: &pyo3::PyAny,
    flags: i32,
) -> PyResult<PyObject> {
    let cv2 = py.import_bound("cv2")?;
    let res = cv2.call_method1("imdecode", (buf, flags))?;
    Ok(res.into())
}

/// imencode() - Encodes an image into a memory buffer.
/// @py: Python interpreter token.
/// @ext: File extension that defines the output format (e.g. \".png\", \".jpg\").
/// @img: Image to be encoded.
/// @params: Format-specific parameters.
///
/// Compresses the image and stores it in the memory buffer.
///
/// Return: A tuple containing (success status, encoded buffer).
#[pyfunction(name = "imencode")]
#[pyo3(signature = (ext, img, params = None))]
pub fn imencode<'py>(
    py: Python<'py>,
    ext: &str,
    img: &pyo3::PyAny,
    params: Option<Vec<i32>>,
) -> PyResult<(PyObject, PyObject)> {
    let cv2 = py.import_bound("cv2")?;
    let res = match params {
        Some(p) => cv2.call_method1("imencode", (ext, img, p))?,
        None => cv2.call_method1("imencode", (ext, img))?,
    };
    let (retval, buf): (PyObject, PyObject) = res.extract()?;
    Ok((retval, buf))
}


