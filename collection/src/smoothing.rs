#[pyfunction]
fn apply_filter2d<'py>(py: Python<'py>, img: ReadonlyArrayDyn<'py, u8>, kernel: PyReadonlyArrayDyn<'py, f64>) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = img.as_array();
    let k_arr = kernel.as_array();
    let k_2d = k_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Kernel must be 2d"))?;
    
    let ndim = arr.ndim();
    let hsape = arr.shape();

    if ndim ==3 {
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let mut out = numpy::ndarray::Array3::<u8>::zeros((h, w, c));

        for ch in 0..c{
            let filtered = helpers::convolve_2d_channel(arr.slice(s![.., .., ch]), k_2d.view());
            out.slice_mut(s![.., .., ch]).assign(&filtered);
        }
        Ok(out.into_pyarray_bound(py).to_dyn().into())
    } else if ndim ==2{
        let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        let filtered = helpers::convolve_2d_channel(channel.view(), k_2d.view());
        return Ok(filtered.into_pyarray_bound(py).to_dyn().into());
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be a 2d or 3d"))
    }
}