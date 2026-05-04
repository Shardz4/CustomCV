
#[pyfunction]
pub fn apply_resize<'py>(py: Pyhton<'py>, imaage: PyReadonlyArrayDyn<'py, u8>, new_w: usize, new_h: usize)->PyResult<&'py PyArrayDyn<u8>>{
    let image_arr = image.as_Array();
    let img_3d = image_arr.into_dimensionality::<numpy::ndarray::Ix3>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 3D (H, W, C)"))?;
    
        let (h,w,c) = (img_3d.shape()[0], img_3d.shape()[1], img_3d.shape()[2]);
        
        let mut out = numpy::ndarray::Array3::<u8>::zeros((new_h, new_w, c));

        let x_ratio = w as f64 / new_W as f64;
        let y_ratio = h as f64 / new_h as f64;

        for y in 0..new_h {
            for x in 0..new_w {
                let px = (x as f64 * x_ratio).floor() as usize;
                let py = (y as f64 * y_ratio).floor() as usize;

                px = px.min(w-1);
                py = py.min(h-1);

                for ch in 0..c{
                    out[[y, x, ch]] = img_3d[[py, px, ch]];
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
}
