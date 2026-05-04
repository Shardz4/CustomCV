use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};

#[pyfunction]
pub fn apply_resize<'py>(py: Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, new_w: usize, new_h: usize)->PyResult<&'py PyArrayDyn<u8>>{
    let image_arr = image.as_array();
    let ndim = image_arr.ndim();

    if ndim == 3 {
        let img_3d = image_arr.into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 3D"))?;
        let (h,w,c) = (img_3d.shape()[0], img_3d.shape()[1], img_3d.shape()[2]);
        let mut out = numpy::ndarray::Array3::<u8>::zeros((new_h, new_w, c));
        let x_ratio = w as f64 / new_w as f64;
        let y_ratio = h as f64 / new_h as f64;

        for y in 0..new_h {
            for x in 0..new_w {
                let px = ((x as f64 * x_ratio).floor() as usize).min(w.saturating_sub(1));
                let py_val = ((y as f64 * y_ratio).floor() as usize).min(h.saturating_sub(1));
                for ch in 0..c{
                    out[[y, x, ch]] = img_3d[[py_val, px, ch]];
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else if ndim == 2 {
        let img_2d = image_arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 2D"))?;
        let (h,w) = (img_2d.shape()[0], img_2d.shape()[1]);
        let mut out = numpy::ndarray::Array2::<u8>::zeros((new_h, new_w));
        let x_ratio = w as f64 / new_w as f64;
        let y_ratio = h as f64 / new_h as f64;

        for y in 0..new_h {
            for x in 0..new_w {
                let px = ((x as f64 * x_ratio).floor() as usize).min(w.saturating_sub(1));
                let py_val = ((y as f64 * y_ratio).floor() as usize).min(h.saturating_sub(1));
                out[[y, x]] = img_2d[[py_val, px]];
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

#[pyfunction]
pub fn apply_translate<'py>(py: Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, tx:isize, ty:isize)->PyResult<&'py PyArrayDyn<u8>>{
    let image_arr = image.as_array();
    let ndim = image_arr.ndim();

    if ndim == 3 {
        let img_3d = image_arr.into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 3D"))?;
        let (h,w,c) = (img_3d.shape()[0], img_3d.shape()[1], img_3d.shape()[2]);
        let mut out = numpy::ndarray::Array3::<u8>::zeros((h, w, c));

        for y in 0..h {
            for x in 0..w {
                let sr_y = (y as isize) - ty;
                let sr_x = (x as isize) - tx;
                if sr_y>=0 && sr_y < h as isize && sr_x>=0 && sr_x < w as isize{
                    for ch in 0..c {
                        out[[y, x, ch]] = img_3d[[sr_y as usize, sr_x as usize, ch]];
                    }
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else if ndim == 2 {
        let img_2d = image_arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 2D"))?;
        let (h,w) = (img_2d.shape()[0], img_2d.shape()[1]);
        let mut out = numpy::ndarray::Array2::<u8>::zeros((h, w));

        for y in 0..h {
            for x in 0..w {
                let sr_y = (y as isize) - ty;
                let sr_x = (x as isize) - tx;
                if sr_y>=0 && sr_y < h as isize && sr_x>=0 && sr_x < w as isize{
                    out[[y, x]] = img_2d[[sr_y as usize, sr_x as usize]];
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

#[pyfunction]
pub fn apply_rotate<'py>(py:Python<'py>, image:PyReadonlyArrayDyn<'py, u8>, angle:f64, center:Option<(isize,isize)>)->PyResult<&'py PyArrayDyn<u8>>{
    let img_arr = image.as_array();
    let ndim = img_arr.ndim();

    if ndim == 3 {
        let img_3d = img_arr.into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 3D"))?;
        let (h,w,c) = (img_3d.shape()[0], img_3d.shape()[1], img_3d.shape()[2]);
        let mut out = numpy::ndarray::Array3::<u8>::zeros((h, w, c));
        let cx = center.map_or((w/2) as f64, |(x,_)| x as f64);
        let cy = center.map_or((h/2) as f64, |(_,y)| y as f64);
        let sin_a = angle.sin();
        let cos_a = angle.cos();

        for y in 0..h {
            for x in 0..w {
                let dx = (x as f64) - cx;
                let dy = (y as f64) - cy;
                let src_x = (dx * cos_a + dy * sin_a) + cx;
                let src_y = (dx * sin_a - dy * cos_a) + cy;
                let px = src_x.round() as isize;
                let py_val = src_y.round() as isize;

                if px >= 0 && px < w as isize && py_val >= 0 && py_val < h as isize{
                    for ch in 0..c {
                        out[[y, x, ch]] = img_3d[[py_val as usize, px as usize, ch]];
                    }
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else if ndim == 2 {
        let img_2d = img_arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 2D"))?;
        let (h,w) = (img_2d.shape()[0], img_2d.shape()[1]);
        let mut out = numpy::ndarray::Array2::<u8>::zeros((h, w));
        let cx = center.map_or((w/2) as f64, |(x,_)| x as f64);
        let cy = center.map_or((h/2) as f64, |(_,y)| y as f64);
        let sin_a = angle.sin();
        let cos_a = angle.cos();

        for y in 0..h {
            for x in 0..w {
                let dx = (x as f64) - cx;
                let dy = (y as f64) - cy;
                let src_x = (dx * cos_a + dy * sin_a) + cx;
                let src_y = (dx * sin_a - dy * cos_a) + cy;
                let px = src_x.round() as isize;
                let py_val = src_y.round() as isize;

                if px >= 0 && px < w as isize && py_val >= 0 && py_val < h as isize{
                    out[[y, x]] = img_2d[[py_val as usize, px as usize]];
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

#[pyfunction]
pub fn apply_warp<'py>(
    py: Python<'py>, 
    img: PyReadonlyArrayDyn<'py, u8>, 
    inv_matrix: PyReadonlyArrayDyn<'py, f64>, 
    out_w: usize, 
    out_h: usize
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = img.as_array();
    let m = inv_matrix.as_array();
    let ndim = arr.ndim();

    // Matrix must be 3x3 for perspective mapping
    let m_2d = m.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Inverse Matrix must be 2D (3x3)"))?;

    if ndim == 3 {
        let img_3d = arr.into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 3D"))?;
        let (h, w, c) = (img_3d.shape()[0], img_3d.shape()[1], img_3d.shape()[2]);
        let mut out = numpy::ndarray::Array3::<u8>::zeros((out_h, out_w, c));

        for y in 0..out_h {
            for x in 0..out_w {
                let x_f = x as f64;
                let y_f = y as f64;

                let denom = m_2d[[2, 0]] * x_f + m_2d[[2, 1]] * y_f + m_2d[[2, 2]];
                if denom.abs() < 1e-6 { continue; }

                let src_x = (m_2d[[0, 0]] * x_f + m_2d[[0, 1]] * y_f + m_2d[[0, 2]]) / denom;
                let src_y = (m_2d[[1, 0]] * x_f + m_2d[[1, 1]] * y_f + m_2d[[1, 2]]) / denom;

                let px = src_x.round() as isize;
                let py_ = src_y.round() as isize;

                if px >= 0 && px < w as isize && py_ >= 0 && py_ < h as isize {
                    for ch in 0..c {
                        out[[y, x, ch]] = img_3d[[py_ as usize, px as usize, ch]];
                    }
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else if ndim == 2 {
        let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 2D"))?;
        let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
        let mut out = numpy::ndarray::Array2::<u8>::zeros((out_h, out_w));

        for y in 0..out_h {
            for x in 0..out_w {
                let x_f = x as f64;
                let y_f = y as f64;

                let denom = m_2d[[2, 0]] * x_f + m_2d[[2, 1]] * y_f + m_2d[[2, 2]];
                if denom.abs() < 1e-6 { continue; }

                let src_x = (m_2d[[0, 0]] * x_f + m_2d[[0, 1]] * y_f + m_2d[[0, 2]]) / denom;
                let src_y = (m_2d[[1, 0]] * x_f + m_2d[[1, 1]] * y_f + m_2d[[1, 2]]) / denom;

                let px = src_x.round() as isize;
                let py_ = src_y.round() as isize;

                if px >= 0 && px < w as isize && py_ >= 0 && py_ < h as isize {
                    out[[y, x]] = img_2d[[py_ as usize, px as usize]];
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}