use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};

fn bilinear_interpolate_3d(img: &numpy::ndarray::ArrayView3<'_, u8>, src_x: f64, src_y: f64, ch: usize) -> u8 {
    let h = img.shape()[0] as isize;
    let w = img.shape()[1] as isize;
    
    let x1 = src_x.floor() as isize;
    let y1 = src_y.floor() as isize;
    let x2 = x1 + 1;
    let y2 = y1 + 1;
    
    let dx = src_x - x1 as f64;
    let dy = src_y - y1 as f64;
    
    let x1_c = x1.clamp(0, w - 1) as usize;
    let x2_c = x2.clamp(0, w - 1) as usize;
    let y1_c = y1.clamp(0, h - 1) as usize;
    let y2_c = y2.clamp(0, h - 1) as usize;
    
    let v11 = img[[y1_c, x1_c, ch]] as f64;
    let v21 = img[[y1_c, x2_c, ch]] as f64;
    let v12 = img[[y2_c, x1_c, ch]] as f64;
    let v22 = img[[y2_c, x2_c, ch]] as f64;
    
    let val = (1.0 - dx) * (1.0 - dy) * v11
            + dx * (1.0 - dy) * v21
            + (1.0 - dx) * dy * v12
            + dx * dy * v22;
            
    val.round().clamp(0.0, 255.0) as u8
}

fn bilinear_interpolate_2d(img: &numpy::ndarray::ArrayView2<'_, u8>, src_x: f64, src_y: f64) -> u8 {
    let h = img.shape()[0] as isize;
    let w = img.shape()[1] as isize;
    
    let x1 = src_x.floor() as isize;
    let y1 = src_y.floor() as isize;
    let x2 = x1 + 1;
    let y2 = y1 + 1;
    
    let dx = src_x - x1 as f64;
    let dy = src_y - y1 as f64;
    
    let x1_c = x1.clamp(0, w - 1) as usize;
    let x2_c = x2.clamp(0, w - 1) as usize;
    let y1_c = y1.clamp(0, h - 1) as usize;
    let y2_c = y2.clamp(0, h - 1) as usize;
    
    let v11 = img[[y1_c, x1_c]] as f64;
    let v21 = img[[y1_c, x2_c]] as f64;
    let v12 = img[[y2_c, x1_c]] as f64;
    let v22 = img[[y2_c, x2_c]] as f64;
    
    let val = (1.0 - dx) * (1.0 - dy) * v11
            + dx * (1.0 - dy) * v21
            + (1.0 - dx) * dy * v12
            + dx * dy * v22;
            
    val.round().clamp(0.0, 255.0) as u8
}

fn bicubic_weight(t: f64) -> f64 {
    let abs_t = t.abs();
    let a = -0.75;
    if abs_t <= 1.0 {
        (a + 2.0) * abs_t * abs_t * abs_t - (a + 3.0) * abs_t * abs_t + 1.0
    } else if abs_t < 2.0 {
        a * abs_t * abs_t * abs_t - 5.0 * a * abs_t * abs_t + 8.0 * a * abs_t - 4.0 * a
    } else {
        0.0
    }
}

fn bicubic_interpolate_3d(img: &numpy::ndarray::ArrayView3<'_, u8>, src_x: f64, src_y: f64, ch: usize) -> u8 {
    let h = img.shape()[0] as isize;
    let w = img.shape()[1] as isize;
    
    let x_i = src_x.floor() as isize;
    let y_i = src_y.floor() as isize;
    
    let mut val = 0.0;
    
    for dy in -1..=2 {
        let py = y_i + dy;
        let wy = bicubic_weight(src_y - py as f64);
        let py_c = py.clamp(0, h - 1) as usize;
        
        for dx in -1..=2 {
            let px = x_i + dx;
            let wx = bicubic_weight(src_x - px as f64);
            let px_c = px.clamp(0, w - 1) as usize;
            
            val += img[[py_c, px_c, ch]] as f64 * wx * wy;
        }
    }
    
    val.round().clamp(0.0, 255.0) as u8
}

fn bicubic_interpolate_2d(img: &numpy::ndarray::ArrayView2<'_, u8>, src_x: f64, src_y: f64) -> u8 {
    let h = img.shape()[0] as isize;
    let w = img.shape()[1] as isize;
    
    let x_i = src_x.floor() as isize;
    let y_i = src_y.floor() as isize;
    
    let mut val = 0.0;
    
    for dy in -1..=2 {
        let py = y_i + dy;
        let wy = bicubic_weight(src_y - py as f64);
        let py_c = py.clamp(0, h - 1) as usize;
        
        for dx in -1..=2 {
            let px = x_i + dx;
            let wx = bicubic_weight(src_x - px as f64);
            let px_c = px.clamp(0, w - 1) as usize;
            
            val += img[[py_c, px_c]] as f64 * wx * wy;
        }
    }
    
    val.round().clamp(0.0, 255.0) as u8
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

fn lanczos_weight(x: f64) -> f64 {
    let abs_x = x.abs();
    if abs_x < 4.0 {
        sinc(abs_x) * sinc(abs_x / 4.0)
    } else {
        0.0
    }
}

fn lanczos_interpolate_3d(img: &numpy::ndarray::ArrayView3<'_, u8>, src_x: f64, src_y: f64, ch: usize) -> u8 {
    let h = img.shape()[0] as isize;
    let w = img.shape()[1] as isize;
    
    let x_i = src_x.floor() as isize;
    let y_i = src_y.floor() as isize;
    
    let mut val = 0.0;
    let mut sum_w = 0.0;
    
    for dy in -3..=4 {
        let py = y_i + dy;
        let wy = lanczos_weight(src_y - py as f64);
        let py_c = py.clamp(0, h - 1) as usize;
        
        for dx in -3..=4 {
            let px = x_i + dx;
            let wx = lanczos_weight(src_x - px as f64);
            let px_c = px.clamp(0, w - 1) as usize;
            
            let weight = wx * wy;
            val += img[[py_c, px_c, ch]] as f64 * weight;
            sum_w += weight;
        }
    }
    
    if sum_w.abs() > 1e-9 {
        (val / sum_w).round().clamp(0.0, 255.0) as u8
    } else {
        let py_c = y_i.clamp(0, h - 1) as usize;
        let px_c = x_i.clamp(0, w - 1) as usize;
        img[[py_c, px_c, ch]]
    }
}

fn lanczos_interpolate_2d(img: &numpy::ndarray::ArrayView2<'_, u8>, src_x: f64, src_y: f64) -> u8 {
    let h = img.shape()[0] as isize;
    let w = img.shape()[1] as isize;
    
    let x_i = src_x.floor() as isize;
    let y_i = src_y.floor() as isize;
    
    let mut val = 0.0;
    let mut sum_w = 0.0;
    
    for dy in -3..=4 {
        let py = y_i + dy;
        let wy = lanczos_weight(src_y - py as f64);
        let py_c = py.clamp(0, h - 1) as usize;
        
        for dx in -3..=4 {
            let px = x_i + dx;
            let wx = lanczos_weight(src_x - px as f64);
            let px_c = px.clamp(0, w - 1) as usize;
            
            let weight = wx * wy;
            val += img[[py_c, px_c]] as f64 * weight;
            sum_w += weight;
        }
    }
    
    if sum_w.abs() > 1e-9 {
        (val / sum_w).round().clamp(0.0, 255.0) as u8
    } else {
        let py_c = y_i.clamp(0, h - 1) as usize;
        let px_c = x_i.clamp(0, w - 1) as usize;
        img[[py_c, px_c]]
    }
}

#[pyfunction]
#[pyo3(signature = (image, new_w, new_h, interpolation = 1))]
pub fn apply_resize<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    new_w: usize,
    new_h: usize,
    interpolation: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let image_arr = image.as_array();
    let ndim = image_arr.ndim();

    if ndim == 3 {
        let img_3d = image_arr.into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 3D"))?;
        let (h, w, c) = (img_3d.shape()[0], img_3d.shape()[1], img_3d.shape()[2]);
        let mut out = numpy::ndarray::Array3::<u8>::zeros((new_h, new_w, c));
        let x_ratio = w as f64 / new_w as f64;
        let y_ratio = h as f64 / new_h as f64;

        for y in 0..new_h {
            for x in 0..new_w {
                let src_x = (x as f64 + 0.5) * x_ratio - 0.5;
                let src_y = (y as f64 + 0.5) * y_ratio - 0.5;
                
                for ch in 0..c {
                    let val = match interpolation {
                        0 => {
                            let px = src_x.round() as isize;
                            let py = src_y.round() as isize;
                            img_3d[[py.clamp(0, h as isize - 1) as usize, px.clamp(0, w as isize - 1) as usize, ch]]
                        }
                        1 => bilinear_interpolate_3d(&img_3d.view(), src_x, src_y, ch),
                        2 => bicubic_interpolate_3d(&img_3d.view(), src_x, src_y, ch),
                        4 => lanczos_interpolate_3d(&img_3d.view(), src_x, src_y, ch),
                        _ => return Err(pyo3::exceptions::PyValueError::new_err("Unsupported interpolation mode")),
                    };
                    out[[y, x, ch]] = val;
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else if ndim == 2 {
        let img_2d = image_arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 2D"))?;
        let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
        let mut out = numpy::ndarray::Array2::<u8>::zeros((new_h, new_w));
        let x_ratio = w as f64 / new_w as f64;
        let y_ratio = h as f64 / new_h as f64;

        for y in 0..new_h {
            for x in 0..new_w {
                let src_x = (x as f64 + 0.5) * x_ratio - 0.5;
                let src_y = (y as f64 + 0.5) * y_ratio - 0.5;
                
                let val = match interpolation {
                    0 => {
                        let px = src_x.round() as isize;
                        let py = src_y.round() as isize;
                        img_2d[[py.clamp(0, h as isize - 1) as usize, px.clamp(0, w as isize - 1) as usize]]
                    }
                    1 => bilinear_interpolate_2d(&img_2d.view(), src_x, src_y),
                    2 => bicubic_interpolate_2d(&img_2d.view(), src_x, src_y),
                    4 => lanczos_interpolate_2d(&img_2d.view(), src_x, src_y),
                    _ => return Err(pyo3::exceptions::PyValueError::new_err("Unsupported interpolation mode")),
                };
                out[[y, x]] = val;
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