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

/// Apply an affine transformation to an image.
///
/// Returns a new warped image.
/// - `M`: 2x3 transformation matrix.
/// - `out_w`: width of the output image.
/// - `out_h`: height of the output image.
/// - `flags`: combination of interpolation and map flags (e.g. 16 for WARP_INVERSE_MAP).
#[pyfunction]
#[allow(non_snake_case)]
#[pyo3(signature = (image, M, out_w, out_h, flags = 0))]
pub fn apply_warp_affine<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    M: PyReadonlyArrayDyn<'py, f64>,
    out_w: usize,
    out_h: usize,
    flags: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = image.as_array();
    let m = M.as_array();
    let ndim = arr.ndim();
    
    let m_2d = m.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Affine matrix must be 2D (2x3)"))?;
        
    if m_2d.shape() != [2, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err("Affine matrix must be of shape 2x3"));
    }
    
    // Check if WARP_INVERSE_MAP is set (OpenCV value is 16)
    let is_inverse = (flags & 16) != 0;
    
    let inv_m = if is_inverse {
        [[m_2d[[0, 0]], m_2d[[0, 1]], m_2d[[0, 2]]],
         [m_2d[[1, 0]], m_2d[[1, 1]], m_2d[[1, 2]]]]
    } else {
        let a = m_2d[[0, 0]];
        let b = m_2d[[0, 1]];
        let tx = m_2d[[0, 2]];
        let c = m_2d[[1, 0]];
        let d = m_2d[[1, 1]];
        let ty = m_2d[[1, 2]];
        
        let det = a * d - b * c;
        if det.abs() < 1e-9 {
            return Err(pyo3::exceptions::PyValueError::new_err("Affine matrix is singular and cannot be inverted"));
        }
        
        [[d / det, -b / det, (b * ty - d * tx) / det],
         [-c / det, a / det, (c * tx - a * ty) / det]]
    };
    
    if ndim == 3 {
        let img_3d = arr.into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 3D"))?;
        let (h, w, c) = (img_3d.shape()[0], img_3d.shape()[1], img_3d.shape()[2]);
        let mut out = numpy::ndarray::Array3::<u8>::zeros((out_h, out_w, c));
        
        for y in 0..out_h {
            for x in 0..out_w {
                let x_f = x as f64;
                let y_f = y as f64;
                
                let src_x = inv_m[0][0] * x_f + inv_m[0][1] * y_f + inv_m[0][2];
                let src_y = inv_m[1][0] * x_f + inv_m[1][1] * y_f + inv_m[1][2];
                
                let px = src_x.round() as isize;
                let py = src_y.round() as isize;
                
                if px >= 0 && px < w as isize && py >= 0 && py < h as isize {
                    for ch in 0..c {
                        out[[y, x, ch]] = img_3d[[py as usize, px as usize, ch]];
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
                
                let src_x = inv_m[0][0] * x_f + inv_m[0][1] * y_f + inv_m[0][2];
                let src_y = inv_m[1][0] * x_f + inv_m[1][1] * y_f + inv_m[1][2];
                
                let px = src_x.round() as isize;
                let py = src_y.round() as isize;
                
                if px >= 0 && px < w as isize && py >= 0 && py < h as isize {
                    out[[y, x]] = img_2d[[py as usize, px as usize]];
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

/// Calculate a 2D rotation matrix center by angle scale.
///
/// Returns a 2x3 affine matrix.
#[pyfunction]
pub fn get_rotation_matrix_2d<'py>(
    py: Python<'py>,
    center: (f64, f64),
    angle: f64,
    scale: f64,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let (cx, cy) = center;
    let theta = angle.to_radians();
    let alpha = scale * theta.cos();
    let beta = scale * theta.sin();
    
    let mut m = numpy::ndarray::Array2::<f64>::zeros((2, 3));
    m[[0, 0]] = alpha;
    m[[0, 1]] = beta;
    m[[0, 2]] = (1.0 - alpha) * cx - beta * cy;
    m[[1, 0]] = -beta;
    m[[1, 1]] = alpha;
    m[[1, 2]] = beta * cx + (1.0 - alpha) * cy;
    
    Ok(m.into_pyarray(py).to_dyn())
}

/// Calculate an affine transform from three pairs of corresponding points.
///
/// Returns a 2x3 affine matrix.
#[pyfunction]
pub fn get_affine_transform<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    dst: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let s = src.as_array();
    let d_arr = dst.as_array();
    
    let s_2d = s.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("src points must be a 2D array of shape 3x2"))?;
    let d_2d = d_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("dst points must be a 2D array of shape 3x2"))?;
        
    if s_2d.shape() != [3, 2] || d_2d.shape() != [3, 2] {
        return Err(pyo3::exceptions::PyValueError::new_err("Points must be of shape 3x2"));
    }
    
    let x0 = s_2d[[0, 0]] as f64;
    let y0 = s_2d[[0, 1]] as f64;
    let x1 = s_2d[[1, 0]] as f64;
    let y1 = s_2d[[1, 1]] as f64;
    let x2 = s_2d[[2, 0]] as f64;
    let y2 = s_2d[[2, 1]] as f64;
    
    let u0 = d_2d[[0, 0]] as f64;
    let v0 = d_2d[[0, 1]] as f64;
    let u1 = d_2d[[1, 0]] as f64;
    let v1 = d_2d[[1, 1]] as f64;
    let u2 = d_2d[[2, 0]] as f64;
    let v2 = d_2d[[2, 1]] as f64;
    
    let det = x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1);
    if det.abs() < 1e-9 {
        return Err(pyo3::exceptions::PyValueError::new_err("src points are collinear; cannot compute affine transform"));
    }
    
    let inv_det = 1.0 / det;
    
    let a00 = (y1 - y2) * inv_det;
    let a01 = (y2 - y0) * inv_det;
    let a02 = (y0 - y1) * inv_det;
    
    let a10 = (x2 - x1) * inv_det;
    let a11 = (x0 - x2) * inv_det;
    let a12 = (x1 - x0) * inv_det;
    
    let a20 = (x1 * y2 - x2 * y1) * inv_det;
    let a21 = (x2 * y0 - x0 * y2) * inv_det;
    let a22 = (x0 * y1 - x1 * y0) * inv_det;
    
    let a = a00 * u0 + a01 * u1 + a02 * u2;
    let b = a10 * u0 + a11 * u1 + a12 * u2;
    let tx = a20 * u0 + a21 * u1 + a22 * u2;
    
    let c = a00 * v0 + a01 * v1 + a02 * v2;
    let d = a10 * v0 + a11 * v1 + a12 * v2;
    let ty = a20 * v0 + a21 * v1 + a22 * v2;
    
    let mut m = numpy::ndarray::Array2::<f64>::zeros((2, 3));
    m[[0, 0]] = a;
    m[[0, 1]] = b;
    m[[0, 2]] = tx;
    m[[1, 0]] = c;
    m[[1, 1]] = d;
    m[[1, 2]] = ty;
    
    Ok(m.into_pyarray(py).to_dyn())
}

// 8x8 Linear system solver using Gaussian Elimination with partial pivoting.
fn solve_linear_system(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Option<[f64; 8]> {
    let n = 8;
    for i in 0..n {
        let mut max_row = i;
        for r in (i + 1)..n {
            if a[r][i].abs() > a[max_row][i].abs() {
                max_row = r;
            }
        }
        
        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }
        
        if a[i][i].abs() < 1e-9 {
            return None;
        }
        
        for r in (i + 1)..n {
            let factor = a[r][i] / a[i][i];
            for c in i..n {
                a[r][c] -= factor * a[i][c];
            }
            b[r] -= factor * b[i];
        }
    }
    
    let mut x = [0.0; 8];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for c in (i + 1)..n {
            sum -= a[i][c] * x[c];
        }
        x[i] = sum / a[i][i];
    }
    
    Some(x)
}

/// Calculate a perspective transform from four pairs of corresponding points.
///
/// Returns a 3x3 perspective matrix.
#[pyfunction]
pub fn get_perspective_transform<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    dst: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let s = src.as_array();
    let d_arr = dst.as_array();
    
    let s_2d = s.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("src points must be a 2D array of shape 4x2"))?;
    let d_2d = d_arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("dst points must be a 2D array of shape 4x2"))?;
        
    if s_2d.shape() != [4, 2] || d_2d.shape() != [4, 2] {
        return Err(pyo3::exceptions::PyValueError::new_err("Points must be of shape 4x2"));
    }
    
    let mut a = [[0.0f64; 8]; 8];
    let mut b = [0.0f64; 8];
    
    for i in 0..4 {
        let sx = s_2d[[i, 0]] as f64;
        let sy = s_2d[[i, 1]] as f64;
        let dx = d_2d[[i, 0]] as f64;
        let dy = d_2d[[i, 1]] as f64;
        
        let r1 = i * 2;
        let r2 = i * 2 + 1;
        
        a[r1][0] = sx;
        a[r1][1] = sy;
        a[r1][2] = 1.0;
        a[r1][3] = 0.0;
        a[r1][4] = 0.0;
        a[r1][5] = 0.0;
        a[r1][6] = -sx * dx;
        a[r1][7] = -sy * dx;
        b[r1] = dx;
        
        a[r2][0] = 0.0;
        a[r2][1] = 0.0;
        a[r2][2] = 0.0;
        a[r2][3] = sx;
        a[r2][4] = sy;
        a[r2][5] = 1.0;
        a[r2][6] = -sx * dy;
        a[r2][7] = -sy * dy;
        b[r2] = dy;
    }
    
    let sol = solve_linear_system(a, b)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Four points are collinear or coplanar; cannot compute perspective transform"))?;
        
    let mut m = numpy::ndarray::Array2::<f64>::zeros((3, 3));
    m[[0, 0]] = sol[0];
    m[[0, 1]] = sol[1];
    m[[0, 2]] = sol[2];
    m[[1, 0]] = sol[3];
    m[[1, 1]] = sol[4];
    m[[1, 2]] = sol[5];
    m[[2, 0]] = sol[6];
    m[[2, 1]] = sol[7];
    m[[2, 2]] = 1.0;
    
    Ok(m.into_pyarray(py).to_dyn())
}

/// Flip an image horizontally, vertically, or both.
///
/// `flip_code`: 0 for vertical flip (x-axis), >0 for horizontal flip (y-axis), <0 for both.
#[pyfunction]
pub fn apply_flip<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    flip_code: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = image.as_array();
    let ndim = arr.ndim();
    
    if ndim == 3 {
        let img_3d = arr.into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 3D"))?;
        let (h, w, c) = (img_3d.shape()[0], img_3d.shape()[1], img_3d.shape()[2]);
        let mut out = numpy::ndarray::Array3::<u8>::zeros((h, w, c));
        
        for y in 0..h {
            for x in 0..w {
                let target_y = if flip_code == 0 || flip_code < 0 { h - 1 - y } else { y };
                let target_x = if flip_code > 0 || flip_code < 0 { w - 1 - x } else { x };
                for ch in 0..c {
                    out[[target_y, target_x, ch]] = img_3d[[y, x, ch]];
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else if ndim == 2 {
        let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 2D"))?;
        let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
        let mut out = numpy::ndarray::Array2::<u8>::zeros((h, w));
        
        for y in 0..h {
            for x in 0..w {
                let target_y = if flip_code == 0 || flip_code < 0 { h - 1 - y } else { y };
                let target_x = if flip_code > 0 || flip_code < 0 { w - 1 - x } else { x };
                out[[target_y, target_x]] = img_2d[[y, x]];
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

/// Transpose an image (swap rows and columns).
#[pyfunction]
pub fn apply_transpose<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = image.as_array();
    let ndim = arr.ndim();
    
    if ndim == 3 {
        let img_3d = arr.into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 3D"))?;
        let (h, w, c) = (img_3d.shape()[0], img_3d.shape()[1], img_3d.shape()[2]);
        let mut out = numpy::ndarray::Array3::<u8>::zeros((w, h, c));
        
        for y in 0..h {
            for x in 0..w {
                for ch in 0..c {
                    out[[x, y, ch]] = img_3d[[y, x, ch]];
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else if ndim == 2 {
        let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 2D"))?;
        let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
        let mut out = numpy::ndarray::Array2::<u8>::zeros((w, h));
        
        for y in 0..h {
            for x in 0..w {
                out[[x, y]] = img_2d[[y, x]];
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

/// Apply a generic geometrical transformation to an image.
///
/// - `map1`: 2D array of source x coordinates (f32).
/// - `map2`: 2D array of source y coordinates (f32).
/// - `interpolation`: interpolation method (0 = nearest, 1 = bilinear, 2 = bicubic, 4 = Lanczos4).
#[pyfunction]
#[pyo3(signature = (image, map1, map2, interpolation = 1))]
pub fn apply_remap<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    map1: PyReadonlyArrayDyn<'py, f32>,
    map2: PyReadonlyArrayDyn<'py, f32>,
    interpolation: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = image.as_array();
    let m1 = map1.as_array();
    let m2 = map2.as_array();
    let ndim = arr.ndim();
    
    let m1_2d = m1.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("map1 must be 2D"))?;
    let m2_2d = m2.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("map2 must be 2D"))?;
        
    let out_h = m1_2d.shape()[0];
    let out_w = m1_2d.shape()[1];
    
    if m2_2d.shape() != [out_h, out_w] {
        return Err(pyo3::exceptions::PyValueError::new_err("map1 and map2 must have the same shape"));
    }
    
    if ndim == 3 {
        let img_3d = arr.into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 3D"))?;
        let (h, w, c) = (img_3d.shape()[0], img_3d.shape()[1], img_3d.shape()[2]);
        let mut out = numpy::ndarray::Array3::<u8>::zeros((out_h, out_w, c));
        
        for y in 0..out_h {
            for x in 0..out_w {
                let src_x = m1_2d[[y, x]] as f64;
                let src_y = m2_2d[[y, x]] as f64;
                
                if src_x >= 0.0 && src_x < w as f64 && src_y >= 0.0 && src_y < h as f64 {
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
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else if ndim == 2 {
        let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 2D"))?;
        let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
        let mut out = numpy::ndarray::Array2::<u8>::zeros((out_h, out_w));
        
        for y in 0..out_h {
            for x in 0..out_w {
                let src_x = m1_2d[[y, x]] as f64;
                let src_y = m2_2d[[y, x]] as f64;
                
                if src_x >= 0.0 && src_x < w as f64 && src_y >= 0.0 && src_y < h as f64 {
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
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

/// Invert an affine transform matrix (2x3).
#[pyfunction]
#[allow(non_snake_case)]
pub fn invert_affine_transform<'py>(
    py: Python<'py>,
    M: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let m = M.as_array();
    let m_2d = m.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Affine matrix must be 2D (2x3)"))?;
        
    if m_2d.shape() != [2, 3] {
        return Err(pyo3::exceptions::PyValueError::new_err("Affine matrix must be of shape 2x3"));
    }
    
    let a = m_2d[[0, 0]];
    let b = m_2d[[0, 1]];
    let tx = m_2d[[0, 2]];
    let c = m_2d[[1, 0]];
    let d = m_2d[[1, 1]];
    let ty = m_2d[[1, 2]];
    
    let det = a * d - b * c;
    if det.abs() < 1e-9 {
        return Err(pyo3::exceptions::PyValueError::new_err("Affine matrix is singular and cannot be inverted"));
    }
    
    let mut out = numpy::ndarray::Array2::<f64>::zeros((2, 3));
    out[[0, 0]] = d / det;
    out[[0, 1]] = -b / det;
    out[[0, 2]] = (b * ty - d * tx) / det;
    out[[1, 0]] = -c / det;
    out[[1, 1]] = a / det;
    out[[1, 2]] = (c * tx - a * ty) / det;
    
    Ok(out.into_pyarray(py).to_dyn())
}

/// Remap an image to linear polar coordinates.
///
/// - `center`: coordinates of the transformation center (cx, cy).
/// - `max_radius`: the bounding circle radius.
/// - `out_w`: optional output width.
/// - `out_h`: optional output height.
/// - `interpolation`: interpolation method (0 = nearest, 1 = bilinear, 2 = bicubic, 4 = Lanczos4).
#[pyfunction]
#[pyo3(signature = (image, center, max_radius, out_w = None, out_h = None, interpolation = 1))]
pub fn apply_linear_polar<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    center: (f64, f64),
    max_radius: f64,
    out_w: Option<usize>,
    out_h: Option<usize>,
    interpolation: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = image.as_array();
    let ndim = arr.ndim();
    
    let (h, w) = (arr.shape()[0], arr.shape()[1]);
    let dest_w = out_w.unwrap_or(w);
    let dest_h = out_h.unwrap_or(h);
    
    let (cx, cy) = center;
    
    if ndim == 3 {
        let img_3d = arr.into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 3D"))?;
        let c = img_3d.shape()[2];
        let mut out = numpy::ndarray::Array3::<u8>::zeros((dest_h, dest_w, c));
        
        for y in 0..dest_h {
            let r = (y as f64) * max_radius / (dest_h as f64);
            for x in 0..dest_w {
                let theta = (x as f64) * 2.0 * std::f64::consts::PI / (dest_w as f64);
                let src_x = cx + r * theta.cos();
                let src_y = cy + r * theta.sin();
                
                if src_x >= 0.0 && src_x < w as f64 && src_y >= 0.0 && src_y < h as f64 {
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
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else if ndim == 2 {
        let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 2D"))?;
        let mut out = numpy::ndarray::Array2::<u8>::zeros((dest_h, dest_w));
        
        for y in 0..dest_h {
            let r = (y as f64) * max_radius / (dest_h as f64);
            for x in 0..dest_w {
                let theta = (x as f64) * 2.0 * std::f64::consts::PI / (dest_w as f64);
                let src_x = cx + r * theta.cos();
                let src_y = cy + r * theta.sin();
                
                if src_x >= 0.0 && src_x < w as f64 && src_y >= 0.0 && src_y < h as f64 {
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
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

/// Remap an image to log-polar coordinates.
///
/// - `center`: coordinates of the transformation center (cx, cy).
/// - `scale`: magnitude scale factor.
/// - `out_w`: optional output width.
/// - `out_h`: optional output height.
/// - `interpolation`: interpolation method (0 = nearest, 1 = bilinear, 2 = bicubic, 4 = Lanczos4).
#[pyfunction]
#[pyo3(signature = (image, center, scale, out_w = None, out_h = None, interpolation = 1))]
pub fn apply_log_polar<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    center: (f64, f64),
    scale: f64,
    out_w: Option<usize>,
    out_h: Option<usize>,
    interpolation: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = image.as_array();
    let ndim = arr.ndim();
    
    let (h, w) = (arr.shape()[0], arr.shape()[1]);
    let dest_w = out_w.unwrap_or(w);
    let dest_h = out_h.unwrap_or(h);
    
    let (cx, cy) = center;
    
    if ndim == 3 {
        let img_3d = arr.into_dimensionality::<numpy::ndarray::Ix3>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 3D"))?;
        let c = img_3d.shape()[2];
        let mut out = numpy::ndarray::Array3::<u8>::zeros((dest_h, dest_w, c));
        
        for y in 0..dest_h {
            let r = ((y as f64) / scale).exp();
            for x in 0..dest_w {
                let theta = (x as f64) * 2.0 * std::f64::consts::PI / (dest_w as f64);
                let src_x = cx + r * theta.cos();
                let src_y = cy + r * theta.sin();
                
                if src_x >= 0.0 && src_x < w as f64 && src_y >= 0.0 && src_y < h as f64 {
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
          }
          Ok(out.into_pyarray(py).to_dyn())
      } else if ndim == 2 {
          let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
              .map_err(|_| pyo3::exceptions::PyValueError::new_err("Failed to cast to 2D"))?;
          let mut out = numpy::ndarray::Array2::<u8>::zeros((dest_h, dest_w));
          
          for y in 0..dest_h {
              let r = ((y as f64) / scale).exp();
              for x in 0..dest_w {
                  let theta = (x as f64) * 2.0 * std::f64::consts::PI / (dest_w as f64);
                  let src_x = cx + r * theta.cos();
                  let src_y = cy + r * theta.sin();
                  
                  if src_x >= 0.0 && src_x < w as f64 && src_y >= 0.0 && src_y < h as f64 {
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
          }
          Ok(out.into_pyarray(py).to_dyn())
      } else {
          Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
      }
}

/// Alias for apply_resize matching OpenCV name.
#[pyfunction(name = "resize")]
#[pyo3(signature = (image, new_w, new_h, interpolation = 1))]
pub fn resize<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    new_w: usize,
    new_h: usize,
    interpolation: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    apply_resize(py, image, new_w, new_h, interpolation)
}

/// Alias for apply_warp_affine matching OpenCV name.
#[pyfunction(name = "warpAffine")]
#[pyo3(signature = (image, M, out_w, out_h, flags = 0))]
pub fn warp_affine<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    M: PyReadonlyArrayDyn<'py, f64>,
    out_w: usize,
    out_h: usize,
    flags: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    apply_warp_affine(py, image, M, out_w, out_h, flags)
}

/// Alias for get_rotation_matrix_2d matching OpenCV name.
#[pyfunction(name = "getRotationMatrix2D")]
pub fn get_rotation_matrix_2d_cv<'py>(
    py: Python<'py>,
    center: (f64, f64),
    angle: f64,
    scale: f64,
) -> PyResult<&'py PyArrayDyn<f64>> {
    get_rotation_matrix_2d(py, center, angle, scale)
}

/// Alias for get_affine_transform matching OpenCV name.
#[pyfunction(name = "getAffineTransform")]
pub fn get_affine_transform_cv<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    dst: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<&'py PyArrayDyn<f64>> {
    get_affine_transform(py, src, dst)
}

/// Alias for get_perspective_transform matching OpenCV name.
#[pyfunction(name = "getPerspectiveTransform")]
pub fn get_perspective_transform_cv<'py>(
    py: Python<'py>,
    src: PyReadonlyArrayDyn<'py, f32>,
    dst: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<&'py PyArrayDyn<f64>> {
    get_perspective_transform(py, src, dst)
}

/// Alias for apply_remap matching OpenCV name.
#[pyfunction(name = "remap")]
#[pyo3(signature = (image, map_x, map_y, interpolation = 1))]
pub fn remap<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    map_x: PyReadonlyArrayDyn<'py, f32>,
    map_y: PyReadonlyArrayDyn<'py, f32>,
    interpolation: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    apply_remap(py, image, map_x, map_y, interpolation)
}

/// Alias for apply_flip matching OpenCV name.
#[pyfunction(name = "flip")]
pub fn flip<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    flip_code: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    apply_flip(py, image, flip_code)
}

/// Alias for apply_transpose matching OpenCV name.
#[pyfunction(name = "transpose")]
pub fn transpose<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<&'py PyArrayDyn<u8>> {
    apply_transpose(py, image)
}

/// Alias for invert_affine_transform matching OpenCV name.
#[pyfunction(name = "invertAffineTransform")]
#[allow(non_snake_case)]
pub fn invert_affine_transform_cv<'py>(
    py: Python<'py>,
    M: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<&'py PyArrayDyn<f64>> {
    invert_affine_transform(py, M)
}

/// Alias for apply_linear_polar matching OpenCV name.
#[pyfunction(name = "linearPolar")]
#[pyo3(signature = (image, center, max_radius, out_w = None, out_h = None, interpolation = 1))]
pub fn linear_polar<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    center: (f64, f64),
    max_radius: f64,
    out_w: Option<usize>,
    out_h: Option<usize>,
    interpolation: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    apply_linear_polar(py, image, center, max_radius, out_w, out_h, interpolation)
}

/// Alias for apply_log_polar matching OpenCV name.
#[pyfunction(name = "logPolar")]
#[pyo3(signature = (image, center, scale = 1.0, out_w = None, out_h = None, interpolation = 1))]
pub fn log_polar<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    center: (f64, f64),
    scale: f64,
    out_w: Option<usize>,
    out_h: Option<usize>,
    interpolation: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    apply_log_polar(py, image, center, scale, out_w, out_h, interpolation)
}