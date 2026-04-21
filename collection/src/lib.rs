use pyo3::prelude::*;
use numpy::{
    ndarray::{s, Array2, Array3, ArrayView1, ArrayView2},
    IntoPyArray, PyArray3, PyArrayDyn, PyArrayMethods, PyReadonlyArray3, PyReadonlyArrayDyn,
};
use num_complex::Complex64;
use std::f64::consts::PI;

// ==========================================
// INTERNAL HELPER FUNCTIONS (Not exported)
// ==========================================

fn get_equalized_map(channel_data: ArrayView2<u8>) -> Vec<u8> {
    let mut hist = [0u32; 256];
    let mut count = 0;

    for &pixel in channel_data {
        hist[pixel as usize] += 1;
        count += 1;
    }

    let mut cdf = [0u32; 256];
    let mut sum = 0;
    for i in 0..256 {
        sum += hist[i];
        cdf[i] = sum;
    }

    let mut lut = vec![0u8; 256];
    let min_cdf = *cdf.iter().find(|&&x| x > 0).unwrap_or(&0);
    
    for i in 0..256 {
        let num = (cdf[i] - min_cdf) as f32;
        let den = (count as u32 - min_cdf) as f32;
        
        if den > 0.0 {
            lut[i] = ((num / den) * 255.0).round() as u8;
        } else {
            lut[i] = i as u8; 
        }
    }
    lut
}

fn get_cdf(channel_data: ArrayView2<u8>) -> Vec<f32> {
    let mut hist = [0u32; 256];
    let mut count = 0;

    for &pixel in channel_data{
        hist[pixel as usize] += 1;
        count += 1;
    }

    let mut cdf = vec![0.0; 256];
    let mut sum = 0;
    for i in 0..256 {
        sum += hist[i];
        cdf[i] = sum as f32 / count as f32;
    }
    cdf
}

fn match_histograms(src_cdf: &[f32], tgt_cdf: &[f32]) -> Vec<u8> {
    let mut lut = vec![0u8; 256];
    let mut j: usize = 0;

    for i in 0..256 {
        let src_val = src_cdf[i];
        while j < 256 && tgt_cdf[j] < src_val {
            j += 1;
        }
        lut[i] = if j < 256 { j as u8 } else { 255 };
    }
    lut
}

fn get_target_cdf(target_pdf: ArrayView1<f32>) -> PyResult<Vec<f32>> {
    let mut cleaned = vec![0.0f32; 256];
    for i in 0..256 {
        let v = target_pdf[i];
        cleaned[i] = if v.is_finite() && v > 0.0 { v } else { 0.0 };
    }

    let total: f32 = cleaned.iter().sum();
    if total <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "target_hist must contain positive finite values",
        ));
    }

    let mut cdf = vec![0.0; 256];
    let mut sum = 0.0;
    for i in 0..256 {
        sum += cleaned[i] / total;
        cdf[i] = sum;
    }
    cdf[255] = 1.0;
    Ok(cdf)
}

fn apply_median_3x3(channel: ArrayView2<u8>) -> Array2<u8> {
    let (h, w) = (channel.shape()[0], channel.shape()[1]);
    let mut out = Array2::<u8>::zeros((h, w));
    
    for y in 1..h-1 {
        for x in 1..w-1 {
            let mut window = [
                channel[[y-1, x-1]], channel[[y-1, x]], channel[[y-1, x+1]],
                channel[[y, x-1]],   channel[[y, x]],   channel[[y, x+1]],
                channel[[y+1, x-1]], channel[[y+1, x]], channel[[y+1, x+1]],
            ];
            window.sort_unstable();
            out[[y, x]] = window[4]; 
        }
    }
    out
}

fn apply_laplacian_3x3(channel: ArrayView2<u8>) -> Array2<u8> {
    let (h, w) = (channel.shape()[0], channel.shape()[1]);
    let mut out = Array2::<u8>::zeros((h, w));
    
    for y in 1..h-1 {
        for x in 1..w-1 {
            let sum =
                -1 * channel[[y-1, x-1]] as i32 - 1 * channel[[y-1, x]] as i32 - 1 * channel[[y-1, x+1]] as i32
                -1 * channel[[y, x-1]] as i32   + 8 * channel[[y, x]] as i32   - 1 * channel[[y, x+1]] as i32
                -1 * channel[[y+1, x-1]] as i32 - 1 * channel[[y+1, x]] as i32 - 1 * channel[[y+1, x+1]] as i32;
            
            out[[y, x]] = sum.clamp(0, 255) as u8;
        }
    }
    out
}
fn calculate_otsu_threshold(channel_data: ArrayView2<u8>) -> u8 {
    let mut hist = [0usize; 256];
    let mut total_pixels = 0;

    // Calculate Histogram
    for &pixel in channel_data.iter() {
        hist[pixel as usize] += 1;
        total_pixels += 1;
    }

    // Calculate total sum for means
    let mut sum = 0.0;
    for i in 0..256 {
        sum += (i as f64) * (hist[i] as f64);
    }

    let mut sum_b = 0.0;
    let mut w_b = 0;
    let mut var_max = 0.0;
    let mut threshold = 0;

    // Iterate through all possible thresholds to maximize between-class variance
    for t in 0..256 {
        w_b += hist[t];
        if w_b == 0 { continue; }
        
        let w_f = total_pixels - w_b;
        if w_f == 0 { break; }

        sum_b += (t as f64) * (hist[t] as f64);

        let m_b = sum_b / (w_b as f64);
        let m_f = (sum - sum_b) / (w_f as f64);

        let var_between = (w_b as f64) * (w_f as f64) * (m_b - m_f).powi(2);

        if var_between > var_max {
            var_max = var_between;
            threshold = t as u8;
        }
    }
    
    threshold
}


#[pyfunction]
fn apply_negative<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let x_array = x.as_array();
    let result = x_array.mapv(|pixel| 255 - pixel);
    Ok(result.into_pyarray(py))
}

#[pyfunction]
fn apply_log<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let x_array = x.as_array();
    let c = 255.0 / (256.0 as f64).ln();
    let result = x_array.mapv(|pixel| {
        let val = c * (pixel as f64 + 1.0).ln();
        val.min(255.0) as u8
    });
    Ok(result.into_pyarray(py))
}

#[pyfunction]
fn apply_gamma<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>, gamma: f64) -> PyResult<&'py PyArrayDyn<u8>> {
    let x_array = x.as_array();
    let result = x_array.mapv(|pixel| {
        let r = pixel as f64 / 255.0;
        let s = r.powf(gamma);
        (s * 255.0).min(255.0) as u8
    });
    Ok(result.into_pyarray(py))
}

#[pyfunction]
fn rgb_to_gray<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let r = arr.slice(s![.., .., 0]).mapv(|v| v as f32);
    let g = arr.slice(s![.., .., 1]).mapv(|v| v as f32);
    let b = arr.slice(s![.., .., 2]).mapv(|v| v as f32);
    let gray = (0.299 * &r + 0.587 * &g + 0.114 * &b).mapv(|v| v as u8);
    Ok(gray.into_pyarray(py).to_dyn())
}

#[pyfunction]
fn apply_threshold<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<u8>, threshold_value: u8) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let result = arr.mapv(|pixel| {
        if pixel > threshold_value { 255 } else { 0 }
    });
    Ok(result.into_pyarray(py))
}


#[pyfunction]
fn hist_equalize_rgb<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let shape = arr.shape();
    let h = shape[0];
    let w = shape[1];
    let mut out_arr = Array3::<u8>::zeros((h, w, 3));

    for c in 0..3 {
        let channel_data = arr.slice(s![.., .., c]);
        let lut = get_equalized_map(channel_data);
        let mut out_channel = out_arr.slice_mut(s![.., .., c]);
        for ((i, j), &pixel_val) in channel_data.indexed_iter() {
            out_channel[[i, j]] = lut[pixel_val as usize];
        }
    }
    Ok(out_arr.into_pyarray(py).to_dyn())
}

#[pyfunction]
fn hist_equalize_gray<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let r = arr.slice(s![.., .., 0]);
    let g = arr.slice(s![.., .., 1]);
    let b = arr.slice(s![.., .., 2]);
    let gray_arr = (&r.mapv(|v| v as u16) + &g.mapv(|v| v as u16) + &b.mapv(|v| v as u16)) / 3;
    let gray_arr = gray_arr.mapv(|v| v as u8);
    let lut = get_equalized_map(gray_arr.view());
    let result = gray_arr.mapv(|pixel| lut[pixel as usize]);
    Ok(result.into_pyarray(py).to_dyn())
}


#[pyfunction]
fn hist_spec_rgb<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>, target_hist: PyReadonlyArrayDyn<'py, f32>) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = x.as_array();
    let tgt = target_hist.as_array();
    let tgt_1d = tgt.view().into_dimensionality().map_err(|_| pyo3::exceptions::PyValueError::new_err("target_hist must be 1D with 256 bins"))?;
    if tgt_1d.len() != 256 {
        return Err(pyo3::exceptions::PyValueError::new_err("target_hist length must be 256"));
    }
    let tgt_cdf = get_target_cdf(tgt_1d)?;
    let shape = arr.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("x must have shape (H, W, 3)"));
    }

    let mut out_arr = Array3::<u8>::zeros((shape[0], shape[1], 3));
    for c in 0..3 {
        let channel_data = arr.slice(s![.., .., c]);
        let src_cdf = get_cdf(channel_data);
        let lut = match_histograms(&src_cdf, &tgt_cdf);
        let mut out_channel = out_arr.slice_mut(s![.., .., c]);
        for ((i, j), &pixel_val) in channel_data.indexed_iter() {
            out_channel[[i, j]] = lut[pixel_val as usize];
        }
    }
    Ok(out_arr.into_pyarray(py).to_dyn().to_owned().into())
}

#[pyfunction]
fn hist_spec_gray<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>, target_hist: PyReadonlyArrayDyn<'py, f32>) -> PyResult<Py<PyArrayDyn<u8>>> {
    let arr = x.as_array();
    let tgt = target_hist.as_array();
    let tgt_1d = tgt.view().into_dimensionality().map_err(|_| pyo3::exceptions::PyValueError::new_err("target_hist must be 1D with 256 bins"))?;
    if tgt_1d.len() != 256 {
        return Err(pyo3::exceptions::PyValueError::new_err("target_hist length must be 256"));
    }
    let tgt_cdf = get_target_cdf(tgt_1d)?;
    let shape = arr.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("x must have shape (H, W, 3)"));
    }

    let r = arr.slice(s![.., .., 0]);
    let g = arr.slice(s![.., .., 1]);
    let b = arr.slice(s![.., .., 2]);
    let gray_arr = (&r.mapv(|v| 77u16 * v as u16) + &g.mapv(|v| 150u16 * v as u16) + &b.mapv(|v| 29u16 * v as u16)) / 256;
    let gray_u8 = gray_arr.mapv(|v| v as u8);

    let src_cdf = get_cdf(gray_u8.view().into_dimensionality().unwrap());
    let lut = match_histograms(&src_cdf, &tgt_cdf);
    let result = gray_u8.mapv(|pixel| lut[pixel as usize]);
    Ok(result.into_pyarray(py).to_dyn().to_owned().into())
}

fn erode_2d(image: ArrayView2<u8>, kernel: ArrayView2<u8>) -> Array2<u8> {
    let (h, w) = (image.shape()[0], image.shape()[1]);
    let (kh, kw) = (kernel.shape()[0], kernel.shape()[1]);
    let pad_h = kh / 2;
    let pad_w = kw / 2;

    let mut out = Array2::<u8>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let mut min_val = 255u8;
            for ky in 0..kh {
                for kx in 0..kw {
                    // Only process active elements in the Structuring Element
                    if kernel[[ky, kx]] > 0 {
                        let iy = y as isize + ky as isize - pad_h as isize;
                        let ix = x as isize + kx as isize - pad_w as isize;

                        // Pad out-of-bounds with 255 (white) so edges don't artificially erode
                        let pixel = if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                            image[[iy as usize, ix as usize]]
                        } else {
                            255 
                        };

                        if pixel < min_val {
                            min_val = pixel;
                        }
                    }
                }
            }
            out[[y, x]] = min_val;
        }
    }
    out
}

fn dilate_2d(image: ArrayView2<u8>, kernel: ArrayView2<u8>) -> Array2<u8> {
    let (h, w) = (image.shape()[0], image.shape()[1]);
    let (kh, kw) = (kernel.shape()[0], kernel.shape()[1]);
    let pad_h = kh / 2;
    let pad_w = kw / 2;

    let mut out = Array2::<u8>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let mut max_val = 0u8;
            for ky in 0..kh {
                for kx in 0..kw {
                    if kernel[[ky, kx]] > 0 {
                        let iy = y as isize + ky as isize - pad_h as isize;
                        let ix = x as isize + kx as isize - pad_w as isize;

                        let pixel = if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                            image[[iy as usize, ix as usize]]
                        } else {
                            0 
                        };

                        if pixel > max_val {
                            max_val = pixel;
                        }
                    }
                }
            }
            out[[y, x]] = max_val;
        }
    }
    out
}

// --- Spatial Filters ---

#[pyfunction]
fn median_filter<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let ndim = arr.ndim();
    if ndim == 2 {
        let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        let filtered = apply_median_3x3(channel.view());
        return Ok(filtered.into_pyarray(py).to_dyn());
    } else if ndim == 3 {
        let (h, w, c) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
        let mut out_arr = Array3::<u8>::zeros((h, w, c));
        for ch in 0..c {
            let channel = arr.slice(s![.., .., ch]);
            out_arr.slice_mut(s![.., .., ch]).assign(&apply_median_3x3(channel));
        }
        return Ok(out_arr.into_pyarray(py).to_dyn());
    }
    panic!("Unsupported image dimensions!");
}

#[pyfunction]
fn laplacian_filter<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = x.as_array();
    let ndim = arr.ndim();
    if ndim == 2 {
        let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        let filtered = apply_laplacian_3x3(channel.view());
        return Ok(filtered.into_pyarray(py).to_dyn());
    } else if ndim == 3 {
        let (h, w, c) = (arr.shape()[0], arr.shape()[1], arr.shape()[2]);
        let mut out_arr = Array3::<u8>::zeros((h, w, c));
        for ch in 0..c {
            let channel = arr.slice(s![.., .., ch]);
            out_arr.slice_mut(s![.., .., ch]).assign(&apply_laplacian_3x3(channel));
        }
        return Ok(out_arr.into_pyarray(py).to_dyn());
    }
    panic!("Unsupported image dimensions!");
}

// --- Frequency Filters ---

#[pyfunction]
fn apply_frequency_filter<'py>(py: Python<'py>, f_shifted: PyReadonlyArray3<'py, Complex64>, d0: f64, filter_type: &str) -> &'py PyArray3<Complex64> {
    let f_arr = f_shifted.as_array();
    let (rows, cols, channels) = (f_arr.shape()[0], f_arr.shape()[1], f_arr.shape()[2]);
    let mut output = Array3::<Complex64>::zeros((rows, cols, channels));
    let center_u = (rows as f64) / 2.0;
    let center_v = (cols as f64) / 2.0;

    for u in 0..rows {
        for v in 0..cols {
            let du = (u as f64) - center_u;
            let dv = (v as f64) - center_v;
            let d = (du * du + dv * dv).sqrt();

            let mask_val = match filter_type {
                "ILPF" => if d <= d0 { 1.0 } else { 0.0 },
                "IHPF" => if d <= d0 { 0.0 } else { 1.0 },
                "GLPF" => (-(d * d) / (2.0 * d0 * d0)).exp(),
                "GHPF" => 1.0 - (-(d * d) / (2.0 * d0 * d0)).exp(),
                _ => 1.0, 
            };

            let complex_mask = Complex64::new(mask_val, 0.0);
            for c in 0..channels {
                output[[u, v, c]] = f_arr[[u, v, c]] * complex_mask;
            }
        }
    }
    output.into_pyarray(py)
}

// --- Edge Detection ---
#[pyfunction]
fn rgb_to_cmy<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<f32>> { // <-- Notice the f32 here!
    let arr = x.as_array();
    
    // Ensure the input is a 3D RGB image
    let shape = arr.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Input must be an RGB image with shape (H, W, 3)"));
    }

    // Convert to CMY floats [0.0, 1.0]
    let result = arr.mapv(|pixel| 1.0 - (pixel as f32 / 255.0));

    Ok(result.into_pyarray(py).to_dyn())
}

#[pyfunction]
fn apply_otsu_threshold<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<(u8, &'py PyArrayDyn<u8>)> {
    let arr = x.as_array();
    let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("GrayScale Required"))?;

    let threshold_value = calculate_otsu_threshold(channel.view());
    let result = channel.mapv(|pixel| {
        if pixel > threshold_value { 255 } else { 0 }
    });

    // Return both the value and the array
    Ok((threshold_value, result.into_pyarray(py).to_dyn()))
}
#[pyfunction]
fn apply_canny<'py>(py: Python<'py>, image: PyReadonlyArray3<'py, f64>, low_thresh: f64, high_thresh: f64) -> &'py PyArray3<f64> {
    let img = image.as_array();
    let (rows, cols, channels) = (img.shape()[0], img.shape()[1], img.shape()[2]);
    let mut output = Array3::<f64>::zeros((rows, cols, channels));

    let gaussian = [
        [2.0, 4.0, 5.0, 4.0, 2.0],
        [4.0, 9.0, 12.0, 9.0, 4.0],
        [5.0, 12.0, 15.0, 12.0, 5.0],
        [4.0, 9.0, 12.0, 9.0, 4.0],
        [2.0, 4.0, 5.0, 4.0, 2.0],
    ];
    let gauss_sum: f64 = 159.0;
    let kx = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let ky = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    for c in 0..channels {
        let mut blurred = vec![vec![0.0; cols]; rows];
        for y in 2..rows - 2 {
            for x in 2..cols - 2 {
                let mut sum = 0.0;
                for dy in 0..5 {
                    for dx in 0..5 {
                        sum += img[[y + dy - 2, x + dx - 2, c]] * gaussian[dy][dx] / gauss_sum;
                    }
                }
                blurred[y][x] = sum;
            }
        }

        let mut mag = vec![vec![0.0; cols]; rows];
        let mut angle = vec![vec![0.0; cols]; rows];
        
        for y in 1..rows - 1 {
            for x in 1..cols - 1 {
                let mut gx = 0.0;
                let mut gy = 0.0;
                for dy in 0..3 {
                    for dx in 0..3 {
                        let val = blurred[y + dy - 1][x + dx - 1];
                        gx += val * kx[dy][dx];
                        gy += val * ky[dy][dx];
                    }
                }
                mag[y][x] = (gx * gx + gy * gy).sqrt();
                let mut theta = gy.atan2(gx) * (180.0 / PI);
                if theta < 0.0 { theta += 180.0; }
                angle[y][x] = theta;
            }
        }

        let mut suppressed = vec![vec![0.0; cols]; rows];
        for y in 1..rows - 1 {
            for x in 1..cols - 1 {
                let q: f64;
                let r: f64;
                let a = angle[y][x];

                if (0.0 <= a && a < 22.5) || (157.5 <= a && a <= 180.0) {
                    q = mag[y][x + 1]; r = mag[y][x - 1];
                } else if 22.5 <= a && a < 67.5 {
                    q = mag[y + 1][x - 1]; r = mag[y - 1][x + 1];
                } else if 67.5 <= a && a < 112.5 {
                    q = mag[y + 1][x]; r = mag[y - 1][x];
                } else {
                    q = mag[y - 1][x - 1]; r = mag[y + 1][x + 1];
                }

                if mag[y][x] >= q && mag[y][x] >= r {
                    suppressed[y][x] = mag[y][x];
                } else {
                    suppressed[y][x] = 0.0;
                }
            }
        }

        let mut strong_edges = vec![];
        let mut final_edges = vec![vec![0.0; cols]; rows];

        for y in 1..rows - 1 {
            for x in 1..cols - 1 {
                let p = suppressed[y][x];
                if p >= high_thresh {
                    final_edges[y][x] = 1.0; 
                    strong_edges.push((y, x));
                } else if p >= low_thresh {
                    final_edges[y][x] = 0.5; 
                }
            }
        }

        while let Some((y, x)) = strong_edges.pop() {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let ny = (y as isize + dy) as usize;
                    let nx = (x as isize + dx) as usize;
                    
                    if ny > 0 && ny < rows - 1 && nx > 0 && nx < cols - 1 {
                        if final_edges[ny][nx] == 0.5 { 
                            final_edges[ny][nx] = 1.0;  
                            strong_edges.push((ny, nx)); 
                        }
                    }
                }
            }
        }

        for y in 0..rows {
            for x in 0..cols {
                if final_edges[y][x] != 1.0 {
                    output[[y, x, c]] = 0.0;
                } else {
                    output[[y, x, c]] = 1.0;
                }
            }
        }
    }

    output.into_pyarray(py)
}


#[pymodule]
fn rust_cv_lib(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_negative, m)?)?;
    m.add_function(wrap_pyfunction!(apply_log, m)?)?;
    m.add_function(wrap_pyfunction!(apply_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(rgb_to_gray, m)?)?;
    m.add_function(wrap_pyfunction!(apply_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(hist_equalize_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(hist_equalize_gray, m)?)?;
    m.add_function(wrap_pyfunction!(hist_spec_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(hist_spec_gray, m)?)?;
    m.add_function(wrap_pyfunction!(median_filter, m)?)?;
    m.add_function(wrap_pyfunction!(laplacian_filter, m)?)?;
    m.add_function(wrap_pyfunction!(apply_frequency_filter, m)?)?;
    m.add_function(wrap_pyfunction!(apply_otsu_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(apply_canny, m)?)?;
    m.add_function(wrap_pyfunction!(rgb_to_cmy, m)?)?;
    
    Ok(())
}