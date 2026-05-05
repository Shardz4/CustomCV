use pyo3::prelude::*;
use numpy::{
    IntoPyArray, PyArray3, PyReadonlyArray3, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods
};
use std::f64::consts::PI;
use crate::helpers;
use numpy::ndarray;

// ==========================================
// CANNY EDGE DETECTION
// ==========================================

#[pyfunction]
pub fn apply_canny<'py>(py: Python<'py>, image: PyReadonlyArray3<'py, f64>, low_thresh: f64, high_thresh: f64) -> &'py PyArray3<f64> {
    let img = image.as_array();
    let (rows, cols, channels) = (img.shape()[0], img.shape()[1], img.shape()[2]);
    let mut output = ndarray::Array3::<f64>::zeros((rows, cols, channels));

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
        // Gaussian blur
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

        // Gradient magnitude and direction
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

        // Non-maximum suppression
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
                }
            }
        }

        // Hysteresis thresholding
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
                output[[y, x, c]] = if final_edges[y][x] == 1.0 { 1.0 } else { 0.0 };
            }
        }
    }

    output.into_pyarray(py)
}

#[pyfunction]
fn harris_corner<'py> (py: Python<'py>, image: PyReadonlyArrayDyn<'py, u8>, window_size: usize, k: i32) -> PyResult<Py<PyArrayDyn<f32>>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D Grayscale"))?;
    let (h,w) = (img_2d.shape()[0], img_2d.shape()[1]);
    let (sxx, syy, sxy) = helpers::compute_structure_tensor(&img_2d, window_size);
    let mut response = ndarray::Array2::<f32>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let det = (sxx[[y,x]] * syy[[y,x]]) - (sxy[[y,x]] * sxy[[y,x]]);
            let trace = sxx[[y,x]] + syy[[y,x]];
            response[[y,x]] = det - (k as f32) * (trace * trace);
        }
    }
    Ok(response.into_pyarray_bound(py).to_dyn().clone().unbind())
}

#[pyfunction]
pub fn shi_tomasi_corners<'py>(py: Python<'py>, image:PyReadonlyArrayDyn<'py, u8>, window_size: usize)-> PyResult<Py<PyArrayDyn<f32>>>{
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D Grayscale"))?;
    let (h,w) = (img_2d.shape()[0], img_2d.shape()[1]);
    let (sxx, syy, sxy) = helpers::compute_structure_tensor(&img_2d, window_size);
    let mut response = ndarray::Array2::<f32>::zeros((h, w));

    for y in 0..h{
        for x in 0..w{
            let a = sxx[[y , x]];
            let b = sxy[[y , x]];
            let c = syy[[y , x]];
            
            let trace = a + c;
            let det = (a * c) - (b * b);

            let gap = ((trace * trace) / 4.0 - det).max(0.0).sqrt();
            let lambda_min = (trace / 2.0) - gap;
            response[[y,x]] = lambda_min;
        }
    }
    Ok(response.into_pyarray_bound(py).to_dyn().clone().unbind())
}

#[pyfunction]
fn hough_lines(image: PyReadonlyArrayDyn<'_, u8>, threshold: u32, theta_res: usize) -> PyResult<Vec<(f64, f64)>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Requires 2D binary/edge image"))?;
    
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
    let max_rho = ((h as f64).powi(2) + (w as f64).powi(2)).sqrt().ceil() as usize;
    
    // Accumulator: Rows are rhos (-max_rho to +max_rho), Cols are thetas (0 to 180 degrees)
    let rho_offset = max_rho;
    let mut accumulator = vec![vec![0u32; 180 / theta_res]; max_rho * 2 + 1];

    // Precompute sine and cosine for thetas
    let mut thetas = Vec::new();
    let mut cos_t = Vec::new();
    let mut sin_t = Vec::new();
    for t in (0..180).step_by(theta_res) {
        let rad = (t as f64).to_radians();
        thetas.push(rad);
        cos_t.push(rad.cos());
        sin_t.push(rad.sin());
    }

    // Voting Process
    for y in 0..h {
        for x in 0..w {
            if img_2d[[y, x]] > 0 { // If it's an edge pixel
                for (i, _) in thetas.iter().enumerate() {
                    let rho = (x as f64 * cos_t[i] + y as f64 * sin_t[i]).round() as isize;
                    let rho_idx = (rho + rho_offset as isize) as usize;
                    accumulator[rho_idx][i] += 1;
                }
            }
        }
    }

    // Extract peaks
    let mut lines = Vec::new();
    for r in 0..(max_rho * 2 + 1) {
        for t_idx in 0..(180 / theta_res) {
            if accumulator[r][t_idx] >= threshold {
                let actual_rho = r as f64 - rho_offset as f64;
                let actual_theta = thetas[t_idx];
                lines.push((actual_rho, actual_theta));
            }
        }
    }

    Ok(lines)
}


#[pyfunction]
fn hough_circles(image: PyReadonlyArrayDyn<'_,u8>, radius: usize, threshold: i32) -> PyResult<Vec<(i32, i32, usize)>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Requires 2D binary/edge image"))?;
    
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
    let r = radius as i32;
    let mut accumulator = vec![vec![0u32; w]; h];

    for y in 0..h {
        for x in 0..w {
            if img_2d[[y, x]] > 0 { // If it's an edge pixel
                // Check points on the circle rim
                for theta_deg in 0..360 {
                    let rad = (theta_deg as f64).to_radians();
                    let cx_offset = r * (rad.cos() as i32);
                    let cy_offset = r * (rad.sin() as i32);
                    
                    let cx = x as i32 + cx_offset;
                    let cy = y as i32 + cy_offset;

                    if cx >= 0 && cx < w as i32 && cy >= 0 && cy < h as i32 {
                        accumulator[cy as usize][cx as usize] += 1;
                    }
                }
            }
        }
    }

    let mut circles = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if accumulator[y][x] >= threshold {
                circles.push((x as i32, y as i32, radius));
            }
        }
    }
    Ok(circles)
}
