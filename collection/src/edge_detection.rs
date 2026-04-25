use pyo3::prelude::*;
use numpy::{
    ndarray::{s, Array3},
    IntoPyArray, PyArray3, PyArrayDyn, PyArrayMethods, PyReadonlyArray3, PyReadonlyArrayDyn,
};
use std::f64::consts::PI;

use crate::helpers::{apply_median_3x3, apply_laplacian_3x3};

// ==========================================
// SPATIAL FILTERS (Median / Laplacian)
// ==========================================

#[pyfunction]
pub fn median_filter<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
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
pub fn laplacian_filter<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<&'py PyArrayDyn<u8>> {
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

// ==========================================
// CANNY EDGE DETECTION
// ==========================================

#[pyfunction]
pub fn apply_canny<'py>(py: Python<'py>, image: PyReadonlyArray3<'py, f64>, low_thresh: f64, high_thresh: f64) -> &'py PyArray3<f64> {
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
