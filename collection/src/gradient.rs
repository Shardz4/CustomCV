use pyo3::prelude::*;
use numpy::{
    ndarray::{s, Array2, Array3},
    IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn,
};

// ==========================================
// INTERNAL HELPERS
// ==========================================

/// Convolve a single 2D channel with an f64 kernel, returning f64 (preserving sign).
fn convolve_f64(channel: numpy::ndarray::ArrayView2<u8>, kernel: &Array2<f64>) -> Array2<f64> {
    let (h, w) = (channel.shape()[0], channel.shape()[1]);
    let (kh, kw) = (kernel.shape()[0], kernel.shape()[1]);
    let pad_h = kh / 2;
    let pad_w = kw / 2;
    let mut out = Array2::<f64>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for ky in 0..kh {
                for kx in 0..kw {
                    let iy = (y as isize + ky as isize - pad_h as isize).clamp(0, h as isize - 1) as usize;
                    let ix = (x as isize + kx as isize - pad_w as isize).clamp(0, w as isize - 1) as usize;
                    sum += channel[[iy, ix]] as f64 * kernel[[ky, kx]];
                }
            }
            out[[y, x]] = sum;
        }
    }
    out
}

/// Build the standard Sobel kernel of the given size for the specified derivative order.
/// Only supports ksize 1, 3, 5, 7.
fn build_sobel_kernel(ksize: usize, dx: usize, dy: usize) -> Array2<f64> {
    if ksize == 1 {
        // ksize=1 is a 1×3 or 3×1 kernel
        if dx == 1 && dy == 0 {
            return Array2::from_shape_vec((1, 3), vec![-1.0, 0.0, 1.0]).unwrap();
        } else if dx == 0 && dy == 1 {
            return Array2::from_shape_vec((3, 1), vec![-1.0, 0.0, 1.0]).unwrap();
        }
    }

    // Build 1D smoothing and derivative kernels
    // For ksize=3: smooth = [1,2,1], deriv = [-1,0,1]
    // For ksize=5: smooth = [1,4,6,4,1], deriv = [-1,-2,0,2,1]
    // For ksize=7: smooth = [1,6,15,20,15,6,1], deriv = [-1,-4,-5,0,5,4,1]
    let (smooth, deriv) = match ksize {
        3 => (vec![1.0, 2.0, 1.0], vec![-1.0, 0.0, 1.0]),
        5 => (vec![1.0, 4.0, 6.0, 4.0, 1.0], vec![-1.0, -2.0, 0.0, 2.0, 1.0]),
        7 => (vec![1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0], vec![-1.0, -4.0, -5.0, 0.0, 5.0, 4.0, 1.0]),
        _ => (vec![1.0, 2.0, 1.0], vec![-1.0, 0.0, 1.0]),
    };

    let n = ksize;
    let mut kernel = Array2::<f64>::zeros((n, n));

    if dx == 1 && dy == 0 {
        // Horizontal gradient: derivative along x, smooth along y
        for row in 0..n {
            for col in 0..n {
                kernel[[row, col]] = smooth[row] * deriv[col];
            }
        }
    } else if dx == 0 && dy == 1 {
        // Vertical gradient: derivative along y, smooth along x
        for row in 0..n {
            for col in 0..n {
                kernel[[row, col]] = deriv[row] * smooth[col];
            }
        }
    } else if dx == 1 && dy == 1 {
        // Mixed: derivative along both
        for row in 0..n {
            for col in 0..n {
                kernel[[row, col]] = deriv[row] * deriv[col];
            }
        }
    } else if dx == 2 && dy == 0 {
        // Second derivative along x: convolve deriv with itself for x, smooth for y
        // For 3×3: second_deriv = [1, -2, 1]
        let second = match ksize {
            3 => vec![1.0, -2.0, 1.0],
            5 => vec![1.0, 0.0, -2.0, 0.0, 1.0],
            7 => vec![1.0, 2.0, -1.0, -4.0, -1.0, 2.0, 1.0],
            _ => vec![1.0, -2.0, 1.0],
        };
        for row in 0..n {
            for col in 0..n {
                kernel[[row, col]] = smooth[row] * second[col];
            }
        }
    } else if dx == 0 && dy == 2 {
        // Second derivative along y
        let second = match ksize {
            3 => vec![1.0, -2.0, 1.0],
            5 => vec![1.0, 0.0, -2.0, 0.0, 1.0],
            7 => vec![1.0, 2.0, -1.0, -4.0, -1.0, 2.0, 1.0],
            _ => vec![1.0, -2.0, 1.0],
        };
        for row in 0..n {
            for col in 0..n {
                kernel[[row, col]] = second[row] * smooth[col];
            }
        }
    }

    kernel
}

/// Build a Laplacian kernel of the given size.
fn build_laplacian_kernel(ksize: usize) -> Array2<f64> {
    match ksize {
        1 => {
            // 3×3 basic 4-connected Laplacian (ksize=1 in OpenCV)
            Array2::from_shape_vec((3, 3), vec![
                0.0,  1.0, 0.0,
                1.0, -4.0, 1.0,
                0.0,  1.0, 0.0,
            ]).unwrap()
        }
        3 => {
            // 3×3 8-connected Laplacian
            Array2::from_shape_vec((3, 3), vec![
                -1.0, -1.0, -1.0,
                -1.0,  8.0, -1.0,
                -1.0, -1.0, -1.0,
            ]).unwrap()
        }
        5 => {
            // 5×5 Laplacian (discrete approximation)
            Array2::from_shape_vec((5, 5), vec![
                -1.0, -3.0, -4.0, -3.0, -1.0,
                -3.0,  0.0,  6.0,  0.0, -3.0,
                -4.0,  6.0, 20.0,  6.0, -4.0,
                -3.0,  0.0,  6.0,  0.0, -3.0,
                -1.0, -3.0, -4.0, -3.0, -1.0,
            ]).unwrap()
        }
        7 => {
            // 7×7 Laplacian (LoG approximation kernel)
            Array2::from_shape_vec((7, 7), vec![
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                -1.0, -1.0, -1.0, 48.0, -1.0, -1.0, -1.0,
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            ]).unwrap()
        }
        _ => build_laplacian_kernel(3), // fallback
    }
}

// ==========================================
// GRADIENT / EDGE OPERATOR FUNCTIONS
// ==========================================

/// apply_sobel() - Sobel gradient operator.
/// @py: Python interpreter token.
/// @x: Input image array (u8).
/// @dx: Order of the derivative x.
/// @dy: Order of the derivative y.
/// @ksize: Size of the extended Sobel kernel; must be 1, 3, 5, or 7.
///
/// Computes the first, second, or third-order image derivatives using an extended Sobel operator.
///
/// Return: 32-bit float gradient image array.
#[pyfunction]
pub fn apply_sobel<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, u8>,
    dx: usize,
    dy: usize,
    ksize: usize,
) -> PyResult<&'py PyArrayDyn<f32>> {
    if dx == 0 && dy == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("At least one of dx, dy must be > 0"));
    }
    if ![1, 3, 5, 7].contains(&ksize) {
        return Err(pyo3::exceptions::PyValueError::new_err("ksize must be 1, 3, 5, or 7"));
    }

    let kernel = build_sobel_kernel(ksize, dx, dy);
    let arr = x.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();

    if ndim == 3 {
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let mut out = Array3::<f32>::zeros((h, w, c));
        for ch in 0..c {
            let conv = convolve_f64(arr.slice(s![.., .., ch]), &kernel);
            for y in 0..h {
                for xi in 0..w {
                    out[[y, xi, ch]] = conv[[y, xi]] as f32;
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else if ndim == 2 {
        let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        let conv = convolve_f64(channel.view(), &kernel);
        Ok(conv.mapv(|v| v as f32).into_pyarray(py).to_dyn())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

/// apply_scharr() - Scharr gradient operator (3×3 only).
/// @py: Python interpreter token.
/// @x: Input image array (u8).
/// @dx: Order of the derivative x.
/// @dy: Order of the derivative y.
///
/// Calculates the first x- or y-image derivative using the 3x3 Scharr operator.
///
/// Return: 32-bit float gradient image array.
#[pyfunction]
pub fn apply_scharr<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, u8>,
    dx: usize,
    dy: usize,
) -> PyResult<&'py PyArrayDyn<f32>> {
    if !((dx == 1 && dy == 0) || (dx == 0 && dy == 1)) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Scharr requires exactly one of dx=1,dy=0 or dx=0,dy=1",
        ));
    }

    // Scharr kernels
    let kernel = if dx == 1 {
        // Horizontal (x) gradient
        Array2::from_shape_vec((3, 3), vec![
            -3.0,  0.0,  3.0,
            -10.0, 0.0, 10.0,
            -3.0,  0.0,  3.0,
        ]).unwrap()
    } else {
        // Vertical (y) gradient
        Array2::from_shape_vec((3, 3), vec![
            -3.0, -10.0, -3.0,
             0.0,   0.0,  0.0,
             3.0,  10.0,  3.0,
        ]).unwrap()
    };

    let arr = x.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();

    if ndim == 3 {
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let mut out = Array3::<f32>::zeros((h, w, c));
        for ch in 0..c {
            let conv = convolve_f64(arr.slice(s![.., .., ch]), &kernel);
            for y in 0..h {
                for xi in 0..w {
                    out[[y, xi, ch]] = conv[[y, xi]] as f32;
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else if ndim == 2 {
        let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        let conv = convolve_f64(channel.view(), &kernel);
        Ok(conv.mapv(|v| v as f32).into_pyarray(py).to_dyn())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}

/// apply_laplacian() - Laplacian operator with configurable kernel size.
/// @py: Python interpreter token.
/// @x: Input image array (u8).
/// @ksize: Aperture size used to compute the second-derivative filters; must be 1, 3, 5, or 7.
///
/// Calculates the Laplacian of an image by summing the second x and y derivatives.
///
/// Return: 32-bit float Laplacian image array.
#[pyfunction]
pub fn apply_laplacian<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, u8>,
    ksize: usize,
) -> PyResult<&'py PyArrayDyn<f32>> {
    if ![1, 3, 5, 7].contains(&ksize) {
        return Err(pyo3::exceptions::PyValueError::new_err("ksize must be 1, 3, 5, or 7"));
    }

    let kernel = build_laplacian_kernel(ksize);
    let arr = x.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();

    if ndim == 3 {
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let mut out = Array3::<f32>::zeros((h, w, c));
        for ch in 0..c {
            let conv = convolve_f64(arr.slice(s![.., .., ch]), &kernel);
            for y in 0..h {
                for xi in 0..w {
                    out[[y, xi, ch]] = conv[[y, xi]] as f32;
                }
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else if ndim == 2 {
        let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
        let conv = convolve_f64(channel.view(), &kernel);
        Ok(conv.mapv(|v| v as f32).into_pyarray(py).to_dyn())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"))
    }
}
