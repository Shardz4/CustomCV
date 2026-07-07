use pyo3::prelude::*;
use numpy::{
    ndarray::{Array2, ArrayView2, ArrayViewMutD},
    IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn,
};

// ==========================================
// INTERNAL HELPERS
// ==========================================

/// Suzuki-Abe (Suzuki85) border following algorithm to find all contours.
/// Returns a list of contours, where each contour is a vector of (x, y) points.
pub fn find_contours_suzuki(image: &ArrayView2<u8>) -> Vec<Vec<(i32, i32)>> {
    let (h, w) = (image.shape()[0], image.shape()[1]);
    
    // Pad the image with a 1-pixel border of 0s to simplify boundary checks.
    // Padded size: (h + 2) x (w + 2)
    let mut grid = Array2::<i32>::zeros((h + 2, w + 2));
    for y in 0..h {
        for x in 0..w {
            if image[[y, x]] > 0 {
                grid[[y + 1, x + 1]] = 1;
            }
        }
    }

    let mut contours: Vec<Vec<(i32, i32)>> = Vec::new();
    let mut nbd = 1;

    // 8-neighbor offsets in clockwise order:
    // 0: West, 1: NW, 2: North, 3: NE, 4: East, 5: SE, 6: South, 7: SW
    const NEIGHBORS: [(isize, isize); 8] = [
        (0, -1),   // 0: West
        (-1, -1),  // 1: NW
        (-1, 0),   // 2: North
        (-1, 1),   // 3: NE
        (0, 1),    // 4: East
        (1, 1),    // 5: SE
        (1, 0),    // 6: South
        (1, -1),   // 7: SW
    ];

    let get_neighbor_dir = |dy: isize, dx: isize| -> Option<usize> {
        for i in 0..8 {
            if NEIGHBORS[i] == (dy, dx) {
                return Some(i);
            }
        }
        None
    };

    for y in 1..=h {
        let mut ln = 1; // Active border ID on this row
        for x in 1..=w {
            let val = grid[[y, x]];
            if val == 0 {
                continue;
            }

            let mut border_started = false;
            let mut start_dir = 0;

            // Check if outer border starts
            if val == 1 && grid[[y, x - 1]] == 0 {
                nbd += 1;
                start_dir = 0; // Start search from West
                border_started = true;
            }
            // Check if hole border starts
            else if val >= 1 && grid[[y, x + 1]] == 0 {
                let val_abs = val.abs();
                if val_abs > ln {
                    nbd += 1;
                    start_dir = 4; // Start search from East
                    border_started = true;
                }
            }

            if border_started {
                let mut contour_pts = Vec::new();
                let (cy, cx) = (y, x);

                // Find first non-zero neighbor clockwise
                let mut found_neighbor = false;
                let mut first_y = 0;
                let mut first_x = 0;

                for k in 0..8 {
                    let dir = (start_dir + k) % 8;
                    let ny = cy as isize + NEIGHBORS[dir].0;
                    let nx = cx as isize + NEIGHBORS[dir].1;
                    if ny >= 0 && ny < (h + 2) as isize && nx >= 0 && nx < (w + 2) as isize {
                        if grid[[ny as usize, nx as usize]] != 0 {
                            found_neighbor = true;
                            first_y = ny as usize;
                            first_x = nx as usize;
                            break;
                        }
                    }
                }

                if !found_neighbor {
                    // Isolated pixel
                    grid[[cy, cx]] = -nbd;
                    contour_pts.push((cx as i32 - 1, cy as i32 - 1));
                } else {
                    let (mut prev_y, mut prev_x) = (cy, cx);
                    let (mut cur_y, mut cur_x) = (first_y, first_x);
                    
                    let start_y = cy;
                    let start_x = cx;
                    let first_const_y = first_y;
                    let first_const_x = first_x;

                    // Push start point
                    contour_pts.push((cx as i32 - 1, cy as i32 - 1));

                    loop {
                        contour_pts.push((cur_x as i32 - 1, cur_y as i32 - 1));

                        // Search counter-clockwise starting from the neighbor next to prev
                        let dy = prev_y as isize - cur_y as isize;
                        let dx = prev_x as isize - cur_x as isize;
                        let prev_dir = get_neighbor_dir(dy, dx).unwrap_or(0);
                        
                        let mut next_y = 0;
                        let mut next_x = 0;
                        let mut found_next = false;

                        for k in 1..=8 {
                            let dir = (prev_dir + 8 - k) % 8;
                            let ny = cur_y as isize + NEIGHBORS[dir].0;
                            let nx = cur_x as isize + NEIGHBORS[dir].1;
                            if ny >= 0 && ny < (h + 2) as isize && nx >= 0 && nx < (w + 2) as isize {
                                if grid[[ny as usize, nx as usize]] != 0 {
                                    next_y = ny as usize;
                                    next_x = nx as usize;
                                    found_next = true;
                                    break;
                                }
                            }
                        }

                        if !found_next {
                            break;
                        }

                        // Update grid value
                        let right_x = cur_x + 1;
                        if right_x < w + 2 && grid[[cur_y, right_x]] == 0 {
                            grid[[cur_y, cur_x]] = -nbd;
                        } else if grid[[cur_y, cur_x]] == 1 {
                            grid[[cur_y, cur_x]] = nbd;
                        }

                        // Check if we completed the loop and are about to revisit the first step
                        if cur_y == start_y && cur_x == start_x && next_y == first_const_y && next_x == first_const_x {
                            break;
                        }

                        prev_y = cur_y;
                        prev_x = cur_x;
                        cur_y = next_y;
                        cur_x = next_x;
                    }
                }

                contours.push(contour_pts);
            }

            // Update ln
            let current_val = grid[[y, x]];
            if current_val != 0 {
                ln = current_val.abs();
            }
        }
    }

    contours
}

/// Bresenham's line algorithm to draw a line on a mutable array view.
fn draw_line_bresenham(
    img: &mut ArrayViewMutD<u8>,
    x1: i32, y1: i32,
    x2: i32, y2: i32,
    color: &[u8],
    thickness: i32,
) {
    let mut x = x1;
    let mut y = y1;
    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();
    let sx = if x1 < x2 { 1 } else { -1 };
    let sy = if y1 < y2 { 1 } else { -1 };
    let mut err = dx - dy;

    let h = img.shape()[0] as i32;
    let w = img.shape()[1] as i32;
    let channels = if img.ndim() == 3 { img.shape()[2] } else { 1 };

    let mut draw_pixel = |px: i32, py: i32| {
        let t = thickness.max(1);
        let radius = t / 2;
        for dy_t in -radius..=radius {
            for dx_t in -radius..=radius {
                let cur_x = px + dx_t;
                let cur_y = py + dy_t;
                if cur_x >= 0 && cur_x < w && cur_y >= 0 && cur_y < h {
                    if channels == 1 {
                        img[[cur_y as usize, cur_x as usize]] = color[0];
                    } else {
                        for c in 0..channels {
                            img[[cur_y as usize, cur_x as usize, c]] = color[c];
                        }
                    }
                }
            }
        }
    };

    loop {
        draw_pixel(x, y);
        if x == x2 && y == y2 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

/// Fill a polygon using the scanline fill algorithm.
fn fill_polygon(img: &mut ArrayViewMutD<u8>, pts: &[(i32, i32)], color: &[u8]) {
    if pts.len() < 3 {
        return;
    }
    let h = img.shape()[0] as i32;
    let w = img.shape()[1] as i32;
    let channels = if img.ndim() == 3 { img.shape()[2] } else { 1 };

    // Find bounding box
    let mut min_y = pts[0].1;
    let mut max_y = pts[0].1;
    for p in pts {
        if p.1 < min_y { min_y = p.1; }
        if p.1 > max_y { max_y = p.1; }
    }

    // Clip bounding box to image bounds
    let min_y = min_y.clamp(0, h - 1);
    let max_y = max_y.clamp(0, h - 1);

    let n = pts.len();

    for y in min_y..=max_y {
        let mut node_x = Vec::new();
        let mut j = n - 1;
        for i in 0..n {
            let py_i = pts[i].1;
            let py_j = pts[j].1;
            let px_i = pts[i].0;
            let px_j = pts[j].0;

            if (py_i < y && py_j >= y) || (py_j < y && py_i >= y) {
                let intersect_x = px_i as f64 + (y - py_i) as f64 * (px_j - px_i) as f64 / (py_j - py_i) as f64;
                node_x.push(intersect_x.round() as i32);
            }
            j = i;
        }

        node_x.sort_unstable();

        for pair in node_x.chunks_exact(2) {
            let start_x = pair[0].clamp(0, w - 1);
            let end_x = pair[1].clamp(0, w - 1);
            for x in start_x..=end_x {
                if channels == 1 {
                    img[[y as usize, x as usize]] = color[0];
                } else {
                    for c in 0..channels {
                        img[[y as usize, x as usize, c]] = color[c];
                    }
                }
            }
        }
    }
}

// ==========================================
// PYFUNCTION EXPORTS
// ==========================================

/// Extract contours from a binary image using the Suzuki-Abe border following algorithm.
///
/// Returns a list of contours, where each contour is a numpy array of shape (N, 2)
/// containing [x, y] coordinates (column, row).
#[pyfunction]
pub fn find_contours<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, u8>) -> PyResult<Vec<&'py PyArrayDyn<i32>>> {
    let arr = x.as_array();
    let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input must be a 2D binary image"))?;

    let contours_raw = find_contours_suzuki(&channel.view());
    
    let mut out = Vec::new();
    for contour in contours_raw {
        let n = contour.len();
        let mut arr_contour = Array2::<i32>::zeros((n, 2));
        for i in 0..n {
            arr_contour[[i, 0]] = contour[i].0;
            arr_contour[[i, 1]] = contour[i].1;
        }
        out.push(arr_contour.into_pyarray(py).to_dyn());
    }
    
    Ok(out)
}

/// Render contours onto a grayscale or color image.
///
/// Returns a new annotated image.
/// - `contours`: list of numpy arrays representing contours.
/// - `contour_idx`: index of the contour to draw. If -1, all contours are drawn.
/// - `color`: color value, either a single int (for grayscale) or a list/tuple (R, G, B).
/// - `thickness`: line thickness. If negative (e.g. -1), the contours are filled.
#[pyfunction]
#[pyo3(signature = (img, contours, contour_idx, color, thickness = 1))]
pub fn draw_contours<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    contours: Vec<PyReadonlyArrayDyn<'py, i32>>,
    contour_idx: i32,
    color: PyObject,
    thickness: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = img.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();
    
    // Create a mutable copy of the image to draw on
    let mut out_arr = if ndim == 3 {
        // ndarray Clone
        arr.to_owned()
    } else if ndim == 2 {
        arr.to_owned()
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"));
    };

    let channels = if ndim == 3 { shape[2] } else { 1 };
    
    // Parse color
    let mut parsed_color = vec![0u8; channels];
    Python::with_gil(|py_gil| {
        if let Ok(val) = color.extract::<u8>(py_gil) {
            for c in 0..channels {
                parsed_color[c] = val;
            }
        } else if let Ok(list) = color.extract::<Vec<u8>>(py_gil) {
            for c in 0..channels.min(list.len()) {
                parsed_color[c] = list[c];
            }
        } else if let Ok(tuple) = color.extract::<(u8, u8, u8)>(py_gil) {
            if channels >= 3 {
                parsed_color[0] = tuple.0;
                parsed_color[1] = tuple.1;
                parsed_color[2] = tuple.2;
            } else {
                parsed_color[0] = tuple.0;
            }
        }
    });

    let mut out_view = out_arr.view_mut().into_dyn();

    let process_contour = |c_arr: &ArrayView2<i32>| -> Vec<(i32, i32)> {
        let n = c_arr.shape()[0];
        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            pts.push((c_arr[[i, 0]], c_arr[[i, 1]]));
        }
        pts
    };

    let draw_single = |view: &mut ArrayViewMutD<u8>, c_arr: &ArrayView2<i32>| {
        let pts = process_contour(c_arr);
        if pts.is_empty() {
            return;
        }

        if thickness < 0 {
            fill_polygon(view, &pts, &parsed_color);
        } else {
            let n = pts.len();
            for i in 0..n {
                let p1 = pts[i];
                let p2 = pts[(i + 1) % n];
                draw_line_bresenham(view, p1.0, p1.1, p2.0, p2.1, &parsed_color, thickness);
            }
        }
    };

    if contour_idx >= 0 {
        if (contour_idx as usize) < contours.len() {
            let c_arr_dyn = contours[contour_idx as usize].as_array();
            let c_arr = c_arr_dyn.into_dimensionality::<numpy::ndarray::Ix2>()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Contour must be 2D array"))?;
            draw_single(&mut out_view, &c_arr.view());
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err("Contour index out of range"));
        }
    } else {
        for c_dyn in &contours {
            let c_arr_dyn = c_dyn.as_array();
            let c_arr = c_arr_dyn.into_dimensionality::<numpy::ndarray::Ix2>()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Contour must be 2D array"))?;
            draw_single(&mut out_view, &c_arr.view());
        }
    }

    Ok(out_arr.into_pyarray(py).to_dyn())
}

/// Compute the area enclosed by a contour.
///
/// - `contour`: a (N, 2) or (N, 1, 2) array of coordinates.
/// - `oriented`: if True, returns signed area indicating clockwise/counter-clockwise orientation.
#[pyfunction]
#[pyo3(signature = (contour, oriented = false))]
pub fn contour_area(contour: PyReadonlyArrayDyn<i32>, oriented: bool) -> PyResult<f64> {
    let arr = contour.as_array();
    let ndim = arr.ndim();
    let n = arr.shape()[0];
    if n < 3 {
        return Ok(0.0);
    }
    
    let mut area = 0.0;
    if ndim == 2 {
        if arr.shape()[1] != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("Contour array must have shape (N, 2) or (N, 1, 2)"));
        }
        for i in 0..n {
            let x1 = arr[[i, 0]] as f64;
            let y1 = arr[[i, 1]] as f64;
            let next_idx = (i + 1) % n;
            let x2 = arr[[next_idx, 0]] as f64;
            let y2 = arr[[next_idx, 1]] as f64;
            area += (x1 * y2) - (x2 * y1);
        }
    } else if ndim == 3 {
        if arr.shape()[1] != 1 || arr.shape()[2] != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("Contour array must have shape (N, 2) or (N, 1, 2)"));
        }
        for i in 0..n {
            let x1 = arr[[i, 0, 0]] as f64;
            let y1 = arr[[i, 0, 1]] as f64;
            let next_idx = (i + 1) % n;
            let x2 = arr[[next_idx, 0, 0]] as f64;
            let y2 = arr[[next_idx, 0, 1]] as f64;
            area += (x1 * y2) - (x2 * y1);
        }
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err("Contour array must have shape (N, 2) or (N, 1, 2)"));
    }

    let final_area = area / 2.0;
    if oriented {
        Ok(final_area)
    } else {
        Ok(final_area.abs())
    }
}

/// Compute the perimeter / arc length of a contour or curve.
///
/// - `curve`: a (N, 2) or (N, 1, 2) array of coordinates.
/// - `closed`: if True, includes the segment from the last point back to the first.
#[pyfunction]
#[pyo3(signature = (curve, closed = true))]
pub fn arc_length(curve: PyReadonlyArrayDyn<i32>, closed: bool) -> PyResult<f64> {
    let arr = curve.as_array();
    let ndim = arr.ndim();
    let n = arr.shape()[0];
    if n < 2 {
        return Ok(0.0);
    }
    
    let mut length = 0.0;
    
    let get_pt = |i: usize| -> (f64, f64) {
        if ndim == 2 {
            (arr[[i, 0]] as f64, arr[[i, 1]] as f64)
        } else {
            (arr[[i, 0, 0]] as f64, arr[[i, 0, 1]] as f64)
        }
    };
    
    if ndim != 2 && ndim != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Curve array must have shape (N, 2) or (N, 1, 2)"));
    }
    if ndim == 2 && arr.shape()[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("Curve array must have shape (N, 2)"));
    }
    if ndim == 3 && (arr.shape()[1] != 1 || arr.shape()[2] != 2) {
        return Err(pyo3::exceptions::PyValueError::new_err("Curve array must have shape (N, 1, 2)"));
    }

    for i in 0..(n - 1) {
        let (x1, y1) = get_pt(i);
        let (x2, y2) = get_pt(i + 1);
        let dx = x2 - x1;
        let dy = y2 - y1;
        length += (dx * dx + dy * dy).sqrt();
    }
    
    if closed {
        let (x1, y1) = get_pt(n - 1);
        let (x2, y2) = get_pt(0);
        let dx = x2 - x1;
        let dy = y2 - y1;
        length += (dx * dx + dy * dy).sqrt();
    }
    
    Ok(length)
}
