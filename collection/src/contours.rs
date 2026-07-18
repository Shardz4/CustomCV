use pyo3::prelude::*;
use numpy::{
    ndarray::{Array2, ArrayView2, ArrayViewMutD},
    IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn,
};

// ==========================================
// INTERNAL HELPERS
// ==========================================

/// find_contours_suzuki() - Suzuki-Abe border following algorithm.
/// @image: Dynamic-dimensional input image slice.
///
/// Implements the Suzuki-Abe (Suzuki85) border following algorithm
/// to extract all outer and hole contours from a binary image.
///
/// Return: A vector of contours, where each contour is a vector of (x, y) coordinates.
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
pub(crate) fn draw_line_bresenham(
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
pub(crate) fn fill_polygon(img: &mut ArrayViewMutD<u8>, pts: &[(i32, i32)], color: &[u8]) {
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

/// find_contours() - Extract contours from a binary image.
/// @py: Python interpreter token.
/// @x: Input 2D binary image array (u8).
///
/// Extracts contours from a binary image using the Suzuki-Abe border following algorithm.
///
/// Return: A list of numpy arrays of shape (N, 2) containing [x, y] coordinates.
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

/// draw_contours() - Render contours onto an image.
/// @py: Python interpreter token.
/// @img: Input grayscale or color image array (u8).
/// @contours: List of contours to draw.
/// @contour_idx: Index of contour to draw. If -1, all are drawn.
/// @color: Color value (int or tuple).
/// @thickness: Line thickness (if negative, contours are filled).
///
/// Renders contours onto the input image.
///
/// Return: Annotated image array.
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

/// contour_area() - Compute the area enclosed by a contour.
/// @contour: Input contour array of shape (N, 2) or (N, 1, 2) containing coordinates.
/// @oriented: If true, returns signed area indicating orientation.
///
/// Computes the area enclosed by a contour using Green's theorem (Shoelace formula).
///
/// Return: Enclosed area.
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

/// arc_length() - Compute the perimeter / arc length of a contour or curve.
/// @curve: Input curve array of shape (N, 2) or (N, 1, 2) containing coordinates.
/// @closed: If true, includes the segment from the last point to the first.
///
/// Computes the perimeter or arc length of a contour.
///
/// Return: Arc length.
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

/// bounding_rect() - Compute the axis-aligned bounding rectangle of a contour.
/// @contour: Input contour array of shape (N, 2) or (N, 1, 2).
///
/// Computes the axis-aligned bounding rectangle of a contour.
///
/// Return: A tuple containing (x, y, width, height).
#[pyfunction]
pub fn bounding_rect(contour: PyReadonlyArrayDyn<i32>) -> PyResult<(i32, i32, i32, i32)> {
    let arr = contour.as_array();
    let ndim = arr.ndim();
    let n = arr.shape()[0];
    if n == 0 {
        return Ok((0, 0, 0, 0));
    }
    
    let get_pt = |i: usize| -> (i32, i32) {
        if ndim == 2 {
            (arr[[i, 0]], arr[[i, 1]])
        } else {
            (arr[[i, 0, 0]], arr[[i, 0, 1]])
        }
    };
    
    let (mut min_x, mut min_y) = get_pt(0);
    let (mut max_x, mut max_y) = get_pt(0);
    
    for i in 1..n {
        let (x, y) = get_pt(i);
        if x < min_x { min_x = x; }
        if x > max_x { max_x = x; }
        if y < min_y { min_y = y; }
        if y > max_y { max_y = y; }
    }
    
    let width = max_x - min_x + 1;
    let height = max_y - min_y + 1;
    
    Ok((min_x, min_y, width, height))
}

/// min_area_rect() - Compute the minimum-area rotated bounding rectangle of a contour.
/// @contour: Input contour array.
///
/// Computes the minimum-area rotated bounding rectangle of a contour.
///
/// Return: A tuple containing ((center_x, center_y), (width, height), angle_in_degrees).
#[pyfunction]
pub fn min_area_rect(contour: PyReadonlyArrayDyn<i32>) -> PyResult<((f64, f64), (f64, f64), f64)> {
    let arr = contour.as_array();
    let ndim = arr.ndim();
    let n = arr.shape()[0];
    if n == 0 {
        return Ok(((0.0, 0.0), (0.0, 0.0), 0.0));
    }
    
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let (x, y) = if ndim == 2 {
            (arr[[i, 0]] as f64, arr[[i, 1]] as f64)
        } else {
            (arr[[i, 0, 0]] as f64, arr[[i, 0, 1]] as f64)
        };
        pts.push((x, y));
    }
    
    let hull = convex_hull_points(pts);
    if hull.len() <= 1 {
        let p = if hull.is_empty() { (0.0, 0.0) } else { hull[0] };
        return Ok(((p.0, p.1), (0.0, 0.0), 0.0));
    }
    
    let mut min_area = f64::MAX;
    let mut best_center = (0.0, 0.0);
    let mut best_size = (0.0, 0.0);
    let mut best_angle = 0.0;
    
    let h_len = hull.len();
    for i in 0..h_len {
        let p1 = hull[i];
        let p2 = hull[(i + 1) % h_len];
        let dx = p2.0 - p1.0;
        let dy = p2.1 - p1.1;
        let angle = dy.atan2(dx);
        
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        
        let mut min_rx = f64::MAX;
        let mut max_rx = f64::MIN;
        let mut min_ry = f64::MAX;
        let mut max_ry = f64::MIN;
        
        for p in &hull {
            let rx = p.0 * cos_a + p.1 * sin_a;
            let ry = -p.0 * sin_a + p.1 * cos_a;
            if rx < min_rx { min_rx = rx; }
            if rx > max_rx { max_rx = rx; }
            if ry < min_ry { min_ry = ry; }
            if ry > max_ry { max_ry = ry; }
        }
        
        let w = max_rx - min_rx;
        let h = max_ry - min_ry;
        let area = w * h;
        
        if area < min_area {
            min_area = area;
            let cx_r = (min_rx + max_rx) / 2.0;
            let cy_r = (min_ry + max_ry) / 2.0;
            let cx = cx_r * cos_a - cy_r * sin_a;
            let cy = cx_r * sin_a + cy_r * cos_a;
            best_center = (cx, cy);
            best_size = (w, h);
            best_angle = angle.to_degrees();
        }
    }
    
    Ok((best_center, best_size, best_angle))
}

/// min_enclosing_circle() - Compute the minimum enclosing circle of a contour.
/// @contour: Input contour array.
///
/// Computes the minimum enclosing circle of a 2D point set using Welzl's algorithm.
///
/// Return: A tuple containing ((center_x, center_y), radius).
#[pyfunction]
pub fn min_enclosing_circle(contour: PyReadonlyArrayDyn<i32>) -> PyResult<((f64, f64), f64)> {
    let arr = contour.as_array();
    let ndim = arr.ndim();
    let n = arr.shape()[0];
    if n == 0 {
        return Ok(((0.0, 0.0), 0.0));
    }
    
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let (x, y) = if ndim == 2 {
            (arr[[i, 0]] as f64, arr[[i, 1]] as f64)
        } else {
            (arr[[i, 0, 0]] as f64, arr[[i, 0, 1]] as f64)
        };
        pts.push((x, y));
    }
    
    shuffle_points(&mut pts);
    let circle = welzl(&pts, Vec::new(), pts.len());
    Ok(circle)
}

/// fit_ellipse() - Fit an ellipse to a 2D point set.
/// @contour: Input contour array containing at least 5 points.
///
/// Fits an ellipse to a 2D point set using the direct least squares method.
///
/// Return: A tuple containing ((center_x, center_y), (axis_width, axis_height), angle_in_degrees).
#[pyfunction]
pub fn fit_ellipse(contour: PyReadonlyArrayDyn<i32>) -> PyResult<((f64, f64), (f64, f64), f64)> {
    let arr = contour.as_array();
    let ndim = arr.ndim();
    let n = arr.shape()[0];
    if n < 5 {
        return Err(pyo3::exceptions::PyValueError::new_err("fitEllipse requires at least 5 points"));
    }
    
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let (x, y) = if ndim == 2 {
            (arr[[i, 0]] as f64, arr[[i, 1]] as f64)
        } else {
            (arr[[i, 0, 0]] as f64, arr[[i, 0, 1]] as f64)
        };
        pts.push((x, y));
    }
    
    let mut s = [[0.0; 5]; 5];
    let mut b = [0.0; 5];
    
    for &(x, y) in &pts {
        let x2 = x * x;
        let xy = x * y;
        let y2 = y * y;
        let row = [x2, xy, y2, x, y];
        for j in 0..5 {
            for k in 0..5 {
                s[j][k] += row[j] * row[k];
            }
            b[j] -= row[j];
        }
    }
    
    let a = match solve_5x5(s, b) {
        Some(sol) => sol,
        None => return Err(pyo3::exceptions::PyValueError::new_err("Could not fit ellipse (singular matrix)")),
    };
    
    let aa = a[0];
    let bb = a[1];
    let cc = a[2];
    let dd = a[3];
    let ee = a[4];
    let ff = 1.0;
    
    let denom = bb * bb - 4.0 * aa * cc;
    if denom >= 0.0 {
        // Conic is not an ellipse
        return Err(pyo3::exceptions::PyValueError::new_err("Fitted conic is not an ellipse"));
    }
    
    // Center of ellipse
    let cx = (bb * ee - 2.0 * cc * dd) / denom;
    let cy = (bb * dd - 2.0 * aa * ee) / denom;
    
    // Translate origin to center
    let f_prime = aa * cx * cx + bb * cx * cy + cc * cy * cy + dd * cx + ee * cy + ff;
    
    // Eigenvalues of quadratic form
    let val_sqrt = ((aa - cc).powi(2) + bb * bb).sqrt();
    let l1 = (aa + cc + val_sqrt) / 2.0;
    let l2 = (aa + cc - val_sqrt) / 2.0;
    
    let semi_a = if -f_prime / l1 > 0.0 { (-f_prime / l1).sqrt() } else { 0.0 };
    let semi_b = if -f_prime / l2 > 0.0 { (-f_prime / l2).sqrt() } else { 0.0 };
    
    let axis_w = 2.0 * semi_a;
    let axis_h = 2.0 * semi_b;
    
    let mut angle = if bb.abs() < 1e-12 {
        if aa < cc { 0.0 } else { 90.0 }
    } else {
        (0.5 * bb.atan2(aa - cc)).to_degrees()
    };
    if angle < 0.0 {
        angle += 180.0;
    }
    
    Ok(((cx, cy), (axis_w, axis_h), angle))
}

// ==========================================
// CONVEX HULL & WELZL HELPERS
// ==========================================

fn convex_hull_points(mut pts: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    if pts.len() <= 1 {
        return pts;
    }
    
    pts.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    
    pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9);
    
    if pts.len() <= 1 {
        return pts;
    }

    let cross = |o: &(f64, f64), a: &(f64, f64), b: &(f64, f64)| -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };

    let mut lower = Vec::new();
    for p in &pts {
        while lower.len() >= 2 && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(*p);
    }

    let mut upper = Vec::new();
    for p in pts.iter().rev() {
        while upper.len() >= 2 && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(*p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    ((p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2)).sqrt()
}

fn is_inside(circle: &((f64, f64), f64), p: (f64, f64)) -> bool {
    distance(circle.0, p) <= circle.1 + 1e-9
}

fn circumcircle(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> ((f64, f64), f64) {
    let d = 2.0 * (a.0 * (b.1 - c.1) + b.0 * (c.1 - a.1) + c.0 * (a.1 - b.1));
    if d.abs() < 1e-9 {
        let d1 = distance(a, b);
        let d2 = distance(b, c);
        let d3 = distance(a, c);
        if d1 >= d2 && d1 >= d3 {
            (((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0), d1 / 2.0)
        } else if d2 >= d1 && d2 >= d3 {
            (((b.0 + c.0) / 2.0, (b.1 + c.1) / 2.0), d2 / 2.0)
        } else {
            (((a.0 + c.0) / 2.0, (a.1 + c.1) / 2.0), d3 / 2.0)
        }
    } else {
        let ux = ((a.0.powi(2) + a.1.powi(2)) * (b.1 - c.1)
            + (b.0.powi(2) + b.1.powi(2)) * (c.1 - a.1)
            + (c.0.powi(2) + c.1.powi(2)) * (a.1 - b.1))
            / d;
        let uy = ((a.0.powi(2) + a.1.powi(2)) * (c.0 - b.0)
            + (b.0.powi(2) + b.1.powi(2)) * (a.0 - c.0)
            + (c.0.powi(2) + c.1.powi(2)) * (b.0 - a.0))
            / d;
        let center = (ux, uy);
        (center, distance(center, a))
    }
}

fn circle_from_boundary(boundary: &[(f64, f64)]) -> ((f64, f64), f64) {
    match boundary.len() {
        0 => ((0.0, 0.0), 0.0),
        1 => (boundary[0], 0.0),
        2 => {
            let a = boundary[0];
            let b = boundary[1];
            (((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0), distance(a, b) / 2.0)
        }
        3 => {
            let a = boundary[0];
            let b = boundary[1];
            let c = boundary[2];
            let dot_a = (b.0 - a.0) * (c.0 - a.0) + (b.1 - a.1) * (c.1 - a.1);
            let dot_b = (a.0 - b.0) * (c.0 - b.0) + (a.1 - b.1) * (c.1 - b.1);
            let dot_c = (a.0 - c.0) * (b.0 - c.0) + (a.1 - c.1) * (b.1 - c.1);

            if dot_a < 0.0 {
                (((b.0 + c.0) / 2.0, (b.1 + c.1) / 2.0), distance(b, c) / 2.0)
            } else if dot_b < 0.0 {
                (((a.0 + c.0) / 2.0, (a.1 + c.1) / 2.0), distance(a, c) / 2.0)
            } else if dot_c < 0.0 {
                (((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0), distance(a, b) / 2.0)
            } else {
                circumcircle(a, b, c)
            }
        }
        _ => ((0.0, 0.0), 0.0),
    }
}

fn welzl(pts: &[(f64, f64)], boundary: Vec<(f64, f64)>, n: usize) -> ((f64, f64), f64) {
    if n == 0 || boundary.len() == 3 {
        return circle_from_boundary(&boundary);
    }
    let p = pts[n - 1];
    let circle = welzl(pts, boundary.clone(), n - 1);
    if is_inside(&circle, p) {
        return circle;
    }
    let mut new_boundary = boundary;
    new_boundary.push(p);
    welzl(pts, new_boundary, n - 1)
}

fn shuffle_points(pts: &mut [(f64, f64)]) {
    let mut state = 123456789u64;
    let mut next_random = || -> usize {
        state ^= state << 12;
        state ^= state >> 25;
        state ^= state << 27;
        state as usize
    };
    let n = pts.len();
    for i in (1..n).rev() {
        let j = next_random() % (i + 1);
        pts.swap(i, j);
    }
}

fn solve_5x5(mut s: [[f64; 5]; 5], mut b: [f64; 5]) -> Option<[f64; 5]> {
    let n = 5;
    for i in 0..n {
        let mut max_row = i;
        for r in (i + 1)..n {
            if s[r][i].abs() > s[max_row][i].abs() {
                max_row = r;
            }
        }
        if s[max_row][i].abs() < 1e-12 {
            return None;
        }
        s.swap(i, max_row);
        b.swap(i, max_row);
        
        for r in (i + 1)..n {
            let factor = s[r][i] / s[i][i];
            for c in i..n {
                s[r][c] -= factor * s[i][c];
            }
            b[r] -= factor * b[i];
        }
    }
    
    let mut x = [0.0; 5];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for c in (i + 1)..n {
            sum -= s[i][c] * x[c];
        }
        x[i] = sum / s[i][i];
    }
    Some(x)
}

/// convex_hull() - Compute the convex hull of a point set.
/// @py: Python interpreter token.
/// @contour: Input contour array.
/// @clockwise: If true, returns hull points in clockwise order.
/// @return_points: If true, returns coordinate array (M, 2). Otherwise, returns index array (M, 1).
///
/// Computes the convex hull of a point set using the Monotone Chain (Andrew's) algorithm.
///
/// Return: An array representing the convex hull.
#[pyfunction]
#[pyo3(signature = (contour, clockwise = false, return_points = true))]
pub fn convex_hull<'py>(
    py: Python<'py>,
    contour: PyReadonlyArrayDyn<'py, i32>,
    clockwise: bool,
    return_points: bool,
) -> PyResult<&'py PyArrayDyn<i32>> {
    let arr = contour.as_array();
    let ndim = arr.ndim();
    let n = arr.shape()[0];
    if n == 0 {
        return Ok(Array2::<i32>::zeros((0, 2)).into_pyarray(py).to_dyn());
    }
    
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let (x, y) = if ndim == 2 {
            (arr[[i, 0]] as f64, arr[[i, 1]] as f64)
        } else {
            (arr[[i, 0, 0]] as f64, arr[[i, 0, 1]] as f64)
        };
        pts.push((x, y));
    }
    
    let mut hull_indices = convex_hull_indexed(&pts);
    if clockwise {
        hull_indices.reverse();
    }
    
    if return_points {
        let m = hull_indices.len();
        let mut out = Array2::<i32>::zeros((m, 2));
        for i in 0..m {
            let idx = hull_indices[i];
            if ndim == 2 {
                out[[i, 0]] = arr[[idx, 0]];
                out[[i, 1]] = arr[[idx, 1]];
            } else {
                out[[i, 0]] = arr[[idx, 0, 0]];
                out[[i, 1]] = arr[[idx, 0, 1]];
            }
        }
        Ok(out.into_pyarray(py).to_dyn())
    } else {
        let m = hull_indices.len();
        let mut out = Array2::<i32>::zeros((m, 1));
        for i in 0..m {
            out[[i, 0]] = hull_indices[i] as i32;
        }
        Ok(out.into_pyarray(py).to_dyn())
    }
}

/// convexity_defects() - Find convexity defects in a contour.
/// @py: Python interpreter token.
/// @contour: Input contour array of shape (N, 2) or (N, 1, 2).
/// @convexhull: Index array of the hull vertices in the contour.
///
/// Finds convexity defects (valleys/indents relative to the convex hull) in a contour.
///
/// Return: A (K, 4) array containing [start_index, end_index, far_index, depth_scaled_by_256].
#[pyfunction]
pub fn convexity_defects<'py>(
    py: Python<'py>,
    contour: PyReadonlyArrayDyn<'py, i32>,
    convexhull: PyReadonlyArrayDyn<'py, i32>,
) -> PyResult<&'py PyArrayDyn<i32>> {
    let arr = contour.as_array();
    let ndim = arr.ndim();
    let n = arr.shape()[0];
    
    let hull_arr = convexhull.as_array();
    let m = hull_arr.shape()[0];
    
    if n < 3 || m < 3 {
        return Ok(Array2::<i32>::zeros((0, 4)).into_pyarray(py).to_dyn());
    }
    
    // Parse hull indices
    let mut hull = Vec::with_capacity(m);
    for i in 0..m {
        let idx = if hull_arr.ndim() == 2 {
            hull_arr[[i, 0]] as usize
        } else {
            hull_arr[[i]] as usize
        };
        hull.push(idx);
    }
    
    let get_pt = |i: usize| -> (f64, f64) {
        if ndim == 2 {
            (arr[[i, 0]] as f64, arr[[i, 1]] as f64)
        } else {
            (arr[[i, 0, 0]] as f64, arr[[i, 0, 1]] as f64)
        }
    };
    
    let mut defects = Vec::new();
    
    for i in 0..m {
        let start = hull[i];
        let end = hull[(i + 1) % m];
        
        let p_start = get_pt(start);
        let p_end = get_pt(end);
        let dx = p_end.0 - p_start.0;
        let dy = p_end.1 - p_start.1;
        let len_sq = dx * dx + dy * dy;
        let len = len_sq.sqrt();
        
        if len < 1e-9 {
            continue;
        }
        
        let mut max_dist = 0.0;
        let mut far_idx = start;
        
        // Walk from start to end along contour
        let mut idx = (start + 1) % n;
        while idx != end {
            let p = get_pt(idx);
            let dist = ((dy * p.0 - dx * p.1 + p_end.0 * p_start.1 - p_end.1 * p_start.0) / len).abs();
            if dist > max_dist {
                max_dist = dist;
                far_idx = idx;
            }
            idx = (idx + 1) % n;
        }
        
        if max_dist > 1.0 {
            let depth_scaled = (max_dist * 256.0).round() as i32;
            defects.push([start as i32, end as i32, far_idx as i32, depth_scaled]);
        }
    }
    
    let k = defects.len();
    let mut out = Array2::<i32>::zeros((k, 4));
    for i in 0..k {
        out[[i, 0]] = defects[i][0];
        out[[i, 1]] = defects[i][1];
        out[[i, 2]] = defects[i][2];
        out[[i, 3]] = defects[i][3];
    }
    
    Ok(out.into_pyarray(py).to_dyn())
}

/// approx_poly_dp() - Approximate a polygonal curve with specified precision.
/// @py: Python interpreter token.
/// @curve: Input curve array of shape (N, 2) or (N, 1, 2).
/// @epsilon: Parameter specifying the approximation accuracy.
/// @closed: If true, the approximated curve is closed.
///
/// Approximates a polygonal curve with specified precision using the Douglas-Peucker algorithm.
///
/// Return: The approximated curve array.
#[pyfunction]
#[pyo3(signature = (curve, epsilon, closed = true))]
pub fn approx_poly_dp<'py>(
    py: Python<'py>,
    curve: PyReadonlyArrayDyn<'py, i32>,
    epsilon: f64,
    closed: bool,
) -> PyResult<&'py PyArrayDyn<i32>> {
    let arr = curve.as_array();
    let ndim = arr.ndim();
    let n = arr.shape()[0];
    if n < 3 {
        return Ok(curve.as_array().to_owned().into_pyarray(py).to_dyn());
    }
    
    let get_pt = |i: usize| -> (f64, f64) {
        if ndim == 2 {
            (arr[[i, 0]] as f64, arr[[i, 1]] as f64)
        } else {
            (arr[[i, 0, 0]] as f64, arr[[i, 0, 1]] as f64)
        }
    };
    
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        pts.push(get_pt(i));
    }
    
    let mut keep = vec![false; n];
    keep[0] = true;
    
    if closed {
        let mut loop_pts = pts.clone();
        loop_pts.push(pts[0]);
        let mut loop_keep = vec![false; n + 1];
        loop_keep[0] = true;
        loop_keep[n] = true;
        
        rdp(&loop_pts, 0, n, epsilon, &mut loop_keep);
        
        for i in 0..n {
            if loop_keep[i] {
                keep[i] = true;
            }
        }
    } else {
        keep[n - 1] = true;
        rdp(&pts, 0, n - 1, epsilon, &mut keep);
    }
    
    let mut indices = Vec::new();
    for i in 0..n {
        if keep[i] {
            indices.push(i);
        }
    }
    
    let m = indices.len();
    let mut out = Array2::<i32>::zeros((m, 2));
    for i in 0..m {
        let idx = indices[i];
        if ndim == 2 {
            out[[i, 0]] = arr[[idx, 0]];
            out[[i, 1]] = arr[[idx, 1]];
        } else {
            out[[i, 0]] = arr[[idx, 0, 0]];
            out[[i, 1]] = arr[[idx, 0, 1]];
        }
    }
    
    Ok(out.into_pyarray(py).to_dyn())
}

fn convex_hull_indexed(pts: &[(f64, f64)]) -> Vec<usize> {
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }
    
    let mut ip: Vec<usize> = (0..n).collect();
    ip.sort_unstable_by(|&i, &j| {
        pts[i].0.partial_cmp(&pts[j].0).unwrap_or(std::cmp::Ordering::Equal)
            .then(pts[i].1.partial_cmp(&pts[j].1).unwrap_or(std::cmp::Ordering::Equal))
    });
    
    let cross = |o: usize, a: usize, b: usize| -> f64 {
        (pts[a].0 - pts[o].0) * (pts[b].1 - pts[o].1) - (pts[a].1 - pts[o].1) * (pts[b].0 - pts[o].0)
    };
    
    let mut lower = Vec::new();
    for &p in &ip {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    
    let mut upper = Vec::new();
    for &p in ip.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    
    if lower.len() > 1 { lower.pop(); }
    if upper.len() > 1 { upper.pop(); }
    lower.extend(upper);
    lower
}

fn rdp(pts: &[(f64, f64)], start: usize, end: usize, epsilon: f64, keep: &mut [bool]) {
    if end <= start + 1 {
        return;
    }
    
    let p1 = pts[start];
    let p2 = pts[end];
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let len_sq = dx * dx + dy * dy;
    
    let mut max_dist_sq = 0.0;
    let mut split_idx = start;
    
    for i in (start + 1)..end {
        let p = pts[i];
        let dist_sq = if len_sq < 1e-9 {
            (p.0 - p1.0).powi(2) + (p.1 - p1.1).powi(2)
        } else {
            let t = ((p.0 - p1.0) * dx + (p.1 - p1.1) * dy) / len_sq;
            let t = t.clamp(0.0, 1.0);
            let proj_x = p1.0 + t * dx;
            let proj_y = p1.1 + t * dy;
            (p.0 - proj_x).powi(2) + (p.1 - proj_y).powi(2)
        };
        
        if dist_sq > max_dist_sq {
            max_dist_sq = dist_sq;
            split_idx = i;
        }
    }
    
    if max_dist_sq > epsilon * epsilon {
        keep[split_idx] = true;
        rdp(pts, start, split_idx, epsilon, keep);
        rdp(pts, split_idx, end, epsilon, keep);
    }
}

// ==========================================
// IMAGE MOMENTS & HU MOMENTS
// ==========================================

struct SpatialMoments {
    m00: f64,
    m10: f64,
    m01: f64,
    m20: f64,
    m11: f64,
    m02: f64,
    m30: f64,
    m21: f64,
    m12: f64,
    m03: f64,
}

fn compute_polygon_moments(pts: &[(f64, f64)]) -> SpatialMoments {
    let n = pts.len();
    let mut m = SpatialMoments {
        m00: 0.0, m10: 0.0, m01: 0.0,
        m20: 0.0, m11: 0.0, m02: 0.0,
        m30: 0.0, m21: 0.0, m12: 0.0, m03: 0.0,
    };
    if n < 3 {
        return m;
    }
    
    for i in 0..n {
        let p1 = pts[i];
        let p2 = pts[(i + 1) % n];
        let a = p1.0 * p2.1 - p2.0 * p1.1;
        
        m.m00 += a;
        m.m10 += a * (p1.0 + p2.0);
        m.m01 += a * (p1.1 + p2.1);
        
        m.m20 += a * (p1.0 * p1.0 + p1.0 * p2.0 + p2.0 * p2.0);
        m.m02 += a * (p1.1 * p1.1 + p1.1 * p2.1 + p2.1 * p2.1);
        m.m11 += a * (2.0 * p1.0 * p1.1 + p1.0 * p2.1 + p2.0 * p1.1 + 2.0 * p2.0 * p2.1);
        
        m.m30 += a * (p1.0 * p1.0 * p1.0 + p1.0 * p1.0 * p2.0 + p1.0 * p2.0 * p2.0 + p2.0 * p2.0 * p2.0);
        m.m03 += a * (p1.1 * p1.1 * p1.1 + p1.1 * p1.1 * p2.1 + p1.1 * p2.1 * p2.1 + p2.1 * p2.1 * p2.1);
        
        m.m21 += a * (3.0 * p1.0 * p1.0 * p1.1 + p1.0 * p1.0 * p2.1 + 2.0 * p1.0 * p2.0 * p1.1 + 2.0 * p1.0 * p2.0 * p2.1 + p2.0 * p2.0 * p1.1 + 3.0 * p2.0 * p2.0 * p2.1);
        m.m12 += a * (3.0 * p1.1 * p1.1 * p1.0 + p1.1 * p1.1 * p2.0 + 2.0 * p1.1 * p2.1 * p1.0 + 2.0 * p1.1 * p2.1 * p2.0 + p2.1 * p2.1 * p1.0 + 3.0 * p2.1 * p2.1 * p2.0);
    }
    
    m.m00 /= 2.0;
    m.m10 /= 6.0;
    m.m01 /= 6.0;
    m.m20 /= 12.0;
    m.m02 /= 12.0;
    m.m11 /= 24.0;
    m.m30 /= 20.0;
    m.m03 /= 20.0;
    m.m21 /= 60.0;
    m.m12 /= 60.0;
    
    m
}

fn compute_raster_moments(image: &ArrayView2<u8>) -> SpatialMoments {
    let (h, w) = (image.shape()[0], image.shape()[1]);
    let mut m = SpatialMoments {
        m00: 0.0, m10: 0.0, m01: 0.0,
        m20: 0.0, m11: 0.0, m02: 0.0,
        m30: 0.0, m21: 0.0, m12: 0.0, m03: 0.0,
    };
    
    for y in 0..h {
        let y_f = y as f64;
        let y2 = y_f * y_f;
        let y3 = y2 * y_f;
        for x in 0..w {
            let val = image[[y, x]] as f64;
            if val == 0.0 {
                continue;
            }
            let x_f = x as f64;
            let x2 = x_f * x_f;
            let x3 = x2 * x_f;
            
            m.m00 += val;
            m.m10 += val * x_f;
            m.m01 += val * y_f;
            m.m20 += val * x2;
            m.m11 += val * x_f * y_f;
            m.m02 += val * y2;
            m.m30 += val * x3;
            m.m21 += val * x2 * y_f;
            m.m12 += val * x_f * y2;
            m.m03 += val * y3;
        }
    }
    m
}

fn build_moments_dict(py: Python<'_>, m: SpatialMoments) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new_bound(py);
    
    dict.set_item("m00", m.m00)?;
    dict.set_item("m10", m.m10)?;
    dict.set_item("m01", m.m01)?;
    dict.set_item("m20", m.m20)?;
    dict.set_item("m11", m.m11)?;
    dict.set_item("m02", m.m02)?;
    dict.set_item("m30", m.m30)?;
    dict.set_item("m21", m.m21)?;
    dict.set_item("m12", m.m12)?;
    dict.set_item("m03", m.m03)?;
    
    let (cx, cy) = if m.m00.abs() > 1e-9 {
        (m.m10 / m.m00, m.m01 / m.m00)
    } else {
        (0.0, 0.0)
    };
    
    let mu20 = m.m20 - cx * m.m10;
    let mu11 = m.m11 - cx * m.m01;
    let mu02 = m.m02 - cy * m.m01;
    
    let mu30 = m.m30 - 3.0 * cx * m.m20 + 2.0 * cx * cx * m.m10;
    let mu21 = m.m21 - 2.0 * cx * m.m11 - cy * m.m20 + 2.0 * cx * cx * m.m01;
    let mu12 = m.m12 - 2.0 * cy * m.m11 - cx * m.m02 + 2.0 * cy * cy * m.m10;
    let mu03 = m.m03 - 3.0 * cy * m.m02 + 2.0 * cy * cy * m.m01;
    
    dict.set_item("mu20", mu20)?;
    dict.set_item("mu11", mu11)?;
    dict.set_item("mu02", mu02)?;
    dict.set_item("mu30", mu30)?;
    dict.set_item("mu21", mu21)?;
    dict.set_item("mu12", mu12)?;
    dict.set_item("mu03", mu03)?;
    
    let get_nu = |mu: f64, p: f64, q: f64| -> f64 {
        if m.m00.abs() > 1e-9 {
            let power = (p + q) / 2.0 + 1.0;
            mu / m.m00.abs().powf(power)
        } else {
            0.0
        }
    };
    
    dict.set_item("nu20", get_nu(mu20, 2.0, 0.0))?;
    dict.set_item("nu11", get_nu(mu11, 1.0, 1.0))?;
    dict.set_item("nu02", get_nu(mu02, 0.0, 2.0))?;
    dict.set_item("nu30", get_nu(mu30, 3.0, 0.0))?;
    dict.set_item("nu21", get_nu(mu21, 2.0, 1.0))?;
    dict.set_item("nu12", get_nu(mu12, 1.0, 2.0))?;
    dict.set_item("nu03", get_nu(mu03, 0.0, 3.0))?;
    
    Ok(dict.unbind().into())
}

fn contour_moments_internal(contour: &PyReadonlyArrayDyn<i32>) -> PyResult<SpatialMoments> {
    let arr = contour.as_array();
    let ndim = arr.ndim();
    let n = arr.shape()[0];
    
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let (px, py_val) = if ndim == 2 {
            if arr.shape()[1] != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err("Contour array must have shape (N, 2) or (N, 1, 2)"));
            }
            (arr[[i, 0]] as f64, arr[[i, 1]] as f64)
        } else if ndim == 3 {
            if arr.shape()[1] != 1 || arr.shape()[2] != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err("Contour array must have shape (N, 2) or (N, 1, 2)"));
            }
            (arr[[i, 0, 0]] as f64, arr[[i, 0, 1]] as f64)
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err("Contour array must have shape (N, 2) or (N, 1, 2)"));
        };
        pts.push((px, py_val));
    }
    Ok(compute_polygon_moments(&pts))
}

fn hu_moments_internal(m: SpatialMoments) -> Vec<f64> {
    let (cx, cy) = if m.m00.abs() > 1e-9 {
        (m.m10 / m.m00, m.m01 / m.m00)
    } else {
        (0.0, 0.0)
    };
    
    let mu20 = m.m20 - cx * m.m10;
    let mu11 = m.m11 - cx * m.m01;
    let mu02 = m.m02 - cy * m.m01;
    
    let mu30 = m.m30 - 3.0 * cx * m.m20 + 2.0 * cx * cx * m.m10;
    let mu21 = m.m21 - 2.0 * cx * m.m11 - cy * m.m20 + 2.0 * cx * cx * m.m01;
    let mu12 = m.m12 - 2.0 * cy * m.m11 - cx * m.m02 + 2.0 * cy * cy * m.m10;
    let mu03 = m.m03 - 3.0 * cy * m.m02 + 2.0 * cy * cy * m.m01;
    
    let get_nu = |mu: f64, p: f64, q: f64| -> f64 {
        if m.m00.abs() > 1e-9 {
            let power = (p + q) / 2.0 + 1.0;
            mu / m.m00.abs().powf(power)
        } else {
            0.0
        }
    };
    
    let n20 = get_nu(mu20, 2.0, 0.0);
    let n11 = get_nu(mu11, 1.0, 1.0);
    let n02 = get_nu(mu02, 0.0, 2.0);
    let n30 = get_nu(mu30, 3.0, 0.0);
    let n21 = get_nu(mu21, 2.0, 1.0);
    let n12 = get_nu(mu12, 1.0, 2.0);
    let n03 = get_nu(mu03, 0.0, 3.0);
    
    let h1 = n20 + n02;
    let h2 = (n20 - n02).powi(2) + 4.0 * n11 * n11;
    let h3 = (n30 - 3.0 * n12).powi(2) + (3.0 * n21 - n03).powi(2);
    let h4 = (n30 + n12).powi(2) + (n21 + n03).powi(2);
    
    let t1 = n30 + n12;
    let t2 = n21 + n03;
    let h5 = (n30 - 3.0 * n12) * t1 * (t1 * t1 - 3.0 * t2 * t2)
        + (3.0 * n21 - n03) * t2 * (3.0 * t1 * t1 - t2 * t2);
        
    let h6 = (n20 - n02) * (t1 * t1 - t2 * t2) + 4.0 * n11 * t1 * t2;
    
    let h7 = (3.0 * n21 - n03) * t1 * (t1 * t1 - 3.0 * t2 * t2)
        - (n30 - 3.0 * n12) * t2 * (3.0 * t1 * t1 - t2 * t2);
        
    vec![h1, h2, h3, h4, h5, h6, h7]
}

/// moments() - Compute spatial, central, and normalized moments.
/// @py: Python interpreter token.
/// @x: Input grayscale image (u8) or 2D/3D contour (i32).
///
/// Computes spatial, central, and normalized moments of a grayscale image or a 2D contour.
///
/// Return: A python dictionary with moment keys (e.g. m00, m10, mu20, nu11, etc.).
#[pyfunction]
pub fn moments<'py>(py: Python<'py>, x: PyObject) -> PyResult<PyObject> {
    if let Ok(arr_u8) = x.extract::<PyReadonlyArrayDyn<u8>>(py) {
        let arr = arr_u8.as_array();
        let ndim = arr.ndim();
        if ndim == 2 {
            let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>().unwrap();
            let m = compute_raster_moments(&channel.view());
            return build_moments_dict(py, m);
        }
    }
    
    if let Ok(arr_i32) = x.extract::<PyReadonlyArrayDyn<i32>>(py) {
        let m = contour_moments_internal(&arr_i32)?;
        return build_moments_dict(py, m);
    }
    
    Err(pyo3::exceptions::PyValueError::new_err(
        "Input must be a 2D u8 image or a 2D/3D i32 contour"
    ))
}

/// hu_moments() - Compute Hu invariant moments of a shape from its moments.
/// @moments_dict: Python dictionary of moments (nu20, nu11, nu02, nu30, nu21, nu12, nu03).
///
/// Computes the 7 Hu invariant moments of a shape, which are invariant to translation, scale, and rotation.
///
/// Return: A vector of 7 Hu invariant moments.
#[pyfunction]
pub fn hu_moments(moments_dict: &pyo3::types::PyDict) -> PyResult<Vec<f64>> {
    let get_val = |key: &str| -> f64 {
        moments_dict.get_item(key).ok().flatten()
            .and_then(|v| v.extract::<f64>().ok())
            .unwrap_or(0.0)
    };
    
    let n20 = get_val("nu20");
    let n11 = get_val("nu11");
    let n02 = get_val("nu02");
    let n30 = get_val("nu30");
    let n21 = get_val("nu21");
    let n12 = get_val("nu12");
    let n03 = get_val("nu03");
    
    let h1 = n20 + n02;
    let h2 = (n20 - n02).powi(2) + 4.0 * n11 * n11;
    let h3 = (n30 - 3.0 * n12).powi(2) + (3.0 * n21 - n03).powi(2);
    let h4 = (n30 + n12).powi(2) + (n21 + n03).powi(2);
    
    let t1 = n30 + n12;
    let t2 = n21 + n03;
    let h5 = (n30 - 3.0 * n12) * t1 * (t1 * t1 - 3.0 * t2 * t2)
        + (3.0 * n21 - n03) * t2 * (3.0 * t1 * t1 - t2 * t2);
        
    let h6 = (n20 - n02) * (t1 * t1 - t2 * t2) + 4.0 * n11 * t1 * t2;
    
    let h7 = (3.0 * n21 - n03) * t1 * (t1 * t1 - 3.0 * t2 * t2)
        - (n30 - 3.0 * n12) * t2 * (3.0 * t1 * t1 - t2 * t2);
        
    Ok(vec![h1, h2, h3, h4, h5, h6, h7])
}

/// match_shapes() - Compare two contour shapes using Hu moments.
/// @contour1: First contour array.
/// @contour2: Second contour array.
/// @method: Comparison method (1, 2, or 3).
///
/// Compares two contour shapes using Hu moments.
///
/// Return: A similarity score (lower is more similar).
#[pyfunction]
#[pyo3(signature = (contour1, contour2, method = 1))]
pub fn match_shapes(
    contour1: PyReadonlyArrayDyn<i32>,
    contour2: PyReadonlyArrayDyn<i32>,
    method: i32,
) -> PyResult<f64> {
    let m1 = contour_moments_internal(&contour1)?;
    let m2 = contour_moments_internal(&contour2)?;
    
    let hu1 = hu_moments_internal(m1);
    let hu2 = hu_moments_internal(m2);
    
    let mut score = 0.0;
    
    let get_m = |h: f64| -> f64 {
        if h.abs() < 1e-20 {
            0.0
        } else {
            h.signum() * h.abs().log10()
        }
    };
    
    let m_a: Vec<f64> = hu1.iter().map(|&h| get_m(h)).collect();
    let m_b: Vec<f64> = hu2.iter().map(|&h| get_m(h)).collect();
    
    match method {
        1 => {
            for i in 0..7 {
                let ma_val = m_a[i];
                let mb_val = m_b[i];
                if ma_val.abs() > 1e-9 && mb_val.abs() > 1e-9 {
                    score += (1.0 / ma_val - 1.0 / mb_val).abs();
                }
            }
        }
        2 => {
            for i in 0..7 {
                score += (m_a[i] - m_b[i]).abs();
            }
        }
        3 => {
            let mut max_val = 0.0;
            for i in 0..7 {
                let ma_val = m_a[i];
                let mb_val = m_b[i];
                if ma_val.abs() > 1e-9 {
                    let val = (ma_val - mb_val).abs() / ma_val.abs();
                    if val > max_val {
                        max_val = val;
                    }
                }
            }
            score = max_val;
        }
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown match method. Must be 1, 2, or 3.")),
    }
    
    Ok(score)
}

/// is_contour_convex() - Test if a contour is convex.
/// @contour: Input contour array.
///
/// Tests if the input contour is convex by checking the sign of the cross products of consecutive edges.
///
/// Return: True if the contour is convex, False otherwise.
#[pyfunction]
pub fn is_contour_convex(contour: PyReadonlyArrayDyn<i32>) -> PyResult<bool> {
    let arr = contour.as_array();
    let ndim = arr.ndim();
    let n = arr.shape()[0];
    if n < 3 {
        return Ok(true);
    }
    
    let get_pt = |i: usize| -> (f64, f64) {
        if ndim == 2 {
            (arr[[i, 0]] as f64, arr[[i, 1]] as f64)
        } else {
            (arr[[i, 0, 0]] as f64, arr[[i, 0, 1]] as f64)
        }
    };
    
    let mut sign = 0.0;
    
    for i in 0..n {
        let a = get_pt(i);
        let b = get_pt((i + 1) % n);
        let c = get_pt((i + 2) % n);
        
        let cross = (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0);
        
        if cross != 0.0 {
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return Ok(false);
            }
        }
    }
    
    Ok(true)
}

/// point_polygon_test() - Test if a point is inside, outside, or on the boundary of a contour.
/// @contour: Input contour array.
/// @pt: The 2D point (x, y).
/// @measure_dist: If true, returns signed Euclidean distance. If false, returns +1.0, -1.0, or 0.0.
///
/// Tests if a point is inside, outside, or on the boundary of a contour.
///
/// Return: The result of the test (distance or indicator float).
#[pyfunction]
pub fn point_polygon_test(
    contour: PyReadonlyArrayDyn<i32>,
    pt: (f64, f64),
    measure_dist: bool,
) -> PyResult<f64> {
    let arr = contour.as_array();
    let ndim = arr.ndim();
    let n = arr.shape()[0];
    if n == 0 {
        return Ok(-1.0);
    }
    
    let get_pt = |i: usize| -> (f64, f64) {
        if ndim == 2 {
            (arr[[i, 0]] as f64, arr[[i, 1]] as f64)
        } else {
            (arr[[i, 0, 0]] as f64, arr[[i, 0, 1]] as f64)
        }
    };
    
    let mut min_dist_sq = f64::MAX;
    let mut on_boundary = false;
    
    for i in 0..n {
        let p1 = get_pt(i);
        let p2 = get_pt((i + 1) % n);
        
        let dx = p2.0 - p1.0;
        let dy = p2.1 - p1.1;
        let len_sq = dx * dx + dy * dy;
        
        let dist_sq = if len_sq < 1e-9 {
            (pt.0 - p1.0).powi(2) + (pt.1 - p1.1).powi(2)
        } else {
            let t = ((pt.0 - p1.0) * dx + (pt.1 - p1.1) * dy) / len_sq;
            let t = t.clamp(0.0, 1.0);
            let proj_x = p1.0 + t * dx;
            let proj_y = p1.1 + t * dy;
            (pt.0 - proj_x).powi(2) + (pt.1 - proj_y).powi(2)
        };
        
        if dist_sq < min_dist_sq {
            min_dist_sq = dist_sq;
        }
        
        if dist_sq < 1e-9 {
            on_boundary = true;
        }
    }
    
    if on_boundary {
        return Ok(0.0);
    }
    
    let mut inside = false;
    for i in 0..n {
        let p1 = get_pt(i);
        let p2 = get_pt((i + 1) % n);
        
        if ((p1.1 > pt.1) != (p2.1 > pt.1)) &&
           (pt.0 < (p2.0 - p1.0) * (pt.1 - p1.1) / (p2.1 - p1.1) + p1.0) {
            inside = !inside;
        }
    }
    
    if measure_dist {
        let min_dist = min_dist_sq.sqrt();
        if inside {
            Ok(min_dist)
        } else {
            Ok(-min_dist)
        }
    } else {
        if inside {
            Ok(1.0)
        } else {
            Ok(-1.0)
        }
    }
}
