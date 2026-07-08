use pyo3::prelude::*;
use numpy::{
    ndarray::{Array2, ArrayViewMutD},
    IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn,
};
use crate::contours::{draw_line_bresenham, fill_polygon};

// Helper to parse color
fn parse_color(py: Python<'_>, color: PyObject, channels: usize) -> Vec<u8> {
    let mut parsed_color = vec![0u8; channels];
    if let Ok(val) = color.extract::<u8>(py) {
        for c in 0..channels {
            parsed_color[c] = val;
        }
    } else if let Ok(list) = color.extract::<Vec<u8>>(py) {
        for c in 0..channels.min(list.len()) {
            parsed_color[c] = list[c];
        }
    } else if let Ok(tuple) = color.extract::<(u8, u8, u8)>(py) {
        if channels >= 3 {
            parsed_color[0] = tuple.0;
            parsed_color[1] = tuple.1;
            parsed_color[2] = tuple.2;
        } else {
            parsed_color[0] = tuple.0;
        }
    }
    parsed_color
}

/// Draw a line segment on an image.
///
/// Returns a new annotated image.
/// - `pt1`: Starting point (x, y) as integer tuple.
/// - `pt2`: Ending point (x, y) as integer tuple.
/// - `color`: color value, either single int or (R, G, B) tuple.
/// - `thickness`: line thickness (default 1).
#[pyfunction]
#[pyo3(signature = (img, pt1, pt2, color, thickness = 1))]
pub fn line<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    pt1: (i32, i32),
    pt2: (i32, i32),
    color: PyObject,
    thickness: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = img.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();
    
    if ndim != 2 && ndim != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"));
    }
    
    let mut out_arr = arr.to_owned();
    let channels = if ndim == 3 { shape[2] } else { 1 };
    let parsed_color = parse_color(py, color, channels);
    
    let mut out_view = out_arr.view_mut().into_dyn();
    draw_line_bresenham(&mut out_view, pt1.0, pt1.1, pt2.0, pt2.1, &parsed_color, thickness);
    
    Ok(out_arr.into_pyarray(py).to_dyn())
}

/// Draw a rectangle on an image.
///
/// Returns a new annotated image.
/// - `pt1`: One corner (x, y) of the rectangle.
/// - `pt2`: Opposite corner (x, y) of the rectangle.
/// - `color`: color value, either single int or (R, G, B) tuple.
/// - `thickness`: line thickness. If negative (e.g. -1), the rectangle is filled.
#[pyfunction]
#[pyo3(signature = (img, pt1, pt2, color, thickness = 1))]
pub fn rectangle<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    pt1: (i32, i32),
    pt2: (i32, i32),
    color: PyObject,
    thickness: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = img.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();
    
    if ndim != 2 && ndim != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"));
    }
    
    let mut out_arr = arr.to_owned();
    let (h, w) = (shape[0] as i32, shape[1] as i32);
    let channels = if ndim == 3 { shape[2] } else { 1 };
    let parsed_color = parse_color(py, color, channels);
    
    let x1 = pt1.0;
    let y1 = pt1.1;
    let x2 = pt2.0;
    let y2 = pt2.1;
    
    let mut out_view = out_arr.view_mut().into_dyn();
    
    if thickness < 0 {
        let start_y = y1.min(y2).clamp(0, h - 1);
        let end_y = y1.max(y2).clamp(0, h - 1);
        let start_x = x1.min(x2).clamp(0, w - 1);
        let end_x = x1.max(x2).clamp(0, w - 1);
        
        for y in start_y..=end_y {
            for x in start_x..=end_x {
                if ndim == 3 {
                    for c in 0..channels {
                        out_arr[[y as usize, x as usize, c]] = parsed_color[c];
                    }
                } else {
                    out_arr[[y as usize, x as usize]] = parsed_color[0];
                }
            }
        }
    } else {
        // Draw 4 border lines
        draw_line_bresenham(&mut out_view, x1, y1, x2, y1, &parsed_color, thickness);
        draw_line_bresenham(&mut out_view, x2, y1, x2, y2, &parsed_color, thickness);
        draw_line_bresenham(&mut out_view, x2, y2, x1, y2, &parsed_color, thickness);
        draw_line_bresenham(&mut out_view, x1, y2, x1, y1, &parsed_color, thickness);
    }
    
    Ok(out_arr.into_pyarray(py).to_dyn())
}

/// Draw a circle on an image.
///
/// Returns a new annotated image.
/// - `center`: Center coordinates (x, y) as integer tuple.
/// - `radius`: Circle radius.
/// - `color`: color value, either single int or (R, G, B) tuple.
/// - `thickness`: line thickness. If negative (e.g. -1), the circle is filled.
#[pyfunction]
#[pyo3(signature = (img, center, radius, color, thickness = 1))]
pub fn circle<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    center: (i32, i32),
    radius: i32,
    color: PyObject,
    thickness: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = img.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();
    
    if ndim != 2 && ndim != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"));
    }
    
    let mut out_arr = arr.to_owned();
    let (h, w) = (shape[0] as i32, shape[1] as i32);
    let channels = if ndim == 3 { shape[2] } else { 1 };
    let parsed_color = parse_color(py, color, channels);
    
    let cx = center.0;
    let cy = center.1;
    let mut out_view = out_arr.view_mut().into_dyn();
    
    if thickness < 0 {
        // Filled circle
        let start_y = (cy - radius).max(0);
        let end_y = (cy + radius).min(h - 1);
        
        for y in start_y..=end_y {
            let dy = y - cy;
            let dx = ((radius * radius - dy * dy) as f64).sqrt().round() as i32;
            let start_x = (cx - dx).max(0);
            let end_x = (cx + dx).min(w - 1);
            
            for x in start_x..=end_x {
                if ndim == 3 {
                    for c in 0..channels {
                        out_arr[[y as usize, x as usize, c]] = parsed_color[c];
                    }
                } else {
                    out_arr[[y as usize, x as usize]] = parsed_color[0];
                }
            }
        }
    } else {
        // Midpoint circle algorithm for outline
        let mut x = 0;
        let mut y = radius;
        let mut d = 3 - 2 * radius;
        
        let mut draw_pixel = |view: &mut ArrayViewMutD<u8>, px: i32, py: i32| {
            let t = thickness.max(1);
            let brush_r = t / 2;
            for dy_t in -brush_r..=brush_r {
                for dx_t in -brush_r..=brush_r {
                    let cur_x = px + dx_t;
                    let cur_y = py + dy_t;
                    if cur_x >= 0 && cur_x < w && cur_y >= 0 && cur_y < h {
                        if ndim == 3 {
                            for c in 0..channels {
                                view[[cur_y as usize, cur_x as usize, c]] = parsed_color[c];
                            }
                        } else {
                            view[[cur_y as usize, cur_x as usize]] = parsed_color[0];
                        }
                    }
                }
            }
        };
        
        let mut draw_8_symmetry = |view: &mut ArrayViewMutD<u8>, px: i32, py: i32| {
            draw_pixel(view, cx + px, cy + py);
            draw_pixel(view, cx - px, cy + py);
            draw_pixel(view, cx + px, cy - py);
            draw_pixel(view, cx - px, cy - py);
            draw_pixel(view, cx + py, cy + px);
            draw_pixel(view, cx - py, cy + px);
            draw_pixel(view, cx + py, cy - px);
            draw_pixel(view, cx - py, cy - px);
        };
        
        draw_8_symmetry(&mut out_view, x, y);
        while y >= x {
            x += 1;
            if d > 0 {
                y -= 1;
                d = d + 4 * (x - y) + 10;
            } else {
                d = d + 4 * x + 6;
            }
            draw_8_symmetry(&mut out_view, x, y);
        }
    }
    
    Ok(out_arr.into_pyarray(py).to_dyn())
}

/// Draw an ellipse or elliptic arc on an image.
///
/// Returns a new annotated image.
/// - `center`: Center coordinates (x, y) as integer tuple.
/// - `axes`: Semi-axes lengths (major_axis_radius, minor_axis_radius) as integer tuple.
/// - `angle`: Ellipse rotation angle in degrees.
/// - `start_angle`: Starting angle of the elliptic arc in degrees.
/// - `end_angle`: Ending angle of the elliptic arc in degrees.
/// - `color`: color value, either single int or (R, G, B) tuple.
/// - `thickness`: line thickness. If negative (e.g. -1), the ellipse is filled.
#[pyfunction]
#[pyo3(signature = (img, center, axes, angle, start_angle, end_angle, color, thickness = 1))]
pub fn ellipse<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    center: (i32, i32),
    axes: (i32, i32),
    angle: f64,
    start_angle: f64,
    end_angle: f64,
    color: PyObject,
    thickness: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = img.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();
    
    if ndim != 2 && ndim != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D"));
    }
    
    let mut out_arr = arr.to_owned();
    let channels = if ndim == 3 { shape[2] } else { 1 };
    let parsed_color = parse_color(py, color, channels);
    
    let cx = center.0 as f64;
    let cy = center.1 as f64;
    let a = axes.0 as f64;
    let b = axes.1 as f64;
    
    let angle_rad = angle.to_radians();
    let cos_angle = angle_rad.cos();
    let sin_angle = angle_rad.sin();
    
    let mut diff = end_angle - start_angle;
    while diff < 0.0 {
        diff += 360.0;
    }
    
    let steps = (diff.abs().round() as usize).max(4);
    let mut pts = Vec::with_capacity(steps + 1);
    
    for i in 0..=steps {
        let alpha = start_angle + (i as f64 / steps as f64) * diff;
        let alpha_rad = alpha.to_radians();
        
        let xp = a * alpha_rad.cos();
        let yp = b * alpha_rad.sin();
        
        let x = cx + xp * cos_angle - yp * sin_angle;
        let y = cy + xp * sin_angle + yp * cos_angle;
        
        pts.push((x.round() as i32, y.round() as i32));
    }
    
    let mut out_view = out_arr.view_mut().into_dyn();
    
    if thickness < 0 {
        // If it's a partial sector, connect end-point to start-point through the center
        if diff < 360.0 {
            pts.push((cx.round() as i32, cy.round() as i32));
        }
        fill_polygon(&mut out_view, &pts, &parsed_color);
    } else {
        let n = pts.len();
        // Draw the arc segments
        for i in 0..(n - 1) {
            draw_line_bresenham(&mut out_view, pts[i].0, pts[i].1, pts[i + 1].0, pts[i + 1].1, &parsed_color, thickness);
        }
        // If it is a full closed ellipse, connect back to start
        if diff >= 360.0 {
            draw_line_bresenham(&mut out_view, pts[n - 1].0, pts[n - 1].1, pts[0].0, pts[0].1, &parsed_color, thickness);
        }
    }
    
    Ok(out_arr.into_pyarray(py).to_dyn())
}
