use pyo3::prelude::*;
use numpy::{
    ndarray::{Array2, ArrayViewMutD},
    IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn,
};
use crate::contours::{draw_line_bresenham, fill_polygon};

const FONT_8X8: [[u8; 8]; 95] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // ' ' (32)
    [0x18, 0x3c, 0x3c, 0x18, 0x18, 0x00, 0x18, 0x00], // '!'
    [0x6c, 0x6c, 0x6c, 0x00, 0x00, 0x00, 0x00, 0x00], // '"'
    [0x24, 0x24, 0x7e, 0x24, 0x7e, 0x24, 0x24, 0x00], // '#'
    [0x18, 0x3e, 0x60, 0x3c, 0x06, 0x7c, 0x18, 0x00], // '$'
    [0x00, 0xc6, 0xcc, 0x18, 0x30, 0x66, 0xc3, 0x00], // '%'
    [0x38, 0x6c, 0x38, 0x76, 0xdc, 0xcc, 0x76, 0x00], // '&'
    [0x30, 0x30, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00], // '''
    [0x0c, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0c, 0x00], // '('
    [0x30, 0x18, 0x0c, 0x0c, 0x0c, 0x18, 0x30, 0x00], // ')'
    [0x00, 0x66, 0x3c, 0xff, 0x3c, 0x66, 0x00, 0x00], // '*'
    [0x00, 0x18, 0x18, 0x7e, 0x18, 0x18, 0x00, 0x00], // '+'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30], // ','
    [0x00, 0x00, 0x00, 0x7e, 0x00, 0x00, 0x00, 0x00], // '-'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00], // '.'
    [0x06, 0x0c, 0x18, 0x30, 0x60, 0xc0, 0x80, 0x00], // '/'
    [0x3c, 0x66, 0x6e, 0x76, 0x66, 0x66, 0x3c, 0x00], // '0'
    [0x18, 0x1c, 0x18, 0x18, 0x18, 0x18, 0x7e, 0x00], // '1'
    [0x3e, 0x66, 0x06, 0x0c, 0x18, 0x30, 0x7e, 0x00], // '2'
    [0x3e, 0x66, 0x06, 0x1c, 0x06, 0x66, 0x3e, 0x00], // '3'
    [0x06, 0x0e, 0x1e, 0x36, 0x7f, 0x06, 0x06, 0x00], // '4'
    [0x7e, 0x60, 0x7c, 0x06, 0x06, 0x66, 0x3c, 0x00], // '5'
    [0x3c, 0x66, 0x60, 0x7c, 0x66, 0x66, 0x3c, 0x00], // '6'
    [0x7e, 0x66, 0x0c, 0x18, 0x30, 0x30, 0x30, 0x00], // '7'
    [0x3c, 0x66, 0x66, 0x3c, 0x66, 0x66, 0x3c, 0x00], // '8'
    [0x3c, 0x66, 0x66, 0x3e, 0x06, 0x66, 0x3c, 0x00], // '9'
    [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00], // ':'
    [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x30, 0x00], // ';'
    [0x0c, 0x18, 0x30, 0x60, 0x30, 0x18, 0x0c, 0x00], // '<'
    [0x00, 0x00, 0x7e, 0x00, 0x7e, 0x00, 0x00, 0x00], // '='
    [0x30, 0x18, 0x0c, 0x06, 0x0c, 0x18, 0x30, 0x00], // '>'
    [0x3c, 0x66, 0x06, 0x0c, 0x18, 0x00, 0x18, 0x00], // '?'
    [0x3c, 0x66, 0x6e, 0x6e, 0x60, 0x62, 0x3c, 0x00], // '@'
    [0x18, 0x3c, 0x66, 0x66, 0x7e, 0x66, 0x66, 0x00], // 'A'
    [0x7c, 0x66, 0x66, 0x7c, 0x66, 0x66, 0x7c, 0x00], // 'B'
    [0x3c, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3c, 0x00], // 'C'
    [0x78, 0x6c, 0x66, 0x66, 0x66, 0x6c, 0x78, 0x00], // 'D'
    [0x7e, 0x60, 0x60, 0x78, 0x60, 0x60, 0x7e, 0x00], // 'E'
    [0x7e, 0x60, 0x60, 0x78, 0x60, 0x60, 0x60, 0x00], // 'F'
    [0x3c, 0x66, 0x60, 0x6e, 0x66, 0x66, 0x3e, 0x00], // 'G'
    [0x66, 0x66, 0x66, 0x7e, 0x66, 0x66, 0x66, 0x00], // 'H'
    [0x3e, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3e, 0x00], // 'I'
    [0x1e, 0x0c, 0x0c, 0x0c, 0x0c, 0xcc, 0x78, 0x00], // 'J'
    [0x66, 0x6c, 0x78, 0x70, 0x78, 0x6c, 0x66, 0x00], // 'K'
    [0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7e, 0x00], // 'L'
    [0x63, 0x77, 0x7f, 0x6b, 0x63, 0x63, 0x63, 0x00], // 'M'
    [0x66, 0x6e, 0x7e, 0x76, 0x66, 0x66, 0x66, 0x00], // 'N'
    [0x3c, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3c, 0x00], // 'O'
    [0x7c, 0x66, 0x66, 0x7c, 0x60, 0x60, 0x60, 0x00], // 'P'
    [0x3c, 0x66, 0x66, 0x66, 0x6e, 0x6c, 0x3e, 0x00], // 'Q'
    [0x7c, 0x66, 0x66, 0x7c, 0x78, 0x6c, 0x66, 0x00], // 'R'
    [0x3e, 0x60, 0x60, 0x3c, 0x06, 0x06, 0x7c, 0x00], // 'S'
    [0x7e, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00], // 'T'
    [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3c, 0x00], // 'U'
    [0x66, 0x66, 0x66, 0x66, 0x66, 0x3c, 0x18, 0x00], // 'V'
    [0x63, 0x63, 0x63, 0x6b, 0x7f, 0x77, 0x63, 0x00], // 'W'
    [0x66, 0x66, 0x3c, 0x18, 0x3c, 0x66, 0x66, 0x00], // 'X'
    [0x66, 0x66, 0x66, 0x3c, 0x18, 0x18, 0x18, 0x00], // 'Y'
    [0x7e, 0x06, 0x0c, 0x18, 0x30, 0x60, 0x7e, 0x00], // 'Z'
    [0x3c, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3c, 0x00], // '['
    [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x00], // '\'
    [0x3c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x3c, 0x00], // ']'
    [0x18, 0x3c, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00], // '^'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x00], // '_'
    [0x30, 0x30, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00], // '`'
    [0x00, 0x00, 0x3c, 0x06, 0x3e, 0x66, 0x3e, 0x00], // 'a'
    [0x60, 0x60, 0x7c, 0x66, 0x66, 0x66, 0x7c, 0x00], // 'b'
    [0x00, 0x00, 0x3e, 0x60, 0x60, 0x66, 0x3e, 0x00], // 'c'
    [0x06, 0x06, 0x3e, 0x66, 0x66, 0x66, 0x3e, 0x00], // 'd'
    [0x00, 0x00, 0x3c, 0x66, 0x7e, 0x60, 0x3e, 0x00], // 'e'
    [0x1c, 0x30, 0x78, 0x30, 0x30, 0x30, 0x30, 0x00], // 'f'
    [0x00, 0x00, 0x3e, 0x66, 0x66, 0x3e, 0x06, 0x3c], // 'g'
    [0x60, 0x60, 0x7c, 0x66, 0x66, 0x66, 0x66, 0x00], // 'h'
    [0x18, 0x00, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00], // 'i'
    [0x0c, 0x00, 0x0c, 0x0c, 0x0c, 0x0c, 0xcc, 0x78], // 'j'
    [0x60, 0x60, 0x66, 0x6c, 0x78, 0x6c, 0x66, 0x00], // 'k'
    [0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x1c, 0x00], // 'l'
    [0x00, 0x00, 0x66, 0x7f, 0x6b, 0x63, 0x63, 0x00], // 'm'
    [0x00, 0x00, 0x7c, 0x66, 0x66, 0x66, 0x66, 0x00], // 'n'
    [0x00, 0x00, 0x3c, 0x66, 0x66, 0x66, 0x3c, 0x00], // 'o'
    [0x00, 0x00, 0x7c, 0x66, 0x66, 0x7c, 0x60, 0x60], // 'p'
    [0x00, 0x00, 0x3e, 0x66, 0x66, 0x3e, 0x06, 0x06], // 'q'
    [0x00, 0x00, 0x7c, 0x66, 0x60, 0x60, 0x60, 0x00], // 'r'
    [0x00, 0x00, 0x3e, 0x60, 0x3c, 0x06, 0x7c, 0x00], // 's'
    [0x30, 0x30, 0x7c, 0x30, 0x30, 0x30, 0x1c, 0x00], // 't'
    [0x00, 0x00, 0x66, 0x66, 0x66, 0x66, 0x3e, 0x00], // 'u'
    [0x00, 0x00, 0x66, 0x66, 0x66, 0x3c, 0x18, 0x00], // 'v'
    [0x00, 0x00, 0x63, 0x63, 0x6b, 0x7f, 0x36, 0x00], // 'w'
    [0x00, 0x00, 0x66, 0x3c, 0x18, 0x3c, 0x66, 0x00], // 'x'
    [0x00, 0x00, 0x66, 0x66, 0x66, 0x3e, 0x06, 0x3c], // 'y'
    [0x00, 0x00, 0x7e, 0x0c, 0x18, 0x30, 0x7e, 0x00], // 'z'
    [0x0c, 0x18, 0x18, 0x30, 0x18, 0x18, 0x0c, 0x00], // '{'
    [0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00], // '|'
    [0x30, 0x18, 0x18, 0x0c, 0x18, 0x18, 0x30, 0x00], // '}'
    [0x76, 0xdc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // '~'
];

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

/// line() - Draw a line segment on an image.
/// @py: Python interpreter token.
/// @img: Input image array (u8).
/// @pt1: Starting point (x, y).
/// @pt2: Ending point (x, y).
/// @color: Line color value (int or RGB tuple).
/// @thickness: Line thickness.
///
/// Draws a line segment between pt1 and pt2 using Bresenham's algorithm.
///
/// Return: Annotated image array.
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

/// rectangle() - Draw a rectangle on an image.
/// @py: Python interpreter token.
/// @img: Input image array (u8).
/// @pt1: One corner of the rectangle (x, y).
/// @pt2: Opposite corner of the rectangle (x, y).
/// @color: Rectangle color.
/// @thickness: Line thickness (if negative, the rectangle is filled).
///
/// Draws a simple, thick, or filled rectangle on the image.
///
/// Return: Annotated image array.
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

/// circle() - Draw a circle on an image.
/// @py: Python interpreter token.
/// @img: Input image array (u8).
/// @center: Center coordinates (x, y).
/// @radius: Circle radius.
/// @color: Circle color.
/// @thickness: Line thickness (if negative, the circle is filled).
///
/// Draws a simple or filled circle using Midpoint Circle algorithm.
///
/// Return: Annotated image array.
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
        
        let draw_pixel = |view: &mut ArrayViewMutD<u8>, px: i32, py: i32| {
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
        
        let draw_8_symmetry = |view: &mut ArrayViewMutD<u8>, px: i32, py: i32| {
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

/// ellipse() - Draw a simple or thick elliptic arc or an ellipsoid sector.
/// @py: Python interpreter token.
/// @img: Input image array (u8).
/// @center: Center of the ellipse (x, y).
/// @axes: Half of the size of the ellipse main axes (major, minor).
/// @angle: Ellipse rotation angle in degrees.
/// @start_angle: Starting angle of the elliptic arc in degrees.
/// @end_angle: Ending angle of the elliptic arc in degrees.
/// @color: Ellipse color.
/// @thickness: Line thickness (if negative, the ellipse is filled).
///
/// Draws a simple or thick elliptic arc or a filled ellipse sector.
///
/// Return: Annotated image array.
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

/// polylines() - Draw one or more polygonal curves (polylines) on an image.
/// @py: Python interpreter token.
/// @img: Input image array (u8).
/// @pts: List of 2D coordinates arrays representing polylines.
/// @is_closed: If true, draws a line connecting the last and first points of each polyline.
/// @color: Line color.
/// @thickness: Line thickness.
///
/// Draws one or more polygonal curves on the image.
///
/// Return: Annotated image array.
#[pyfunction]
#[pyo3(signature = (img, pts, is_closed, color, thickness = 1))]
pub fn polylines<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    pts: Vec<PyReadonlyArrayDyn<'py, i32>>,
    is_closed: bool,
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
    
    for poly_dyn in &pts {
        let poly_arr = poly_dyn.as_array();
        let poly_ndim = poly_arr.ndim();
        let n = poly_arr.shape()[0];
        if n < 2 {
            continue;
        }
        
        let get_pt = |i: usize| -> (i32, i32) {
            if poly_ndim == 2 {
                (poly_arr[[i, 0]], poly_arr[[i, 1]])
            } else {
                (poly_arr[[i, 0, 0]], poly_arr[[i, 0, 1]])
            }
        };
        
        for i in 0..(n - 1) {
            let p1 = get_pt(i);
            let p2 = get_pt(i + 1);
            draw_line_bresenham(&mut out_view, p1.0, p1.1, p2.0, p2.1, &parsed_color, thickness);
        }
        
        if is_closed {
            let p1 = get_pt(n - 1);
            let p2 = get_pt(0);
            draw_line_bresenham(&mut out_view, p1.0, p1.1, p2.0, p2.1, &parsed_color, thickness);
        }
    }
    
    Ok(out_arr.into_pyarray(py).to_dyn())
}

/// fill_poly() - Fill the area bounded by several polygonal contours on an image.
/// @py: Python interpreter token.
/// @img: Input image array (u8).
/// @pts: List of 2D coordinates arrays representing polygons to fill.
/// @color: Fill color.
///
/// Fills the interior of one or more polygons on the image.
///
/// Return: Annotated image array.
#[pyfunction]
pub fn fill_poly<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    pts: Vec<PyReadonlyArrayDyn<'py, i32>>,
    color: PyObject,
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
    
    for poly_dyn in &pts {
        let poly_arr = poly_dyn.as_array();
        let poly_ndim = poly_arr.ndim();
        let n = poly_arr.shape()[0];
        if n < 3 {
            continue;
        }
        
        let mut poly_pts = Vec::with_capacity(n);
        for i in 0..n {
            let pt = if poly_ndim == 2 {
                (poly_arr[[i, 0]], poly_arr[[i, 1]])
            } else {
                (poly_arr[[i, 0, 0]], poly_arr[[i, 0, 1]])
            };
            poly_pts.push(pt);
        }
        
        fill_polygon(&mut out_view, &poly_pts, &parsed_color);
    }
    
    Ok(out_arr.into_pyarray(py).to_dyn())
}

/// arrowed_line() - Draw an arrowed line segment from pt1 to pt2 on an image.
/// @py: Python interpreter token.
/// @img: Input image array (u8).
/// @pt1: Starting point (x, y).
/// @pt2: Ending point (x, y).
/// @color: Arrow color.
/// @thickness: Line thickness.
/// @tip_length: Length of the arrow tip relative to the line segment length.
///
/// Draws an arrowed line segment with the tip pointing to pt2.
///
/// Return: Annotated image array.
#[pyfunction]
#[pyo3(signature = (img, pt1, pt2, color, thickness = 1, tip_length = 0.1))]
pub fn arrowed_line<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    pt1: (i32, i32),
    pt2: (i32, i32),
    color: PyObject,
    thickness: i32,
    tip_length: f64,
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
    
    // Draw the main line
    draw_line_bresenham(&mut out_view, pt1.0, pt1.1, pt2.0, pt2.1, &parsed_color, thickness);
    
    let dx = (pt2.0 - pt1.0) as f64;
    let dy = (pt2.1 - pt1.1) as f64;
    let len = (dx * dx + dy * dy).sqrt();
    
    if len > 0.0 {
        let t_len = len * tip_length;
        let angle = dy.atan2(dx);
        
        let angle_left = angle + std::f64::consts::PI - (35.0f64).to_radians();
        let angle_right = angle + std::f64::consts::PI + (35.0f64).to_radians();
        
        let p_left_x = (pt2.0 as f64 + t_len * angle_left.cos()).round() as i32;
        let p_left_y = (pt2.1 as f64 + t_len * angle_left.sin()).round() as i32;
        
        let p_right_x = (pt2.0 as f64 + t_len * angle_right.cos()).round() as i32;
        let p_right_y = (pt2.1 as f64 + t_len * angle_right.sin()).round() as i32;
        
        draw_line_bresenham(&mut out_view, pt2.0, pt2.1, p_left_x, p_left_y, &parsed_color, thickness);
        draw_line_bresenham(&mut out_view, pt2.0, pt2.1, p_right_x, p_right_y, &parsed_color, thickness);
    }
    
    Ok(out_arr.into_pyarray(py).to_dyn())
}

/// put_text() - Render text on an image.
/// @py: Python interpreter token.
/// @img: Input image array (u8).
/// @text: String to be written.
/// @org: Bottom-left corner coordinates of the text string in the image.
/// @font_scale: Font scale factor.
/// @color: Text color.
/// @thickness: Line/fill thickness.
///
/// Renders the specified text string on the image.
///
/// Return: Annotated image array.
#[pyfunction]
#[pyo3(signature = (img, text, org, font_scale, color, thickness = 1))]
pub fn put_text<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    text: String,
    org: (i32, i32),
    font_scale: f64,
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
    
    let scale = font_scale.max(0.1);
    let char_w = (8.0 * scale).round() as i32;
    let char_h = (8.0 * scale).round() as i32;
    
    let mut cur_x = org.0;
    
    for ch in text.chars() {
        let ascii = ch as usize;
        if ascii >= 32 && ascii < 128 {
            let bitmap = FONT_8X8[ascii - 32];
            let start_y = org.1 - char_h;
            
            for row in 0..8 {
                let row_data = bitmap[row];
                for col in 0..8 {
                    if (row_data & (1 << (7 - col))) != 0 {
                        let px_start = cur_x + (col as f64 * scale).round() as i32;
                        let py_start = start_y + (row as f64 * scale).round() as i32;
                        
                        let px_end = cur_x + ((col + 1) as f64 * scale).round() as i32;
                        let py_end = start_y + ((row + 1) as f64 * scale).round() as i32;
                        
                        let t_offset = (thickness - 1).max(0);
                        
                        for py_draw in (py_start - t_offset)..py_end {
                            for px_draw in (px_start - t_offset)..px_end {
                                if px_draw >= 0 && px_draw < w && py_draw >= 0 && py_draw < h {
                                    if ndim == 3 {
                                        for c in 0..channels {
                                            out_arr[[py_draw as usize, px_draw as usize, c]] = parsed_color[c];
                                        }
                                    } else {
                                        out_arr[[py_draw as usize, px_draw as usize]] = parsed_color[0];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        cur_x += char_w;
    }
    
    Ok(out_arr.into_pyarray(py).to_dyn())
}
