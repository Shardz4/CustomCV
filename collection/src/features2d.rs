use pyo3::prelude::*;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, IntoPyArray};
use ndarray::Zip;

#[pyclass]
#[derive(Clone, Debug)]
pub struct KeyPoint {
    #[pyo3(get, set)]
    pub pt: (f64, f64), // (x, y)
    #[pyo3(get, set)]
    pub size: f32,
    #[pyo3(get, set)]
    pub angle: f32,
    #[pyo3(get, set)]
    pub response: f32,
    #[pyo3(get, set)]
    pub octave: i32,
    #[pyo3(get, set)]
    pub class_id: i32,
}

#[pymethods]
impl KeyPoint {
    #[new]
    #[pyo3(signature = (x, y, size = 0.0, angle = -1.0, response = 0.0, octave = 0, class_id = -1))]
    pub fn new(
        x: f64,
        y: f64,
        size: f32,
        angle: f32,
        response: f32,
        octave: i32,
        class_id: i32,
    ) -> Self {
        KeyPoint {
            pt: (x, y),
            size,
            angle,
            response,
            octave,
            class_id,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "KeyPoint(x={}, y={}, size={}, angle={}, response={}, octave={}, class_id={})",
            self.pt.0, self.pt.1, self.size, self.angle, self.response, self.octave, self.class_id
        )
    }
}

/// FAST corner detection.
#[pyfunction]
#[pyo3(signature = (image, threshold = 10, nonmax_suppression = true))]
pub fn fast_detect<'py>(
    _py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    threshold: i32,
    nonmax_suppression: bool,
) -> PyResult<Vec<KeyPoint>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D Grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
    
    // Circle offsets relative to candidate (dy, dx)
    let circle_offsets: [(isize, isize); 16] = [
        (-3, 0),  // 1
        (-3, 1),  // 2
        (-2, 2),  // 3
        (-1, 3),  // 4
        (0, 3),   // 5
        (1, 3),   // 6
        (2, 2),   // 7
        (3, 1),   // 8
        (3, 0),   // 9
        (3, -1),  // 10
        (2, -2),  // 11
        (1, -3),  // 12
        (0, -3),  // 13
        (-1, -3), // 14
        (-2, -2), // 15
        (-3, -1), // 16
    ];
    
    let mut candidates = Vec::new();
    let mut scores = numpy::ndarray::Array2::<f32>::zeros((h, w));
    
    for y in 3..h.saturating_sub(3) {
        for x in 3..w.saturating_sub(3) {
            let ip = img_2d[[y, x]] as i32;
            
            // Fast check: check 1, 9, 5, 13
            let v1 = img_2d[[(y as isize + circle_offsets[0].0) as usize, (x as isize + circle_offsets[0].1) as usize]] as i32;
            let v9 = img_2d[[(y as isize + circle_offsets[8].0) as usize, (x as isize + circle_offsets[8].1) as usize]] as i32;
            let v5 = img_2d[[(y as isize + circle_offsets[4].0) as usize, (x as isize + circle_offsets[4].1) as usize]] as i32;
            let v13 = img_2d[[(y as isize + circle_offsets[12].0) as usize, (x as isize + circle_offsets[12].1) as usize]] as i32;
            
            let mut count_brighter = 0;
            let mut count_darker = 0;
            
            if v1 > ip + threshold { count_brighter += 1; } else if v1 < ip - threshold { count_darker += 1; }
            if v9 > ip + threshold { count_brighter += 1; } else if v9 < ip - threshold { count_darker += 1; }
            if v5 > ip + threshold { count_brighter += 1; } else if v5 < ip - threshold { count_darker += 1; }
            if v13 > ip + threshold { count_brighter += 1; } else if v13 < ip - threshold { count_darker += 1; }
            
            if count_brighter < 2 && count_darker < 2 {
                continue;
            }
            
            // Full check: 16 pixels
            let mut circle_vals = [0i32; 16];
            for i in 0..16 {
                circle_vals[i] = img_2d[[(y as isize + circle_offsets[i].0) as usize, (x as isize + circle_offsets[i].1) as usize]] as i32;
            }
            
            // Check for contiguous brighter/darker sequence of length >= 9
            let mut extended = [0i32; 32];
            for i in 0..16 {
                extended[i] = circle_vals[i];
                extended[i + 16] = circle_vals[i];
            }
            
            let mut is_corner = false;
            let mut max_score = 0.0f32;
            
            // Check brighter
            let mut seq_len = 0;
            for i in 0..32 {
                if extended[i] > ip + threshold {
                    seq_len += 1;
                    if seq_len >= 9 {
                        is_corner = true;
                        let mut sum_diff = 0.0f32;
                        for j in (i + 1 - seq_len)..=i {
                            sum_diff += (extended[j] - ip - threshold).abs() as f32;
                        }
                        if sum_diff > max_score {
                            max_score = sum_diff;
                        }
                    }
                } else {
                    seq_len = 0;
                }
            }
            
            // Check darker
            seq_len = 0;
            for i in 0..32 {
                if extended[i] < ip - threshold {
                    seq_len += 1;
                    if seq_len >= 9 {
                        is_corner = true;
                        let mut sum_diff = 0.0f32;
                        for j in (i + 1 - seq_len)..=i {
                            sum_diff += (ip - extended[j] - threshold).abs() as f32;
                        }
                        if sum_diff > max_score {
                            max_score = sum_diff;
                        }
                    }
                } else {
                    seq_len = 0;
                }
            }
            
            if is_corner {
                candidates.push((y, x));
                scores[[y, x]] = max_score;
            }
        }
    }
    
    let mut keypoints = Vec::new();
    for &(y, x) in &candidates {
        if nonmax_suppression {
            let score = scores[[y, x]];
            let mut is_max = true;
            'outer: for dy in -1..=1 {
                for dx in -1..=1 {
                    if dy == 0 && dx == 0 { continue; }
                    let ny = (y as isize + dy) as usize;
                    let nx = (x as isize + dx) as usize;
                    if scores[[ny, nx]] > score {
                        is_max = false;
                        break 'outer;
                    }
                }
            }
            if is_max {
                keypoints.push(KeyPoint::new(x as f64, y as f64, 7.0, -1.0, score, 0, -1));
            }
        } else {
            keypoints.push(KeyPoint::new(x as f64, y as f64, 7.0, -1.0, scores[[y, x]], 0, -1));
        }
    }
    
    Ok(keypoints)
}
