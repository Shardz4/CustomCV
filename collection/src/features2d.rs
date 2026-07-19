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
    /// KeyPoint::new() - Constructor for KeyPoint.
    /// @x: X-coordinate of the keypoint.
    /// @y: Y-coordinate of the keypoint.
    /// @size: Keypoint diameter.
    /// @angle: Keypoint orientation.
    /// @response: Keypoint strength/response.
    /// @octave: Image octave (scale space layer).
    /// @class_id: Object class ID.
    ///
    /// Creates a new KeyPoint instance.
    ///
    /// Return: A KeyPoint instance.
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

#[pyclass]
#[derive(Clone, Debug)]
pub struct DMatch {
    #[pyo3(get, set)]
    pub query_idx: i32,
    #[pyo3(get, set)]
    pub train_idx: i32,
    #[pyo3(get, set)]
    pub img_idx: i32,
    #[pyo3(get, set)]
    pub distance: f32,
}

#[pymethods]
impl DMatch {
    /// DMatch::new() - Constructor for DMatch.
    /// @query_idx: Query descriptor index.
    /// @train_idx: Train descriptor index.
    /// @distance: Distance between descriptors.
    /// @img_idx: Train image index.
    ///
    /// Creates a new DMatch instance representing a descriptor match.
    ///
    /// Return: A DMatch instance.
    #[new]
    #[pyo3(signature = (query_idx, train_idx, distance, img_idx = 0))]
    pub fn new(query_idx: i32, train_idx: i32, distance: f32, img_idx: i32) -> Self {
        Self { query_idx, train_idx, img_idx, distance }
    }

    fn __repr__(&self) -> String {
        format!(
            "DMatch(queryIdx={}, trainIdx={}, distance={}, imgIdx={})",
            self.query_idx, self.train_idx, self.distance, self.img_idx
        )
    }
}

/// fast_detect() - FAST corner detection.
/// @_py: Python interpreter token.
/// @image: Input 2D grayscale image (u8).
/// @threshold: Threshold on difference between intensity of the center pixel and pixels on a circle around it.
/// @nonmax_suppression: If true, non-maximum suppression is applied.
///
/// Detects corners using the FAST (Features from Accelerated Segment Test) method.
///
/// Return: A vector of detected KeyPoints.
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

/// good_features_to_track() - Good Features to Track detector (Shi-Tomasi/Harris).
/// @_py: Python interpreter token.
/// @image: Input 2D grayscale image (u8).
/// @max_corners: Maximum number of corners to return.
/// @quality_level: Parameter characterizing the minimal accepted quality of image corners.
/// @min_distance: Minimum possible Euclidean distance between the returned corners.
/// @block_size: Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.
/// @use_harris_detector: Parameter indicating whether to use a Harris detector or Shi-Tomasi.
/// @k: Free parameter of the Harris detector.
///
/// Finds the most prominent corners in an image using the Shi-Tomasi or Harris method.
///
/// Return: A vector of detected KeyPoints.
#[pyfunction]
#[pyo3(signature = (image, max_corners = 1000, quality_level = 0.01, min_distance = 10.0, block_size = 3, use_harris_detector = false, k = 0.04))]
pub fn good_features_to_track<'py>(
    _py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    max_corners: usize,
    quality_level: f64,
    min_distance: f64,
    block_size: usize,
    use_harris_detector: bool,
    k: f64,
) -> PyResult<Vec<KeyPoint>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D Grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
    
    let (sxx, syy, sxy) = crate::helpers::compute_structure_tensor(&img_2d, block_size);
    let mut response = numpy::ndarray::Array2::<f32>::zeros((h, w));
    let mut max_resp = 0.0f32;
    
    for y in 0..h {
        for x in 0..w {
            let a = sxx[[y, x]];
            let b = sxy[[y, x]];
            let c = syy[[y, x]];
            
            let val = if use_harris_detector {
                let det = (a * c) - (b * b);
                let trace = a + c;
                det - (k as f32) * (trace * trace)
            } else {
                let trace = a + c;
                let det = (a * c) - (b * b);
                let gap = ((trace * trace) / 4.0 - det).max(0.0).sqrt();
                (trace / 2.0) - gap
            };
            
            response[[y, x]] = val;
            if val > max_resp {
                max_resp = val;
            }
        }
    }
    
    let threshold = quality_level as f32 * max_resp;
    let mut candidates = Vec::new();
    
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let val = response[[y, x]];
            if val > threshold {
                let mut is_local_max = true;
                'check: for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dy == 0 && dx == 0 { continue; }
                        if response[[(y as isize + dy) as usize, (x as isize + dx) as usize]] > val {
                            is_local_max = false;
                            break 'check;
                        }
                    }
                }
                if is_local_max {
                    candidates.push(KeyPoint::new(x as f64, y as f64, block_size as f32, -1.0, val, 0, -1));
                }
            }
        }
    }
    
    candidates.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut accepted: Vec<KeyPoint> = Vec::new();
    let min_dist_sq = min_distance * min_distance;
    
    for kp in candidates {
        let (cx, cy) = kp.pt;
        let mut too_close = false;
        for acc in &accepted {
            let (ax, ay) = acc.pt;
            let dist_sq = (cx - ax) * (cx - ax) + (cy - ay) * (cy - ay);
            if dist_sq < min_dist_sq {
                too_close = true;
                break;
            }
        }
        if !too_close {
            accepted.push(kp);
            if accepted.len() >= max_corners {
                break;
            }
        }
    }
    
    Ok(accepted)
}

/// Helper function to generate BRIEF pixel pairs deterministically.
fn get_brief_pairs() -> Vec<((i32, i32), (i32, i32))> {
    let mut pairs = Vec::with_capacity(256);
    let mut seed: u32 = 0x12345678;
    let mut rand_coord = |seed: &mut u32| -> i32 {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((*seed / 65536) % 31) as i32 - 15
    };
    for _ in 0..256 {
        let x1 = rand_coord(&mut seed);
        let y1 = rand_coord(&mut seed);
        let x2 = rand_coord(&mut seed);
        let y2 = rand_coord(&mut seed);
        pairs.push(((x1, y1), (x2, y2)));
    }
    pairs
}

/// orb_detect_and_compute() - ORB keypoint detector and descriptor extractor.
/// @py: Python interpreter token.
/// @image: Input 2D grayscale image (u8).
/// @max_features: Maximum number of features to retain.
/// @threshold: FAST threshold.
///
/// Detects keypoints and computes BRIEF descriptors for them using ORB.
///
/// Return: A tuple of (detected KeyPoints vector, descriptor matrix array).
#[pyfunction]
#[pyo3(signature = (image, max_features = 500, threshold = 20))]
pub fn orb_detect_and_compute<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    max_features: usize,
    threshold: i32,
) -> PyResult<(Vec<KeyPoint>, &'py PyArrayDyn<u8>)> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D Grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
    
    let raw_kps = fast_detect(py, image.clone(), threshold, true)?;
    
    let mut kps = raw_kps;
    kps.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
    
    let brief_pairs = get_brief_pairs();
    
    let mut valid_kps = Vec::new();
    let mut descriptors_vec = Vec::new();
    
    let border = 16;
    for mut kp in kps {
        let (kx, ky) = (kp.pt.0 as isize, kp.pt.1 as isize);
        if kx < border || kx >= (w as isize - border) || ky < border || ky >= (h as isize - border) {
            continue;
        }
        
        let mut m10 = 0.0f64;
        let mut m01 = 0.0f64;
        for dy in -15..=15 {
            for dx in -15..=15 {
                if dx * dx + dy * dy <= 225 {
                    let val = img_2d[[(ky + dy) as usize, (kx + dx) as usize]] as f64;
                    m10 += dx as f64 * val;
                    m01 += dy as f64 * val;
                }
            }
        }
        let angle = m01.atan2(m10);
        kp.angle = angle.to_degrees() as f32;
        
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let mut desc = [0u8; 32];
        
        for i in 0..256 {
            let ((dx1, dy1), (dx2, dy2)) = brief_pairs[i];
            
            let rx1 = (dx1 as f64 * cos_a - dy1 as f64 * sin_a).round() as isize;
            let ry1 = (dx1 as f64 * sin_a + dy1 as f64 * cos_a).round() as isize;
            let rx2 = (dx2 as f64 * cos_a - dy2 as f64 * sin_a).round() as isize;
            let ry2 = (dx2 as f64 * sin_a + dy2 as f64 * cos_a).round() as isize;
            
            let val1 = img_2d[[(ky + ry1) as usize, (kx + rx1) as usize]];
            let val2 = img_2d[[(ky + ry2) as usize, (kx + rx2) as usize]];
            
            if val1 < val2 {
                desc[i / 8] |= 1 << (i % 8);
            }
        }
        
        valid_kps.push(kp);
        descriptors_vec.extend_from_slice(&desc);
        
        if valid_kps.len() >= max_features {
            break;
        }
    }
    
    let num_desc = valid_kps.len();
    let mut desc_arr = numpy::ndarray::Array2::<u8>::zeros((num_desc, 32));
    for i in 0..num_desc {
        for j in 0..32 {
            desc_arr[[i, j]] = descriptors_vec[i * 32 + j];
        }
    }
    
    Ok((valid_kps, desc_arr.into_pyarray(py).to_dyn()))
}

/// sift_detect_and_compute() - SIFT keypoint detector and descriptor extractor.
/// @py: Python interpreter token.
/// @image: Input 2D grayscale image (u8).
/// @max_features: Maximum number of features to retain.
///
/// Detects keypoints and computes SIFT descriptors for them.
///
/// Return: A tuple of (detected KeyPoints vector, descriptor matrix array).
#[pyfunction]
#[pyo3(signature = (image, max_features = 500))]
pub fn sift_detect_and_compute<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    max_features: usize,
) -> PyResult<(Vec<KeyPoint>, &'py PyArrayDyn<f32>)> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D Grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
    
    let mut all_kps = Vec::new();
    
    let kps0 = good_features_to_track(py, image.clone(), max_features, 0.01, 10.0, 3, false, 0.04)?;
    for mut kp in kps0 {
        kp.octave = 0;
        kp.size = 1.6;
        all_kps.push((kp, 1.0));
    }
    
    if h > 30 && w > 30 {
        let mut img1 = numpy::ndarray::Array2::<u8>::zeros((h / 2, w / 2));
        for y in 0..h/2 {
            for x in 0..w/2 {
                img1[[y, x]] = img_2d[[y * 2, x * 2]];
            }
        }
        let py_img1 = img1.into_pyarray(py).to_dyn();
        let kps1 = good_features_to_track(py, py_img1.readonly(), max_features, 0.01, 10.0, 3, false, 0.04)?;
        for mut kp in kps1 {
            kp.octave = 1;
            kp.size = 3.2;
            let (kx, ky) = kp.pt;
            kp.pt = (kx * 2.0, ky * 2.0);
            all_kps.push((kp, 0.5));
        }
    }
    
    if h > 60 && w > 60 {
        let mut img2 = numpy::ndarray::Array2::<u8>::zeros((h / 4, w / 4));
        for y in 0..h/4 {
            for x in 0..w/4 {
                img2[[y, x]] = img_2d[[y * 4, x * 4]];
            }
        }
        let py_img2 = img2.into_pyarray(py).to_dyn();
        let kps2 = good_features_to_track(py, py_img2.readonly(), max_features, 0.01, 10.0, 3, false, 0.04)?;
        for mut kp in kps2 {
            kp.octave = 2;
            kp.size = 6.4;
            let (kx, ky) = kp.pt;
            kp.pt = (kx * 4.0, ky * 4.0);
            all_kps.push((kp, 0.25));
        }
    }
    
    all_kps.sort_by(|a, b| b.0.response.partial_cmp(&a.0.response).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut valid_kps = Vec::new();
    let mut descriptors_vec = Vec::new();
    
    let border = 16;
    for (mut kp, scale) in all_kps {
        let (orig_x, orig_y) = kp.pt;
        let kx = (orig_x * scale).round() as isize;
        let ky = (orig_y * scale).round() as isize;
        
        let local_h = (h as f64 * scale) as isize;
        let local_w = (w as f64 * scale) as isize;
        
        if kx < border || kx >= (local_w - border) || ky < border || ky >= (local_h - border) {
            continue;
        }
        
        let mut orient_hist = [0.0f64; 36];
        for dy in -8..=8 {
            for dx in -8..=8 {
                let step = (1.0 / scale) as isize;
                let ox = (orig_x as isize + dx * step) as usize;
                let oy = (orig_y as isize + dy * step) as usize;
                
                if ox > 0 && ox < w - 1 && oy > 0 && oy < h - 1 {
                    let dx_val = img_2d[[oy, ox + 1]] as f64 - img_2d[[oy, ox - 1]] as f64;
                    let dy_val = img_2d[[oy + 1, ox]] as f64 - img_2d[[oy - 1, ox]] as f64;
                    let mag = (dx_val * dx_val + dy_val * dy_val).sqrt();
                    let mut theta = dy_val.atan2(dx_val);
                    if theta < 0.0 {
                        theta += 2.0 * std::f64::consts::PI;
                    }
                    let bin = ((theta.to_degrees() / 10.0).round() as usize) % 36;
                    let weight = (-(dx * dx + dy * dy) as f64 / 32.0).exp();
                    orient_hist[bin] += mag * weight;
                }
            }
        }
        
        let mut max_val = 0.0;
        let mut dom_bin = 0;
        for i in 0..36 {
            if orient_hist[i] > max_val {
                max_val = orient_hist[i];
                dom_bin = i;
            }
        }
        let dom_angle = (dom_bin as f64 * 10.0).to_radians();
        kp.angle = dom_angle.to_degrees() as f32;
        
        let mut desc = vec![0.0f32; 128];
        let cos_a = dom_angle.cos();
        let sin_a = dom_angle.sin();
        
        for dy in -8..8 {
            for dx in -8..8 {
                let rx = dx as f64 * cos_a - dy as f64 * sin_a;
                let ry = dx as f64 * sin_a + dy as f64 * cos_a;
                
                let ox = (orig_x + rx / scale).round() as usize;
                let oy = (orig_y + ry / scale).round() as usize;
                
                if ox > 0 && ox < w - 1 && oy > 0 && oy < h - 1 {
                    let dx_val = img_2d[[oy, ox + 1]] as f64 - img_2d[[oy, ox - 1]] as f64;
                    let dy_val = img_2d[[oy + 1, ox]] as f64 - img_2d[[oy - 1, ox]] as f64;
                    let mag = (dx_val * dx_val + dy_val * dy_val).sqrt();
                    let mut theta = dy_val.atan2(dx_val) - dom_angle;
                    if theta < 0.0 {
                        theta += 2.0 * std::f64::consts::PI;
                    }
                    let theta_deg = theta.to_degrees();
                    let bin = ((theta_deg / 45.0).round() as usize) % 8;
                    
                    let sub_x = ((rx + 8.0) / 4.0).floor() as isize;
                    let sub_y = ((ry + 8.0) / 4.0).floor() as isize;
                    
                    if sub_x >= 0 && sub_x < 4 && sub_y >= 0 && sub_y < 4 {
                        let sub_idx = (sub_y * 4 + sub_x) as usize;
                        let weight = (-(dx * dx + dy * dy) as f64 / 128.0).exp();
                        desc[sub_idx * 8 + bin] += (mag * weight) as f32;
                    }
                }
            }
        }
        
        let mut sum_sq = 0.0f32;
        for &v in &desc { sum_sq += v * v; }
        let norm = sum_sq.sqrt();
        if norm > 1e-6 {
            for v in &mut desc {
                *v = (*v / norm).min(0.2f32);
            }
            let mut sum_sq2 = 0.0f32;
            for &v in &desc { sum_sq2 += v * v; }
            let norm2 = sum_sq2.sqrt();
            if norm2 > 1e-6 {
                for v in &mut desc {
                    *v /= norm2;
                }
            }
        }
        
        valid_kps.push(kp);
        descriptors_vec.push(desc);
        
        if valid_kps.len() >= max_features {
            break;
        }
    }
    
    let num_kps = valid_kps.len();
    let mut desc_arr = numpy::ndarray::Array2::<f32>::zeros((num_kps, 128));
    for i in 0..num_kps {
        for j in 0..128 {
            desc_arr[[i, j]] = descriptors_vec[i][j];
        }
    }
    
    Ok((valid_kps, desc_arr.into_pyarray(py).to_dyn()))
}

/// Helper to get BRISK sampling pattern coordinates (60 points)
fn get_brisk_sampling_pattern() -> Vec<(f64, f64)> {
    let mut pattern = Vec::with_capacity(60);
    pattern.push((0.0, 0.0));
    
    for i in 0..4 {
        let angle = (i as f64) * 2.0 * std::f64::consts::PI / 4.0;
        pattern.push((1.5 * angle.cos(), 1.5 * angle.sin()));
    }
    for i in 0..10 {
        let angle = (i as f64) * 2.0 * std::f64::consts::PI / 10.0;
        pattern.push((3.5 * angle.cos(), 3.5 * angle.sin()));
    }
    for i in 0..18 {
        let angle = (i as f64) * 2.0 * std::f64::consts::PI / 18.0;
        pattern.push((6.5 * angle.cos(), 6.5 * angle.sin()));
    }
    for i in 0..28 {
        let angle = (i as f64) * 2.0 * std::f64::consts::PI / 28.0;
        pattern.push((10.5 * angle.cos(), 10.5 * angle.sin()));
    }
    pattern
}

/// Helper to get BRISK short pairs (512 pairs) and long pairs deterministically
fn get_brisk_pairs(pattern: &[(f64, f64)]) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let mut short_pairs = Vec::new();
    let mut long_pairs = Vec::new();
    
    let n = pattern.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = pattern[i].0 - pattern[j].0;
            let dy = pattern[i].1 - pattern[j].1;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < 3.5 {
                short_pairs.push((i, j));
            } else if dist > 6.0 {
                long_pairs.push((i, j));
            }
        }
    }
    
    short_pairs.truncate(512);
    let mut seed: u32 = 0x87654321;
    while short_pairs.len() < 512 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let i = (seed as usize) % n;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let j = (seed as usize) % n;
        if i != j {
            short_pairs.push((i, j));
        }
    }
    (short_pairs, long_pairs)
}

/// brisk_detect_and_compute() - BRISK keypoint detector and descriptor extractor.
/// @py: Python interpreter token.
/// @image: Input 2D grayscale image (u8).
/// @max_features: Maximum number of features to retain.
/// @threshold: FAST/AGAST detection threshold.
///
/// Detects keypoints and computes BRISK descriptors for them.
///
/// Return: A tuple of (detected KeyPoints vector, descriptor matrix array).
#[pyfunction]
#[pyo3(signature = (image, max_features = 500, threshold = 20))]
pub fn brisk_detect_and_compute<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    max_features: usize,
    threshold: i32,
) -> PyResult<(Vec<KeyPoint>, &'py PyArrayDyn<u8>)> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D Grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
    
    let mut all_kps = Vec::new();
    
    let kps0 = fast_detect(py, image.clone(), threshold, true)?;
    for mut kp in kps0 {
        kp.octave = 0;
        kp.size = 1.0;
        all_kps.push((kp, 1.0));
    }
    
    if h > 30 && w > 30 {
        let mut img1 = numpy::ndarray::Array2::<u8>::zeros((h / 2, w / 2));
        for y in 0..h/2 {
            for x in 0..w/2 {
                img1[[y, x]] = img_2d[[y * 2, x * 2]];
            }
        }
        let py_img1 = img1.into_pyarray(py).to_dyn();
        let kps1 = fast_detect(py, py_img1.readonly(), threshold, true)?;
        for mut kp in kps1 {
            kp.octave = 1;
            kp.size = 2.0;
            let (kx, ky) = kp.pt;
            kp.pt = (kx * 2.0, ky * 2.0);
            all_kps.push((kp, 0.5));
        }
    }
    
    if h > 60 && w > 60 {
        let mut img2 = numpy::ndarray::Array2::<u8>::zeros((h / 4, w / 4));
        for y in 0..h/4 {
            for x in 0..w/4 {
                img2[[y, x]] = img_2d[[y * 4, x * 4]];
            }
        }
        let py_img2 = img2.into_pyarray(py).to_dyn();
        let kps2 = fast_detect(py, py_img2.readonly(), threshold, true)?;
        for mut kp in kps2 {
            kp.octave = 2;
            kp.size = 4.0;
            let (kx, ky) = kp.pt;
            kp.pt = (kx * 4.0, ky * 4.0);
            all_kps.push((kp, 0.25));
        }
    }
    
    all_kps.sort_by(|a, b| b.0.response.partial_cmp(&a.0.response).unwrap_or(std::cmp::Ordering::Equal));
    
    let pattern = get_brisk_sampling_pattern();
    let (short_pairs, long_pairs) = get_brisk_pairs(&pattern);
    
    let mut valid_kps = Vec::new();
    let mut descriptors_vec = Vec::new();
    
    let border = 16;
    for (mut kp, scale) in all_kps {
        let (orig_x, orig_y) = kp.pt;
        let kx = (orig_x * scale).round() as isize;
        let ky = (orig_y * scale).round() as isize;
        
        let local_h = (h as f64 * scale) as isize;
        let local_w = (w as f64 * scale) as isize;
        
        if kx < border || kx >= (local_w - border) || ky < border || ky >= (local_h - border) {
            continue;
        }
        
        let mut intensities = vec![0.0f64; pattern.len()];
        for i in 0..pattern.len() {
            let px = orig_x + pattern[i].0 / scale;
            let py_val = orig_y + pattern[i].1 / scale;
            
            let x0 = px.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let y0 = py_val.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let dx = px - x0 as f64;
            let dy = py_val - y0 as f64;
            
            let val00 = img_2d[[y0, x0]] as f64;
            let val01 = img_2d[[y0, x1]] as f64;
            let val10 = img_2d[[y1, x0]] as f64;
            let val11 = img_2d[[y1, x1]] as f64;
            
            intensities[i] = (1.0 - dx) * (1.0 - dy) * val00
                + dx * (1.0 - dy) * val01
                + (1.0 - dx) * dy * val10
                + dx * dy * val11;
        }
        
        let mut rx = 0.0;
        let mut ry = 0.0;
        for &(i, j) in &long_pairs {
            let diff = intensities[j] - intensities[i];
            let dx = pattern[j].0 - pattern[i].0;
            let dy = pattern[j].1 - pattern[i].1;
            let len = (dx * dx + dy * dy).sqrt();
            if len > 1e-6 {
                rx += (dx / len) * diff;
                ry += (dy / len) * diff;
            }
        }
        let angle = ry.atan2(rx);
        kp.angle = angle.to_degrees() as f32;
        
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        
        let mut rot_intensities = vec![0.0f64; pattern.len()];
        for i in 0..pattern.len() {
            let rx = pattern[i].0 * cos_a - pattern[i].1 * sin_a;
            let ry = pattern[i].0 * sin_a + pattern[i].1 * cos_a;
            let px = orig_x + rx / scale;
            let py_val = orig_y + ry / scale;
            
            let x0 = px.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let y0 = py_val.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let dx = px - x0 as f64;
            let dy = py_val - y0 as f64;
            
            let val00 = img_2d[[y0, x0]] as f64;
            let val01 = img_2d[[y0, x1]] as f64;
            let val10 = img_2d[[y1, x0]] as f64;
            let val11 = img_2d[[y1, x1]] as f64;
            
            rot_intensities[i] = (1.0 - dx) * (1.0 - dy) * val00
                + dx * (1.0 - dy) * val01
                + (1.0 - dx) * dy * val10
                + dx * dy * val11;
        }
        
        let mut desc = [0u8; 64];
        for b in 0..512 {
            let (i, j) = short_pairs[b];
            if rot_intensities[i] < rot_intensities[j] {
                desc[b / 8] |= 1 << (b % 8);
            }
        }
        
        valid_kps.push(kp);
        descriptors_vec.extend_from_slice(&desc);
        
        if valid_kps.len() >= max_features {
            break;
        }
    }
    
    let num_desc = valid_kps.len();
    let mut desc_arr = numpy::ndarray::Array2::<u8>::zeros((num_desc, 64));
    for i in 0..num_desc {
        for j in 0..64 {
            desc_arr[[i, j]] = descriptors_vec[i * 64 + j];
        }
    }
    
    Ok((valid_kps, desc_arr.into_pyarray(py).to_dyn()))
}

/// Helper function to perform a fast grayscale bilateral filter.
fn bilateral_filter_gray(img: &numpy::ndarray::Array2<u8>, sigma_s: f64, sigma_r: f64) -> numpy::ndarray::Array2<u8> {
    let (h, w) = (img.shape()[0], img.shape()[1]);
    let mut out = numpy::ndarray::Array2::<u8>::zeros((h, w));
    let r = (sigma_s * 3.0).ceil() as isize;
    
    let mut spatial_weights = vec![vec![0.0f64; (2 * r + 1) as usize]; (2 * r + 1) as usize];
    for dy in -r..=r {
        for dx in -r..=r {
            spatial_weights[(dy + r) as usize][(dx + r) as usize] = (-((dx * dx + dy * dy) as f64) / (2.0 * sigma_s * sigma_s)).exp();
        }
    }
    
    for y in 0..h {
        for x in 0..w {
            let mut sum_val = 0.0;
            let mut sum_w = 0.0;
            let center_val = img[[y, x]] as f64;
            
            for dy in -r..=r {
                for dx in -r..=r {
                    let ny = y as isize + dy;
                    let nx = x as isize + dx;
                    if ny >= 0 && ny < h as isize && nx >= 0 && nx < w as isize {
                        let val = img[[ny as usize, nx as usize]] as f64;
                        let range_w = (-((val - center_val) * (val - center_val)) / (2.0 * sigma_r * sigma_r)).exp();
                        let w_val = spatial_weights[(dy + r) as usize][(dx + r) as usize] * range_w;
                        sum_val += val * w_val;
                        sum_w += w_val;
                    }
                }
            }
            out[[y, x]] = if sum_w > 1e-6 { (sum_val / sum_w).round().clamp(0.0, 255.0) as u8 } else { img[[y, x]] };
        }
    }
    out
}

/// akaze_detect_and_compute() - AKAZE keypoint detector and descriptor extractor.
/// @py: Python interpreter token.
/// @image: Input 2D grayscale image (u8).
/// @max_features: Maximum number of features to retain.
///
/// Detects keypoints and computes AKAZE descriptors for them using nonlinear scale space.
///
/// Return: A tuple of (detected KeyPoints vector, descriptor matrix array).
#[pyfunction]
#[pyo3(signature = (image, max_features = 500))]
pub fn akaze_detect_and_compute<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    max_features: usize,
) -> PyResult<(Vec<KeyPoint>, &'py PyArrayDyn<u8>)> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D Grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
    
    let level0 = img_2d.to_owned();
    let level1 = bilateral_filter_gray(&level0, 1.5, 20.0);
    let level2 = bilateral_filter_gray(&level1, 3.0, 40.0);
    
    let mut all_kps = Vec::new();
    
    let py_l0 = level0.clone().into_pyarray(py).to_dyn();
    let kps0 = good_features_to_track(py, py_l0.readonly(), max_features, 0.01, 10.0, 3, false, 0.04)?;
    for mut kp in kps0 {
        kp.octave = 0;
        kp.size = 1.0;
        all_kps.push(kp);
    }
    
    let py_l1 = level1.clone().into_pyarray(py).to_dyn();
    let kps1 = good_features_to_track(py, py_l1.readonly(), max_features, 0.01, 10.0, 3, false, 0.04)?;
    for mut kp in kps1 {
        kp.octave = 1;
        kp.size = 2.0;
        all_kps.push(kp);
    }
    
    let py_l2 = level2.clone().into_pyarray(py).to_dyn();
    let kps2 = good_features_to_track(py, py_l2.readonly(), max_features, 0.01, 10.0, 3, false, 0.04)?;
    for mut kp in kps2 {
        kp.octave = 2;
        kp.size = 4.0;
        all_kps.push(kp);
    }
    
    all_kps.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut valid_kps = Vec::new();
    let mut descriptors_vec = Vec::new();
    
    let border = 16;
    for mut kp in all_kps {
        let (kx, ky) = (kp.pt.0 as isize, kp.pt.1 as isize);
        if kx < border || kx >= (w as isize - border) || ky < border || ky >= (h as isize - border) {
            continue;
        }
        
        let src_img = match kp.octave {
            1 => &level1,
            2 => &level2,
            _ => &level0,
        };
        
        let mut m10 = 0.0;
        let mut m01 = 0.0;
        for dy in -15..=15 {
            for dx in -15..=15 {
                if dx * dx + dy * dy <= 225 {
                    let val = src_img[[(ky + dy) as usize, (kx + dx) as usize]] as f64;
                    m10 += dx as f64 * val;
                    m01 += dy as f64 * val;
                }
            }
        }
        let angle = m01.atan2(m10);
        kp.angle = angle.to_degrees() as f32;
        
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        
        let mut patch = numpy::ndarray::Array2::<f64>::zeros((16, 16));
        let mut patch_dx = numpy::ndarray::Array2::<f64>::zeros((16, 16));
        let mut patch_dy = numpy::ndarray::Array2::<f64>::zeros((16, 16));
        
        for y_idx in 0..16 {
            for x_idx in 0..16 {
                let dx = x_idx as f64 - 8.0;
                let dy = y_idx as f64 - 8.0;
                
                let rx = dx * cos_a - dy * sin_a;
                let ry = dx * sin_a + dy * cos_a;
                
                let px = (kp.pt.0 + rx).round() as usize;
                let py_val = (kp.pt.1 + ry).round() as usize;
                
                if px > 0 && px < w - 1 && py_val > 0 && py_val < h - 1 {
                    patch[[y_idx, x_idx]] = src_img[[py_val, px]] as f64;
                    patch_dx[[y_idx, x_idx]] = src_img[[py_val, px + 1]] as f64 - src_img[[py_val, px - 1]] as f64;
                    patch_dy[[y_idx, x_idx]] = src_img[[py_val + 1, px]] as f64 - src_img[[py_val - 1, px]] as f64;
                }
            }
        }
        
        let mut cells_2x2_i = vec![0.0; 4];
        let mut cells_2x2_dx = vec![0.0; 4];
        let mut cells_2x2_dy = vec![0.0; 4];
        for cy in 0..2 {
            for cx in 0..2 {
                let mut sum_i = 0.0;
                let mut sum_dx = 0.0;
                let mut sum_dy = 0.0;
                for dy in 0..8 {
                    for dx in 0..8 {
                        sum_i += patch[[cy * 8 + dy, cx * 8 + dx]];
                        sum_dx += patch_dx[[cy * 8 + dy, cx * 8 + dx]];
                        sum_dy += patch_dy[[cy * 8 + dy, cx * 8 + dx]];
                    }
                }
                let idx = cy * 2 + cx;
                cells_2x2_i[idx] = sum_i / 64.0;
                cells_2x2_dx[idx] = sum_dx / 64.0;
                cells_2x2_dy[idx] = sum_dy / 64.0;
            }
        }
        
        let mut cells_3x3_i = vec![0.0; 9];
        let mut cells_3x3_dx = vec![0.0; 9];
        let mut cells_3x3_dy = vec![0.0; 9];
        for cy in 0..3 {
            for cx in 0..3 {
                let mut sum_i = 0.0;
                let mut sum_dx = 0.0;
                let mut sum_dy = 0.0;
                let y_start = cy * 5;
                let y_end = (cy * 5 + 6).min(16);
                let x_start = cx * 5;
                let x_end = (cx * 5 + 6).min(16);
                let count = ((y_end - y_start) * (x_end - x_start)) as f64;
                for dy in y_start..y_end {
                    for dx in x_start..x_end {
                        sum_i += patch[[dy, dx]];
                        sum_dx += patch_dx[[dy, dx]];
                        sum_dy += patch_dy[[dy, dx]];
                    }
                }
                let idx = cy * 3 + cx;
                cells_3x3_i[idx] = sum_i / count;
                cells_3x3_dx[idx] = sum_dx / count;
                cells_3x3_dy[idx] = sum_dy / count;
            }
        }
        
        let mut desc = [0u8; 16];
        let mut bit_idx = 0;
        
        for i in 0..4 {
            for j in (i + 1)..4 {
                if cells_2x2_i[i] < cells_2x2_i[j] { desc[bit_idx / 8] |= 1 << (bit_idx % 8); }
                bit_idx += 1;
                if cells_2x2_dx[i] < cells_2x2_dx[j] { desc[bit_idx / 8] |= 1 << (bit_idx % 8); }
                bit_idx += 1;
                if cells_2x2_dy[i] < cells_2x2_dy[j] { desc[bit_idx / 8] |= 1 << (bit_idx % 8); }
                bit_idx += 1;
            }
        }
        
        for i in 0..9 {
            for j in (i + 1)..9 {
                if cells_3x3_i[i] < cells_3x3_i[j] { desc[bit_idx / 8] |= 1 << (bit_idx % 8); }
                bit_idx += 1;
                if cells_3x3_dx[i] < cells_3x3_dx[j] { desc[bit_idx / 8] |= 1 << (bit_idx % 8); }
                bit_idx += 1;
                if cells_3x3_dy[i] < cells_3x3_dy[j] { desc[bit_idx / 8] |= 1 << (bit_idx % 8); }
                bit_idx += 1;
            }
        }
        
        valid_kps.push(kp);
        descriptors_vec.extend_from_slice(&desc);
        
        if valid_kps.len() >= max_features {
            break;
        }
    }
    
    let num_desc = valid_kps.len();
    let mut desc_arr = numpy::ndarray::Array2::<u8>::zeros((num_desc, 16));
    for i in 0..num_desc {
        for j in 0..16 {
            desc_arr[[i, j]] = descriptors_vec[i * 16 + j];
        }
    }
    
    Ok((valid_kps, desc_arr.into_pyarray(py).to_dyn()))
}

/// mser_detect() - MSER (Maximally Stable Extremal Regions) blob detector.
/// @_py: Python interpreter token.
/// @image: Input 2D grayscale image (u8).
/// @delta: Delta parameter.
/// @min_area: Minimum area of a region.
/// @max_area: Maximum area of a region.
/// @max_variation: Maximum variation to accept a region.
///
/// Detects Maximally Stable Extremal Regions (MSER) in an image.
///
/// Return: A list of regions, where each region is a list of pixel coordinates.
#[pyfunction]
#[pyo3(signature = (image, delta = 5, min_area = 60, max_area = 14400, max_variation = 0.25))]
pub fn mser_detect<'py>(
    _py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    delta: usize,
    min_area: usize,
    max_area: usize,
    max_variation: f64,
) -> PyResult<Vec<Vec<(usize, usize)>>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D Grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
    
    let steps: Vec<usize> = (delta..=255-delta).step_by(delta).collect();
    let mut all_labeled = Vec::new();
    let mut all_pixels = Vec::new();
    
    for &t in &steps {
        let mut binary = numpy::ndarray::Array2::<u8>::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                if img_2d[[y, x]] as usize <= t {
                    binary[[y, x]] = 255;
                }
            }
        }
        
        let mut labeled = numpy::ndarray::Array2::<i32>::zeros((h, w));
        let mut next_label = 1;
        let mut component_pixels = Vec::new();
        component_pixels.push(Vec::new());
        
        for y in 0..h {
            for x in 0..w {
                if binary[[y, x]] == 255 && labeled[[y, x]] == 0 {
                    let label = next_label;
                    next_label += 1;
                    let mut pixels = Vec::new();
                    
                    let mut queue = std::collections::VecDeque::new();
                    queue.push_back((y, x));
                    labeled[[y, x]] = label;
                    
                    while let Some((cy, cx)) = queue.pop_front() {
                        pixels.push((cx, cy));
                        
                        let neighbors = [
                            (cy as isize - 1, cx as isize),
                            (cy as isize + 1, cx as isize),
                            (cy as isize, cx as isize - 1),
                            (cy as isize, cx as isize + 1),
                        ];
                        for &(ny, nx) in &neighbors {
                            if ny >= 0 && ny < h as isize && nx >= 0 && nx < w as isize {
                                let (ny_u, nx_u) = (ny as usize, nx as usize);
                                if binary[[ny_u, nx_u]] == 255 && labeled[[ny_u, nx_u]] == 0 {
                                    labeled[[ny_u, nx_u]] = label;
                                    queue.push_back((ny_u, nx_u));
                                }
                            }
                        }
                    }
                    component_pixels.push(pixels);
                }
            }
        }
        all_labeled.push(labeled);
        all_pixels.push(component_pixels);
    }
    
    let mut stable_regions = Vec::new();
    
    for idx in 1..steps.len() - 1 {
        let comps_t = &all_pixels[idx];
        let labeled_prev = &all_labeled[idx - 1];
        let comps_prev = &all_pixels[idx - 1];
        let labeled_next = &all_labeled[idx + 1];
        let comps_next = &all_pixels[idx + 1];
        
        for (label_t, comp) in comps_t.iter().enumerate() {
            if label_t == 0 { continue; }
            let area = comp.len();
            if area < min_area || area > max_area {
                continue;
            }
            
            let mut prev_counts = std::collections::HashMap::new();
            for &(cx, cy) in comp {
                let lbl = labeled_prev[[cy, cx]];
                if lbl > 0 {
                    *prev_counts.entry(lbl).or_insert(0) += 1;
                }
            }
            let best_prev_label = prev_counts.into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(lbl, _)| lbl);
            
            let mut next_counts = std::collections::HashMap::new();
            for &(cx, cy) in comp {
                let lbl = labeled_next[[cy, cx]];
                if lbl > 0 {
                    *next_counts.entry(lbl).or_insert(0) += 1;
                }
            }
            let best_next_label = next_counts.into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(lbl, _)| lbl);
                
            if let (Some(l_prev), Some(l_next)) = (best_prev_label, best_next_label) {
                let area_prev = comps_prev[l_prev as usize].len();
                let area_next = comps_next[l_next as usize].len();
                
                let variation = (area_next as f64 - area_prev as f64) / area as f64;
                if variation <= max_variation {
                    stable_regions.push(comp.clone());
                }
            }
        }
    }
    
    Ok(stable_regions)
}

/// Helper to compute the convex hull area of a set of 2D points.
fn convex_hull_area(points: &[(usize, usize)]) -> f64 {
    if points.len() < 3 {
        return points.len() as f64;
    }
    let mut pts = points.iter().map(|&(x, y)| (x as f64, y as f64)).collect::<Vec<_>>();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)));
        
    let cross = |o: &(f64, f64), a: &(f64, f64), b: &(f64, f64)| -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    
    let mut lower = Vec::new();
    for p in &pts {
        while lower.len() >= 2 && cross(&lower[lower.len()-2], &lower[lower.len()-1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(*p);
    }
    
    let mut upper = Vec::new();
    for p in pts.iter().rev() {
        while upper.len() >= 2 && cross(&upper[upper.len()-2], &upper[upper.len()-1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(*p);
    }
    
    lower.pop();
    upper.pop();
    lower.extend(upper);
    
    let mut area = 0.0;
    let n = lower.len();
    if n < 3 { return points.len() as f64; }
    for i in 0..n {
        let j = (i + 1) % n;
        area += lower[i].0 * lower[j].1 - lower[j].0 * lower[i].1;
    }
    (area.abs() / 2.0)
}

/// simple_blob_detect() - Extract blobs (connected components) based on color, size, circularity, inertia, and convexity.
/// @_py: Python interpreter token.
/// @image: Input 2D grayscale image (u8).
/// @min_threshold: Minimum threshold.
/// @max_threshold: Maximum threshold.
/// @threshold_step: Step for threshold iteration.
/// @min_dist_between_blobs: Minimum distance between centers of two blobs.
/// @min_repeatability: Minimum repeatability to accept a blob.
/// @filter_by_color: If true, filters by blob color.
/// @blob_color: Blob color to search (0 or 255).
/// @filter_by_area: If true, filters by blob area size.
/// @min_area: Minimum area.
/// @max_area: Maximum area.
/// @filter_by_circularity: If true, filters by circularity ratio.
/// @min_circularity: Minimum circularity.
/// @max_circularity: Maximum circularity.
/// @filter_by_inertia: If true, filters by inertia ratio.
/// @min_inertia_ratio: Minimum inertia ratio.
/// @max_inertia_ratio: Maximum inertia ratio.
/// @filter_by_convexity: If true, filters by convexity ratio.
/// @min_convexity: Minimum convexity.
/// @max_convexity: Maximum convexity.
///
/// Detects blobs in a grayscale image based on various shape features.
///
/// Return: A vector of detected KeyPoints.
#[pyfunction]
#[pyo3(signature = (
    image,
    min_threshold = 50.0,
    max_threshold = 220.0,
    threshold_step = 10.0,
    min_dist_between_blobs = 10.0,
    min_repeatability = 2,
    filter_by_color = true,
    blob_color = 0,
    filter_by_area = true,
    min_area = 25.0,
    max_area = 5000.0,
    filter_by_circularity = false,
    min_circularity = 0.8,
    max_circularity = 3.4e38,
    filter_by_inertia = false,
    min_inertia_ratio = 0.1,
    max_inertia_ratio = 3.4e38,
    filter_by_convexity = false,
    min_convexity = 0.95,
    max_convexity = 3.4e38,
))]
pub fn simple_blob_detect<'py>(
    _py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    min_threshold: f64,
    max_threshold: f64,
    threshold_step: f64,
    min_dist_between_blobs: f64,
    min_repeatability: usize,
    filter_by_color: bool,
    blob_color: u8,
    filter_by_area: bool,
    min_area: f64,
    max_area: f64,
    filter_by_circularity: bool,
    min_circularity: f64,
    max_circularity: f64,
    filter_by_inertia: bool,
    min_inertia_ratio: f64,
    max_inertia_ratio: f64,
    filter_by_convexity: bool,
    min_convexity: f64,
    max_convexity: f64,
) -> PyResult<Vec<KeyPoint>> {
    let arr = image.as_array();
    let img_2d = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be 2D Grayscale"))?;
    let (h, w) = (img_2d.shape()[0], img_2d.shape()[1]);
    
    let mut thresholds = Vec::new();
    let mut t = min_threshold;
    while t <= max_threshold {
        thresholds.push(t as u8);
        t += threshold_step;
    }
    
    let mut candidates = Vec::new();
    
    for &thresh in &thresholds {
        let mut binary = numpy::ndarray::Array2::<u8>::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                if img_2d[[y, x]] <= thresh {
                    binary[[y, x]] = 255;
                }
            }
        }
        
        let mut labeled = numpy::ndarray::Array2::<i32>::zeros((h, w));
        let mut next_label = 1;
        
        for y in 0..h {
            for x in 0..w {
                if binary[[y, x]] == 255 && labeled[[y, x]] == 0 {
                    let label = next_label;
                    next_label += 1;
                    
                    let mut pixels = Vec::new();
                    let mut queue = std::collections::VecDeque::new();
                    queue.push_back((y, x));
                    labeled[[y, x]] = label;
                    
                    while let Some((cy, cx)) = queue.pop_front() {
                        pixels.push((cx, cy));
                        let neighbors = [
                            (cy as isize - 1, cx as isize),
                            (cy as isize + 1, cx as isize),
                            (cy as isize, cx as isize - 1),
                            (cy as isize, cx as isize + 1),
                        ];
                        for &(ny, nx) in &neighbors {
                            if ny >= 0 && ny < h as isize && nx >= 0 && nx < w as isize {
                                let (ny_u, nx_u) = (ny as usize, nx as usize);
                                if binary[[ny_u, nx_u]] == 255 && labeled[[ny_u, nx_u]] == 0 {
                                    labeled[[ny_u, nx_u]] = label;
                                    queue.push_back((ny_u, nx_u));
                                }
                            }
                        }
                    }
                    
                    let area = pixels.len() as f64;
                    if filter_by_area && (area < min_area || area > max_area) {
                        continue;
                    }
                    
                    let mut sum_x = 0.0;
                    let mut sum_y = 0.0;
                    for &(cx, cy) in &pixels {
                        sum_x += cx as f64;
                        sum_y += cy as f64;
                    }
                    let cx_val = sum_x / area;
                    let cy_val = sum_y / area;
                    
                    if filter_by_color {
                        let center_val = img_2d[[cy_val.round() as usize, cx_val.round() as usize]];
                        if (blob_color == 0 && center_val > 128) || (blob_color == 255 && center_val <= 128) {
                            continue;
                        }
                    }
                    
                    if filter_by_circularity {
                        let mut perimeter = 0.0;
                        for &(px, py) in &pixels {
                            let mut is_boundary = false;
                            let neighbors = [
                                (py as isize - 1, px as isize),
                                (py as isize + 1, px as isize),
                                (py as isize, px as isize - 1),
                                (py as isize, px as isize + 1),
                            ];
                            for &(ny, nx) in &neighbors {
                                if ny < 0 || ny >= h as isize || nx < 0 || nx >= w as isize {
                                    is_boundary = true;
                                    break;
                                } else if binary[[ny as usize, nx as usize]] == 0 {
                                    is_boundary = true;
                                    break;
                                }
                            }
                            if is_boundary {
                                perimeter += 1.0;
                            }
                        }
                        let circularity = if perimeter > 0.0 {
                            4.0 * std::f64::consts::PI * area / (perimeter * perimeter)
                        } else {
                            0.0
                        };
                        if circularity < min_circularity || circularity > max_circularity {
                            continue;
                        }
                    }
                    
                    if filter_by_inertia {
                        let mut mu20 = 0.0;
                        let mut mu02 = 0.0;
                        let mut mu11 = 0.0;
                        for &(px, py) in &pixels {
                            let dx = px as f64 - cx_val;
                            let dy = py as f64 - cy_val;
                            mu20 += dx * dx;
                            mu02 += dy * dy;
                            mu11 += dx * dy;
                        }
                        let d = ((mu20 - mu02) * (mu20 - mu02) + 4.0 * mu11 * mu11).sqrt();
                        let l1 = (mu20 + mu02 + d) / 2.0;
                        let l2 = (mu20 + mu02 - d) / 2.0;
                        let inertia_ratio = if l1 > 1e-6 {
                            (l2 / l1).sqrt()
                        } else {
                            0.0
                        };
                        if inertia_ratio < min_inertia_ratio || inertia_ratio > max_inertia_ratio {
                            continue;
                        }
                    }
                    
                    if filter_by_convexity {
                        let hull_area = convex_hull_area(&pixels);
                        let convexity = if hull_area > 0.0 {
                            area / hull_area
                        } else {
                            0.0
                        };
                        if convexity < min_convexity || convexity > max_convexity {
                            continue;
                        }
                    }
                    
                    candidates.push(KeyPoint {
                        pt: (cx_val, cy_val),
                        size: 2.0 * (area / std::f64::consts::PI).sqrt() as f32,
                        angle: -1.0,
                        response: area as f32,
                        octave: 0,
                        class_id: -1,
                    });
                }
            }
        }
    }
    
    // Group candidates that are closer than min_dist_between_blobs
    let mut groups: Vec<Vec<KeyPoint>> = Vec::new();
    for cand in candidates {
        let mut found_group = false;
        for gp in &mut groups {
            let mut close = false;
            for member in gp.iter() {
                let dx = member.pt.0 - cand.pt.0;
                let dy = member.pt.1 - cand.pt.1;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < min_dist_between_blobs {
                    close = true;
                    break;
                }
            }
            if close {
                gp.push(cand.clone());
                found_group = true;
                break;
            }
        }
        if !found_group {
            groups.push(vec![cand]);
        }
    }
    
    let mut final_kps = Vec::new();
    for gp in groups {
        if gp.len() >= min_repeatability {
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_size = 0.0;
            let mut sum_resp = 0.0;
            for member in &gp {
                sum_x += member.pt.0;
                sum_y += member.pt.1;
                sum_size += member.size;
                sum_resp += member.response;
            }
            let n = gp.len() as f64;
            let n_f32 = gp.len() as f32;
            final_kps.push(KeyPoint {
                pt: (sum_x / n, sum_y / n),
                size: sum_size / n_f32,
                angle: -1.0,
                response: sum_resp / n_f32,
                octave: 0,
                class_id: -1,
            });
        }
    }
    
    Ok(final_kps)
}

/// bf_match() - Brute-force descriptor matcher.
/// @_py: Python interpreter token.
/// @query_descriptors: Matrix of query descriptors (u8 or f32).
/// @train_descriptors: Matrix of train descriptors (u8 or f32).
/// @cross_check: If true, performs bidirectional matching.
/// @norm_type: Normalization type ("L1", "L2", or "HAMMING").
///
/// Finds the best matches for each descriptor in the query set from the train set.
///
/// Return: A vector of DMatch instances.
#[pyfunction]
#[pyo3(signature = (query_descriptors, train_descriptors, cross_check = false, norm_type = "L2"))]
pub fn bf_match<'py>(
    _py: Python<'py>,
    query_descriptors: &pyo3::PyAny,
    train_descriptors: &pyo3::PyAny,
    cross_check: bool,
    norm_type: &str,
) -> PyResult<Vec<DMatch>> {
    if let Ok(q_u8) = query_descriptors.extract::<PyReadonlyArrayDyn<u8>>() {
        let t_u8 = train_descriptors.extract::<PyReadonlyArrayDyn<u8>>()?;
        let q_arr = q_u8.as_array();
        let t_arr = t_u8.as_array();
        let q_2d = q_arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Descriptors must be 2D"))?;
        let t_2d = t_arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Descriptors must be 2D"))?;
        
        let q_rows = q_2d.shape()[0];
        let t_rows = t_2d.shape()[0];
        let cols = q_2d.shape()[1];
        
        let mut q_to_t = vec![( -1, f32::MAX ); q_rows];
        for i in 0..q_rows {
            let mut best_dist = u32::MAX;
            let mut best_idx = -1;
            let q_row = q_2d.row(i);
            for j in 0..t_rows {
                let t_row = t_2d.row(j);
                let mut dist = 0;
                for k in 0..cols {
                    dist += (q_row[k] ^ t_row[k]).count_ones();
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j as i32;
                }
            }
            q_to_t[i] = (best_idx, best_dist as f32);
        }
        
        let mut matches = Vec::new();
        if cross_check {
            let mut t_to_q = vec![( -1, f32::MAX ); t_rows];
            for j in 0..t_rows {
                let mut best_dist = u32::MAX;
                let mut best_idx = -1;
                let t_row = t_2d.row(j);
                for i in 0..q_rows {
                    let q_row = q_2d.row(i);
                    let mut dist = 0;
                    for k in 0..cols {
                        dist += (q_row[k] ^ t_row[k]).count_ones();
                    }
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = i as i32;
                    }
                }
                t_to_q[j] = (best_idx, best_dist as f32);
            }
            
            for i in 0..q_rows {
                let (best_t, dist) = q_to_t[i];
                if best_t != -1 {
                    let (best_q, _) = t_to_q[best_t as usize];
                    if best_q == i as i32 {
                        matches.push(DMatch {
                            query_idx: i as i32,
                            train_idx: best_t,
                            img_idx: 0,
                            distance: dist,
                        });
                    }
                }
            }
        } else {
            for i in 0..q_rows {
                let (best_t, dist) = q_to_t[i];
                if best_t != -1 {
                    matches.push(DMatch {
                        query_idx: i as i32,
                        train_idx: best_t,
                        img_idx: 0,
                        distance: dist,
                    });
                }
            }
        }
        Ok(matches)
    } else if let Ok(q_f32) = query_descriptors.extract::<PyReadonlyArrayDyn<f32>>() {
        let t_f32 = train_descriptors.extract::<PyReadonlyArrayDyn<f32>>()?;
        let q_arr = q_f32.as_array();
        let t_arr = t_f32.as_array();
        let q_2d = q_arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Descriptors must be 2D"))?;
        let t_2d = t_arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Descriptors must be 2D"))?;
        
        let q_rows = q_2d.shape()[0];
        let t_rows = t_2d.shape()[0];
        let cols = q_2d.shape()[1];
        
        let is_l1 = norm_type == "L1";
        
        let mut q_to_t = vec![( -1, f32::MAX ); q_rows];
        for i in 0..q_rows {
            let mut best_dist = f32::MAX;
            let mut best_idx = -1;
            let q_row = q_2d.row(i);
            for j in 0..t_rows {
                let t_row = t_2d.row(j);
                let mut dist = 0.0;
                for k in 0..cols {
                    let diff = q_row[k] - t_row[k];
                    if is_l1 {
                        dist += diff.abs();
                    } else {
                        dist += diff * diff;
                    }
                }
                if !is_l1 {
                    dist = dist.sqrt();
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j as i32;
                }
            }
            q_to_t[i] = (best_idx, best_dist);
        }
        
        let mut matches = Vec::new();
        if cross_check {
            let mut t_to_q = vec![( -1, f32::MAX ); t_rows];
            for j in 0..t_rows {
                let mut best_dist = f32::MAX;
                let mut best_idx = -1;
                let t_row = t_2d.row(j);
                for i in 0..q_rows {
                    let q_row = q_2d.row(i);
                    let mut dist = 0.0;
                    for k in 0..cols {
                        let diff = q_row[k] - t_row[k];
                        if is_l1 {
                            dist += diff.abs();
                        } else {
                            dist += diff * diff;
                        }
                    }
                    if !is_l1 {
                        dist = dist.sqrt();
                    }
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = i as i32;
                    }
                }
                t_to_q[j] = (best_idx, best_dist);
            }
            
            for i in 0..q_rows {
                let (best_t, dist) = q_to_t[i];
                if best_t != -1 {
                    let (best_q, _) = t_to_q[best_t as usize];
                    if best_q == i as i32 {
                        matches.push(DMatch {
                            query_idx: i as i32,
                            train_idx: best_t,
                            img_idx: 0,
                            distance: dist,
                        });
                    }
                }
            }
        } else {
            for i in 0..q_rows {
                let (best_t, dist) = q_to_t[i];
                if best_t != -1 {
                    matches.push(DMatch {
                        query_idx: i as i32,
                        train_idx: best_t,
                        img_idx: 0,
                        distance: dist,
                    });
                }
            }
        }
        Ok(matches)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("Descriptors must be either np.uint8 or np.float32"))
    }
}

/// knn_match() - K-Nearest Neighbor descriptor matcher.
/// @_py: Python interpreter token.
/// @query_descriptors: Matrix of query descriptors (u8 or f32).
/// @train_descriptors: Matrix of train descriptors (u8 or f32).
/// @k: Number of nearest neighbors to search.
/// @norm_type: Normalization type ("L1", "L2", or "HAMMING").
///
/// Finds the k best matches for each descriptor in the query set.
///
/// Return: A vector of vectors containing up to k matches per query descriptor.
#[pyfunction]
#[pyo3(signature = (query_descriptors, train_descriptors, k, norm_type = "L2"))]
pub fn knn_match<'py>(
    _py: Python<'py>,
    query_descriptors: &pyo3::PyAny,
    train_descriptors: &pyo3::PyAny,
    k: usize,
    norm_type: &str,
) -> PyResult<Vec<Vec<DMatch>>> {
    if let Ok(q_u8) = query_descriptors.extract::<PyReadonlyArrayDyn<u8>>() {
        let t_u8 = train_descriptors.extract::<PyReadonlyArrayDyn<u8>>()?;
        let q_arr = q_u8.as_array();
        let t_arr = t_u8.as_array();
        let q_2d = q_arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Descriptors must be 2D"))?;
        let t_2d = t_arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Descriptors must be 2D"))?;
        
        let q_rows = q_2d.shape()[0];
        let t_rows = t_2d.shape()[0];
        let cols = q_2d.shape()[1];
        
        let mut all_matches = Vec::with_capacity(q_rows);
        
        for i in 0..q_rows {
            let q_row = q_2d.row(i);
            let mut best_matches = Vec::with_capacity(k);
            
            for j in 0..t_rows {
                let t_row = t_2d.row(j);
                let mut dist = 0;
                for idx in 0..cols {
                    dist += (q_row[idx] ^ t_row[idx]).count_ones();
                }
                
                let d_match = DMatch {
                    query_idx: i as i32,
                    train_idx: j as i32,
                    img_idx: 0,
                    distance: dist as f32,
                };
                
                let pos = best_matches.binary_search_by(|m: &DMatch| m.distance.partial_cmp(&d_match.distance).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or_else(|x| x);
                if pos < k {
                    best_matches.insert(pos, d_match);
                    if best_matches.len() > k {
                        best_matches.pop();
                    }
                }
            }
            all_matches.push(best_matches);
        }
        Ok(all_matches)
    } else if let Ok(q_f32) = query_descriptors.extract::<PyReadonlyArrayDyn<f32>>() {
        let t_f32 = train_descriptors.extract::<PyReadonlyArrayDyn<f32>>()?;
        let q_arr = q_f32.as_array();
        let t_arr = t_f32.as_array();
        let q_2d = q_arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Descriptors must be 2D"))?;
        let t_2d = t_arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Descriptors must be 2D"))?;
        
        let q_rows = q_2d.shape()[0];
        let t_rows = t_2d.shape()[0];
        let cols = q_2d.shape()[1];
        
        let is_l1 = norm_type == "L1";
        let mut all_matches = Vec::with_capacity(q_rows);
        
        for i in 0..q_rows {
            let q_row = q_2d.row(i);
            let mut best_matches = Vec::with_capacity(k);
            
            for j in 0..t_rows {
                let t_row = t_2d.row(j);
                let mut dist = 0.0;
                for idx in 0..cols {
                    let diff = q_row[idx] - t_row[idx];
                    if is_l1 {
                        dist += diff.abs();
                    } else {
                        dist += diff * diff;
                    }
                }
                if !is_l1 {
                    dist = dist.sqrt();
                }
                
                let d_match = DMatch {
                    query_idx: i as i32,
                    train_idx: j as i32,
                    img_idx: 0,
                    distance: dist,
                };
                
                let pos = best_matches.binary_search_by(|m: &DMatch| m.distance.partial_cmp(&d_match.distance).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or_else(|x| x);
                if pos < k {
                    best_matches.insert(pos, d_match);
                    if best_matches.len() > k {
                        best_matches.pop();
                    }
                }
            }
            all_matches.push(best_matches);
        }
        Ok(all_matches)
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("Descriptors must be either np.uint8 or np.float32"))
    }
}

// ==========================================
// DRAWING & HOMOGRAPHY HELPERS
// ==========================================

fn draw_line_3d(arr: &mut ndarray::Array3<u8>, pt1: (i32, i32), pt2: (i32, i32), color: [u8; 3]) {
    let (h, w) = (arr.shape()[0] as i32, arr.shape()[1] as i32);
    let x1 = pt1.0;
    let y1 = pt1.1;
    let x2 = pt2.0;
    let y2 = pt2.1;
    
    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();
    let sx = if x1 < x2 { 1 } else { -1 };
    let sy = if y1 < y2 { 1 } else { -1 };
    let mut err = dx - dy;
    
    let mut cx = x1;
    let mut cy = y1;
    
    loop {
        if cx >= 0 && cx < w && cy >= 0 && cy < h {
            arr[[cy as usize, cx as usize, 0]] = color[0];
            arr[[cy as usize, cx as usize, 1]] = color[1];
            arr[[cy as usize, cx as usize, 2]] = color[2];
        }
        if cx == x2 && cy == y2 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            cx += sx;
        }
        if e2 < dx {
            err += dx;
            cy += sy;
        }
    }
}

fn draw_circle_3d(arr: &mut ndarray::Array3<u8>, center: (i32, i32), radius: i32, color: [u8; 3]) {
    let (h, w) = (arr.shape()[0] as i32, arr.shape()[1] as i32);
    let cx = center.0;
    let cy = center.1;
    
    let mut x = 0;
    let mut y = radius;
    let mut d = 3 - 2 * radius;
    
    let mut draw_pts = |px: i32, py: i32| {
        let pts = [
            (cx + px, cy + py), (cx - px, cy + py),
            (cx + px, cy - py), (cx - px, cy - py),
            (cx + py, cy + px), (cx - py, cy + px),
            (cx + py, cy - px), (cx - py, cy - px),
        ];
        for &(tx, ty) in &pts {
            if tx >= 0 && tx < w && ty >= 0 && ty < h {
                arr[[ty as usize, tx as usize, 0]] = color[0];
                arr[[ty as usize, tx as usize, 1]] = color[1];
                arr[[ty as usize, tx as usize, 2]] = color[2];
            }
        }
    };
    
    draw_pts(x, y);
    while y >= x {
        x += 1;
        if d > 0 {
            y -= 1;
            d = d + 4 * (x - y) + 10;
        } else {
            d = d + 4 * x + 6;
        }
        draw_pts(x, y);
    }
}

fn extract_points(points: &pyo3::PyAny) -> PyResult<Vec<(f64, f64)>> {
    if let Ok(arr_f64) = points.extract::<PyReadonlyArrayDyn<f64>>() {
        let arr = arr_f64.as_array();
        let shape = arr.shape();
        if shape.len() == 2 && shape[1] == 2 {
            let mut v = Vec::new();
            for i in 0..shape[0] {
                v.push((arr[[i, 0]], arr[[i, 1]]));
            }
            return Ok(v);
        } else if shape.len() == 3 && shape[1] == 1 && shape[2] == 2 {
            let mut v = Vec::new();
            for i in 0..shape[0] {
                v.push((arr[[i, 0, 0]], arr[[i, 0, 1]]));
            }
            return Ok(v);
        }
    } else if let Ok(arr_f32) = points.extract::<PyReadonlyArrayDyn<f32>>() {
        let arr = arr_f32.as_array();
        let shape = arr.shape();
        if shape.len() == 2 && shape[1] == 2 {
            let mut v = Vec::new();
            for i in 0..shape[0] {
                v.push((arr[[i, 0]] as f64, arr[[i, 1]] as f64));
            }
            return Ok(v);
        } else if shape.len() == 3 && shape[1] == 1 && shape[2] == 2 {
            let mut v = Vec::new();
            for i in 0..shape[0] {
                v.push((arr[[i, 0, 0]] as f64, arr[[i, 0, 1]] as f64));
            }
            return Ok(v);
        }
    } else if let Ok(list) = points.extract::<Vec<(f64, f64)>>() {
        return Ok(list);
    }
    Err(pyo3::exceptions::PyTypeError::new_err("Points must be Nx2 or Nx1x2 array or list of coordinate pairs"))
}

fn normalize_points(pts: &[(f64, f64)]) -> (Vec<(f64, f64)>, ndarray::Array2<f64>) {
    let n = pts.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for &(x, y) in pts {
        sum_x += x;
        sum_y += y;
    }
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    
    let mut sum_dist = 0.0;
    for &(x, y) in pts {
        let dx = x - mean_x;
        let dy = y - mean_y;
        sum_dist += (dx * dx + dy * dy).sqrt();
    }
    let mean_dist = sum_dist / n;
    let scale = if mean_dist > 0.0 {
        2.0f64.sqrt() / mean_dist
    } else {
        1.0
    };
    
    let mut norm_pts = Vec::with_capacity(pts.len());
    for &(x, y) in pts {
        norm_pts.push(((x - mean_x) * scale, (y - mean_y) * scale));
    }
    
    let mut t = ndarray::Array2::<f64>::zeros((3, 3));
    t[[0, 0]] = scale;
    t[[0, 2]] = -mean_x * scale;
    t[[1, 1]] = scale;
    t[[1, 2]] = -mean_y * scale;
    t[[2, 2]] = 1.0;
    
    (norm_pts, t)
}

fn invert_transform(t: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let scale = t[[0, 0]];
    let mean_x = -t[[0, 2]] / scale;
    let mean_y = -t[[1, 2]] / scale;
    
    let mut inv = ndarray::Array2::<f64>::zeros((3, 3));
    inv[[0, 0]] = 1.0 / scale;
    inv[[0, 2]] = mean_x;
    inv[[1, 1]] = 1.0 / scale;
    inv[[1, 2]] = mean_y;
    inv[[2, 2]] = 1.0;
    inv
}

fn matmul_3x3(a: &ndarray::Array2<f64>, b: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let mut c = ndarray::Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += a[[i, k]] * b[[k, j]];
            }
            c[[i, j]] = sum;
        }
    }
    c
}

fn jacobi_eigen(mut a: ndarray::Array2<f64>) -> (ndarray::Array1<f64>, ndarray::Array2<f64>) {
    let n = 9;
    let mut v = ndarray::Array2::<f64>::eye(n);
    let max_rotations = 100;
    let eps = 1e-15;
    
    for _ in 0..max_rotations {
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 0;
        for i in 0..n {
            for j in (i+1)..n {
                let abs_val = a[[i, j]].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    p = i;
                    q = j;
                }
            }
        }
        
        if max_val < eps {
            break;
        }
        
        let app = a[[p, p]];
        let aqq = a[[q, q]];
        let apq = a[[p, q]];
        let phi = 0.5 * (2.0 * apq).atan2(aqq - app);
        let c = phi.cos();
        let s = phi.sin();
        
        let mut next_a = a.clone();
        next_a[[p, p]] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
        next_a[[q, q]] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
        next_a[[p, q]] = 0.0;
        next_a[[q, p]] = 0.0;
        
        for i in 0..n {
            if i != p && i != q {
                let aip = a[[i, p]];
                let aiq = a[[i, q]];
                next_a[[i, p]] = c * aip - s * aiq;
                next_a[[p, i]] = next_a[[i, p]];
                next_a[[i, q]] = s * aip + c * aiq;
                next_a[[q, i]] = next_a[[i, q]];
            }
        }
        a = next_a;
        
        for i in 0..n {
            let vip = v[[i, p]];
            let viq = v[[i, q]];
            v[[i, p]] = c * vip - s * viq;
            v[[i, q]] = s * vip + c * viq;
        }
    }
    
    let mut eigenvalues = ndarray::Array1::<f64>::zeros(n);
    for i in 0..n {
        eigenvalues[i] = a[[i, i]];
    }
    (eigenvalues, v)
}

fn compute_homography_dlt(src: &[(f64, f64)], dst: &[(f64, f64)]) -> Option<ndarray::Array2<f64>> {
    if src.len() < 4 {
        return None;
    }
    let (src_norm, t_src) = normalize_points(src);
    let (dst_norm, t_dst) = normalize_points(dst);
    
    let n = src.len();
    let mut a = ndarray::Array2::<f64>::zeros((2 * n, 9));
    for i in 0..n {
        let (x, y) = src_norm[i];
        let (xp, yp) = dst_norm[i];
        a[[2 * i, 0]] = -x;
        a[[2 * i, 1]] = -y;
        a[[2 * i, 2]] = -1.0;
        a[[2 * i, 6]] = x * xp;
        a[[2 * i, 7]] = y * xp;
        a[[2 * i, 8]] = xp;
        
        a[[2 * i + 1, 3]] = -x;
        a[[2 * i + 1, 4]] = -y;
        a[[2 * i + 1, 5]] = -1.0;
        a[[2 * i + 1, 6]] = x * yp;
        a[[2 * i + 1, 7]] = y * yp;
        a[[2 * i + 1, 8]] = yp;
    }
    
    let mut ata = ndarray::Array2::<f64>::zeros((9, 9));
    for j in 0..9 {
        for k in 0..9 {
            let mut sum = 0.0;
            for i in 0..(2 * n) {
                sum += a[[i, j]] * a[[i, k]];
            }
            ata[[j, k]] = sum;
        }
    }
    
    let (eigenvals, eigenvectors) = jacobi_eigen(ata);
    
    let mut min_val = f64::MAX;
    let mut min_idx = 0;
    for i in 0..9 {
        if eigenvals[i] < min_val {
            min_val = eigenvals[i];
            min_idx = i;
        }
    }
    
    let mut h_norm = ndarray::Array2::<f64>::zeros((3, 3));
    for r in 0..3 {
        for c in 0..3 {
            h_norm[[r, c]] = eigenvectors[[r * 3 + c, min_idx]];
        }
    }
    
    let t_dst_inv = invert_transform(&t_dst);
    let tmp = matmul_3x3(&t_dst_inv, &h_norm);
    let mut h = matmul_3x3(&tmp, &t_src);
    
    let h22 = h[[2, 2]];
    if h22.abs() > 1e-9 {
        h.mapv_inplace(|x| x / h22);
    }
    Some(h)
}

struct KdNode {
    idx: usize,
    axis: usize,
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
}

fn build_kdtree(indices: &mut [usize], descriptors: &ndarray::ArrayView2<f32>, depth: usize) -> Option<Box<KdNode>> {
    if indices.is_empty() {
        return None;
    }
    let k = descriptors.shape()[1];
    let axis = depth % k;
    
    indices.sort_by(|&a, &b| {
        descriptors[[a, axis]].partial_cmp(&descriptors[[b, axis]]).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    let median = indices.len() / 2;
    let idx = indices[median];
    
    let left = build_kdtree(&mut indices[..median], descriptors, depth + 1);
    let right = build_kdtree(&mut indices[median + 1..], descriptors, depth + 1);
    
    Some(Box::new(KdNode {
        idx,
        axis,
        left,
        right,
    }))
}

fn kdtree_search(
    node: &Option<Box<KdNode>>,
    query: &ndarray::ArrayView1<f32>,
    descriptors: &ndarray::ArrayView2<f32>,
    best_idx: &mut usize,
    best_dist: &mut f32,
) {
    let node = match node {
        Some(n) => n,
        None => return,
    };
    
    let d_val = descriptors.row(node.idx);
    let mut dist = 0.0;
    for i in 0..query.len() {
        dist += (query[i] - d_val[i]).powi(2);
    }
    dist = dist.sqrt();
    
    if dist < *best_dist {
        *best_dist = dist;
        *best_idx = node.idx;
    }
    
    let axis = node.axis;
    let diff = query[axis] - d_val[axis];
    
    let (first, second) = if diff < 0.0 {
        (&node.left, &node.right)
    } else {
        (&node.right, &node.left)
    };
    
    kdtree_search(first, query, descriptors, best_idx, best_dist);
    
    if diff.abs() < *best_dist {
        kdtree_search(second, query, descriptors, best_idx, best_dist);
    }
}

// ==========================================
// EXPORTED FUNCTIONS
// ==========================================

/// draw_keypoints() - Draws keypoints on an image.
/// @py: Python interpreter token.
/// @image: Input image array (u8). Can be 2D grayscale or 3D color.
/// @keypoints: Vector of KeyPoints to draw.
/// @color: Color of the keypoints (scalar or tuple). If None, random colors are used.
/// @flags: Flags setting drawing features.
///     - 0 (DEFAULT): draws keypoints as circles.
///     - 4 (DRAW_RICH_KEYPOINTS): draws circle with size and orientation.
///
/// Draws a circle at each keypoint location. If flags is DRAW_RICH_KEYPOINTS,
/// the circle's radius matches keypoint size, and a line is drawn indicating orientation.
///
/// Return: Converted/drawn image array (u8).
#[pyfunction]
#[pyo3(name = "drawKeypoints")]
#[pyo3(signature = (image, keypoints, color = None, flags = 0))]
pub fn draw_keypoints<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    keypoints: Vec<KeyPoint>,
    color: Option<PyObject>,
    flags: i32,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr = image.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();
    
    let mut out = if ndim == 2 {
        let h = shape[0];
        let w = shape[1];
        let mut three_chan = ndarray::Array3::<u8>::zeros((h, w, 3));
        for r in 0..h {
            for c in 0..w {
                let val = arr[[r, c]];
                three_chan[[r, c, 0]] = val;
                three_chan[[r, c, 1]] = val;
                three_chan[[r, c, 2]] = val;
            }
        }
        three_chan
    } else if ndim == 3 && shape[2] == 3 {
        arr.into_dimensionality::<numpy::ndarray::Ix3>().unwrap().to_owned()
    } else if ndim == 3 && shape[2] == 1 {
        let h = shape[0];
        let w = shape[1];
        let mut three_chan = ndarray::Array3::<u8>::zeros((h, w, 3));
        for r in 0..h {
            for c in 0..w {
                let val = arr[[r, c, 0]];
                three_chan[[r, c, 0]] = val;
                three_chan[[r, c, 1]] = val;
                three_chan[[r, c, 2]] = val;
            }
        }
        three_chan
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err("Image must be 2D or 3D 1/3-channel"));
    };
    
    let parsed_color: Option<[u8; 3]> = if let Some(c_obj) = color {
        if let Ok(tuple) = c_obj.extract::<(u8, u8, u8)>(py) {
            Some([tuple.0, tuple.1, tuple.2])
        } else if let Ok(list) = c_obj.extract::<Vec<u8>>(py) {
            if list.len() >= 3 {
                Some([list[0], list[1], list[2]])
            } else if list.len() == 1 {
                Some([list[0], list[0], list[0]])
            } else {
                None
            }
        } else if let Ok(val) = c_obj.extract::<u8>(py) {
            Some([val, val, val])
        } else {
            None
        }
    } else {
        None
    };
    
    for kp in keypoints {
        let pt = (kp.pt.0.round() as i32, kp.pt.1.round() as i32);
        let r_val = ((kp.pt.0 * 123.456 + kp.pt.1 * 456.789) as u32 % 256) as u8;
        let g_val = ((kp.pt.0 * 789.123 + kp.pt.1 * 123.789) as u32 % 256) as u8;
        let b_val = ((kp.pt.0 * 456.123 + kp.pt.1 * 789.789) as u32 % 256) as u8;
        let k_color = parsed_color.unwrap_or([b_val, g_val, r_val]);
        
        if flags == 4 {
            // DRAW_RICH_KEYPOINTS
            let radius = (kp.size / 2.0).round() as i32;
            let radius = if radius < 1 { 3 } else { radius };
            draw_circle_3d(&mut out, pt, radius, k_color);
            if kp.angle >= 0.0 {
                let angle_rad = kp.angle.to_radians() as f64;
                let line_end_x = (kp.pt.0 + radius as f64 * angle_rad.cos()).round() as i32;
                let line_end_y = (kp.pt.1 + radius as f64 * angle_rad.sin()).round() as i32;
                draw_line_3d(&mut out, pt, (line_end_x, line_end_y), k_color);
            }
        } else {
            // DEFAULT
            draw_circle_3d(&mut out, pt, 3, k_color);
        }
    }
    
    Ok(out.into_pyarray(py).to_dyn())
}

/// draw_matches() - Draws keypoint matches from two images side-by-side.
/// @py: Python interpreter token.
/// @img1: First input image (u8).
/// @keypoints1: Keypoints from the first image.
/// @img2: Second input image (u8).
/// @keypoints2: Keypoints from the second image.
/// @matches: Matches between keypoints (vector of DMatch).
/// @match_color: Color of matches and connection lines. If None, random colors are used.
/// @single_point_color: Color of single keypoints (not matched). If None, random colors are used.
///
/// Concatenates the two images side-by-side and draws lines connecting matched keypoints.
///
/// Return: Combined image array with drawn matches (u8).
#[pyfunction]
#[pyo3(name = "drawMatches")]
#[pyo3(signature = (img1, keypoints1, img2, keypoints2, matches, match_color = None, single_point_color = None))]
pub fn draw_matches<'py>(
    py: Python<'py>,
    img1: PyReadonlyArrayDyn<'py, u8>,
    keypoints1: Vec<KeyPoint>,
    img2: PyReadonlyArrayDyn<'py, u8>,
    keypoints2: Vec<KeyPoint>,
    matches: Vec<DMatch>,
    match_color: Option<PyObject>,
    single_point_color: Option<PyObject>,
) -> PyResult<&'py PyArrayDyn<u8>> {
    let arr1 = img1.as_array();
    let arr2 = img2.as_array();
    let (h1, w1) = (arr1.shape()[0], arr1.shape()[1]);
    let (h2, w2) = (arr2.shape()[0], arr2.shape()[1]);
    
    let max_h = h1.max(h2);
    let total_w = w1 + w2;
    let mut out = ndarray::Array3::<u8>::zeros((max_h, total_w, 3));
    
    // Copy img1 to left side
    let ndim1 = arr1.ndim();
    for r in 0..h1 {
        for c in 0..w1 {
            let (b, g, r_val) = if ndim1 == 2 {
                let v = arr1[[r, c]];
                (v, v, v)
            } else if arr1.shape()[2] == 3 {
                (arr1[[r, c, 0]], arr1[[r, c, 1]], arr1[[r, c, 2]])
            } else {
                let v = arr1[[r, c, 0]];
                (v, v, v)
            };
            out[[r, c, 0]] = b;
            out[[r, c, 1]] = g;
            out[[r, c, 2]] = r_val;
        }
    }
    
    // Copy img2 to right side
    let ndim2 = arr2.ndim();
    for r in 0..h2 {
        for c in 0..w2 {
            let (b, g, r_val) = if ndim2 == 2 {
                let v = arr2[[r, c]];
                (v, v, v)
            } else if arr2.shape()[2] == 3 {
                (arr2[[r, c, 0]], arr2[[r, c, 1]], arr2[[r, c, 2]])
            } else {
                let v = arr2[[r, c, 0]];
                (v, v, v)
            };
            out[[r, c + w1, 0]] = b;
            out[[r, c + w1, 1]] = g;
            out[[r, c + w1, 2]] = r_val;
        }
    }
    
    let parsed_match_color: Option<[u8; 3]> = if let Some(c_obj) = match_color {
        if let Ok(tuple) = c_obj.extract::<(u8, u8, u8)>(py) {
            Some([tuple.0, tuple.1, tuple.2])
        } else if let Ok(list) = c_obj.extract::<Vec<u8>>(py) {
            if list.len() >= 3 { Some([list[0], list[1], list[2]]) } else { None }
        } else { None }
    } else {
        None
    };
    
    let parsed_single_color: Option<[u8; 3]> = if let Some(c_obj) = single_point_color {
        if let Ok(tuple) = c_obj.extract::<(u8, u8, u8)>(py) {
            Some([tuple.0, tuple.1, tuple.2])
        } else if let Ok(list) = c_obj.extract::<Vec<u8>>(py) {
            if list.len() >= 3 { Some([list[0], list[1], list[2]]) } else { None }
        } else { None }
    } else {
        None
    };
    
    // Set of matched indices to draw single keypoints later
    let mut matched_kps1 = vec![false; keypoints1.len()];
    let mut matched_kps2 = vec![false; keypoints2.len()];
    
    for m in &matches {
        let q_idx = m.query_idx as usize;
        let t_idx = m.train_idx as usize;
        if q_idx < keypoints1.len() && t_idx < keypoints2.len() {
            matched_kps1[q_idx] = true;
            matched_kps2[t_idx] = true;
            
            let pt1 = (keypoints1[q_idx].pt.0.round() as i32, keypoints1[q_idx].pt.1.round() as i32);
            let pt2 = ((keypoints2[t_idx].pt.0.round() as i32) + w1 as i32, keypoints2[t_idx].pt.1.round() as i32);
            
            let r_val = ((keypoints1[q_idx].pt.0 * 123.456 + keypoints1[q_idx].pt.1 * 456.789) as u32 % 256) as u8;
            let g_val = ((keypoints1[q_idx].pt.0 * 789.123 + keypoints1[q_idx].pt.1 * 123.789) as u32 % 256) as u8;
            let b_val = ((keypoints1[q_idx].pt.0 * 456.123 + keypoints1[q_idx].pt.1 * 789.789) as u32 % 256) as u8;
            let k_color = parsed_match_color.unwrap_or([b_val, g_val, r_val]);
            
            draw_circle_3d(&mut out, pt1, 4, k_color);
            draw_circle_3d(&mut out, pt2, 4, k_color);
            draw_line_3d(&mut out, pt1, pt2, k_color);
        }
    }
    
    // Draw unmatched keypoints on image 1
    for (i, kp) in keypoints1.iter().enumerate() {
        if !matched_kps1[i] {
            let pt = (kp.pt.0.round() as i32, kp.pt.1.round() as i32);
            let r_val = ((kp.pt.0 * 123.456 + kp.pt.1 * 456.789) as u32 % 256) as u8;
            let g_val = ((kp.pt.0 * 789.123 + kp.pt.1 * 123.789) as u32 % 256) as u8;
            let b_val = ((kp.pt.0 * 456.123 + kp.pt.1 * 789.789) as u32 % 256) as u8;
            let k_color = parsed_single_color.unwrap_or([b_val, g_val, r_val]);
            draw_circle_3d(&mut out, pt, 3, k_color);
        }
    }
    
    // Draw unmatched keypoints on image 2
    for (i, kp) in keypoints2.iter().enumerate() {
        if !matched_kps2[i] {
            let pt = ((kp.pt.0.round() as i32) + w1 as i32, kp.pt.1.round() as i32);
            let r_val = ((kp.pt.0 * 123.456 + kp.pt.1 * 456.789) as u32 % 256) as u8;
            let g_val = ((kp.pt.0 * 789.123 + kp.pt.1 * 123.789) as u32 % 256) as u8;
            let b_val = ((kp.pt.0 * 456.123 + kp.pt.1 * 789.789) as u32 % 256) as u8;
            let k_color = parsed_single_color.unwrap_or([b_val, g_val, r_val]);
            draw_circle_3d(&mut out, pt, 3, k_color);
        }
    }
    
    Ok(out.into_pyarray(py).to_dyn())
}

/// find_homography() - Find homography matrix using DLT and optional RANSAC.
/// @py: Python interpreter token.
/// @src_points: Source coordinates (list of tuples or Nx2 array).
/// @dst_points: Destination coordinates (list of tuples or Nx2 array).
/// @method: Parameter estimation method. 0 = Least Squares (DLT), 8 = RANSAC.
/// @ransac_reproj_threshold: Maximum allowed reprojection error to treat a point pair as an inlier.
///
/// Computes a perspective transform matrix between source and destination planes.
/// RANSAC robustly handles mismatched coordinates.
///
/// Return: A tuple of (homography_matrix_3x3, inlier_mask).
#[pyfunction]
#[pyo3(name = "findHomography")]
#[pyo3(signature = (src_points, dst_points, method = 0, ransac_reproj_threshold = 3.0))]
pub fn find_homography<'py>(
    py: Python<'py>,
    src_points: &pyo3::PyAny,
    dst_points: &pyo3::PyAny,
    method: i32,
    ransac_reproj_threshold: f64,
) -> PyResult<(&'py PyArrayDyn<f64>, &'py PyArrayDyn<u8>)> {
    let src = extract_points(src_points)?;
    let dst = extract_points(dst_points)?;
    if src.len() != dst.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("Source and destination point counts must match"));
    }
    if src.len() < 4 {
        return Err(pyo3::exceptions::PyValueError::new_err("At least 4 points are required to compute a homography"));
    }
    
    let n = src.len();
    let mut best_h = ndarray::Array2::<f64>::eye(3);
    let mut best_inliers = ndarray::Array1::<u8>::zeros(n);
    let mut max_inlier_count = 0;
    
    if method == 8 {
        let mut seed = 123456789u32;
        let mut rand_idx = |s: &mut u32| -> usize {
            *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (*s as usize) % n
        };
        
        let num_iterations = 2000;
        for _ in 0..num_iterations {
            let mut sample_idx = [0; 4];
            let mut count = 0;
            let mut attempts = 0;
            while count < 4 && attempts < 100 {
                let idx = rand_idx(&mut seed);
                if !sample_idx[..count].contains(&idx) {
                    sample_idx[count] = idx;
                    count += 1;
                }
                attempts += 1;
            }
            if count < 4 { continue; }
            
            let sample_src = [src[sample_idx[0]], src[sample_idx[1]], src[sample_idx[2]], src[sample_idx[3]]];
            let sample_dst = [dst[sample_idx[0]], dst[sample_idx[1]], dst[sample_idx[2]], dst[sample_idx[3]]];
            
            if let Some(h) = compute_homography_dlt(&sample_src, &sample_dst) {
                let mut inliers = ndarray::Array1::<u8>::zeros(n);
                let mut inlier_count = 0;
                for i in 0..n {
                    let (x, y) = src[i];
                    let (xp, yp) = dst[i];
                    let w = h[[2, 0]] * x + h[[2, 1]] * y + h[[2, 2]];
                    if w.abs() > 1e-9 {
                        let proj_x = (h[[0, 0]] * x + h[[0, 1]] * y + h[[0, 2]]) / w;
                        let proj_y = (h[[1, 0]] * x + h[[1, 1]] * y + h[[1, 2]]) / w;
                        let err = ((xp - proj_x).powi(2) + (yp - proj_y).powi(2)).sqrt();
                        if err <= ransac_reproj_threshold {
                            inliers[i] = 1;
                            inlier_count += 1;
                        }
                    }
                }
                if inlier_count > max_inlier_count {
                    max_inlier_count = inlier_count;
                    best_inliers = inliers;
                    best_h = h;
                }
            }
        }
        
        let mut final_src = Vec::new();
        let mut final_dst = Vec::new();
        for i in 0..n {
            if best_inliers[i] == 1 {
                final_src.push(src[i]);
                final_dst.push(dst[i]);
            }
        }
        if final_src.len() >= 4 {
            if let Some(h) = compute_homography_dlt(&final_src, &final_dst) {
                best_h = h;
            }
        }
    } else {
        if let Some(h) = compute_homography_dlt(&src, &dst) {
            best_h = h;
            best_inliers.fill(1);
        }
    }
    
    Ok((best_h.into_pyarray(py).to_dyn(), best_inliers.into_pyarray(py).to_dyn()))
}

/// flann_match() - Approximate nearest-neighbor matcher (FLANN-like).
/// @py: Python interpreter token.
/// @query_descriptors: Matrix of query descriptors (u8 or f32).
/// @train_descriptors: Matrix of train descriptors (u8 or f32).
///
/// For f32 descriptors (e.g. SIFT), builds a KD-tree to perform fast approximate
/// nearest neighbor search.  For u8 binary descriptors, falls back to brute force
/// Hamming distance matching.
///
/// Return: A vector of DMatch instances.
#[pyfunction]
pub fn flann_match<'py>(
    py: Python<'py>,
    query_descriptors: &pyo3::PyAny,
    train_descriptors: &pyo3::PyAny,
) -> PyResult<Vec<DMatch>> {
    if let Ok(q_f32) = query_descriptors.extract::<PyReadonlyArrayDyn<f32>>() {
        let t_f32 = train_descriptors.extract::<PyReadonlyArrayDyn<f32>>()?;
        let q_arr = q_f32.as_array();
        let t_arr = t_f32.as_array();
        let q_2d = q_arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Descriptors must be 2D"))?;
        let t_2d = t_arr.into_dimensionality::<numpy::ndarray::Ix2>()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Descriptors must be 2D"))?;
            
        let q_rows = q_2d.shape()[0];
        let t_rows = t_2d.shape()[0];
        
        let mut indices: Vec<usize> = (0..t_rows).collect();
        let tree = build_kdtree(&mut indices, &t_2d, 0);
        
        let mut matches = Vec::with_capacity(q_rows);
        for i in 0..q_rows {
            let q_row = q_2d.row(i);
            let mut best_idx = 0;
            let mut best_dist = f32::MAX;
            kdtree_search(&tree, &q_row, &t_2d, &mut best_idx, &mut best_dist);
            matches.push(DMatch {
                query_idx: i as i32,
                train_idx: best_idx as i32,
                img_idx: 0,
                distance: best_dist,
            });
        }
        Ok(matches)
    } else {
        bf_match(py, query_descriptors, train_descriptors, false, "HAMMING")
    }
}
