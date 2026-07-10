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

/// Good Features to Track (Shi-Tomasi or Harris wrapper with NMS and distance sorting).
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

/// ORB keypoint detector and descriptor extractor.
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

/// SIFT keypoint detector and descriptor extractor.
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

/// BRISK keypoint detector and descriptor extractor.
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
