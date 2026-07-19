use numpy::ndarray::{Array2, ArrayView2};

// ==========================================
// BORDER HANDLING HELPERS
// ==========================================

/// get_border_index() - Maps an out-of-bounds index to an in-bounds index based on border mode.
pub fn get_border_index(i: isize, len: usize, border_type: &str) -> Option<usize> {
    let len_i = len as isize;
    if i >= 0 && i < len_i {
        return Some(i as usize);
    }
    match border_type {
        "reflect" | "BORDER_REFLECT" => {
            if i < 0 {
                let mut temp = -i - 1;
                if temp >= len_i { temp = len_i - 1; }
                Some(temp as usize)
            } else {
                let mut temp = 2 * len_i - i - 1;
                if temp < 0 { temp = 0; }
                Some(temp as usize)
            }
        }
        "reflect_101" | "BORDER_REFLECT_101" | "BORDER_DEFAULT" | "default" => {
            if i < 0 {
                let mut temp = -i;
                if temp >= len_i { temp = len_i - 1; }
                Some(temp as usize)
            } else {
                let mut temp = 2 * len_i - i - 2;
                if temp < 0 { temp = 0; }
                Some(temp as usize)
            }
        }
        "replicate" | "BORDER_REPLICATE" => {
            Some(i.clamp(0, len_i - 1) as usize)
        }
        "wrap" | "BORDER_WRAP" => {
            if i < 0 {
                Some(((i % len_i + len_i) % len_i) as usize)
            } else {
                Some((i % len_i) as usize)
            }
        }
        "constant" | "BORDER_CONSTANT" => {
            None
        }
        _ => {
            // Default to reflect_101 (OpenCV default)
            if i < 0 {
                let mut temp = -i;
                if temp >= len_i { temp = len_i - 1; }
                Some(temp as usize)
            } else {
                let mut temp = 2 * len_i - i - 2;
                if temp < 0 { temp = 0; }
                Some(temp as usize)
            }
        }
    }
}

/// get_border_pixel() - Resolves pixel value at (y, x), supporting border padding modes.
pub fn get_border_pixel(channel: &ArrayView2<u8>, y: isize, x: isize, border_type: &str, border_value: u8) -> u8 {
    let h = channel.shape()[0];
    let w = channel.shape()[1];
    let iy = match get_border_index(y, h, border_type) {
        Some(val) => val,
        None => return border_value,
    };
    let ix = match get_border_index(x, w, border_type) {
        Some(val) => val,
        None => return border_value,
    };
    channel[[iy, ix]]
}

/// apply_median_3x3() - Applies a 3×3 median filter to a single 2D channel.
/// @channel: 2D input image channel slice (u8).
/// @border_type: Border padding mode ("reflect", "replicate", "wrap", "constant").
/// @border_value: Padding value for constant border.
///
/// Computes the median value in a 3x3 window around each pixel.
///
/// Return: Filtered 2D channel.
pub fn apply_median_3x3(channel: ArrayView2<u8>, border_type: &str, border_value: u8) -> Array2<u8> {
    let (h, w) = (channel.shape()[0], channel.shape()[1]);
    let mut out = Array2::<u8>::zeros((h, w));
    
    for y in 0..h {
        for x in 0..w {
            let mut window = [0u8; 9];
            let mut idx = 0;
            for ky in -1..=1 {
                for kx in -1..=1 {
                    window[idx] = get_border_pixel(&channel, y as isize + ky, x as isize + kx, border_type, border_value);
                    idx += 1;
                }
            }
            window.sort_unstable();
            out[[y, x]] = window[4]; 
        }
    }
    out
}

/// apply_laplacian_3x3() - Applies a 3×3 Laplacian filter to a single 2D channel.
/// @channel: 2D input image channel slice (u8).
/// @border_type: Border padding mode.
/// @border_value: Constant border value.
///
/// Approximates the Laplacian using a discrete differentiation operator.
///
/// Return: Filtered 2D channel.
pub fn apply_laplacian_3x3(channel: ArrayView2<u8>, border_type: &str, border_value: u8) -> Array2<u8> {
    let (h, w) = (channel.shape()[0], channel.shape()[1]);
    let mut out = Array2::<u8>::zeros((h, w));
    
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0i32;
            let kernel = [
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1],
            ];
            for ky in -1..=1 {
                for kx in -1..=1 {
                    let val = get_border_pixel(&channel, y as isize + ky, x as isize + kx, border_type, border_value);
                    sum += val as i32 * kernel[(ky + 1) as usize][(kx + 1) as usize];
                }
            }
            out[[y, x]] = sum.clamp(0, 255) as u8;
        }
    }
    out
}

/// calculate_otsu_threshold() - Calculates the optimal threshold using Otsu's method.
/// @channel_data: 2D input image channel slice (u8).
///
/// Chooses a threshold that minimizes the intra-class variance of the thresholded black and white pixels.
///
/// Return: Optimal threshold value (0-255).
pub fn calculate_otsu_threshold(channel_data: ArrayView2<u8>) -> u8 {
    let mut hist = [0usize; 256];
    let mut total_pixels = 0;

    // Calculate Histogram
    for &pixel in channel_data.iter() {
        hist[pixel as usize] += 1;
        total_pixels += 1;
    }

    // Calculate total sum for means
    let mut sum = 0.0;
    for i in 0..256 {
        sum += (i as f64) * (hist[i] as f64);
    }

    let mut sum_b = 0.0;
    let mut w_b = 0;
    let mut var_max = 0.0;
    let mut threshold = 0;

    // Iterate through all possible thresholds to maximize between-class variance
    for t in 0..256 {
        w_b += hist[t];
        if w_b == 0 { continue; }
        
        let w_f = total_pixels - w_b;
        if w_f == 0 { break; }

        sum_b += (t as f64) * (hist[t] as f64);

        let m_b = sum_b / (w_b as f64);
        let m_f = (sum - sum_b) / (w_f as f64);

        let var_between = (w_b as f64) * (w_f as f64) * (m_b - m_f).powi(2);

        if var_between > var_max {
            var_max = var_between;
            threshold = t as u8;
        }
    }
    
    threshold
}

/// calculate_triangle_threshold() - Triangle threshold algorithm.
/// @channel_data: 2D input image channel slice (u8).
///
/// Draws a line between the histogram peak and the farthest non-zero bin,
/// then picks the threshold at the bin with maximum perpendicular distance.
///
/// Return: Optimal threshold value (0-255).
pub fn calculate_triangle_threshold(channel_data: ArrayView2<u8>) -> u8 {
    // Build histogram
    let mut hist = [0usize; 256];
    for &pixel in channel_data.iter() {
        hist[pixel as usize] += 1;
    }

    // Find the peak (mode) of the histogram
    let mut peak_idx = 0usize;
    let mut peak_val = 0usize;
    for i in 0..256 {
        if hist[i] > peak_val {
            peak_val = hist[i];
            peak_idx = i;
        }
    }

    // Find the farthest non-zero bin from the peak
    let mut left_nz = 0usize;
    let mut right_nz = 255usize;
    for i in 0..256 {
        if hist[i] > 0 { left_nz = i; break; }
    }
    for i in (0..256).rev() {
        if hist[i] > 0 { right_nz = i; break; }
    }

    // Determine which tail is longer (the line goes from peak to far end)
    let (start, end, flip) = if (peak_idx - left_nz) < (right_nz - peak_idx) {
        // Right tail is longer: line from peak to right_nz
        (peak_idx, right_nz, false)
    } else {
        // Left tail is longer: line from left_nz to peak
        (left_nz, peak_idx, true)
    };

    if start == end {
        return peak_idx as u8;
    }

    // Line from (start, hist[start]) to (end, hist[end])
    let x1 = start as f64;
    let y1 = hist[start] as f64;
    let x2 = end as f64;
    let y2 = hist[end] as f64;
    let dx = x2 - x1;
    let dy = y2 - y1;
    let line_len = (dx * dx + dy * dy).sqrt();

    // Find the bin with maximum perpendicular distance from the line
    let mut max_dist = 0.0f64;
    let mut threshold = start;
    for i in start..=end {
        let px = i as f64;
        let py = hist[i] as f64;
        let dist = ((dy * px - dx * py + x2 * y1 - y2 * x1) / line_len).abs();
        if dist > max_dist {
            max_dist = dist;
            threshold = i;
        }
    }

    // If we used the left tail, the threshold is on the left side of the peak
    if flip { threshold += 1; }

    threshold as u8
}

/// compute_structure_tensor() - Computes the structure tensor of a grayscale image.
/// @image: 2D grayscale image slice.
/// @window_size: Integration window size.
///
/// Computes structure tensor components Sxx, Syy, and Sxy using Sobel derivatives.
///
/// Return: A tuple containing (Sxx, Syy, Sxy) Array2 matrix components.
pub fn compute_structure_tensor(image: &numpy::ndarray::ArrayView2<u8>, window_size: usize) -> (numpy::ndarray::Array2<f32>, numpy::ndarray::Array2<f32>, numpy::ndarray::Array2<f32>) {
    let (h, w) = (image.shape()[0], image.shape()[1]);

    let mut ix = numpy::ndarray::Array2::<f32>::zeros((h,w));
    let mut iy = numpy::ndarray::Array2::<f32>::zeros((h,w));
    
    for y in 1..h-1{
        for x in 1..w-1{
            let p00 = image[[y-1, x-1]] as f32;
            let p01 = image[[y-1, x]] as f32;
            let p02 = image[[y-1, x+1]] as f32;
            let p10 = image[[y, x-1]] as f32;
            // centre
            let p12 = image[[y, x+1]] as f32;
            let p20 = image[[y+1, x-1]] as f32;
            let p21 = image[[y+1, x]] as f32;
            let p22 = image[[y+1, x+1]] as f32;

            ix[[y,x]] = -p00 + p02 - 2.0*p10 + 2.0*p12 - p20 + p22;
            iy[[y,x]] = -p00 - 2.0*p01 - p02 + p20 + 2.0*p21 + p22;
        }
    }

    let ixx = ix.mapv(|v| v * v);
    let iyy = iy.mapv(|v| v * v);
    let ixy = numpy::ndarray::Zip::from(&ix).and(&iy).map_collect(|&x, &y| x * y);

    let pad = window_size / 2;
    
    let mut sxx = numpy::ndarray::Array2::<f32>::zeros((h,w));
    let mut syy = numpy::ndarray::Array2::<f32>::zeros((h,w));
    let mut sxy = numpy::ndarray::Array2::<f32>::zeros((h,w));
    
    for y in pad.. h - pad {
        for x in pad..w - pad {
            let mut sum_xx = 0.0;
            let mut sum_yy = 0.0;
            let mut sum_xy = 0.0;

            for wy in 0..window_size {
                for wx in 0..window_size {
                    let ny = y + wy - pad;
                    let nx = x + wx - pad;
                    
                    sum_xx += ixx[[ny, nx]];
                    sum_yy += iyy[[ny, nx]];
                    sum_xy += ixy[[ny, nx]];
                }
            }
            sxx[[y,x]] = sum_xx;
            syy[[y,x]] = sum_yy;
            sxy[[y,x]] = sum_xy;
        }
    }
    (sxx, syy, sxy)
    
}

/// convolve_2d_channel() - Convolve a 2D channel with a 2D float kernel.
/// @channel: 2D input image channel slice (u8).
/// @kernel: 2D float convolution kernel.
/// @border_type: Border padding mode.
/// @border_value: Constant border value.
///
/// Convolves the 2D channel with the specified kernel, clamping the results to [0, 255].
///
/// Return: Convolved 2D channel array.
pub fn convolve_2d_channel(
    channel: numpy::ndarray::ArrayView2<u8>,
    kernel: numpy::ndarray::ArrayView2<f64>,
    border_type: &str,
    border_value: u8,
) -> numpy::ndarray::Array2<u8> {
    let (h, w) = (channel.shape()[0], channel.shape()[1]);
    let (kh, kw) = (kernel.shape()[0], kernel.shape()[1]);
    
    let pad_h = kh / 2;
    let pad_w = kw / 2;
    let mut out = numpy::ndarray::Array2::<u8>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for ky in 0..kh {
                for kx in 0..kw {
                    let iy = y as isize + ky as isize - pad_h as isize;
                    let ix = x as isize + kx as isize - pad_w as isize;
                    let val = get_border_pixel(&channel, iy, ix, border_type, border_value);
                    sum += val as f64 * kernel[[ky, kx]];
                }
            }
            out[[y, x]] = sum.clamp(0.0, 255.0) as u8;
        }
    }
    out
}
