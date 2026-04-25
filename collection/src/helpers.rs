use numpy::ndarray::{Array2, ArrayView2};

/// Applies a 3×3 median filter to a single 2D channel.
pub fn apply_median_3x3(channel: ArrayView2<u8>) -> Array2<u8> {
    let (h, w) = (channel.shape()[0], channel.shape()[1]);
    let mut out = Array2::<u8>::zeros((h, w));
    
    for y in 1..h-1 {
        for x in 1..w-1 {
            let mut window = [
                channel[[y-1, x-1]], channel[[y-1, x]], channel[[y-1, x+1]],
                channel[[y, x-1]],   channel[[y, x]],   channel[[y, x+1]],
                channel[[y+1, x-1]], channel[[y+1, x]], channel[[y+1, x+1]],
            ];
            window.sort_unstable();
            out[[y, x]] = window[4]; 
        }
    }
    out
}

/// Applies a 3×3 Laplacian filter to a single 2D channel.
pub fn apply_laplacian_3x3(channel: ArrayView2<u8>) -> Array2<u8> {
    let (h, w) = (channel.shape()[0], channel.shape()[1]);
    let mut out = Array2::<u8>::zeros((h, w));
    
    for y in 1..h-1 {
        for x in 1..w-1 {
            let sum =
                -1 * channel[[y-1, x-1]] as i32 - 1 * channel[[y-1, x]] as i32 - 1 * channel[[y-1, x+1]] as i32
                -1 * channel[[y, x-1]] as i32   + 8 * channel[[y, x]] as i32   - 1 * channel[[y, x+1]] as i32
                -1 * channel[[y+1, x-1]] as i32 - 1 * channel[[y+1, x]] as i32 - 1 * channel[[y+1, x+1]] as i32;
            
            out[[y, x]] = sum.clamp(0, 255) as u8;
        }
    }
    out
}

/// Calculates the optimal threshold using Otsu's method.
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
