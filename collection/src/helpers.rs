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

pub fn convolve_2d_channel(channel:numpy::ndarray::ArrayView2<u8>, kernel:numpy::ndarray::ArrayView2<f64>) -> numpy::ndarray::Array2<u8> {
    let (h,w) = (channel.shape()[0], channel.shape()[1]);
    let (kh, kw) = (kernel.shape()[0], kernel.shape()[1]);
    
    let pad_h = kh/2;
    let pad_w = kw/2;
    let mut out = numpy::ndarray::Array2::<u8>::zeros((h, w));

    for y in 0..h{
        for x in 0..w{
            let mut sum = 0.0;
            for ky in 0..kh{
                for kx in 0..kw {

                    let iy = (y as isize + ky as isize - pad_h as isize).clamp(0, h as isize - 1) as usize;
                    let ix = (x as isize + kx as isize - pad_w as isize).clamp(0, w as isize - 1) as usize;

                    sum += channel[[iy, ix]] as f64 * kernel[[ky, kx]];
                }
            }
            out[[y,x]] = sum.clamp(0.0, 255.0) as u8;
        }
    }
    out
}
