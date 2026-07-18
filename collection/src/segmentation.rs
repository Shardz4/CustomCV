use pyo3::prelude::*;
use numpy::{
    ndarray::{Array2, Array3, ArrayView2, IxDyn},
    IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn,
};
use std::collections::{BinaryHeap, VecDeque};

// ==========================================
// DATA STRUCTURES & HELPERS
// ==========================================

struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self { parent: (0..size).collect() }
    }
    
    fn find(&mut self, mut i: usize) -> usize {
        while self.parent[i] != i {
            self.parent[i] = self.parent[self.parent[i]];
            i = self.parent[i];
        }
        i
    }
    
    fn union(&mut self, i: usize, j: usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            self.parent[root_i] = root_j;
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct WatershedItem {
    priority: i32,
    y: usize,
    x: usize,
}

impl Ord for WatershedItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap priority
        other.priority.cmp(&self.priority)
    }
}

impl PartialOrd for WatershedItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ==========================================
// CORE ALGORITHMS
// ==========================================

/// Connected component labeling using Disjoint-Set (Union-Find).
fn connected_components_impl(
    image: &ArrayView2<u8>,
    connectivity: i32,
) -> (i32, Array2<i32>) {
    let (h, w) = (image.shape()[0], image.shape()[1]);
    let mut labels = Array2::<i32>::zeros((h, w));
    
    let mut ds = DisjointSet::new(h * w / 2 + 2);
    let mut next_label = 1;
    
    for y in 0..h {
        for x in 0..w {
            if image[[y, x]] == 0 {
                continue;
            }
            
            let mut neighbors = Vec::new();
            if x > 0 && labels[[y, x - 1]] > 0 {
                neighbors.push(labels[[y, x - 1]] as usize);
            }
            if y > 0 && labels[[y - 1, x]] > 0 {
                neighbors.push(labels[[y - 1, x]] as usize);
            }
            
            if connectivity == 8 {
                if y > 0 && x > 0 && labels[[y - 1, x - 1]] > 0 {
                    neighbors.push(labels[[y - 1, x - 1]] as usize);
                }
                if y > 0 && x + 1 < w && labels[[y - 1, x + 1]] > 0 {
                    neighbors.push(labels[[y - 1, x + 1]] as usize);
                }
            }
            
            if neighbors.is_empty() {
                labels[[y, x]] = next_label as i32;
                next_label += 1;
                if next_label >= ds.parent.len() {
                    ds.parent.push(ds.parent.len());
                }
            } else {
                let min_label = neighbors.iter().cloned().min().unwrap();
                labels[[y, x]] = min_label as i32;
                for &nb in &neighbors {
                    ds.union(nb, min_label);
                }
            }
        }
    }
    
    let mut unique_labels = std::collections::HashMap::new();
    let mut final_count = 1; // 0 is background
    
    for y in 0..h {
        for x in 0..w {
            let label = labels[[y, x]] as usize;
            if label == 0 {
                continue;
            }
            let root = ds.find(label);
            let final_lbl = *unique_labels.entry(root).or_insert_with(|| {
                let current = final_count;
                final_count += 1;
                current
            });
            labels[[y, x]] = final_lbl;
        }
    }
    
    (final_count, labels)
}

// ==========================================
// PYFUNCTION EXPORTS
// ==========================================

/// connected_components() - Label connected components in a binary image.
/// @py: Python interpreter token.
/// @image: 2D binary image array (u8).
/// @connectivity: 8-way or 4-way connectivity flag (default 8).
///
/// Computes the connected components labeled image of boolean image.
///
/// Return: 2D labeled image array (i32).
#[pyfunction]
#[pyo3(signature = (image, connectivity = 8))]
pub fn connected_components<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    connectivity: i32,
) -> PyResult<&'py PyArrayDyn<i32>> {
    let arr = image.as_array();
    let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input must be a 2D binary image"))?;
    
    let (_, labels) = connected_components_impl(&channel.view(), connectivity);
    Ok(labels.into_pyarray(py).to_dyn())
}

/// connected_components_with_stats() - Label connected components and compute statistics.
/// @py: Python interpreter token.
/// @image: 2D binary image array (u8).
/// @connectivity: 8-way or 4-way connectivity flag (default 8).
///
/// Computes the connected components labeled image and statistics for each label.
///
/// Return: A tuple containing (num_labels, labels, stats, centroids).
#[pyfunction]
#[pyo3(signature = (image, connectivity = 8))]
pub fn connected_components_with_stats<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    connectivity: i32,
) -> PyResult<(i32, &'py PyArrayDyn<i32>, &'py PyArrayDyn<i32>, &'py PyArrayDyn<f64>)> {
    let arr = image.as_array();
    let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input must be a 2D binary image"))?;
    
    let (num_labels, labels) = connected_components_impl(&channel.view(), connectivity);
    
    let h = channel.shape()[0];
    let w = channel.shape()[1];
    
    let n = num_labels as usize;
    let mut left = vec![i32::MAX; n];
    let mut right = vec![i32::MIN; n];
    let mut top = vec![i32::MAX; n];
    let mut bottom = vec![i32::MIN; n];
    let mut area = vec![0; n];
    let mut sum_x = vec![0.0; n];
    let mut sum_y = vec![0.0; n];
    
    for y in 0..h {
        for x in 0..w {
            let lbl = labels[[y, x]] as usize;
            area[lbl] += 1;
            let xi = x as i32;
            let yi = y as i32;
            if xi < left[lbl] { left[lbl] = xi; }
            if xi > right[lbl] { right[lbl] = xi; }
            if yi < top[lbl] { top[lbl] = yi; }
            if yi > bottom[lbl] { bottom[lbl] = yi; }
            sum_x[lbl] += x as f64;
            sum_y[lbl] += y as f64;
        }
    }
    
    // Build stats array (num_labels, 5)
    // Columns: [left, top, width, height, area]
    let mut stats = Array2::<i32>::zeros((n, 5));
    let mut centroids = Array2::<f64>::zeros((n, 2));
    
    // Background stats
    stats[[0, 0]] = 0;
    stats[[0, 1]] = 0;
    stats[[0, 2]] = w as i32;
    stats[[0, 3]] = h as i32;
    stats[[0, 4]] = area[0];
    
    if area[0] > 0 {
        centroids[[0, 0]] = sum_x[0] / area[0] as f64;
        centroids[[0, 1]] = sum_y[0] / area[0] as f64;
    }
    
    for i in 1..n {
        stats[[i, 0]] = left[i];
        stats[[i, 1]] = top[i];
        stats[[i, 2]] = right[i] - left[i] + 1;
        stats[[i, 3]] = bottom[i] - top[i] + 1;
        stats[[i, 4]] = area[i];
        
        if area[i] > 0 {
            centroids[[i, 0]] = sum_x[i] / area[i] as f64;
            centroids[[i, 1]] = sum_y[i] / area[i] as f64;
        }
    }
    
    Ok((
        num_labels,
        labels.into_pyarray(py).to_dyn(),
        stats.into_pyarray(py).to_dyn(),
        centroids.into_pyarray(py).to_dyn(),
    ))
}

/// distance_transform() - Compute the distance transform of a binary image.
/// @py: Python interpreter token.
/// @image: 2D binary image array (u8).
///
/// Computes the distance to the nearest zero pixel for every pixel in a binary image using the Chamfer metric.
///
/// Return: 2D distance transform image array (f32).
#[pyfunction]
pub fn distance_transform<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
) -> PyResult<&'py PyArrayDyn<f32>> {
    let arr = image.as_array();
    let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Input must be a 2D binary image"))?;
    
    let (h, w) = (channel.shape()[0], channel.shape()[1]);
    let mut dist = Array2::<f32>::from_elem((h, w), f32::INFINITY);
    
    // Initialize background to 0
    for y in 0..h {
        for x in 0..w {
            if channel[[y, x]] == 0 {
                dist[[y, x]] = 0.0;
            }
        }
    }
    
    // Forward pass
    for y in 0..h {
        for x in 0..w {
            let mut d = dist[[y, x]];
            if d == 0.0 {
                continue;
            }
            if x > 0 {
                d = d.min(dist[[y, x - 1]] + 1.0);
            }
            if y > 0 {
                d = d.min(dist[[y - 1, x]] + 1.0);
                if x > 0 {
                    d = d.min(dist[[y - 1, x - 1]] + 1.414);
                }
                if x + 1 < w {
                    d = d.min(dist[[y - 1, x + 1]] + 1.414);
                }
            }
            dist[[y, x]] = d;
        }
    }
    
    // Backward pass
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let mut d = dist[[y, x]];
            if d == 0.0 {
                continue;
            }
            if x + 1 < w {
                d = d.min(dist[[y, x + 1]] + 1.0);
            }
            if y + 1 < h {
                d = d.min(dist[[y + 1, x]] + 1.0);
                if x + 1 < w {
                    d = d.min(dist[[y + 1, x + 1]] + 1.414);
                }
                if x > 0 {
                    d = d.min(dist[[y + 1, x - 1]] + 1.414);
                }
            }
            dist[[y, x]] = d;
        }
    }
    
    // Clean up any remaining infinity values
    for y in 0..h {
        for x in 0..w {
            if dist[[y, x]].is_infinite() {
                dist[[y, x]] = 0.0;
            }
        }
    }
    
    Ok(dist.into_pyarray(py).to_dyn())
}

/// flood_fill() - Region growing flood fill algorithm.
/// @py: Python interpreter token.
/// @image: Grayscale (2D) or Color (3D) u8 image.
/// @seed_point: Starting (x, y) coordinate.
/// @new_val: Color to fill with.
/// @lo_diff: Lower boundary difference.
/// @up_diff: Upper boundary difference.
///
/// Fills a connected component with the given color starting from the seed point.
///
/// Return: A tuple containing (number of filled pixels, updated image array).
#[pyfunction]
#[pyo3(signature = (image, seed_point, new_val, lo_diff = 0, up_diff = 0))]
pub fn flood_fill<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    seed_point: (i32, i32),
    new_val: PyObject,
    lo_diff: u8,
    up_diff: u8,
) -> PyResult<(i32, &'py PyArrayDyn<u8>)> {
    let arr = image.as_array();
    let ndim = arr.ndim();
    let shape = arr.shape();
    
    let (h, w) = (shape[0], shape[1]);
    let channels = if ndim == 3 { shape[2] } else { 1 };
    
    let mut out_arr = arr.to_owned();
    
    // Parse color
    let mut parsed_color = vec![0u8; channels];
    Python::with_gil(|py_gil| {
        if let Ok(val) = new_val.extract::<u8>(py_gil) {
            for c in 0..channels {
                parsed_color[c] = val;
            }
        } else if let Ok(list) = new_val.extract::<Vec<u8>>(py_gil) {
            for c in 0..channels.min(list.len()) {
                parsed_color[c] = list[c];
            }
        }
    });

    let (sx, sy) = (seed_point.0, seed_point.1);
    if sx < 0 || sx >= w as i32 || sy < 0 || sy >= h as i32 {
        return Err(pyo3::exceptions::PyValueError::new_err("Seed point outside image boundary"));
    }
    
    let mut visited = Array2::<bool>::from_elem((h, w), false);
    let mut queue = VecDeque::new();
    
    let (sx_u, sy_u) = (sx as usize, sy as usize);
    queue.push_back((sy_u, sx_u));
    visited[[sy_u, sx_u]] = true;
    
    let mut filled_count = 0;
    
    // Seed color
    let mut seed_color = vec![0u8; channels];
    if ndim == 3 {
        for c in 0..channels {
            seed_color[c] = out_arr[[sy_u, sx_u, c]];
        }
    } else {
        seed_color[0] = out_arr[[sy_u, sx_u]];
    }
    
    while let Some((cy, cx)) = queue.pop_front() {
        // Color update
        if ndim == 3 {
            for c in 0..channels {
                out_arr[[cy, cx, c]] = parsed_color[c];
            }
        } else {
            out_arr[[cy, cx]] = parsed_color[0];
        }
        filled_count += 1;
        
        let neighbors = [
            (cy as isize - 1, cx as isize),
            (cy as isize + 1, cx as isize),
            (cy as isize, cx as isize - 1),
            (cy as isize, cx as isize + 1),
        ];
        
        for &(ny, nx) in &neighbors {
            if ny >= 0 && ny < h as isize && nx >= 0 && nx < w as isize {
                let ny_u = ny as usize;
                let nx_u = nx as usize;
                if !visited[[ny_u, nx_u]] {
                    // Check difference
                    let mut match_val = true;
                    if ndim == 3 {
                        for c in 0..channels {
                            let diff = (out_arr[[ny_u, nx_u, c]] as i16 - seed_color[c] as i16).abs();
                            if diff > lo_diff as i16 || diff > up_diff as i16 {
                                match_val = false;
                                break;
                            }
                        }
                    } else {
                        let diff = (out_arr[[ny_u, nx_u]] as i16 - seed_color[0] as i16).abs();
                        if diff > lo_diff as i16 || diff > up_diff as i16 {
                            match_val = false;
                        }
                    }
                    
                    if match_val {
                        visited[[ny_u, nx_u]] = true;
                        queue.push_back((ny_u, nx_u));
                    }
                }
            }
        }
    }
    
    Ok((filled_count, out_arr.into_pyarray(py).to_dyn()))
}

/// watershed() - Meyer's marker-based watershed segmentation algorithm.
/// @py: Python interpreter token.
/// @image: Grayscale 2D image array (u8).
/// @markers: 2D i32 array of markers where positive labels indicate seeds.
///
/// Performs marker-based image segmentation using the watershed algorithm.
///
/// Return: Segmented markers image array (i32).
#[pyfunction]
pub fn watershed<'py>(
    py: Python<'py>,
    image: PyReadonlyArrayDyn<'py, u8>,
    markers: PyReadonlyArrayDyn<'py, i32>,
) -> PyResult<&'py PyArrayDyn<i32>> {
    let arr = image.as_array();
    let channel = arr.into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Image must be a 2D grayscale image"))?;
    
    let mut markers_arr = markers.as_array().into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Markers must be a 2D i32 image"))?
        .to_owned();
        
    let (h, w) = (channel.shape()[0], channel.shape()[1]);
    let mut pq = BinaryHeap::new();
    
    // Initial boundaries next to markers
    for y in 0..h {
        for x in 0..w {
            let label = markers_arr[[y, x]];
            if label > 0 {
                let neighbors = [
                    (y as isize - 1, x as isize),
                    (y as isize + 1, x as isize),
                    (y as isize, x as isize - 1),
                    (y as isize, x as isize + 1),
                ];
                for &(ny, nx) in &neighbors {
                    if ny >= 0 && ny < h as isize && nx >= 0 && nx < w as isize {
                        let ny_u = ny as usize;
                        let nx_u = nx as usize;
                        if markers_arr[[ny_u, nx_u]] == 0 {
                            markers_arr[[ny_u, nx_u]] = -2; // status: in queue
                            pq.push(WatershedItem {
                                priority: channel[[ny_u, nx_u]] as i32,
                                y: ny_u,
                                x: nx_u,
                            });
                        }
                    }
                }
            }
        }
    }
    
    while let Some(WatershedItem { priority: _, y, x }) = pq.pop() {
        let neighbors = [
            (y as isize - 1, x as isize),
            (y as isize + 1, x as isize),
            (y as isize, x as isize - 1),
            (y as isize, x as isize + 1),
        ];
        
        let mut adj_label = 0;
        let mut is_watershed = false;
        
        for &(ny, nx) in &neighbors {
            if ny >= 0 && ny < h as isize && nx >= 0 && nx < w as isize {
                let n_label = markers_arr[[ny as usize, nx as usize]];
                if n_label > 0 {
                    if adj_label == 0 {
                        adj_label = n_label;
                    } else if n_label != adj_label {
                        is_watershed = true;
                    }
                }
            }
        }
        
        if is_watershed {
            markers_arr[[y, x]] = -1;
        } else if adj_label > 0 {
            markers_arr[[y, x]] = adj_label;
            for &(ny, nx) in &neighbors {
                if ny >= 0 && ny < h as isize && nx >= 0 && nx < w as isize {
                    let ny_u = ny as usize;
                    let nx_u = nx as usize;
                    if markers_arr[[ny_u, nx_u]] == 0 {
                        markers_arr[[ny_u, nx_u]] = -2;
                        pq.push(WatershedItem {
                            priority: channel[[ny_u, nx_u]] as i32,
                            y: ny_u,
                            x: nx_u,
                        });
                    }
                }
            }
        } else {
            markers_arr[[y, x]] = 0;
        }
    }
    
    // Re-label any remaining -2 to 0
    for y in 0..h {
        for x in 0..w {
            if markers_arr[[y, x]] == -2 {
                markers_arr[[y, x]] = 0;
            }
        }
    }
    
    Ok(markers_arr.into_pyarray(py).to_dyn())
}

/// grab_cut() - GrabCut image segmentation.
/// @py: Python interpreter token.
/// @img: 3D color BGR/RGB image array.
/// @mask: 2D u8 GrabCut status mask.
/// @rect: Bounding box containing the foreground object.
/// @bgd_model: Background model.
/// @fgd_model: Foreground model.
/// @iter_count: Number of iterations.
/// @mode: Operation mode.
///
/// Extracts foreground objects from the color image.
///
/// Return: A tuple containing (updated mask, bgd_model, fgd_model).
#[pyfunction]
#[pyo3(signature = (img, mask, rect, bgd_model, fgd_model, iter_count = 5, mode = 1))]
pub fn grab_cut<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    mask: PyReadonlyArrayDyn<'py, u8>,
    rect: (i32, i32, i32, i32),
    bgd_model: PyObject,
    fgd_model: PyObject,
    iter_count: i32,
    mode: i32,
) -> PyResult<(&'py PyArrayDyn<u8>, PyObject, PyObject)> {
    let img_arr = img.as_array();
    let ndim = img_arr.ndim();
    if ndim != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("GrabCut requires a 3D color image"));
    }
    let shape = img_arr.shape();
    let (h, w, channels) = (shape[0], shape[1], shape[2]);
    if channels != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("GrabCut requires exactly 3 channels (RGB/BGR)"));
    }
    
    let mut mask_arr = mask.as_array().into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("Mask must be a 2D u8 array"))?
        .to_owned();
        
    // Mode 0: INIT_WITH_RECT
    if mode == 0 {
        let (rx, ry, rw, rh) = rect;
        for y in 0..h {
            for x in 0..w {
                let xi = x as i32;
                let yi = y as i32;
                if xi >= rx && xi < rx + rw && yi >= ry && yi < ry + rh {
                    mask_arr[[y, x]] = 3; // GC_PR_FGD (probable foreground)
                } else {
                    mask_arr[[y, x]] = 0; // GC_BGD (obvious background)
                }
            }
        }
    }
    
    // We train a color histogram of 16x16x16 bins (total 4096 bins)
    for _iter in 0..iter_count {
        let mut bg_hist = vec![0.0f64; 4096];
        let mut fg_hist = vec![0.0f64; 4096];
        let mut bg_total = 0.0;
        let mut fg_total = 0.0;
        
        for y in 0..h {
            for x in 0..w {
                let r = img_arr[[y, x, 0]] as usize;
                let g = img_arr[[y, x, 1]] as usize;
                let b = img_arr[[y, x, 2]] as usize;
                let bin = (r / 16) * 256 + (g / 16) * 16 + (b / 16);
                
                let lbl = mask_arr[[y, x]];
                if lbl == 0 || lbl == 2 {
                    bg_hist[bin] += 1.0;
                    bg_total += 1.0;
                } else {
                    fg_hist[bin] += 1.0;
                    fg_total += 1.0;
                }
            }
        }
        
        // Normalize histograms
        if bg_total > 0.0 {
            for val in &mut bg_hist {
                *val /= bg_total;
            }
        }
        if fg_total > 0.0 {
            for val in &mut fg_hist {
                *val /= fg_total;
            }
        }
        
        // ICM (Iterated Conditional Modes) label updates
        let beta = 1.5; // Spatial smoothness parameter
        
        for y in 0..h {
            for x in 0..w {
                let lbl = mask_arr[[y, x]];
                // Only update probable background (2) or probable foreground (3)
                if lbl == 2 || lbl == 3 {
                    let r = img_arr[[y, x, 0]] as usize;
                    let g = img_arr[[y, x, 1]] as usize;
                    let b = img_arr[[y, x, 2]] as usize;
                    let bin = (r / 16) * 256 + (g / 16) * 16 + (b / 16);
                    
                    let p_bg = bg_hist[bin].max(1e-9);
                    let p_fg = fg_hist[bin].max(1e-9);
                    
                    // Spatial smoothness neighbors
                    let neighbors = [
                        (y as isize - 1, x as isize),
                        (y as isize + 1, x as isize),
                        (y as isize, x as isize - 1),
                        (y as isize, x as isize + 1),
                    ];
                    
                    let mut bg_neighbors = 0;
                    let mut fg_neighbors = 0;
                    
                    for &(ny, nx) in &neighbors {
                        if ny >= 0 && ny < h as isize && nx >= 0 && nx < w as isize {
                            let n_lbl = mask_arr[[ny as usize, nx as usize]];
                            if n_lbl == 0 || n_lbl == 2 {
                                bg_neighbors += 1;
                            } else {
                                fg_neighbors += 1;
                            }
                        }
                    }
                    
                    // Energy: -ln P + beta * neighbors with opposite label
                    let e_bg = -p_bg.ln() + beta * fg_neighbors as f64;
                    let e_fg = -p_fg.ln() + beta * bg_neighbors as f64;
                    
                    if e_bg < e_fg {
                        mask_arr[[y, x]] = 2; // GC_PR_BGD
                    } else {
                        mask_arr[[y, x]] = 3; // GC_PR_FGD
                    }
                }
            }
        }
    }
    
    Ok((mask_arr.into_pyarray(py).to_dyn(), bgd_model, fgd_model))
}

/// grab_cut_cv() - GrabCut image segmentation alias.
/// @py: Python interpreter token.
/// @img: 3D color BGR/RGB image array.
/// @mask: 2D u8 GrabCut status mask.
/// @rect: Bounding box containing the foreground object.
/// @bgd_model: Background model.
/// @fgd_model: Foreground model.
/// @iter_count: Number of iterations.
/// @mode: Operation mode.
///
/// Alias for grab_cut. Extracts foreground objects from the color image.
///
/// Return: A tuple containing (updated mask, bgd_model, fgd_model).
#[pyfunction(name = "grabCut")]
#[pyo3(signature = (img, mask, rect, bgd_model, fgd_model, iter_count = 5, mode = 1))]
pub fn grab_cut_cv<'py>(
    py: Python<'py>,
    img: PyReadonlyArrayDyn<'py, u8>,
    mask: PyReadonlyArrayDyn<'py, u8>,
    rect: (i32, i32, i32, i32),
    bgd_model: PyObject,
    fgd_model: PyObject,
    iter_count: i32,
    mode: i32,
) -> PyResult<(&'py PyArrayDyn<u8>, PyObject, PyObject)> {
    grab_cut(py, img, mask, rect, bgd_model, fgd_model, iter_count, mode)
}

