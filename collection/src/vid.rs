use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (device_index = 0, save_path = None))]
pub fn video_capture<'py>(
    py: Python<'py>,
    device_index: i32,
    save_path: Option<String>,
) -> PyResult<()> {
    let cv2 = py.import_bound("cv2")?;
    let cap = cv2.call_method1("VideoCapture", (device_index,))?;
    if !cap.call_method0("isOpened")?.extract::<bool>()? {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to open video capture device at index {}",
            device_index
        )));
    }
    
    let mut writer: Option<Bound<PyAny>> = None;
    if let Some(ref path) = save_path {
        let width = cap.call_method1("get", (3,))?.extract::<f64>()? as i32; // cv2.CAP_PROP_FRAME_WIDTH
        let height = cap.call_method1("get", (4,))?.extract::<f64>()? as i32; // cv2.CAP_PROP_FRAME_HEIGHT
        let fourcc = cv2.call_method1("VideoWriter_fourcc", ("m", "p", "4", "v"))?;
        writer = Some(cv2.call_method1("VideoWriter", (path, &fourcc, 20.0, (width, height)))?);
    }
    
    let window_name = "Video Capture (Press 'q' or 'ESC' to exit)";
    while cap.call_method0("isOpened")?.extract::<bool>()? {
        let result = cap.call_method0("read")?;
        let (ret, frame): (bool, Bound<PyAny>) = result.extract()?;
        if !ret {
            break;
        }
        cv2.call_method1("imshow", (window_name, &frame))?;
        
        if let Some(ref w) = writer {
            w.call_method1("write", (&frame,))?;
        }
        
        let key = cv2.call_method1("waitKey", (1,))?.extract::<i32>()?;
        if key == 27 || key == b'q' as i32 { // ESC or 'q'
            break;
        }
    }
    
    cap.call_method0("release")?;
    if let Some(ref w) = writer {
        w.call_method0("release")?;
    }
    cv2.call_method0("destroyAllWindows")?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (video_path, output_dir, frame_interval = 1))]
pub fn extract_images_from_video<'py>(
    py: Python<'py>,
    video_path: &str,
    output_dir: &str,
    frame_interval: usize,
) -> PyResult<()> {
    let cv2 = py.import_bound("cv2")?;
    
    std::fs::create_dir_all(output_dir)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to create output directory: {}", e)))?;
        
    let cap = cv2.call_method1("VideoCapture", (video_path,))?;
    if !cap.call_method0("isOpened")?.extract::<bool>()? {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to open video file: {}",
            video_path
        )));
    }
    
    let mut count = 0;
    let mut saved_count = 0;
    while cap.call_method0("isOpened")?.extract::<bool>()? {
        let result = cap.call_method0("read")?;
        let (ret, frame): (bool, Bound<PyAny>) = result.extract()?;
        if !ret {
            break;
        }
        
        if count % frame_interval == 0 {
            let filename = format!("{}/frame_{:05}.png", output_dir, saved_count);
            cv2.call_method1("imwrite", (&filename, &frame))?;
            saved_count += 1;
        }
        count += 1;
    }
    cap.call_method0("release")?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (image_paths, output_video_path, fps = 20.0))]
pub fn extract_video_from_images<'py>(
    py: Python<'py>,
    image_paths: Vec<String>,
    output_video_path: &str,
    fps: f64,
) -> PyResult<()> {
    if image_paths.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Image paths list is empty"));
    }
    let cv2 = py.import_bound("cv2")?;
    
    let mut sorted_paths = image_paths;
    sorted_paths.sort();
    
    let first_frame = cv2.call_method1("imread", (&sorted_paths[0],))?;
    if first_frame.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Failed to read the first image: {}",
            sorted_paths[0]
        )));
    }
    
    let shape = first_frame.getattr("shape")?.extract::<Vec<usize>>()?;
    if shape.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid image shape"));
    }
    let height = shape[0];
    let width = shape[1];
    
    let fourcc = cv2.call_method1("VideoWriter_fourcc", ("m", "p", "4", "v"))?;
    let writer = cv2.call_method1("VideoWriter", (output_video_path, &fourcc, fps, (width, height)))?;
    if !writer.call_method0("isOpened")?.extract::<bool>()? {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("Failed to open VideoWriter"));
    }
    
    for path in sorted_paths {
        let frame = cv2.call_method1("imread", (&path,))?;
        if !frame.is_none() {
            writer.call_method1("write", (&frame,))?;
        }
    }
    writer.call_method0("release")?;
    Ok(())
}

/// Applies MOG2 background subtraction to a video file.
///
/// Processes each frame through a Mixture of Gaussians (v2) background model,
/// returning a foreground mask. The mask is cleaned with morphological opening
/// to remove noise.
///
/// Mask values: 0 = background, 127 = shadow (when detectShadows=true), 255 = foreground.
///
/// If `output_path` is provided, the foreground mask video is saved to that path.
/// Otherwise, the mask is displayed in a window (press 'q' or ESC to exit).
#[pyfunction]
#[pyo3(signature = (video_path, history = 500, var_threshold = 16.0, detect_shadows = true, kernel_size = 5, output_path = None))]
pub fn background_subtract_mog2<'py>(
    py: Python<'py>,
    video_path: &str,
    history: i32,
    var_threshold: f64,
    detect_shadows: bool,
    kernel_size: i32,
    output_path: Option<String>,
) -> PyResult<()> {
    let cv2 = py.import_bound("cv2")?;

    // Create the MOG2 background subtractor (positional args: history, varThreshold, detectShadows)
    let mog2 = cv2.call_method1(
        "createBackgroundSubtractorMOG2",
        (history, var_threshold, detect_shadows),
    )?;

    // Open the video
    let cap = cv2.call_method1("VideoCapture", (video_path,))?;
    if !cap.call_method0("isOpened")?.extract::<bool>()? {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to open video file: {}",
            video_path
        )));
    }

    // Build the morphological kernel (elliptical)
    let morph_ellipse = cv2.getattr("MORPH_ELLIPSE")?.extract::<i32>()?;
    let kernel = cv2.call_method1(
        "getStructuringElement",
        (morph_ellipse, (kernel_size, kernel_size)),
    )?;

    // Set up an optional writer for saving the mask video
    let mut writer: Option<Bound<PyAny>> = None;
    if let Some(ref path) = output_path {
        let width = cap.call_method1("get", (3,))?.extract::<f64>()? as i32;
        let height = cap.call_method1("get", (4,))?.extract::<f64>()? as i32;
        let fps = cap.call_method1("get", (5,))?.extract::<f64>()?;
        let fps = if fps > 0.0 { fps } else { 20.0 };
        let fourcc = cv2.call_method1("VideoWriter_fourcc", ("m", "p", "4", "v"))?;
        // Positional args: filename, fourcc, fps, frameSize, isColor
        // isColor = false since the mask is single-channel grayscale
        writer = Some(cv2.call_method1(
            "VideoWriter",
            (path, &fourcc, fps, (width, height), false),
        )?);
    }

    let morph_open = cv2.getattr("MORPH_OPEN")?.extract::<i32>()?;
    let window_name = "MOG2 Background Subtraction (Press 'q' or 'ESC' to exit)";

    while cap.call_method0("isOpened")?.extract::<bool>()? {
        let result = cap.call_method0("read")?;
        let (ret, frame): (bool, Bound<PyAny>) = result.extract()?;
        if !ret {
            break;
        }

        // Apply MOG2 — produces the foreground mask
        let mask = mog2.call_method1("apply", (&frame,))?;

        // Clean up with morphological opening
        let mask = cv2.call_method1("morphologyEx", (&mask, morph_open, &kernel))?;

        if let Some(ref w) = writer {
            w.call_method1("write", (&mask,))?;
        } else {
            cv2.call_method1("imshow", (window_name, &mask))?;
            let key = cv2.call_method1("waitKey", (1,))?.extract::<i32>()?;
            if key == 27 || key == b'q' as i32 {
                break;
            }
        }
    }

    cap.call_method0("release")?;
    if let Some(ref w) = writer {
        w.call_method0("release")?;
    }
    if output_path.is_none() {
        cv2.call_method0("destroyAllWindows")?;
    }
    Ok(())
}

