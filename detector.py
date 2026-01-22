"""
Synthetic Image Detection using Gradient Fields and PCA

This module provides a lightweight, interpretable method to detect AI-generated
(diffusion-based) images by analyzing luminance gradient field characteristics.

Real photos produce coherent gradient fields tied to physical lighting and sensor noise,
while diffusion-generated images exhibit unstable high-frequency structures from denoising.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Union
import os


def rgb_to_luminance(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to luminance using ITU-R BT.709 coefficients.
    
    Args:
        image: RGB image array of shape (H, W, 3)
        
    Returns:
        Luminance array of shape (H, W)
    """
    # ITU-R BT.709 luminance coefficients
    coefficients = np.array([0.2126, 0.7152, 0.0722])
    return np.dot(image[..., :3], coefficients)


def compute_gradients(luminance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial gradients (horizontal and vertical) of luminance field.
    
    Args:
        luminance: 2D luminance array
        
    Returns:
        Tuple of (gradient_x, gradient_y) arrays
    """
    # Horizontal gradient (Sobel-like, but simpler central difference)
    gradient_x = np.zeros_like(luminance)
    gradient_x[:, 1:-1] = luminance[:, 2:] - luminance[:, :-2]
    
    # Vertical gradient
    gradient_y = np.zeros_like(luminance)
    gradient_y[1:-1, :] = luminance[2:, :] - luminance[:-2, :]
    
    return gradient_x, gradient_y


def analyze_gradient_pca(gradient_x: np.ndarray, gradient_y: np.ndarray) -> dict:
    """
    Perform PCA analysis on the gradient field and extract discriminative features.
    
    Args:
        gradient_x: Horizontal gradient array
        gradient_y: Vertical gradient array
        
    Returns:
        Dictionary containing PCA-derived metrics
    """
    # Flatten and stack gradients into feature matrix
    gx_flat = gradient_x.flatten()
    gy_flat = gradient_y.flatten()
    
    # Create gradient feature matrix (N x 2)
    gradient_matrix = np.vstack([gx_flat, gy_flat]).T
    
    # Center the data
    mean = np.mean(gradient_matrix, axis=0)
    centered = gradient_matrix - mean
    
    # Compute covariance matrix
    cov_matrix = np.cov(centered.T)
    
    # Eigendecomposition for PCA
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute discriminative metrics
    total_variance = np.sum(eigenvalues)
    
    # Variance ratio (how much variance is captured by first component)
    variance_ratio = eigenvalues[0] / (total_variance + 1e-10)
    
    # Eigenvalue ratio (anisotropy of gradient field)
    eigenvalue_ratio = eigenvalues[0] / (eigenvalues[1] + 1e-10)
    
    # Project data onto first principal component
    projection = centered @ eigenvectors[:, 0]
    
    # Reshape projection back to 2D for visualization
    h, w = gradient_x.shape
    projection_map = projection.reshape(h, w)
    
    # High-frequency instability metric (kurtosis of projection)
    projection_std = np.std(projection)
    projection_mean = np.mean(projection)
    kurtosis = np.mean(((projection - projection_mean) / (projection_std + 1e-10)) ** 4)
    
    # Gradient magnitude statistics
    gradient_magnitude = np.sqrt(gx_flat**2 + gy_flat**2)
    magnitude_mean = np.mean(gradient_magnitude)
    magnitude_std = np.std(gradient_magnitude)
    
    # Coefficient of variation (normalized dispersion)
    coeff_variation = magnitude_std / (magnitude_mean + 1e-10)
    
    # High-frequency energy ratio (top 10% magnitude vs total)
    sorted_magnitudes = np.sort(gradient_magnitude)[::-1]
    top_10_percent = int(len(sorted_magnitudes) * 0.1)
    high_freq_ratio = np.sum(sorted_magnitudes[:top_10_percent]) / (np.sum(sorted_magnitudes) + 1e-10)
    
    return {
        'variance_ratio': variance_ratio,
        'eigenvalue_ratio': eigenvalue_ratio,
        'kurtosis': kurtosis,
        'coeff_variation': coeff_variation,
        'high_freq_ratio': high_freq_ratio,
        'magnitude_mean': magnitude_mean,
        'magnitude_std': magnitude_std,
        'eigenvalues': eigenvalues,
        'total_variance': total_variance,
        'projection_map': projection_map,
        'gradient_x': gradient_x,
        'gradient_y': gradient_y
    }


def detect_synthetic_image(
    image_input: Union[str, np.ndarray, Image.Image],
    return_details: bool = False,
    thresholds: dict = None
) -> Union[str, Tuple[str, dict]]:
    """
    Detect whether an image is real or diffusion-generated using gradient field PCA analysis.
    
    This function analyzes the luminance gradient field of an image to determine authenticity.
    Real photographs exhibit coherent gradient structures from physical lighting and sensor
    characteristics, while diffusion-generated images show distinctive high-frequency
    instabilities from the denoising process.
    
    Args:
        image_input: Input image as file path (str), numpy array (H,W,3), or PIL Image
        return_details: If True, return detailed analysis metrics along with classification
        thresholds: Optional dictionary to override default detection thresholds
        
    Returns:
        If return_details is False:
            str: "Real" or "Diffusion Generated"
        If return_details is True:
            Tuple[str, dict]: (classification, metrics_dictionary)
            
    Raises:
        ValueError: If image cannot be loaded or has invalid format
        FileNotFoundError: If image path does not exist
        
    Example:
        >>> result = detect_synthetic_image("photo.jpg")
        >>> print(result)
        "Real"
        
        >>> result, details = detect_synthetic_image("ai_image.png", return_details=True)
        >>> print(f"{result}, confidence metrics: {details['confidence_score']:.2f}")
    """
    # Load and validate image
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        try:
            image = Image.open(image_input).convert('RGB')
            image_array = np.array(image, dtype=np.float64) / 255.0
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
        image_array = np.array(image, dtype=np.float64) / 255.0
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim != 3 or image_input.shape[2] < 3:
            raise ValueError("Image array must have shape (H, W, 3) or (H, W, 4)")
        image_array = image_input[:, :, :3].astype(np.float64)
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    # Validate image dimensions
    if image_array.shape[0] < 32 or image_array.shape[1] < 32:
        raise ValueError("Image must be at least 32x32 pixels")
    
    # Step 1: Convert to luminance
    luminance = rgb_to_luminance(image_array)
    
    # Step 2: Compute spatial gradients
    gradient_x, gradient_y = compute_gradients(luminance)
    
    # Step 3: PCA analysis on gradient field
    metrics = analyze_gradient_pca(gradient_x, gradient_y)
    
    # Step 4: Classification using interpretable thresholds
    # These thresholds are derived from the characteristic differences between
    # real photos (coherent gradients) and diffusion outputs (unstable high-freq patterns)
    
    # Classification decision
    # Default thresholds or overrides
    th = {
        'cv': 1.8,
        'kurtosis': 4.5,
        'hf': 0.35,
        'ev_low': 1.5,
        'ev_high': 50.0
    }
    if thresholds:
        th.update(thresholds)
    
    # Scoring system based on multiple metrics
    score = 0.0
    
    # Metric 1: Coefficient of variation
    if metrics['coeff_variation'] > th['cv']:
        score += 0.25
    
    # Metric 2: Kurtosis of PCA projection
    if metrics['kurtosis'] > th['kurtosis']:
        score += 0.25
    
    # Metric 3: High-frequency energy ratio
    if metrics['high_freq_ratio'] > th['hf']:
        score += 0.25
    
    # Metric 4: Eigenvalue ratio (gradient field anisotropy)
    if metrics['eigenvalue_ratio'] < th['ev_low'] or metrics['eigenvalue_ratio'] > th['ev_high']:
        score += 0.25
    
    # Classification decision
    classification_threshold = 0.5
    is_synthetic = score >= classification_threshold
    
    # Prepare result
    classification = "Diffusion Generated" if is_synthetic else "Real"
    
    if return_details:
        details = {
            **metrics,
            'detection_score': score,
            'confidence_score': abs(score - 0.5) * 2,  # 0 to 1 confidence
            'thresholds': {
                'cv_threshold': th['cv'],
                'kurtosis_threshold': th['kurtosis'],
                'hf_threshold': th['hf'],
                'ev_threshold_low': th['ev_low'],
                'ev_threshold_high': th['ev_high'],
                'classification_threshold': classification_threshold
            },
            'image_dimensions': image_array.shape[:2]
        }
        return classification, details
    
    return classification


def batch_detect(image_paths: list, verbose: bool = True) -> list:
    """
    Process multiple images and return detection results.
    
    Args:
        image_paths: List of image file paths
        verbose: Print progress if True
        
    Returns:
        List of tuples: [(path, classification, confidence_score), ...]
    """
    results = []
    total = len(image_paths)
    
    for i, path in enumerate(image_paths):
        try:
            classification, details = detect_synthetic_image(path, return_details=True)
            results.append((path, classification, details['confidence_score']))
            if verbose:
                print(f"[{i+1}/{total}] {os.path.basename(path)}: {classification} "
                      f"(confidence: {details['confidence_score']:.2f})")
        except Exception as e:
            results.append((path, "Error", str(e)))
            if verbose:
                print(f"[{i+1}/{total}] {os.path.basename(path)}: Error - {e}")
    
    return results


# Quick test function
def _self_test():
    """Generate a simple synthetic test to verify the pipeline works."""
    print("Running self-test...")
    
    # Create a simple gradient test image
    test_image = np.zeros((256, 256, 3), dtype=np.float64)
    for i in range(256):
        test_image[i, :, :] = i / 255.0
    
    result, details = detect_synthetic_image(test_image, return_details=True)
    print(f"Test image result: {result}")
    print(f"Detection score: {details['detection_score']:.3f}")
    print(f"Coefficient of variation: {details['coeff_variation']:.3f}")
    print(f"Kurtosis: {details['kurtosis']:.3f}")
    print("Self-test completed successfully!")
    return True


if __name__ == "__main__":
    _self_test()
