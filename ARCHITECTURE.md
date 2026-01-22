# Architecture & Logic

This document explains the technical architecture and the theoretical logic behind the Synthetic Image Detector.

## 🧠 Core Logic: Gradient Field Analysis

The detector works on the principle that Generative AI models (like Stable Diffusion, Midjourney) and physical cameras construct images differently.

### 1. The Physical Model (Real Images)
*   **Origin**: Light hitting a physical sensor.
*   **Characteristics**: Lighting creates smooth, coherent gradients. Noise is typically Poissonian (shot noise) or Gaussian (read noise), which is uniform and unstructured.
*   **Gradient Field**: Isotropic (similar properties in all directions) or structured by the physical geometry of the scene.

### 2. The Diffusion Model (AI Images)
*   **Origin**: Iterative denoising from pure Gaussian noise.
*   **Characteristics**: The reverse diffusion process often leaves high-frequency residual artifacts.
*   **Gradient Field**: Anisotropic instablities and "ridges" in the gradient covariance matrix that don't match physical lighting falloff.

## ⚙️ Implementation Pipeline

The logic is implemented in `detector.py` following this pipeline:

```mermaid
graph TD
    A[Input Image] --> B[RGB to Luminance]
    B --> C[Compute Spatial Gradients]
    C --> D[Gradient Covariance Matrix]
    D --> E[PCA Decomposition]
    E --> F[Feature Extraction]
    F --> G[Classification Rules]
    G --> H[Result: Real / Fake]
```

### Sequence Diagram

```mermaid
sequenceDiagram
    participant CLI as Main CLI
    participant DET as Detector Logic
    participant IMG as Image Processor
    
    CLI->>CLI: Parse Arguments
    CLI->>DET: detect_synthetic_image(path)
    DET->>IMG: Load & Verify Image
    IMG-->>DET: Image Array
    
    DET->>DET: rgb_to_luminance()
    DET->>DET: compute_gradients()
    DET->>DET: analyze_gradient_pca()
    
    Note over DET: Calculates Variance, Kurtosis,<br/>and Spectral Entropy
    
    DET->>DET: Apply Thresholds
    DET-->>CLI: Return Classification & Metrics
    
    CLI->>CLI: Format & Print Output
```

### Step 1: Preprocessing (`detector.rgb_to_luminance`)
We convert the HxWx3 color image to a single HxW luminance channel using ITU-R BT.709 coefficients. Color information is discarded to focus purely on structural texture.

### Step 2: Gradient Computation (`detector.compute_gradients`)
We calculate the first-order derivatives (pixel differences) in X and Y directions.
*   $G_x(i,j) = I(i, j+1) - I(i, j-1)$
*   $G_y(i,j) = I(i+1, j) - I(i-1, j)$

### Step 3: PCA Analysis (`detector.analyze_gradient_pca`)
We treat every pixel's gradient $(g_x, g_y)$ as a data point in 2D space.
1.  Form the covariance matrix of these gradients.
2.  Compute Eigenvalues ($\lambda_1, \lambda_2$).
3.  Project gradients onto the principal component.

### Step 4: Decision Metrics
We extract specific features that separate Real from Fake:

| Metric | Real Behavior | AI/Diffusion Behavior |
| :--- | :--- | :--- |
| **Coefficient of Variation** | Lower | **Higher** (Unstable local gradients) |
| **Kurtosis** | Normal | **High** (Heavy tails/outliers in gradient distribution) |
| **High-Freq Ratio** | Balanced | **Concentrated** (Pixel-level noise artifacts) |
| **Eigenvalue Ratio** | Anisotropic | **Mixed** (Often overly isotropic or oddly skewed) |

> **Note:** The scoring system is **Weighted**. Structural metrics (Eigenvalue Ratio, CV) have higher priority than Noise metrics (Kurtosis, HF Ratio). This prevents noisy real photos (e.g., high ISO) from being misclassified as fake.

## 📂 File Structure

*   `main.py`: **Entry Point**. Handles CLI arguments, file I/O, and user display.
*   `detector.py`: **Core Logic**. Contains the math, image processing, and classification rules. Pure functional design.
*   `requirements.txt`: Python package dependencies.
