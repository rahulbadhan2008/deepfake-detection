# Synthetic Image Detector

A lightweight, interpretable Python tool to detect AI-generated images (specifically diffusion models) by analyzing luminance gradient fields.

## 🚀 Overview

This tool implements a physics-based approach to image forensics. It assumes:
- **Real Photos**: Have coherent gradient fields derived from physical lighting and sensor characteristics.
- **AI Images**: Exhibit unstable high-frequency structures and anisotropy in their gradient fields due to the denoising process.

By projecting image gradients into a lower-dimensional space using PCA, we can identify these structural anomalies without training complex machine learning models.

## 🧠 How it Works

The detector uses **Luminance-Gradient PCA Analysis**.

1.  **Convert to Luminance**: Strip color to focus on structure.
2.  **Compute Gradients**: Measure pixel-to-pixel changes.
3.  **Analyze Covariance**: Use PCA to find the "shape" of the gradient field.

Real photos have coherent, isotropic gradient fields (from physical light). Diffusion models often leave specific anisotropic "fingerprints" in the noise.

👉 **[Read the full logical breakdown and diagrams in ARCHITECTURE.md](./ARCHITECTURE.md)**

## 📦 Installation

1.  **Clone or Download** this repository.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Requires Python 3.8+*

## 🛠️ Usage

### Analyze a Single Image
Run the detector on any image file:

```bash
python main.py path/to/image.jpg
```

**Example Output:**
```text
╔══════════════════════════════════════════════════════════════╗
║       🔍 Synthetic Image Detection using Gradient Fields      ║
║                    Luminance-Gradient PCA Analysis             ║
╚══════════════════════════════════════════════════════════════╝

📷 Analyzing: path/to/image.jpg
--------------------------------------------------

🎯 Result: ⚠️ Diffusion Generated
   Confidence: 85.0%
   Detection Score: 0.92/1.00

📊 Detailed Metrics:
   • Coefficient of Variation: 2.1504
   • Kurtosis: 5.1203
   ...
```

### Analyze a Directory
Scan an entire folder of images:

```bash
python main.py --dir ./my_images
```

### Options
- `--verbose, -v`: Show detailed PCA metrics (ON by default for single images).
- `--quiet, -q`: Output only the classification result (useful for scripts).
- `--test`: Run a built-in self-test to verify installation.

## 📊 Visualization
You can generate visual charts of the analysis (Gradient Map, PCA Projection) using the `--visualize` flag.
```bash
python main.py image.jpg --visualize
```
This saves multiple analysis images in the `analysis/` folder:
1.  `*_comparison.png`: Side-by-Side "Neck-to-Neck" view.
2.  `*_gradient_dist.png`: Gradient Magnitude Distribution (Log-Log).

### ⚔️ Comparison Report (Head-to-Head)
To generate a detailed forensic comparison between a specific Real and Fake image:

```bash
python compare.py --real path/to/real.jpg --fake path/to/fake.jpg
```
This generates `comparison_report.png` in the current directory.

### 🧠 Interpreting the Graphs (Visual Guide)

#### 1. Pixel Anomaly Comparison (Heatmap)
| What to look for | Status | Meaning |
| :--- | :--- | :--- |
| **Dark / "Ghost"** | ✅ **Real** | Deviations are low and follow edges. |
| **Bright Red / "Lava"** | ⚠️ **Fake** | High anomaly scores across the image. |

#### 2. Gradient Distribution (Log-Log Plot)
| Shape | Status | Meaning |
| :--- | :--- | :--- |
| **Straight Line** | ✅ **Real** | Follows natural 1/f Power Law statistics. |
| **Curved / Drop-off** | ⚠️ **Fake** | Lacks high-frequency natural gradients (denoising artifact). |

## ⚙️ Configuration
You can fine-tune the detection sensitivity by adjusting the internal scoring thresholds via CLI arguments:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--th-cv` | 1.8 | Coefficient of Variation Threshold |
| `--th-kurt` | 50.0 | Kurtosis Threshold (Higher = more tolerance for noise) |
| `--th-hf` | 0.45 | High-Frequency Energy Ratio Threshold |
| `--th-ev-low` | 1.5 | Eigenvalue Ratio Lower Bound |
| `--th-ev-high` | 50.0 | Eigenvalue Ratio Upper Bound |

**Example:**
```bash
python main.py image.jpg --th-cv 2.0 --th-kurt 5.0
```

## 🔧 Troubleshooting

If you encounter `ImportError: cannot import name 'Image' from 'PIL'`, ensure you have the correct version of Pillow installed:
```bash
pip uninstall Pillow
pip install Pillow --upgrade
```

## 📝 License

MIT License
