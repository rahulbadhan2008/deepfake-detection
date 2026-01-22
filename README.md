# Synthetic Image Detector

A lightweight, interpretable Python tool to detect AI-generated images (specifically diffusion models) by analyzing luminance gradient fields.

## 🚀 Overview

This tool implements a physics-based approach to image forensics. It assumes:
- **Real Photos**: Have coherent gradient fields derived from physical lighting and sensor characteristics.
- **AI Images**: Exhibit unstable high-frequency structures and anisotropy in their gradient fields due to the denoising process.

By projecting image gradients into a lower-dimensional space using PCA, we can identify these structural anomalies without training complex machine learning models.

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

## 🔧 Troubleshooting

If you encounter `ImportError: cannot import name 'Image' from 'PIL'`, ensure you have the correct version of Pillow installed:
```bash
pip uninstall Pillow
pip install Pillow --upgrade
```

## 📝 License

MIT License
