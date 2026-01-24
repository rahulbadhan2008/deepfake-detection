#!/usr/bin/env python3
"""
Forensic Image Comparison Report Generator
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from detector import detect_synthetic_image

def get_anomaly_map(details):
    proj_map = details['projection_map']
    # Calculate Absolute Anomaly Magnitude
    anomaly_map = np.abs(proj_map)
    # Robust normalization (0-1)
    vmax = np.percentile(anomaly_map, 99.0) # slightly relaxed percentile for comparison
    anomaly_norm = np.clip(anomaly_map / (vmax + 1e-10), 0, 1)
    return anomaly_norm

def plot_gradient_dist(ax, details, color, label):
    gx = details['gradient_x'].flatten()
    gy = details['gradient_y'].flatten()
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Sort descending
    sorted_indices = np.argsort(magnitude)[::-1]
    sorted_magnitude = magnitude[sorted_indices]
    
    # Filter
    valid_mask = sorted_magnitude > 1e-6
    sorted_magnitude = sorted_magnitude[valid_mask]
    
    # Rank
    ranks = np.arange(1, len(sorted_magnitude) + 1)
    
    # Downsample
    if len(ranks) > 10000:
        indices = np.linspace(0, len(ranks)-1, 10000).astype(int)
        ranks = ranks[indices]
        sorted_magnitude = sorted_magnitude[indices]
        
    ax.loglog(ranks, sorted_magnitude, color=color, linewidth=2, label=label)
    
    # Reference
    ref_x = ranks
    ref_y = sorted_magnitude[0] * (ranks[0] / ref_x)
    ax.loglog(ref_x, ref_y, 'k--', alpha=0.3, label='Theoretical 1/f')

def create_report(real_path, fake_path, output_path="comparison_report.png"):
    print("running comparison...")
    
    # Run analysis
    res_real, det_real = detect_synthetic_image(real_path, return_details=True)
    res_fake, det_fake = detect_synthetic_image(fake_path, return_details=True)
    
    # Setup Figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.4])
    
    # --- ROW 1: REAL IMAGE ANALYSIS ---
    ax_r_img = fig.add_subplot(gs[0, 0])
    ax_r_map = fig.add_subplot(gs[0, 1])
    ax_r_graph = fig.add_subplot(gs[0, 2])
    ax_r_stats = fig.add_subplot(gs[0, 3])
    
    # Image
    img_real = Image.open(real_path).convert('RGB')
    ax_r_img.imshow(img_real)
    ax_r_img.set_title(f"Reference Image (Real)\n{os.path.basename(real_path)}", fontsize=12, fontweight='bold', color='green')
    ax_r_img.axis('off')
    
    # Map
    map_real = get_anomaly_map(det_real)
    ax_r_map.imshow(map_real, cmap='inferno', vmin=0, vmax=1)
    ax_r_map.set_title("Anomaly Heatmap\n(Expect: Dark/Ghost)", fontsize=11)
    ax_r_map.axis('off')
    
    # Graph
    plot_gradient_dist(ax_r_graph, det_real, 'green', 'Real')
    ax_r_graph.set_title("Gradient Distribution\n(Expect: Straight Line)", fontsize=11)
    ax_r_graph.grid(True, which='both', alpha=0.3)
    
    # Stats Text
    stats_text_real = (
        f"Result: {res_real}\n"
        f"Score: {det_real['detection_score']:.2f}\n\n"
        f"Key Metrics:\n"
        f"• Kurtosis: {det_real['kurtosis']:.1f} (Low)\n"
        f"• CV: {det_real['coeff_variation']:.2f} (Low)\n"
        f"• High-Freq: {det_real['high_freq_ratio']:.2f}"
    )
    ax_r_stats.text(0.1, 0.5, stats_text_real, fontsize=12, va='center', family='monospace')
    ax_r_stats.axis('off')

    # --- ROW 2: FAKE IMAGE ANALYSIS ---
    ax_f_img = fig.add_subplot(gs[1, 0])
    ax_f_map = fig.add_subplot(gs[1, 1])
    ax_f_graph = fig.add_subplot(gs[1, 2])
    ax_f_stats = fig.add_subplot(gs[1, 3])
    
    # Image
    img_fake = Image.open(fake_path).convert('RGB')
    ax_f_img.imshow(img_fake)
    ax_f_img.set_title(f"Suspect Image (Fake)\n{os.path.basename(fake_path)}", fontsize=12, fontweight='bold', color='red')
    ax_f_img.axis('off')
    
    # Map
    map_fake = get_anomaly_map(det_fake)
    ax_f_map.imshow(map_fake, cmap='inferno', vmin=0, vmax=1)
    ax_f_map.set_title("Anomaly Heatmap\n(Expect: Bright/Noise)", fontsize=11)
    ax_f_map.axis('off')
    
    # Graph
    plot_gradient_dist(ax_f_graph, det_fake, 'red', 'Fake')
    ax_f_graph.set_title("Gradient Distribution\n(Expect: Curved/Drop-off)", fontsize=11)
    ax_f_graph.grid(True, which='both', alpha=0.3)
    
    # Stats Text
    stats_text_fake = (
        f"Result: {res_fake}\n"
        f"Score: {det_fake['detection_score']:.2f}\n\n"
        f"Key Metrics:\n"
        f"• Kurtosis: {det_fake['kurtosis']:.1f} (High)\n"
        f"• CV: {det_fake['coeff_variation']:.2f} (High)\n"
        f"• High-Freq: {det_fake['high_freq_ratio']:.2f}"
    )
    ax_f_stats.text(0.1, 0.5, stats_text_fake, fontsize=12, va='center', family='monospace')
    ax_f_stats.axis('off')

    # --- ROW 3: SUMMARY BANNER ---
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    explanation = (
        "INTERPRETATION GUIDE:\n"
        "1. Heatmaps: Real images show structure (edges). Fake images show random 'snow' or grid patterns.\n"
        "2. Graphs: Real images follow a straight 1/f Power Law line. Fake images often curve downwards purely.\n"
        "3. Metrics: Kurtosis measures 'spikeyness' of noise. High Kurtosis (>50) is a strong indicator of diffusion models."
    )
    
    ax_summary.text(0.5, 0.5, explanation, ha='center', va='center', fontsize=14, 
                    bbox=dict(facecolor='#f0f0f0', edgecolor='gray', pad=10))

    plt.suptitle("Forensic Image Analysis: Real vs Synthetic Comparison", fontsize=24, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig(output_path, dpi=100)
    print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', required=True, help='Path to real image')
    parser.add_argument('--fake', required=True, help='Path to fake image')
    parser.add_argument('--output', default='comparison_report.png', help='Output filename')
    args = parser.parse_args()
    
    create_report(args.real, args.fake, args.output)
