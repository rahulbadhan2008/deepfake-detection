#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Image Detection CLI

Command-line interface for detecting AI-generated images using gradient field analysis.
"""

import argparse
import sys
import os
import numpy as np
from detector import detect_synthetic_image, batch_detect

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def print_banner():
    """Display ASCII banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║       🔍 Synthetic Image Detection using Gradient Fields      ║
║                    Luminance-Gradient PCA Analysis             ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def format_result(classification, details=None):
    """Format detection result for display."""
    if classification == "Real":
        icon = "✅"
        color_code = "\033[92m"  # Green
    else:
        icon = "⚠️"
        color_code = "\033[93m"  # Yellow
    
    reset_code = "\033[0m"
    
    result = f"{color_code}{icon} {classification}{reset_code}"
    
    if details:
        result += f"\n   Confidence: {details['confidence_score']:.1%}"
        result += f"\n   Detection Score: {details['detection_score']:.2f}/1.00"
    
    return result


def save_visualization(image_path, details, output_path=None):
    """Generate and save analysis plots."""
    if not HAS_MATPLOTLIB:
        print("⚠️  Matplotlib not installed. Skipping visualization.")
        return

    if output_path is None:
        # Create visualization directory
        vis_dir = "visualization"
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
            
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(vis_dir, f"{base}_analysis.png")

    try:
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Original Image (Load again for display)
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # 2. Gradient Relief (Embossed effect like the user's reference)
        # Combine Gx and Gy to create a lighting effect (e.g., light from top-left)
        # Normalizing to 0-1 range for display, centered at 0.5
        gx = details['gradient_x']
        gy = details['gradient_y']
        
        # Simple relief: Gx + Gy
        relief = gx + gy
        
        # Normalize robustly
        vmax = np.percentile(np.abs(relief), 98)
        relief_norm = np.clip(relief / (vmax + 1e-10), -1, 1) * 0.5 + 0.5
        
        axes[1].imshow(relief_norm, cmap='gray')
        axes[1].set_title("Luminance Gradient Field (Relief)")
        axes[1].axis('off')
        
        # 3. PCA Projection (The "Artifact" Map)
        # We normalize specific ranges to highlight outliers
        proj_map = details['projection_map']
        vmax = np.percentile(np.abs(proj_map), 99) # Slightly tighter clamp
        im3 = axes[2].imshow(proj_map, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[2].set_title("PCA Projection (Structural Anomalies)")
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Add a super title with classification and score
        score = details['detection_score']
        classification = "Diffusion Generated" if score >= 0.5 else "Real"
        plt.suptitle(f"Result: {classification} (Score: {score:.2f})", fontsize=16)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"   📊 Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")


def save_gradient_distribution_plot(image_path, details):
    """Generate a Gradient Magnitude Distribution Plot (Log-Log)."""
    if not HAS_MATPLOTLIB:
        return

    # Create analysis directory
    analysis_dir = "analysis"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
        
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(analysis_dir, f"{base}_gradient_dist.png")
    
    try:
        # Get gradients
        gx = details['gradient_x'].flatten()
        gy = details['gradient_y'].flatten()
        
        # Calculate Magnitude
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Sort descending (for rank-ordered plot)
        sorted_indices = np.argsort(magnitude)[::-1]
        sorted_magnitude = magnitude[sorted_indices]
        
        # Filter zero/near-zero values for Log plot
        valid_mask = sorted_magnitude > 1e-6
        sorted_magnitude = sorted_magnitude[valid_mask]
        
        # Create Rank vector (1 to N)
        ranks = np.arange(1, len(sorted_magnitude) + 1)
        
        # Downsample for plotting performance (max 10k points)
        if len(ranks) > 10000:
            indices = np.linspace(0, len(ranks)-1, 10000).astype(int)
            ranks = ranks[indices]
            sorted_magnitude = sorted_magnitude[indices]
            
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Classification color
        score = details['detection_score']
        classification = "Diffusion Generated" if score >= 0.5 else "Real"
        color = 'red' if classification == 'Diffusion Generated' else 'green'
        
        # Log-Log Line Plot
        ax.loglog(ranks, sorted_magnitude, color=color, linewidth=2, alpha=0.8, label=classification)
        
        ax.set_title(f"Gradient Magnitude Distribution (Log-Log)\n{os.path.basename(image_path)}", fontsize=14)
        ax.set_xlabel("Rank (Log Scale)")
        ax.set_ylabel("Gradient Magnitude (Log Scale)")
        ax.grid(True, which="both", ls="--", alpha=0.3)
        
        # Add a reference "Perfect Power Law" line (1/x) for visual comparison
        # Arbitrary intercept to match data scale roughly
        ref_x = ranks
        ref_y = sorted_magnitude[0] * (ranks[0] / ref_x) # y = k/x
        ax.loglog(ref_x, ref_y, 'k--', alpha=0.3, label='Theoretical Natural 1/f')
        
        ax.legend()
        
        plt.figtext(0.5, 0.02, f"Result: {classification} (Score: {score:.2f})", 
                    ha="center", fontsize=12, bbox={"facecolor":color, "alpha":0.2, "pad":5})
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"   📈 Gradient Distribution Plot saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving distribution plot: {e}")


def save_side_by_side(image_path, details):
    """Generate a Side-by-Side (Neck-to-Neck) comparison."""
    if not HAS_MATPLOTLIB:
        return

    # Create analysis directory
    analysis_dir = "analysis"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
        
    base = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(analysis_dir, f"{base}_comparison.png")
    
    try:
        # 1. Original Image
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        
        # 2. Anomaly Map
        proj_map = details['projection_map']
        anomaly_map = np.abs(proj_map)
        
        # Robust normalization (0-1)
        vmax = np.percentile(anomaly_map, 99.5) 
        anomaly_norm = np.clip(anomaly_map / (vmax + 1e-10), 0, 1)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Original
        axes[0].imshow(img)
        axes[0].set_title("Original Image", fontsize=16)
        axes[0].axis('off')
        
        # Right: Anomaly Map
        im = axes[1].imshow(anomaly_norm, cmap='inferno', vmin=0, vmax=1)
        axes[1].set_title("Pixel Anomaly Analysis", fontsize=16)
        axes[1].axis('off')
        
        # Classification Result
        score = details['detection_score']
        classification = "Diffusion Generated" if score >= 0.5 else "Real"
        color = 'red' if classification == 'Diffusion Generated' else 'green'
        
        # Add visual classification banner
        plt.suptitle(f"Analysis Result: {classification}\nScore: {score:.2f}", 
                     fontsize=20, color='white', backgroundcolor=color) #, weight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # Make room for suptitle
        plt.savefig(output_path, dpi=100)
        plt.close(fig)
        
        print(f"   🆚 Comparison Graph saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving comparison: {e}")


def analyze_single(image_path, verbose=True, visualize=False, thresholds=None):
    """Analyze a single image."""
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        sys.exit(1)
    
    if verbose:
        print(f"\n📷 Analyzing: {image_path}")
        print("-" * 50)
    
    try:
        classification, details = detect_synthetic_image(image_path, return_details=True, thresholds=thresholds)
        
        print(f"\n🎯 Result: {format_result(classification, details)}")
        
        if verbose:
            print(f"\n📊 Detailed Metrics:")
            print(f"   • Coefficient of Variation: {details['coeff_variation']:.4f}")
            print(f"   • Kurtosis: {details['kurtosis']:.4f}")
            print(f"   • High-Freq Ratio: {details['high_freq_ratio']:.4f}")
            print(f"   • Eigenvalue Ratio: {details['eigenvalue_ratio']:.4f}")
            print(f"   • Total Variance: {details['total_variance']:.6f}")
            
            meta = details.get('metadata', {})
            print(f"   • Metadata Check: {meta.get('metadata_desc', 'N/A')}")
            if meta.get('camera') != "Unknown":
                print(f"     -> Camera: {meta['camera']}")
            if meta.get('software') != "Unknown":
                print(f"     -> Software: {meta['software']}")
                
            print(f"   • Image Size: {details['image_dimensions'][0]}x{details['image_dimensions'][1]}")
            
        if visualize:
            save_visualization(image_path, details)
            save_gradient_distribution_plot(image_path, details)
            save_side_by_side(image_path, details)
        
        return classification, details
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        # import traceback
        # traceback.print_exc()
        sys.exit(1)


def analyze_directory(dir_path, extensions=None, visualize=False, thresholds=None):
    """Analyze all images in a directory."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']
    
    if not os.path.isdir(dir_path):
        print(f"❌ Error: Directory not found: {dir_path}")
        sys.exit(1)
    
    # Collect image files
    image_files = []
    for filename in os.listdir(dir_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in extensions:
            image_files.append(os.path.join(dir_path, filename))
    
    if not image_files:
        print(f"⚠️ No image files found in: {dir_path}")
        return
    
    print(f"\n📁 Found {len(image_files)} images in: {dir_path}")
    print(f"{'Filename':<25} | {'Result':<10} | {'Score':<5} | {'Kurt':<5} | {'CV':<5} | {'HF':<5} | {'EV':<5} | {'Metadata':<20}")
    print("-" * 115)
    
    results = []
    real_count = 0
    synthetic_count = 0
    error_count = 0
    
    for i, path in enumerate(image_files):
        filename = os.path.basename(path)
        # Truncate filename if too long
        fname_display = (filename[:22] + '..') if len(filename) > 24 else filename
        
        try:
            classification, details = detect_synthetic_image(path, return_details=True, thresholds=thresholds)
            
            # Store result
            results.append((path, classification, details['confidence_score']))
            
            # Count
            if classification == "Real":
                real_count += 1
                status = "Real"
            else:
                synthetic_count += 1
                status = "Fake" # Shorten for table
                
            # Metadata summary for table
            meta = details.get('metadata', {})
            meta_str = "No Data"
            if meta.get('software') != "Unknown":
                meta_str = f"Soft: {meta['software'][:15]}"
            elif meta.get('camera') != "Unknown":
                meta_str = f"Cam: {meta['camera'][:15]}"
            
            # Print row
            print(f"{fname_display:<25} | {status:<10} | {details['detection_score']:.2f}  | {details['kurtosis']:<5.1f} | {details['coeff_variation']:<5.2f} | {details['high_freq_ratio']:<5.2f} | {details['eigenvalue_ratio']:<5.1f} | {meta_str:<20}")
            
            if visualize:
                save_visualization(path, details)
                save_gradient_distribution_plot(path, details)
                save_side_by_side(path, details)
                
        except Exception as e:
            error_count += 1
            print(f"{fname_display:<25} | Error: {e}")

    print("-" * 115)
    print("📊 SUMMARY")
    print(f"   ✅ Real: {real_count}")
    print(f"   ⚠️  Synthetic: {synthetic_count}")
    if error_count:
        print(f"   ❌ Errors: {error_count}")
    print(f"   📷 Total: {len(image_files)}")
    if visualize:
        print(f"   📂 Visualizations saved to: visualization/")


def main():
    parser = argparse.ArgumentParser(
        description="Detect synthetic (AI-generated) images using gradient field PCA analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py image.jpg                  Analyze a single image
  python main.py --dir ./photos             Analyze all images in directory
  python main.py image.png --visualize      Generate analysis charts
  python main.py image.jpg --th-cv 2.0      Set custom Coeff of Variation threshold
        """
    )
    
    parser.add_argument(
        'image',
        nargs='?',
        help='Path to image file to analyze'
    )
    
    parser.add_argument(
        '--dir', '-d',
        metavar='DIRECTORY',
        help='Analyze all images in directory'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed analysis metrics'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output (just the result)'
    )
    
    parser.add_argument(
        '--visualize', '-vis',
        action='store_true',
        help='Generate analysis visualization images'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run self-test to verify installation'
    )

    # Threshold arguments
    parser.add_argument('--th-cv', type=float, help='Threshold for Coefficient of Variation (default: 1.8)')
    parser.add_argument('--th-kurt', type=float, help='Threshold for Kurtosis (default: 50.0)')
    parser.add_argument('--th-hf', type=float, help='Threshold for High-Freq Ratio (default: 0.45)')
    parser.add_argument('--th-ev-low', type=float, help='Threshold for Eigenvalue Ratio Low (default: 1.5)')
    parser.add_argument('--th-ev-high', type=float, help='Threshold for Eigenvalue Ratio High (default: 50.0)')
    
    args = parser.parse_args()
    
    # Print banner unless quiet mode
    if not args.quiet:
        print_banner()
    
    # Handle test mode
    if args.test:
        from detector import _self_test
        _self_test()
        return
    
    # Collect thresholds
    thresholds = {}
    if args.th_cv is not None: thresholds['cv'] = args.th_cv
    if args.th_kurt is not None: thresholds['kurtosis'] = args.th_kurt
    if args.th_hf is not None: thresholds['hf'] = args.th_hf
    if args.th_ev_low is not None: thresholds['ev_low'] = args.th_ev_low
    if args.th_ev_high is not None: thresholds['ev_high'] = args.th_ev_high
    
    if not thresholds:
        thresholds = None
    
    # Handle directory mode
    if args.dir:
        analyze_directory(args.dir, visualize=args.visualize, thresholds=thresholds)
        return
    
    # Handle single image
    if args.image:
        # Default to True for single image unless quiet is requested, 
        # OR if user explicitly passed arg (which is False by default)
        # We'll make it: if --visualize is NOT passed, check if we should auto-enable.
        # But commonly we just want it ON by default for single image.
        visualize = True if args.visualize or not args.quiet else False
        analyze_single(args.image, verbose=not args.quiet, visualize=visualize, thresholds=thresholds)
        return
    
    # No input provided
    parser.print_help()


if __name__ == "__main__":
    main()
