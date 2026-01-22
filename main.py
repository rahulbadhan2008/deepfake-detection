#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Image Detection CLI

Command-line interface for detecting AI-generated images using gradient field analysis.
"""

import argparse
import sys
import os
from detector import detect_synthetic_image, batch_detect


def print_banner():
    """Display ASCII banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║       🔍 Synthetic Image Detection using Gradient Fields      ║
║                    Luminance-Gradient PCA Analysis             ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def format_result(classification: str, details: dict = None) -> str:
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


def analyze_single(image_path: str, verbose: bool = True):
    """Analyze a single image."""
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        sys.exit(1)
    
    if verbose:
        print(f"\n📷 Analyzing: {image_path}")
        print("-" * 50)
    
    try:
        classification, details = detect_synthetic_image(image_path, return_details=True)
        
        print(f"\n🎯 Result: {format_result(classification, details)}")
        
        if verbose:
            print(f"\n📊 Detailed Metrics:")
            print(f"   • Coefficient of Variation: {details['coeff_variation']:.4f}")
            print(f"   • Kurtosis: {details['kurtosis']:.4f}")
            print(f"   • High-Freq Ratio: {details['high_freq_ratio']:.4f}")
            print(f"   • Eigenvalue Ratio: {details['eigenvalue_ratio']:.4f}")
            print(f"   • Total Variance: {details['total_variance']:.6f}")
            print(f"   • Image Size: {details['image_dimensions'][0]}x{details['image_dimensions'][1]}")
        
        return classification, details
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        sys.exit(1)


def analyze_directory(dir_path: str, extensions: list = None):
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
    print("-" * 60)
    
    results = batch_detect(image_files, verbose=True)
    
    # Summary
    real_count = sum(1 for _, cls, _ in results if cls == "Real")
    synthetic_count = sum(1 for _, cls, _ in results if cls == "Diffusion Generated")
    error_count = sum(1 for _, cls, _ in results if cls == "Error")
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"   ✅ Real: {real_count}")
    print(f"   ⚠️  Synthetic: {synthetic_count}")
    if error_count:
        print(f"   ❌ Errors: {error_count}")
    print(f"   📷 Total: {len(image_files)}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect synthetic (AI-generated) images using gradient field PCA analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py image.jpg                  Analyze a single image
  python main.py --dir ./photos             Analyze all images in directory
  python main.py image.png --verbose        Show detailed metrics
  python main.py --test                     Run self-test
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
        '--test',
        action='store_true',
        help='Run self-test to verify installation'
    )
    
    args = parser.parse_args()
    
    # Print banner unless quiet mode
    if not args.quiet:
        print_banner()
    
    # Handle test mode
    if args.test:
        from detector import _self_test
        _self_test()
        return
    
    # Handle directory mode
    if args.dir:
        analyze_directory(args.dir)
        return
    
    # Handle single image
    if args.image:
        analyze_single(args.image, verbose=not args.quiet)
        return
    
    # No input provided
    parser.print_help()


if __name__ == "__main__":
    main()
