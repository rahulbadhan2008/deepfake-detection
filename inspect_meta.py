from PIL import Image, ExifTags
import sys
import os

def inspect(path):
    print(f"\n--- Inspecting: {os.path.basename(path)} ---")
    try:
        img = Image.open(path)
        print(f"Format: {img.format}")
        print(f"Mode: {img.mode}")
        
        # 1. info dictionary
        if hasattr(img, 'info'):
            print("image.info keys:", list(img.info.keys()))
            for k, v in img.info.items():
                if isinstance(v, (str, int, float)):
                    print(f"  info[{k}]: {v}")
                else:
                    print(f"  info[{k}]: <{type(v).__name__}> (Length: {len(v) if hasattr(v, '__len__') else 'N/A'})")
        
        # 2. getexif()
        exif = img.getexif()
        if exif:
            print("getexif() found:")
            for k, v in exif.items():
                tag = ExifTags.TAGS.get(k, k)
                print(f"  Exif[{tag}]: {v}")
        else:
            print("getexif() returned None/Empty")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        inspect(arg)
