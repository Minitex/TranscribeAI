#!/usr/bin/env python3
"""
Preprocess images for OCR.

This script reads image paths from a tracking file and processes them,
saving the results as JPEG files in the specified output folder.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

VALID_EXTS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')

def preprocess_image_for_ocr(input_path, output_path, blur_ksize=3, block_size=15, c=10, simple=False):
    """
    Process an image for OCR. Converts to grayscale and applies preprocessing.
    """
    image = cv2.imread(input_path)
    if image is None:
        tqdm.write(f"❌ Error: Unable to load image {input_path}")
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if simple:
        success = cv2.imwrite(output_path, gray)
    else:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        blurred = cv2.medianBlur(gray, blur_ksize)
        thresh = cv2.adaptiveThreshold(
            blurred,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=c
        )
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        success = cv2.imwrite(output_path, processed)

    if success:
        tqdm.write(f"✅ Saved: {output_path}")
    else:
        tqdm.write(f"❌ Error: Failed to save image to {output_path}")
    
    return success

def update_tracking_file(tracking_file, processed_image):
    """
    Remove a processed image's entry from the tracking file.
    """
    try:
        with open(tracking_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(tracking_file, "w", encoding="utf-8") as f:
            for line in lines:
                if line.strip() != processed_image:
                    f.write(line)
    except Exception as e:
        tqdm.write(f"⚠️ Warning: Could not update tracking file {tracking_file}. Error: {e}")

def process_images(output_folder, tracking_file, blur_ksize, block_size, c, simple):
    """
    Process images listed in the tracking file and save them in the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(tracking_file):
        tqdm.write("No images to process. Please populate the tracking file with image paths.")
        return

    with open(tracking_file, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    to_process = [p for p in paths if os.path.splitext(p)[1].lower() in VALID_EXTS]
    if not to_process:
        tqdm.write("No valid images found to process.")
        return

    progress_bar = tqdm(total=len(to_process), desc="Processing images", ncols=80)
    for image_path in to_process.copy():
        filename = os.path.basename(image_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}.jpeg")
        progress_bar.set_description(f"Processing: {filename}")
        preprocess_image_for_ocr(image_path, output_path, blur_ksize, block_size, c, simple)
        update_tracking_file(tracking_file, image_path)
        progress_bar.update(1)
    progress_bar.close()

def print_instructions():
    """Print usage instructions."""
    print(
        "\nUsage Instructions for preprocess_to_jpeg.py\n"
        "This script processes images listed in a tracking file and saves the processed images to the output folder.\n"
        "Populate the tracking file with one image path per line. Valid extensions are: "
        f"{', '.join(VALID_EXTS)}\n\n"
        "Required Argument:\n"
        "  output_folder  → Folder where processed JPEG images will be saved.\n"
        "Optional Arguments:\n"
        "  --tracking_file <file>  → Path to tracking file (default: PreprocessPostprocess/tracking_file.txt)\n"
        "  --simple                → Convert images to grayscale only\n"
        "  --blur <size>           → Median blur kernel size (default: 3, must be odd)\n"
        "  --blockSize <size>      → Adaptive thresholding block size (default: 15, must be odd and >1)\n"
        "  --C <value>             → Constant subtracted in adaptive thresholding (default: 10)\n\n"
        "Example:\n"
        "  python3 preprocess_to_jpeg.py /path/to/output_folder --tracking_file my_tracking.txt --simple\n"
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Preprocess images for OCR using a tracking file.",
        add_help=False
    )
    parser.add_argument("output_folder", nargs="?", help="Output folder for processed JPEG images.")
    parser.add_argument("--tracking_file", default="PreprocessPostprocess/tracking_file.txt",
                        help="Path to tracking file (default: PreprocessPostprocess/tracking_file.txt)")
    parser.add_argument("--blur", type=int, default=3,
                        help="Kernel size for median blur (default: 3, must be odd)")
    parser.add_argument("--blockSize", type=int, default=15,
                        help="Adaptive thresholding block size (default: 15, must be odd and >1)")
    parser.add_argument("--C", type=int, default=10,
                        help="Constant subtracted in adaptive thresholding (default: 10)")
    parser.add_argument("--simple", action="store_true",
                        help="Convert images to grayscale only")
    parser.add_argument("-h", "--help", action="store_true", help="Show usage instructions")
    
    args = parser.parse_args()
    if args.help or not args.output_folder:
        print_instructions()
        return

    process_images(args.output_folder, args.tracking_file, args.blur, args.blockSize, args.C, args.simple)

if __name__ == "__main__":
    main()