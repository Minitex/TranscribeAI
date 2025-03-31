#!/usr/bin/env python3
"""
Orchestrate the preprocessing and transcription of document images.

This script:
  1. Optionally removes existing tracking files if the --new flag is set.
  2. Populates a tracking file with the full paths of images from the input folder,
     unless the tracking file already indicates processing is complete.
  3. Calls the preprocessing script to generate processed images.
  4. Gathers processed image paths into a Gemini pending file.
  5. Runs the Gemini transcription script.
  6. Removes tracking files once transcription completes.
"""

import os
import subprocess
import argparse
import sys

def remove_tracking_files(base_dir):
    """Remove tracking files used for preprocessing and transcription."""
    tracking_file = os.path.join(base_dir, "PreprocessPostprocess", "optimized_image_path_for_ocr.txt")
    if os.path.exists(tracking_file):
        os.remove(tracking_file)
        print(f"Removed tracking file: {tracking_file}")

    gemini_dir = os.path.join(base_dir, "GeminiImageTranscription")
    pending_file = os.path.join(gemini_dir, "pending_image_paths_gemini.txt")
    if os.path.exists(pending_file):
        os.remove(pending_file)
        print(f"Removed pending Gemini file: {pending_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate preprocessing and transcription of document images."
    )
    parser.add_argument("input_path", help="Path to the folder containing original images to preprocess.")
    parser.add_argument("--new", action="store_true", help="Remove tracking files to start processing fresh.")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.dirname(__file__))

    if args.new:
        remove_tracking_files(base_dir)

    # Create output directory if it doesn't exist.
    output_dir = os.path.join(base_dir, "OptimizedImagesForOCR")
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------
    # Populate the tracking file with full image paths.
    # -------------------------------
    preprocess_dir = os.path.join(base_dir, "PreprocessPostprocess")
    os.makedirs(preprocess_dir, exist_ok=True)
    tracking_file = os.path.join(preprocess_dir, "optimized_image_path_for_ocr.txt")
    
    valid_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    repopulate = True

    if os.path.exists(tracking_file):
        # Check the first non-comment line.
        with open(tracking_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    if stripped == "$$done$$":
                        print("Tracking file indicates processing is complete. Skipping repopulation.")
                        repopulate = False
                    break

    if repopulate:
        image_paths = []
        for file in os.listdir(args.input_path):
            if file.lower().endswith(valid_exts):
                abs_path = os.path.abspath(os.path.join(args.input_path, file))
                image_paths.append(abs_path)
        with open(tracking_file, "w", encoding="utf-8") as f:
            for path in image_paths:
                f.write(path + "\n")
            f.write("$$done$$\n")
        print(f"Written {len(image_paths)} image paths to tracking file: {tracking_file}")

    # -------------------------------
    # Run the preprocessing script.
    # -------------------------------
    with open(tracking_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                break
    if stripped == "$$done$$":
        print("Preprocessing already complete based on tracking file marker. Skipping preprocessing step.")
    else:
        preprocess_script = os.path.join(base_dir, "PreprocessPostprocess", "preprocess_to_jpeg.py")
        # The new preprocess script expects --tracking_file before the output folder.
        preprocess_command = [
            sys.executable,
            preprocess_script,
            "--tracking_file", tracking_file,
            output_dir,
            "--simple"
        ]
        print("Starting image preprocessing in simple mode...")
        try:
            subprocess.run(preprocess_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during preprocessing: {e}")
            sys.exit(1)

    # -------------------------------
    # Gather processed image paths.
    # -------------------------------
    processed_images = []
    for file in os.listdir(output_dir):
        if file.lower().endswith(".jpeg"):
            abs_path = os.path.abspath(os.path.join(output_dir, file))
            processed_images.append(abs_path)
    if not processed_images:
        print("No processed images found. Exiting.")
        sys.exit(0)

    gemini_dir = os.path.join(base_dir, "GeminiImageTranscription")
    os.makedirs(gemini_dir, exist_ok=True)
    pending_file = os.path.join(gemini_dir, "pending_image_paths_gemini.txt")
    if args.new or not os.path.exists(pending_file):
        with open(pending_file, "w", encoding="utf-8") as f:
            for path in processed_images:
                f.write(path + "\n")
        print(f"Written {len(processed_images)} image paths to {pending_file}")
    else:
        print(f"Pending file {pending_file} already exists. Skipping generation.")

    # -------------------------------
    # Run the Gemini transcription script.
    # -------------------------------
    flash_script = os.path.join(base_dir, "GeminiImageTranscription", "flash_process_local_dir.py")
    if not os.path.exists(flash_script):
        print(f"ERROR: Transcription script not found at {flash_script}")
        sys.exit(1)
    flash_command = [
        sys.executable,
        flash_script
    ]
    print("Starting Gemini transcription process...")
    try:
        subprocess.run(flash_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during transcription process: {e}")
        sys.exit(1)

    print("🎉 Document transcription complete.")
    remove_tracking_files(base_dir)
    print("All tracking files removed.")

if __name__ == "__main__":
    main()