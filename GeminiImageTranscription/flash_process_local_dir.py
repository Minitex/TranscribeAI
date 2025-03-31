#!/usr/bin/env python3
"""
Transcribe images using Gemini API, with document context.

This script reads image paths from a pending file, processes each image
by sending it to the Gemini model for transcription, and saves the transcriptions.
It uses context from the OcrDocumentContext folder as follows:
  - If a file named "ALL_DOCUMENT_CONTEXT.txt" exists, its content is applied to all images.
  - Additionally, if an individual context file (named "<basename>_context.txt") exists
    for a given image, its content is appended to the global context.
If neither exists, no context is applied.
Processed images are removed from the pending file.
"""

import os
import sys
import time
from dotenv import load_dotenv
from tqdm import tqdm
import PIL.Image
import google.generativeai as genai
import logging

# Configuration constants
REVERSE_ORDER = True         # Only changes processing order, not pending file order
MAX_RETRIES = 3              # Number of retries before failing
RETRY_DELAY = 5              # Initial delay (in seconds) for exponential backoff

# Get script directory and load environment variables
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, "../API Keys/.env")
load_dotenv(ENV_PATH)

if not os.path.exists(ENV_PATH):
    tqdm.write(f"ERROR: .env file not found: {ENV_PATH}")
    sys.exit(1)

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    tqdm.write("ERROR: API key not found!")
    sys.exit(1)

# Suppress logs for Gemini and absl libraries
logging.getLogger("google").setLevel(logging.CRITICAL)
logging.getLogger("absl").setLevel(logging.CRITICAL)

genai.configure(api_key=API_KEY)

# Define file paths
PENDING_IMAGES_FILE = os.path.join(SCRIPT_DIR, "pending_image_paths_gemini.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "TranscriptionOutputGemini")
CONTEXT_FOLDER = os.path.join(SCRIPT_DIR, "OcrDocumentContext")

# Ensure necessary files and directories exist
if not os.path.exists(PENDING_IMAGES_FILE):
    with open(PENDING_IMAGES_FILE, "w", encoding="utf-8") as f:
        f.write("# List image paths here, one per line\n")
    tqdm.write(f"Pending images file not found. Created new file: {PENDING_IMAGES_FILE}")

if not os.path.exists(PENDING_IMAGES_FILE):
    tqdm.write(f"ERROR: Pending images file not found: {PENDING_IMAGES_FILE}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
tqdm.write(f"Output directory verified/created: {OUTPUT_DIR}")

# Read image paths from pending file (preserving order)
with open(PENDING_IMAGES_FILE, "r", encoding="utf-8") as f:
    image_paths = [line.strip() for line in f if line.strip()]

if not image_paths:
    tqdm.write("No images to process.")
    sys.exit(0)

processing_list = list(reversed(image_paths)) if REVERSE_ORDER else list(image_paths)

# -------------------------------
# Load Document Context (if available)
# -------------------------------
global_context = ""
context_mapping = {}  # Maps image basename to individual context text

if os.path.isdir(CONTEXT_FOLDER):
    # Check for a global context file named "ALL_DOCUMENT_CONTEXT.txt"
    global_context_path = os.path.join(CONTEXT_FOLDER, "ALL_DOCUMENT_CONTEXT.txt")
    if os.path.exists(global_context_path):
        with open(global_context_path, "r", encoding="utf-8") as gf:
            global_context = gf.read().strip()
        tqdm.write("Using global context from 'ALL_DOCUMENT_CONTEXT.txt'.")
    # Load individual context files (if any)
    individual_context_files = [
        f for f in os.listdir(CONTEXT_FOLDER)
        if f.lower().endswith("_context.txt") and f.lower() != "all_document_context.txt"
    ]
    for cf in individual_context_files:
        # Expected naming: <image_basename>_context.txt
        base = os.path.splitext(cf)[0].replace("_context", "")
        path = os.path.join(CONTEXT_FOLDER, cf)
        with open(path, "r", encoding="utf-8") as file:
            context_mapping[base] = file.read().strip()
    if individual_context_files:
        tqdm.write(f"Loaded individual context for {len(context_mapping)} images.")
    if not global_context and not individual_context_files:
        tqdm.write("No context files found in OcrDocumentContext; proceeding without context.")
else:
    tqdm.write("Context folder 'OcrDocumentContext' does not exist; proceeding without context.")

def compose_prompt(image_basename):
    """
    Compose the transcription prompt with context if available.
    
    If a global context exists, it is applied to all images.
    Additionally, if an individual context exists for the image (matching its basename),
    it is appended to the global context.
    """
    context_text = ""
    if global_context:
        context_text = global_context
    if image_basename in context_mapping:
        # Append individual context if available, separated by a newline.
        context_text = f"{context_text}\n{context_mapping[image_basename]}" if context_text else context_mapping[image_basename]
    
    base_prompt = (
        "Please transcribe the text from the uploaded image. "
        "Ensure that the transcription follows correct English spelling, grammar, and sentence structure. "
        "If a word is completely missing, mark it as [blank]. "
        "If a word is unclear or unreadable, make an educated guess based on context rather than providing gibberish, "
        "and mark it as [unsure]."
    )
    if context_text:
        return f"Given the context: {context_text}\n{base_prompt}"
    return base_prompt

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')
total_images = len(image_paths)

# Process images with a progress bar
with tqdm(total=total_images, desc="Processing Images", unit="file", ncols=80) as pbar:
    for image_path in processing_list:
        if not os.path.exists(image_path):
            tqdm.write(f"WARNING: Image file not found: {image_path}")
            # Remove missing files from pending file
            with open(PENDING_IMAGES_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(PENDING_IMAGES_FILE, "w", encoding="utf-8") as f:
                for line in lines:
                    if line.strip() != image_path:
                        f.write(line)
            pbar.update(1)
            continue

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
        pbar.set_description(f"Processing: {os.path.basename(image_path)}")

        prompt = compose_prompt(base_name)

        success = False
        for attempt in range(MAX_RETRIES):
            try:
                img = PIL.Image.open(image_path)
                response = model.generate_content([prompt, img], stream=True)
                if not response:
                    raise ValueError("No response received from API")
                with open(output_path, "w", encoding="utf-8") as out_file:
                    for chunk in response:
                        out_file.write(chunk.text)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    tqdm.write(f"✅ Transcription saved: {output_path}")
                    success = True
                    break
                else:
                    raise ValueError("Transcription file is empty or not created")
            except Exception as e:
                tqdm.write(f"❌ ERROR processing {image_path} (Attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (3 ** attempt)
                    tqdm.write(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    tqdm.write(f"🚨 FAILED after {MAX_RETRIES} attempts: {image_path} will remain in pending list.")
                    success = False
                    break
        if success:
            with open(PENDING_IMAGES_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(PENDING_IMAGES_FILE, "w", encoding="utf-8") as f:
                for line in lines:
                    if line.strip() != image_path:
                        f.write(line)
        pbar.update(1)

tqdm.write("Processing complete. Check logs for any errors.")