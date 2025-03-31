#!/usr/bin/env python3
"""
Detect and remove extraneous phrases from files in a given directory.

This script uses a SentenceTransformer model to compute the semantic similarity
between the first and last lines of each file and defined key phrases.
If a line's similarity exceeds a given threshold, the line is either flagged
or removed. Flagged files are reported for further review.

Usage:
    python3 detect_extraneous_phrases.py <directory_path>
"""

import os
import sys
import time
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Load a pre-trained SBERT model optimized for short sentence similarity
MODEL = SentenceTransformer('sentence-transformers/nli-roberta-base-v2')

# Define key phrases and thresholds
START_PHRASE = "Here is the transcribed text from the image"
END_PHRASE = "Let me know if you need any modifications or formatting adjustments"

SIMILARITY_THRESHOLD = 0.5  # Flag for review if similarity exceeds this value
REMOVAL_THRESHOLD = 0.7     # Remove the line if similarity exceeds this value

def compute_similarity(text: str, reference_text: str) -> float:
    """
    Compute semantic similarity between two texts using SBERT.

    Args:
        text (str): The text to compare.
        reference_text (str): The reference text.

    Returns:
        float: The cosine similarity score.
    """
    embeddings = MODEL.encode([text, reference_text], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity_score

def process_files_in_directory(directory: str) -> list:
    """
    Process files in the specified directory by detecting and removing extraneous phrases.

    For each file, the first and last lines are compared against predefined key phrases.
    Lines exceeding the removal threshold are removed. Files with similarity scores above
    the review threshold are flagged for further review.

    Args:
        directory (str): The path to the directory containing the files.

    Returns:
        list: A list of dictionaries containing details of flagged files.
    """
    flagged_files = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue

        try:
            with open(filepath, 'rb') as file:
                content = file.read().decode('utf-8', errors='ignore')
            lines = content.splitlines()

            if not lines:
                continue

            first_line = lines[0].strip()
            last_line = lines[-1].strip()

            # Compute similarity scores for the first and last lines
            first_line_similarity = compute_similarity(first_line, START_PHRASE)
            last_line_similarity = compute_similarity(last_line, END_PHRASE)

            remove_first = first_line_similarity > REMOVAL_THRESHOLD
            remove_last = last_line_similarity > REMOVAL_THRESHOLD

            # Flag file if either score exceeds the review threshold
            if (first_line_similarity > SIMILARITY_THRESHOLD or 
                last_line_similarity > SIMILARITY_THRESHOLD):
                file_info = {"filename": filename}
                if first_line_similarity > SIMILARITY_THRESHOLD:
                    file_info["first_line"] = first_line
                    file_info["first_line_number"] = 1
                    file_info["first_line_similarity"] = round(first_line_similarity, 2)
                if last_line_similarity > SIMILARITY_THRESHOLD:
                    file_info["last_line"] = last_line
                    file_info["last_line_number"] = len(lines)
                    file_info["last_line_similarity"] = round(last_line_similarity, 2)
                flagged_files.append(file_info)

            # Remove lines if they exceed the removal threshold
            if remove_first:
                tqdm.write(f"Removing first line from {filename} (Similarity: {round(first_line_similarity, 2)})")
                lines = lines[1:]
            if remove_last:
                tqdm.write(f"Removing last line from {filename} (Similarity: {round(last_line_similarity, 2)})")
                lines = lines[:-1]

            # Save the cleaned content back to the file
            with open(filepath, "w", encoding="utf-8") as file:
                file.write("\n".join(lines) + "\n")

        except Exception as e:
            tqdm.write(f"Error processing {filename}: {e}")

    return flagged_files

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 detect_extraneous_phrases.py <directory_path>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.exists(directory):
        print("Error: The provided directory does not exist.")
        sys.exit(1)

    flagged_files = process_files_in_directory(directory)

    if flagged_files:
        print("\nFiles containing potential extraneous phrases (before removal):")
        for file_info in flagged_files:
            print(f"\nFilename: {file_info['filename']}")
            if "first_line" in file_info:
                print(f" - First Line (Line {file_info['first_line_number']}): {file_info['first_line']}")
                print(f"   -> Similarity Score: {file_info['first_line_similarity']}")
            if "last_line" in file_info:
                print(f" - Last Line (Line {file_info['last_line_number']}): {file_info['last_line']}")
                print(f"   -> Similarity Score: {file_info['last_line_similarity']}")
    else:
        print("\nNo files contained the specified phrases.")

if __name__ == "__main__":
    main()