# TranscribeAI

TranscribeAI is an open source project that leverages Large Language Models (LLMs) to transcribe images from documents with exceptional accuracy. This project originated from a need to improve OCR transcription for the [Minnesota Digital Library](https://mndigital.org/) and was initially implemented using GPT-4o mini from our organization's instance to enhance accessibility for screen reader users and improve searchability of content that traditional OCR often fails to capture accurately. This current version uses the free alternative, Gemini Flash 2.0, offering a robust, cost-effective solution.

TranscribeAI can handle various document types—including typed text, handwritten text, tables, and mixed layouts—and it significantly outperforms traditional OCR solutions (e.g., Tesseract) in accuracy. One key factor behind its improved performance is the integration of contextual and metadata information about the document. By providing additional context, the model better interprets ambiguous or degraded text, distinguishes between similar characters, and generates more coherent, complete transcriptions.

## Technologies & Libraries

- **[python-dotenv](https://github.com/theskumar/python-dotenv)** – For managing environment variables.
- **[TQDM](https://github.com/tqdm/tqdm)** – Provides progress bars in the terminal.
- **[Pillow](https://pillow.readthedocs.io/en/stable/)** – A fork of the Python Imaging Library (PIL) for image processing.
- **[google-generativeai](https://github.com/google/generative-ai)** – Used for accessing the Gemini API.
- **[whisper](https://github.com/openai/whisper)** – For speech recognition.
- **[opencv-python](https://pypi.org/project/opencv-python/)** – For computer vision and image processing tasks.

## Project Structure

```plaintext
TranscribeAI/
├── GeminiImageTranscription/
│   ├── flash_process_local_dir.py       # Gemini transcription script
│   ├── OcrDocumentContext/              # Context files for documents
│   │   ├── ALL_DOCUMENT_CONTEXT.txt     # Global context for all images
│   │   ├── handwritten1_context.txt     # Context for a specific image (handwritten1.jpeg)
│   │   └── handwritten2_context.txt     # Context for a specific image (handwritten2.jpeg)
│   └── pending_image_paths_gemini.txt   # Pending file for transcription
├── OptimizedImagesForOCR/               # Output folder for processed images
├── PreprocessPostprocess/
│   ├── preprocess_to_jpeg.py            # Preprocessing script
│   └── optimized_image_path_for_ocr.txt  # Tracking file for preprocessing
├── transcribe_document_main.py          # Orchestration script
└── requirements.txt                     # Project dependencies
```

**Directory Descriptions:**

- **GeminiImageTranscription:** Contains the transcription script and supporting files. The `OcrDocumentContext` folder holds context files that inform the transcription prompt.
- **OptimizedImagesForOCR:** Stores images that have been preprocessed and are ready for transcription.
- **PreprocessPostprocess:** Contains the preprocessing script and its tracking file, which manages the list of images to process.
- **transcribe_document_main.py:** Orchestrates the full pipeline from preprocessing to transcription.

## Getting Started

Follow these steps to set up and run TranscribeAI on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/Minitex/TranscribeAI.git
cd TranscribeAI
```

### 2. Set Up the Virtual Environment

Create and activate a Python virtual environment:

```bash
python -m venv myenv
```

- On macOS/Linux:
  ```bash
  source myenv/bin/activate
  ```
- On Windows:
  ```bash
  myenv\Scripts\activate
  ```

### 3. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create an `.env` from `.env.example` and update it with your actual API key. For example, run:

```bash
cp .env.example .env
```
Then open the .env file and replace your-api-key with your real Google API key:

```bash
GOOGLE_API_KEY=your_actual_api_key_here
```

### 5. Prepare Your Image and Context Files

- **Source Images:**  
  Place your source images (handwritten or typed documents) in a folder (e.g., `/path/to/input_images`).

- **Tracking File (Preprocessing):**  
  The orchestration script automatically scans your input folder and populates the tracking file (`PreprocessPostprocess/optimized_image_path_for_ocr.txt`) with the full paths of the images. Running the orchestration script with the `--new` flag repopulates the tracking file.

- **Context Files (Optional):**  
  - To apply a **global context** to all images, place a file named `ALL_DOCUMENT_CONTEXT.txt` in the `GeminiImageTranscription/OcrDocumentContext` folder.
  - To provide **individual context** for specific images, create files named `<image_basename>_context.txt` (e.g., `handwritten1_context.txt`, `handwritten2_context.txt`) in the same folder.
  - When processing, the script logs which context is applied for each image.

### 6. Run the Full Pipeline

Run the orchestration script to execute the entire pipeline:

```bash
python3 transcribe_document_main.py /path/to/input_images --new
```

**Example Walkthrough:**

1. **Source Images:**  
   You have a folder `/Users/Minitex/Downloads/TestImages` containing 4 images.

2. **Run the Pipeline:**  
   Execute the following command:
   ```bash
   python3 transcribe_document_main.py /Users/Minitex/Downloads/TestImages --new
   ```
   - The script scans `/Users/Minitex/Downloads/TestImages` and populates the tracking file in `PreprocessPostprocess/optimized_image_path_for_ocr.txt` with the full paths of these images.
   - If a global context file exists in `GeminiImageTranscription/OcrDocumentContext/ALL_DOCUMENT_CONTEXT.txt`, its content is applied to all images.
   - If individual context files exist (e.g., `handwritten1_context.txt`), they are appended to the global context for the corresponding image.
   - Preprocessing converts the images and saves the outputs in the `OptimizedImagesForOCR` folder.
   - Processed image paths are then gathered into the pending file for transcription in `GeminiImageTranscription/pending_image_paths_gemini.txt`.
   - Finally, the Gemini transcription script is run, and upon successful transcription, all tracking files are removed.

### 7. Review the Output

- **Transcriptions:**  
  Processed transcriptions are saved as `.txt` files in the `OptimizedImagesForOCR` folder.
- **Logs:**  
  The terminal logs (via `tqdm.write`) provide details on which context (global, individual, or both) was applied to each image, along with any errors or retry information.

## Customization

- **Transcription Prompt:**  
  You can modify the prompt in the `compose_prompt` function within the orchestration script (`transcribe_document_main.py`) to suit your needs.

- **Retry Settings:**  
  Adjust the `MAX_RETRIES` and `RETRY_DELAY` constants in the scripts to customize the retry behavior.

- **Context Files:**
  - **Global Context:**  
    To apply the same context to every document, create a file named `ALL_DOCUMENT_CONTEXT.txt` in the `GeminiImageTranscription/OcrDocumentContext` folder. For example, if your documents share a common background—such as "a handwritten letter discussing family news and travel"—you can include that description in this file. The content of this file will be applied to all images during transcription.

  - **Individual Context:**  
    For documents that require unique context, create separate files for each image. Each file should be named using the image’s basename with `_context.txt` appended. For example:
    - If you have an image named `handwritten1.jpeg`, create a file named `handwritten1_context.txt` with context specific to that document (e.g., "a personal letter written by Nellie McCluer to Mrs. Osborne discussing Easter celebrations").
    - If you have an image named `invoice2022.png`, create a file named `invoice2022_context.txt` with context relevant to that invoice (e.g., "an invoice detailing services rendered in Q1 2022").

    When processing, TranscribeAI will first check for a global context file. If it exists, its content is applied to all images. Then, for each image, the script checks for an individual context file that matches the image’s basename. If one is found, its content is appended to the global context for that image—providing tailored transcription instructions for images that need additional context.


## License

MIT License

## Contributing

Contributions are welcome! Please open issues or submit pull requests with improvements or fixes.