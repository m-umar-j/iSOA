# iSOA

## Image Segmentation and Object Analysis Pipeline

## Overview
This project provides an end-to-end pipeline for image segmentation, object detection, text recognition, and caption generation using various models and tools. The pipeline includes modules for segmentation, detection, OCR, and captioning, and integrates with a Gradio interface for user interaction.

## Files

### `segmentation.py`
- Loads the SAM model and generates masks for input images.
- Saves masks and cropped images to specified directories.
- Stores metadata in an SQLite database.

### `object_detection.py`
- Loads a YOLO model for object detection.
- Updates the SQLite database with detected object labels.

### `caption.py`
- Uses the BLIP model to generate captions for cropped images.
- Updates the SQLite database with generated captions.

### `ocr.py`
- Placeholder for OCR functionality (implementation not provided).

### `main.py`
- Integrates all functionalities: image upload, mask generation, object detection, captioning, and text detection.
- Uses Gradio to create a web interface for uploading images and viewing results.

## Setup

1. **Install Dependencies**
   ```bash
   pip install torch torchvision opencv-python-headless numpy gradio transformers Pillow sqlite3 ultralytics segment-anything
