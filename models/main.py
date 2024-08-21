from caption import generate_captions
from object_detection import detect_objects
from ocr import detect_text
from segmentation_model import generate_masks

import sqlite3
import torch
import torchvision
import numpy as np
import cv2
import sys
import os
from ultralytics import YOLO

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import gradio as gr
from PIL import Image
import io


sam_checkpoint = '//home/azureuser/iSOA/SAM_weights/sam_vit_h_4b8939.pth'
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.append("..")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.96,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=1000,
)
def save_image(image: Image.Image, save_path: str):
    # Save the PIL image to a file
    image.save(save_path)
    return save_path


def process_image(image:Image.Image):
    #image_np = np.array(image)

    # Define paths and other configurations
    image_path = "/home/azureuser/iSOA/data/input_images/uploaded_image.png"
    mask_dir = "/home/azureuser/iSOA/data/segmented_objects"
    crop_dir = "/home/azureuser/iSOA/data/cropped_objects"
    metadata = "/home/azureuser/iSOA/data/output/metadata.db"

    saved_image_path = save_image(image, image_path)
    generate_masks(saved_image_path, mask_dir, crop_dir, metadata)
    generate_masks(image, mask_dir, crop_dir, metadata)
    model = YOLO("/home/azureuser/iSOA/runs/detect/train/weights/best.pt")
    detect_objects()

    # Generate captions
    generate_captions(metadata)

    # Detect text
    detect_text(metadata)
    # I have used BLIP to generate captions
    generate_captions(metadata)

    detect_text(metadata)
    # Load and return the cropped images and metadata
    cropped_images = [Image.open(os.path.join(crop_dir, f"crop{i+1}.png")) for i in range(25)]

    return cropped_images, metadata

iface = gr.Interface(
    fn=process_image,  # Function to call
    inputs=gr.Image(type="pil"),  # Input type
    outputs=[gr.Gallery(label="Cropped Images"), gr.Textbox(label="Metadata")],  # Output types
    live=True  # Optional: update output in real-time
)

iface.launch()



