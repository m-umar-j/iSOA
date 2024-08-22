from caption import generate_captions
from object_detection import detect_objects
from ocr import detect_text
from segmentation_model import generate_masks

import sys
import os
import gradio as gr
from PIL import Image
import sqlite3

sys.path.append("..")

def save_image(image: Image.Image, save_path: str):
    
    image.save(save_path)



def extract_table(metadata_db_path: str) -> str:
    connection = sqlite3.connect(metadata_db_path)
    cursor = connection.cursor()
    
    cursor.execute("SELECT * FROM objects")
    rows = cursor.fetchall()
    
    
    column_names = [description[0] for description in cursor.description]
    
    
    table_str = "\t".join(column_names) + "\n"
    table_str += "\n".join(["\t".join(map(str, row)) for row in rows])
    
    connection.close()
    return table_str

def process_image(image:Image.Image):
    
    image_path = "/home/azureuser/iSOA/data/input_images/uploaded_image.png"
    mask_dir = "/home/azureuser/iSOA/data/segmented_objects"
    crop_dir = "/home/azureuser/iSOA/data/cropped_objects"
    metadata = "/home/azureuser/iSOA/data/output/metadata.db"
    masked_image_output = "/home/azureuser/iSOA/data/output/masked_image.png"


    save_image(image, image_path)
    generate_masks(image_path, mask_dir, crop_dir, metadata)
    
    #labelling cropped images
    detect_objects(metadata)

    # I have used BLIP to generate captions
    generate_captions(metadata)

    detect_text(metadata)
    # Load and return the cropped images and metadata
    cropped_images = [Image.open(os.path.join(crop_dir, f"crop{i+1}.png")) for i in range(25)]
    masked_image = Image.open(masked_image_output)
    metadata_table = extract_table(metadata)

    return masked_image, cropped_images, metadata_table

iface = gr.Interface(
    fn=process_image,  
    
    inputs=gr.Image(type="pil"),  
    outputs=[gr.Image(label="Masked Image"), gr.Gallery(label="Cropped Images"), gr.Textbox(label="Metadata")], 
    live=True  
)

iface.launch()



