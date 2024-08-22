from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# Load the processor and model
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

# importing sqlite3 to save captions for each image
import sqlite3

def generate_captions(metadata):
    connection = sqlite3.connect(metadata)
    cursor = connection.cursor()
    cursor.execute('''ALTER TABLE objects ADD COLUMN caption TEXT''')
    cursor.execute('''SELECT cropped_object FROM objects''')
    rows = cursor.fetchall()

    for row in rows:
        image_path = row[0]
        image = Image.open(image_path)
        
        inputs = processor(images=image, return_tensors="pt")

        # Generate a caption
        output = model.generate(**inputs, max_length = 50)
        # Decode the output

        caption = processor.decode(output[0], skip_special_tokens=True)
        cursor.execute(f'''UPDATE objects SET caption = ? WHERE cropped_object = ?'''
                    , (caption, image_path) )
        

    connection.commit()
    connection.close()
