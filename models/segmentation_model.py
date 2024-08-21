import torch
import torchvision
import numpy as np
import cv2
import sys
import os
import sqlite3

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

#image_path = '/home/azureuser/iSOA/data/input_images/yolotest.png'

#output_path = '/home/azureuser/iSOA/data/segmented_objects/output_image4.png'

sam_checkpoint = '/home/azureuser/iSOA/SAM_weights/sam_vit_h_4b8939.pth'
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

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



def generate_masks(image_path, mask_dir, crop_dir, metadata):
    sys.path.append("..")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)


    connection = sqlite3.connect(metadata)
    cursor = connection.cursor()
    master_id = image_path.split('/')[-1]
    cursor.execute('''CREATE TABLE IF NOT EXISTS objects(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                master_id TEXT NOT NULL,
                object_id TEXT NOT NULL,
                mask_location TEXT NOT NULL,
                cropped_image TEXT NOT NULL,
                area INTEGER NOT NULL,
                x_min INTEGER NOT NULL,
                x_max INTEGER NOT NULL,
                y_min INTEGER NOT NULL,
                y_max INTEGER NOT NULL)''')


    os.makedirs(mask_dir, exist_ok=True)

    # creating sqlite table for storing metadata  such as area, bbox etc
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    master_id = image_path.split('/')[-1]  


    masks.sort(key = lambda x:x['area'], reverse = True)
    
    masks = masks[:25] 


    for i, mask_dict in enumerate(masks):
        
        mask_img = mask_dict['segmentation'].astype(np.uint8) * 255  
        bbox = mask_dict['bbox']
        area = mask_dict['area']

        img = cv2.imread(image_path)
        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
        cropped_img = img[y_min:y_max, x_min:x_max]

        crop_filename = os.path.join(crop_dir, f"crop{i+1}.png")
        cv2.imwrite(crop_filename, cropped_img)

        mask_filename = os.path.join(mask_dir, f"mask{i+1}.png")
        cv2.imwrite(mask_filename, mask_img)
        
        object_id = f"object{i+1}"
        cursor.execute('''INSERT INTO objects (master_id, object_id, mask_location, cropped_image, area, x_min, x_max, y_min, y_max) VALUES (?,?,?,?,?,?,?,?,?)'''
                    ,(master_id, object_id, mask_filename, crop_filename, area, x_min, x_max, y_min, y_max))

    connection.commit()
    connection.close()
