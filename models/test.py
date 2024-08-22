from caption import generate_captions
from object_detection import detect_objects
from ocr import detect_text
from segmentation_model import generate_masks



image_path = '/home/azureuser/iSOA/data/input_images/yolotest.png'
mask_dir = "/home/azureuser/iSOA/data/segmented_objects"
crop_dir = "/home/azureuser/iSOA/data/cropped_objects"
metadata = "/home/azureuser/iSOA/data/output/metadata.db"  # change the path accordingly


generate_masks(image_path, mask_dir, crop_dir, metadata)

detect_objects(metadata)

# I have used BLIP to generate captions
generate_captions(metadata)

detect_text(metadata)