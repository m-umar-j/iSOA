import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# Perform object detection
model = YOLO("/home/azureuser/iSOA/runs/detect/train/weights/best.pt")

results = model("/home/azureuser/iSOA/data/input_images/yolotest.png")

for i, result in enumerate(results):
    # Save and show the annotated image
    result.save('output3.jpg')
    result.show()

    # Extract bounding boxes, class IDs, and masks
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    masks = result.masks  # Assuming masks are available
    names = result.names

    # Print the names dictionary for reference
    print("Class names dictionary:", names)

    # Load the original image for overlay
    orig_image = cv2.imread("/home/azureuser/iSOA/data/input_images/yolotest.png")

    # Print bounding boxes, labels, and masks
    print(f"Bounding boxes shape: {boxes.shape}")
    for box, class_id in zip(boxes, class_ids):
        x_min, y_min, x_max, y_max = box
        label = names[int(class_id)]
        print(f"Bounding Box: [{x_min}, {y_min}, {x_max}, {y_max}] - Label: {label}")
        
        # Display masks if available
        if masks is not None:
            for mask in masks:
                # Assuming mask is a binary mask
                mask_img = mask.cpu().numpy().astype(np.uint8) * 255  # Convert mask to uint8 image
                mask_img = cv2.resize(mask_img, (orig_image.shape[1], orig_image.shape[0]))  # Resize to original image size
                masked_image = cv2.bitwise_and(orig_image, orig_image, mask=mask_img)  # Apply mask
                masked_image_pil = Image.fromarray(masked_image)

                # Save or display the masked image
                masked_image_pil.save(f"masked_object_{i}.png")
                masked_image_pil.show()
