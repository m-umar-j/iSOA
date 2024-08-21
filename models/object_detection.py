from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("/home/azureuser/iSOA/runs/detect/train/weights/best.pt")
#model.train(data='coco128.yaml', epochs=3)  

import sqlite3
def detect_objects(metadata):
    connection = sqlite3.connect(metadata)
    cursor = connection.cursor()

    cursor.execute('''SELECT cropped_image FROM objects''')
    rows = cursor.fetchall()

    #cursor.execute('''ALTER TABLE objects ADD COLUMN label TEXT''')

    for row in rows:
        # Load image
        image_path = row[0]
        #print(row)
        image = cv2.imread(image_path)

        # Perform detection
        results = model(image)


        for result in results:
            # Display bounding boxes, class names, and confidence scores
            result.save('output.jpg')
            boxes = result.boxes.xyxy.cpu().numpy()  # Convert to numpy array
            class_ids = result.boxes.cls.cpu().numpy()
            names = result.names

            for  class_id in  class_ids:
                
                label = names[int(class_id)]  # Get the class name
                cursor.execute('''UPDATE objects SET label = ? WHERE cropped_image = ?'''
                            ,(label, image_path))
                print(label)
    connection.commit()

    # Close the connection
    connection.close()