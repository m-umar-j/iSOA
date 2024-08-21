import easyocr
import cv2
import numpy as np
import sqlite3
import os
reader = easyocr.Reader(['en', 'hi'])

#image_path = "/home/azureuser/iSOA/data/input_images/hindibanner.jpg"

def detect_text(metadata):
    connection = sqlite3.connect(metadata)
    cursor = connection.cursor()
    cursor.execute('''ALTER TABLE objects ADD COLUMN text_from_image TEXT''')
    cursor.execute('''SELECT cropped_object from objects''')
    rows = cursor.fetchall()
    for row in rows:
        image_path = row[0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = reader.readtext(image, detail = 0)
        cursor.execute('''UPDATE objects SET text_from_image = ? WHERE cropped_image = ? ''',
                       (result, image_path))
