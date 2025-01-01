from transformers import pipeline
import os
from ultralytics import YOLO
from gtts import gTTS
import math
import yaml
import cv2
from PIL import Image

image_filename = "captured_image.jpg"
model = YOLO('models/yolov8m-seg.pt')
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

def convert_opencv_to_pil(opencv_image):
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    return pil_image

def check_location(x_min, y_min, x_max, y_max, frame_width, frame_height):
    object_center_x = (x_min + x_max) // 2
    object_center_y = (y_min + y_max) // 2
    
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    
    margin_of_error = 10
    
    if object_center_x < frame_center_x - margin_of_error:
        return "left"
    elif object_center_x > frame_center_x + margin_of_error:
        return "right"
    else:
        return "front"

def scan_mode():
    try:
        cap = cv2.VideoCapture(1)
        ret, frame = cap.read()
        cv2.imwrite(image_filename, frame)
        frame = cv2.imread(image_filename)

        if frame is None:
            print("Error: Unable to load image.")
            return

        results = model(frame)
        names = model.names
        frame_height, frame_width, _ = frame.shape
    
        for result in results:
            for box in result.boxes:
                coordinates = box.xyxy
                name = names[int(box.cls)]
                x_min, y_min, x_max, y_max = coordinates[0][0].item(),coordinates[0][1].item(),coordinates[0][2].item(),coordinates[0][3].item()
                position = check_location(int(x_min), int(y_min), int(x_max), int(y_max), frame_width, frame_height)
                image = convert_opencv_to_pil(frame[int(y_min):int(y_max), int(x_min):int(x_max)])
                distance = pipe(image)["depth"]
                print(distance)
                message = f"There is a {name}, around {distance:.2f} centimeters away and it is in your {position}."
                #speak(message)
                print(message)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

scan_mode()
