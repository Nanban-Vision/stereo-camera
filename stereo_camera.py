import os
from ultralytics import YOLO
import numpy as np
import math
import cv2

image_filename = os.path.join(os.path.expanduser("~/Documents/Blind_Linux/"), "captured_image.jpg")
model = YOLO('models/yolov8m-seg.pt')

data = np.load('calibration_data.npz')
mtx_left = data['mtx_left']
dist_left = data['dist_left']
mtx_right = data['mtx_right']
dist_right = data['dist_right']
R = data['R']
T = data['T']

def calculate_distance(frame_left, frame_right, x_min, y_min, x_max, y_max):
    frame_left_undistorted = cv2.undistort(frame_left, mtx_left, dist_left)
    frame_right_undistorted = cv2.undistort(frame_right, mtx_right, dist_right)

    gray_left = cv2.cvtColor(frame_left_undistorted, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right_undistorted, cv2.COLOR_BGR2GRAY)

    roi_left = gray_left[int(y_min):int(y_max), int(x_min):int(x_max)]
    roi_right = gray_right[int(y_min):int(y_max), int(x_min):int(x_max)]

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx_left, dist_left, 
        mtx_right, dist_right, 
        gray_left.shape[::-1], R, T
    )

    map1_left, map2_left = cv2.initUndistortRectifyMap(
        mtx_left, dist_left, R1, P1, gray_left.shape[::-1], cv2.CV_32FC1
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        mtx_right, dist_right, R2, P2, gray_right.shape[::-1], cv2.CV_32FC1
    )

    rect_left = cv2.remap(roi_left, map1_left, map2_left, cv2.INTER_LINEAR)
    rect_right = cv2.remap(roi_right, map1_right, map2_right, cv2.INTER_LINEAR)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*5,
        blockSize=15,
        P1=8 * 3 * 15**2,
        P2=32 * 3 * 15**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=1
    )

    disparity = stereo.compute(rect_left, rect_right)

    focal_length = P1[0, 0]  
    baseline = np.linalg.norm(T)

    disparity = np.where(disparity == 0, 0.001, disparity)
    depth = (focal_length * baseline) / disparity.mean()

    return depth

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
    left_camera = cv2.VideoCapture(1)   
    right_camera = cv2.VideoCapture(3)  

    try:
        while True:
            ret_left, frame_left = left_camera.read()
            ret_right, frame_right = right_camera.read()

            if not ret_left or not ret_right:
                print("Error capturing frames")
                break

            frame_height, frame_width, _ = frame_left.shape

            results = model(frame_left)

            for result in results:
                for box in result.boxes:
                    x_min, y_min, x_max, y_max = box.xyxy[0]
                    
                    name = model.names[int(box.cls)]
                    
                    distance = calculate_distance(frame_left, frame_right, x_min, y_min, x_max, y_max)
                    location = check_location(x_min, y_min, x_max, y_max, frame_width, frame_height)
                    
                    if distance is not None:
                        print(f"Detected {name} at {distance:.2f} cm, located on the {location}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Scanning stopped by user")
    finally:
        left_camera.release()
        right_camera.release()
        cv2.destroyAllWindows()

scan_mode()
