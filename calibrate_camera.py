import numpy as np
import cv2

CHESSBOARD_SIZE = (3, 7)  

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

objpoints_left, objpoints_right = [], []  
imgpoints_left, imgpoints_right = [], []  

cap_left = cv2.VideoCapture(1)
if not cap_left.isOpened():
    print("Error opening left camera") 

cap_right = cv2.VideoCapture(3) 
if not cap_right.isOpened():
    print("Error opening right camera")

start_time = cv2.getTickCount()
while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    if not ret_left or not ret_right:
        print("Error reading from cameras during initial phase")
        continue
    
    cv2.imshow('Left Camera Feed', frame_left)
    cv2.imshow('Right Camera Feed', frame_right)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    current_time = cv2.getTickCount()
    if (current_time - start_time) / cv2.getTickFrequency() * 1000 > 15000:
        break

print("Starting stereo calibration...")

for i in range(100):  
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    if not ret_left or not ret_right:
        continue
    
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHESSBOARD_SIZE, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHESSBOARD_SIZE, None)

    if ret_left and ret_right:
        objpoints_left.append(objp)
        objpoints_right.append(objp)
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)
        cv2.drawChessboardCorners(frame_left, CHESSBOARD_SIZE, corners_left, ret_left)
        cv2.drawChessboardCorners(frame_right, CHESSBOARD_SIZE, corners_right, ret_right)
    
    cv2.imshow('Left Camera Feed', frame_left)
    cv2.imshow('Right Camera Feed', frame_right)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints_left, imgpoints_left, gray_left.shape[::-1], None, None)

ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints_right, imgpoints_right, gray_right.shape[::-1], None, None)

ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
    objpoints_left,  
    imgpoints_left,
    imgpoints_right,
    mtx_left,
    dist_left,
    mtx_right,
    dist_right,
    gray_left.shape[::-1],
    flags=cv2.CALIB_FIX_INTRINSIC
)

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

np.savez('calibration_data.npz',
         mtx_left=mtx_left, 
         dist_left=dist_left,
         mtx_right=mtx_right,
         dist_right=dist_right,
         R=R,
         T=T,
         E=E,
         F=F)
print("Calibration data saved.")
