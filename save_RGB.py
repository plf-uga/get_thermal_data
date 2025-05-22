import os
import cv2
import time
from datetime import datetime

def run_RGB():

    # Open the USB camera
    cap = cv2.VideoCapture(0)

    # Change working directory
    os.chdir("/home/alveslab/RGB_Imgs")

    width = 300
    height = 200
    interval = 1
    duration = 30  # seconds

    start_time = time.time()
    camera_name = "camera1"

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    for i in range (1):
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            continue  # Skip the rest of the loop if no frame is captured

        frame = cv2.resize(frame, (width, height))

        current_timestamp = time.time() - 2
        current_time = time.strftime('%H%M%S', time.localtime(current_timestamp))

        output_folder = os.path.join(os.getcwd(), f"{camera_name}_{datetime.now().strftime('%Y_%m_%d')}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created directory: {output_folder}")

        frame_path = os.path.join(output_folder, f"{camera_name}_{current_time}.jpg")
        success = cv2.imwrite(frame_path, frame)
        if not success:
            print("Error: Frame could not be saved at", frame_path)
        else:
            print("Saved frame to:", frame_path)

        

    # Release the capture
    cap.release()

