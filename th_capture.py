#!/usr/bin/env python3
import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
import configparser

##########################################################################
#----------------------- CONFIG / GLOBALS --------------------------------#
##########################################################################

BUF_SIZE = 2
W = 256
H = 392
last_log_time = 0

def write_log(info, message, verbose=1):
    """Log message to file and optionally print to console."""
    current_time = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())
    cout = f'{info} {current_time} : {message}\n'
    with open('thermal_outputs.log', 'a') as file:
        if verbose:
            print(cout.strip())
        file.write(cout)

def load_config():
    """Load parameters from thermal.par."""
    config = configparser.ConfigParser()
    config.read('thermal.par')
    cam_name = config.get('Camera', 'cam_name')
    h = config.getint('Camera', 'img_height')
    w = config.getint('Camera', 'img_width')
    temp_max = config.getint('Temperature', 'temp_max')
    temp_min = config.getint('Temperature', 'temp_min')
    thresh = config.getfloat('Temperature', 'temp_trigger')  
    duration = config.getint('Time', 'duration')
    output = config.get('Output', 'folder')
    display = config.get('Display', 'status')
    return cam_name, h, w, temp_max, temp_min, thresh, duration, output, display

##########################################################################
#----------------------- TC001 FUNCTIONS ---------------------------------#
##########################################################################

def get_tc001_camera(device_path):
    """Open TC001 camera and configure."""
    cap = cv2.VideoCapture(device_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Y16 '))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, False)
    ret, frame = cap.read()
    if ret:
        try:
            # just test split
            thdata, imdata = np.array_split(frame, 2, axis=0)
            write_log("INFO", f"TC001 found at {device_path}")
            return cap
        except Exception as e:
            write_log("ERROR", f"Error extracting thermal data: {e}")
            cap.release()
    else:
        write_log("ERROR", "Failed to capture from TC001")
        cap.release()
    raise RuntimeError(f"TC001 camera not found at {device_path}")

def get_tc001_index():
    """
    Try to open the RGB camera with OpenCV by checking the number of channels (expected 2 channels for 16 bit gray image).
    """
    max_cameras = 5  # You can adjust this depending on how many cameras you expect to connect
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Y16 '))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, False)

        try:
            
            ret, frame = cap.read()
            thdata, imdata = np.array_split(frame, 2)
            
            if thdata.shape[2] == 2:
                print(f"TC001 camera found at index {index}")
                cap.release()                
                i = index
                break
                
             
                
        except:
            write_log("ERROR", f"Failed to capture frame at index {index}")
            cap.release()
            continue
    return i

def extract_thermal(frame):

    """Extract raw 16 bit frame"""
    
    thdata, imdata = np.array_split(frame,2)
    lsb = thdata[:,:,0] 
    hsb = thdata[:,:,1].astype(np.uint16) <<8  
    
    return lsb + hsb


def raw_to_celsius(raw):
    """
    Since no calibration data is provided by the vendors. 
    These regression parameters were obtained using a linear regression
    temperature in Celsius in reference objects was measured with IR handheld digital thermometer
    Tests were made at a distance of 0.5 m and room temperature of 25C
    These are just rough values for reference, and should be updated in real conditions

    """

    #Define regression parameters
    a = 0.0251859495852    
    b = -100
    
    return  a * raw + b

def get_heatmap(temp_c, min_temp, max_temp):
    norm = np.clip((temp_c - min_temp) / (max_temp - min_temp), 0, 1)
    norm = (norm * 255).astype(np.uint8)    
    norm = norm.astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    return colored


def display_temperature(img, val_c, loc, color, start_point, end_point):
    """
    Display the temperature at the given location on the image.
    """
    
    cv2.putText(img, "{0:.2f}C".format(val_c), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)
    cv2.rectangle(img, start_point, end_point, color, 1)
    cv2.imshow('Thermal',img)


##########################################################################
#----------------------- RGB CAMERA --------------------------------------#
##########################################################################

def get_rgb_camera(device_path):
    """Open RGB camera."""
    cap = cv2.VideoCapture(device_path)
    if cap.isOpened():
        write_log("INFO", f"RGB camera found at {device_path}")
        return cap
    else:
        write_log("ERROR", f"Failed to open RGB camera at {device_path}")
        cap.release()
        raise RuntimeError(f"RGB camera not found at {device_path}")

def capture_rgb(current_time):
    """Save RGB frame asynchronously."""
    ret, frame = rgb_cap.read()
    if not ret:
        write_log("ERROR", "RGB frame failed")
        return
    frame = cv2.resize(frame, (w, h))
    path = os.path.join(rgb_folder, f"{current_time}.jpg")
    cv2.imwrite(path, frame)

##########################################################################
#----------------------- MAIN PROGRAM ------------------------------------#
##########################################################################

# Load config
cam_name, h, w, max_temp, min_temp, thresh, duration, output, display = load_config()

# Setup folders
output_folder = os.path.join(output, f"{cam_name}_{datetime.now().strftime('%Y_%m_%d')}")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, "color_thermal"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "raw_thermal"), exist_ok=True)
rgb_folder = os.path.join(output_folder, "RGB")
os.makedirs(rgb_folder, exist_ok=True)

# Detect camera devices
tc_index = get_tc001_index()
THERMAL_DEVICE = '/dev/video' + str(tc_index)
RGB_DEVICE = '/dev/video' + str(1 - tc_index)  # assumes second camera

# Init cameras
try:
    cap = get_tc001_camera(THERMAL_DEVICE)
    rgb_cap = get_rgb_camera(RGB_DEVICE)
except RuntimeError as e:
    write_log("ERROR", f"Camera initialization failed: {e}")
    exit(1)

##########################################################################
#----------------------- CAMERA LOOP -------------------------------------#
##########################################################################

def main():
    global last_log_time
    start_time = time.time()
    box_size = 100  # half-size of square ROI
    cooldown_sec = 2
    last_trigger = 0

    while True:
        if (time.time() - start_time) > duration * 60:
            write_log("INFO", "Duration completed")
            break

        ret, frame = cap.read()
        if not ret:
            write_log("ERROR", "Thermal frame read failed")
            continue

        raw = extract_thermal(frame)
        temp_c = raw_to_celsius(raw)
        thermal_norm = get_heatmap(temp_c, min_temp, max_temp)
        h_img, w_img, ch = thermal_norm.shape

        # ROI in center
        cx, cy = w_img // 2, h_img // 2
        #x1, y1 = max(cx - roi_size, 0), max(cy - roi_size, 0)
        #x2, y2 = min(cx + roi_size, w_img), min(cy + roi_size, h_img)
        top_left_x = cx - box_size // 2
        top_left_y = cy - box_size // 2
        bottom_right_x = cx + box_size // 2
        bottom_right_y = cy + box_size // 2
        roi = temp_c[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        #roi = thermal_norm[y1:y2, x1:x2]
        roi_max = roi.max()
        max_temp_pos = np.unravel_index(np.argmax(roi), roi.shape)
        max_temp_x = top_left_x + max_temp_pos[1]
        max_temp_y = top_left_y + max_temp_pos[0]

        # Trigger based on relative threshold
        if roi_max > thresh and (time.time() - last_trigger) > cooldown_sec:
            timestamp = datetime.now().strftime("%H%M%S")
            # Save thermal
            cv2.imwrite(os.path.join(output_folder, "color_thermal", f"{timestamp}.jpg"),
                        thermal_norm)
            np.save(os.path.join(output_folder, "raw_thermal", f"{timestamp}.npy"), raw)
            # Save RGB asynchronously
            threading.Thread(target=capture_rgb, args=(timestamp,)).start()
            write_log("INFO", f"Trigger detected! ROI mean={roi_max:.3f} | saved {timestamp}")
            last_trigger = time.time()

        # Optional display
        if int(display) == 1:                                
            
            display_temperature(thermal_norm, roi_max, (max_temp_x, max_temp_y), (255, 255, 255), (top_left_x,top_left_y), (bottom_right_x, bottom_right_y))               

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    rgb_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
