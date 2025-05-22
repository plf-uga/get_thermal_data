import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from libuvc_wrapper import *
import configparser
from queue import Queue

BUF_SIZE = 2
q = Queue(BUF_SIZE)

os.chdir("/mnt/external/get_thermal_data")

##########################################################################
#---------------Util Functions for the program---------------------------#
##########################################################################


def write_log(info, message, verbose):
    current_timestamp = time.time()   
    # Convert the timestamp to the desired format
    current_time = time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(current_timestamp))
    cout = f'{info} {current_time} : {message}  \n'
    with open('thermal_outputs.log', 'a') as file:
        if verbose == 1:
            print(cout)
        file.write(cout)  

def display_temperature(img, val_k, loc, color):
    """
    Display the temperature at the given location on the image.
    """
    val = ktoc(val_k)
    cv2.putText(img, "{0:.1f} degC".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)


# Configuration
def load_config():
    config = configparser.ConfigParser()
    config.read('thermal.par')
    cam_name = config.get('Camera', 'cam_name')
    h = config.getint('Camera', 'img_height')
    w = config.getint('Camera', 'img_width')
    index = config.getint('Camera', 'index')
    temp_max = config.getint('Temperature', 'temp_max')
    temp_min = config.getint('Temperature','temp_min')
    thresh = config.getint('Temperature','temp_trigger')
    duration = config.getint('Time','duration')
    output = config.get('Output', 'folder')
    return cam_name, h, w, temp_max, temp_min, thresh, duration, index, output



def raw_to_colored(data, min_temp, max_temp):
    """
    Convert the raw thermal data to a standardized 8-bit colored image.
    
    Args:
        data: 16-bit raw thermal data.
        min_temp: The minimum temperature (in °C) for normalization.
        max_temp: The maximum temperature (in °C) for normalization.
        
    Returns:
        A colored image representing the normalized thermal data.
    """
    data_celsius = ktoc(data)
    data_normalized = np.clip((data_celsius - min_temp) / (max_temp - min_temp), 0, 1)
    data_normalized = (data_normalized * 255).astype(np.uint8)
    colored_img = cv2.applyColorMap(data_normalized, cv2.COLORMAP_HOT)
    return colored_img

# Convert raw data to temperature
def ktoc(val):
    return (val - 27315) / 100.0


def get_rgb_camera():
    """
    Try to open the RGB camera with OpenCV by checking the first available index.
    """
    max_cameras = 10  # You can adjust this depending on how many cameras you expect to connect
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            write_log(info = "INFO",message=f"RGB camera found at index {index}", verbose = 1)
            return cap
        else:
            cap.release()
    print("Error: No RGB camera found.")
    return cap

# RGB Camera Capture Thread
def capture_rgb(current_time):
    width, height = w, h
    ret, frame = cap.read()
    if not ret:
        write_log(info = "ERROR", message = "Error: Could not read RGB frame.", verbose = 1)
        return

    frame = cv2.resize(frame, (width, height))
   

    # Ensure "RGB" folder exists inside output_folder

    
    frame_path = os.path.join(rgb_folder, f"{current_time}.jpg")
    
    success = cv2.imwrite(frame_path, frame)  # Save frame
    if success:
        write_log(info = "INFO", message = f"Saved RGB frame: {frame_path}", verbose = 1)
    else:
        write_log(info = "ERROR", message = f"Error: Could not save RGB frame at {frame_path}", verbose = 1)


# Thermal Frame Callback
def py_frame_callback(frame, userptr):
    array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(array_pointer.contents, dtype=np.uint16).reshape(frame.contents.height, frame.contents.width)

    if not q.full():
        q.put(data)

##########################################################################
#---------------End of Util Functions -----------------------------------#
##########################################################################
#cap = get_rgb_camera()
#cap = cv2.VideoCapture('/dev/v4l/by-id/usb-H264_USB_Camera_H264_USB_Camera_2020032801-video-index1')

# Open RGB camera source
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    write_log(info = "ERROR", message = "Could not open RGB camera", verbose = 1)
    exit()

#Import program parameters
cam_name, h, w, temp_max, temp_min, thresh, duration, index, output = load_config()

# Ensure output folder exists
output_folder = os.path.join(output, f"{cam_name}_{datetime.now().strftime('%Y_%m_%d')}")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, "color_thermal"), exist_ok = True)
os.makedirs(os.path.join(output_folder, "raw_thermal"), exist_ok = True)
rgb_folder = os.path.join(output_folder, "RGB")
os.makedirs(rgb_folder, exist_ok=True)  # <-- FIX
write_log(info = "INFO", message = f"Created directory: {output_folder}", verbose = 1)


PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)



##########################################################################
#---------------Begin of Main Program -----------------------------------#
##########################################################################

def main():  
    
    start_time = time.time()
    
    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()
    ctrl = uvc_stream_ctrl()
    ctx, dev, devh, ctrl = POINTER(uvc_context)(), POINTER(uvc_device)(), POINTER(uvc_device_handle)(), uvc_stream_ctrl()

    res = libuvc.uvc_init(byref(ctx), 0)
    if res < 0:
        write_log(info = "ERROR",message ="uvc_init error", verbose = 1)
        exit(1)

    try:
        res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
        if res < 0:
            write_info(info = "ERROR", message = "uvc_find_device error", verbose = 1)
            exit(1)

        res = libuvc.uvc_open(dev, byref(devh))
        if res < 0:
            write_info(info = "ERROR", message= "uvc_open error", verbose = 1)
            exit(1)

        write_log(info = "INFO", message = "Thermal camera started", verbose = 1)
        frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
        libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16, frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval))
        libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
        
        ind = 0
        try:
            while True:
                if (time.time() - start_time) > duration*60: 
                    #print(time.time() - start_time)
                    write_log(info = "INFO", message ="Duration Completed.", verbose = 1)
                    break
                data = q.get(True, 500)
                data2 = cv2.resize(data[:,:], (w, h))
                colored_img = raw_to_colored(data=data2, min_temp=temp_min, max_temp=temp_max)
                if data is None:
                    break
                # Define bounding box size (e.g., 100x100 pixels)
                box_size = 100
                center_x, center_y = colored_img.shape[1] // 2, colored_img.shape[0] // 2
                top_left_x = center_x - box_size // 2
                top_left_y = center_y - box_size // 2
                bottom_right_x = center_x + box_size // 2
                bottom_right_y = center_y + box_size // 2
                cv2.rectangle(colored_img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 255, 255), 2)
                roi = data2[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                max_temp_val = np.max(roi)
                max_temp_celsius = ktoc(max_temp_val)
                
                # Find the maximum temperature in the ROI and its location
                max_temp_val = np.max(roi)
                max_temp_celsius = ktoc(max_temp_val)
                max_temp_pos = np.unravel_index(np.argmax(roi), roi.shape)  # Get position of max temp in ROI
                
                # Convert the position of max temperature in the ROI to the coordinates in the full image
                max_temp_x = top_left_x + max_temp_pos[1]
                max_temp_y = top_left_y + max_temp_pos[0]
                # Display the maximum temperature at the exact location within the bounding box
                display_temperature(colored_img, max_temp_val, (max_temp_x, max_temp_y), (255, 255, 255))

                

                # Trigger RGB capture in a new thread
                if max_temp_celsius > thresh:
                    ind = ind + 1
                    current_time = time.strftime('%H%M%S', time.localtime(time.time()))
                    write_log(info = "INFO", message = f"Heat Signature Detected: {max_temp_celsius}°C", verbose = 1)
                    frame_path = os.path.join(output_folder, "color_thermal", f"{cam_name}_{current_time}.jpg")
                    np.save(os.path.join(output_folder, "raw_thermal", f"{cam_name}_{current_time}.npy"), data)
                    success = cv2.imwrite(frame_path, colored_img)
                    rgb_thread = threading.Thread(target=capture_rgb(current_time = current_time))
                    rgb_thread.start()
		    

        finally:
            libuvc.uvc_stop_streaming(devh)

    finally:
        libuvc.uvc_exit(ctx)
        cap.release()

if __name__ == '__main__':
    main()


##########################################################################
#-----------------End of Main Program -----------------------------------#
##########################################################################


