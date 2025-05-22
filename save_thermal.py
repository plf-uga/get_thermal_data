#!/usr/bin/env python
# -*- coding: utf-8 -*-

from libuvc_wrapper import *
from save_RGB import *
import time
import cv2
import numpy as np
import configparser
from datetime import datetime
try:
  from queue import Queue
except ImportError:
  from Queue import Queue
import platform

"""
    The utils below were copied from the factory github page
"""


BUF_SIZE = 2
q = Queue(BUF_SIZE)

def py_frame_callback(frame, userptr):

  array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
  data = np.frombuffer(
    array_pointer.contents, dtype=np.dtype(np.uint16)
  ).reshape(
    frame.contents.height, frame.contents.width
  ) # no copy

  if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
    return

  if not q.full():
    q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

def ktof(val):
  return (1.8 * ktoc(val) + 32.0)
def ktoc(val):
  return (val - 27315) / 100.0

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

def display_temperature(img, val_k, loc, color):
    """
    Display the temperature at the given location on the image.
    """
    val = ktoc(val_k)
    cv2.putText(img, "{0:.1f} degC".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

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


"""
  Start the main function!!!
"""


def main():
  
	
  #Loading the configuration parameters  
  cam_name, h, w, temp_max, temp_min, thresh, duration, index, output = load_config()
  start_time = time.time()
  duration = duration * 60  # convert to seconds
  ctx = POINTER(uvc_context)()
  dev = POINTER(uvc_device)()
  devh = POINTER(uvc_device_handle)()
  ctrl = uvc_stream_ctrl()
  
  #cap = cv2.VideoCapture(2)
  # Check if the camera opened successfully
  #if not cap.isOpened():
  #  print("Error: Could not open RGB camera")
  #  exit()
  

  res = libuvc.uvc_init(byref(ctx), 0)
  if res < 0:
    print("uvc_init error")
    exit(1)

  try:
    res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
    if res < 0:
      print("uvc_find_device error")
      exit(1)

    try:
      res = libuvc.uvc_open(dev, byref(devh))
      if res < 0:
        print("uvc_open error")
        exit(1)

      print("device opened!")
      print_device_info(devh)
      print_device_formats(devh)

      frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
      if len(frame_formats) == 0:
        print("device does not support Y16")
        exit(1)

      libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
                                              frame_formats[0].wWidth, frame_formats[0].wHeight,
                                              int(1e7 / frame_formats[0].dwDefaultFrameInterval))

      res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
      if res < 0:
        print("uvc_start_streaming failed: {0}".format(res))
        exit(1)

      try:
        while True:
          data = q.get(True, 500)
          if data is None:
            break

          if time.time() - start_time > duration:
              print("INFO:", "Duration Completed.")
              break
          #ret, frame = cap.read()

          #if not ret:
          #  print("Error: Could not read RGB frame.")
          #  continue  # Skip the rest of the loop if no frame is captured

          data = cv2.resize(data[:,:], (w, h))
          colored_img = raw_to_colored(data=data, min_temp=temp_min, max_temp=temp_max)

          # Define bounding box size (e.g., 100x100 pixels)
          box_size = 100
          center_x, center_y = colored_img.shape[1] // 2, colored_img.shape[0] // 2
          top_left_x = center_x - box_size // 2
          top_left_y = center_y - box_size // 2
          bottom_right_x = center_x + box_size // 2
          bottom_right_y = center_y + box_size // 2

          # Draw bounding box on the image
          #cv2.rectangle(colored_img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 255, 255), 2)

          # Extract the region of interest (ROI) from the image
          roi = data[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

          # Find the maximum temperature in the ROI and its location
          max_temp_val = np.max(roi)
          max_temp_celsius = ktoc(max_temp_val)
          max_temp_pos = np.unravel_index(np.argmax(roi), roi.shape)  # Get position of max temp in ROI

          # Convert the position of max temperature in the ROI to the coordinates in the full image
          max_temp_x = top_left_x + max_temp_pos[1]
          max_temp_y = top_left_y + max_temp_pos[0]

          # Display the maximum temperature at the exact location within the bounding box
          #display_temperature(colored_img, max_temp_val, (max_temp_x, max_temp_y), (255, 255, 255))

          # If the maximum temperature is above threshold, log the event
          if max_temp_celsius > thresh:
              print(f"Heat Signature Detected: {max_temp_celsius} degC")
              current_timestamp = time.time()
              current_time = time.strftime('%H%M%S', time.localtime(current_timestamp))
              output_folder = os.path.join(output, f"{cam_name}_{datetime.now().strftime('%Y_%m_%d')}")
              index += 1
              run_RGB()

              if not os.path.exists(output_folder):
                  os.makedirs(os.path.join(output_folder, "RGB_thermal"), exist_ok = True)
                  os.makedirs(os.path.join(output_folder, "raw_thermal"), exist_ok = True)
                  print(f"Created directory: {output_folder}")

	      
              
              frame_path = os.path.join(output_folder, "RGB_thermal", f"{cam_name}_{current_time}_{index}.jpg")
              success = cv2.imwrite(frame_path, colored_img)

          # Display the image
          #cv2.imshow(f'{cam_name}', colored_img)
          #cv2.waitKey(1)

      finally:
        libuvc.uvc_stop_streaming(devh)

      print("done")
    finally:
      libuvc.uvc_unref_device(dev)
  finally:
    libuvc.uvc_exit(ctx)

if __name__ == '__main__':
  main()

