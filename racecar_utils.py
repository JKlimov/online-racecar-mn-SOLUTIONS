#############################
#### Imports
#############################

# General 
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ROS 
try:
    import rclpy
    from sensor_msgs.msg import LaserScan
    from sensor_msgs.msg import Image
    from ackermann_msgs.msg import AckermannDriveStamped
except:
    print('ROS is not installed')

# iPython Display
import PIL.Image
from io import BytesIO
import IPython.display
import time

# Used for HSV select
import threading
try:
    import ipywidgets as widgets
except:
    print('ipywidgets is not installed')
    
import pyrealsense2 as rs

#############################
#### Parameters
#############################

# Video Capture Port
video_port = 2

# Display ID
current_display_id = 1 # keeps track of display id

# Resize dimensions
resize_width = 640
resize_height = 480

#############################
#### Racecar ROS Class
#############################

# Starter code class that handles the fancy stuff. No need to modify this!

#############################
#### General Display
#############################

def show_inline(img):
    '''Displays an image inline.'''
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r,g,b])
    plt.imshow(rgb_img)
    plt.xticks([]), plt.yticks([])
    plt.show()

def show_frame(frame):
    global display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    f = BytesIO()
    PIL.Image.fromarray(frame).save(f, 'jpeg')
    img = IPython.display.Image(data=f.getvalue())
    display.update(img)

def resize_cap(cap, width, height):
    cap.set(3,width)
    cap.set(4,height)

#############################
#### Identify Cone
#############################

def show_video(func, time_limit = 10, use_both_frames = False, show_video = True):
    global current_display_id
    display = IPython.display.display('', display_id=current_display_id)
    current_display_id += 1
    
    def display_frames(frames):
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not color_frame or not depth_frame:
            return
        
        # Convert image to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        if use_both_frames:
            processed_img = func(color_image, depth_image)
        else:
            processed_img = func(color_image)

        if show_video:
            f = BytesIO()
            PIL.Image.fromarray(processed_img).save(f, 'jpeg')
            img = IPython.display.Image(data=f.getvalue())
            display.update(img)
            time.sleep(0.2)
        
    withRealSenseFrames(display_frames, time_limit)

def show_image(func):
    global display, current_display_id
    display = IPython.display.display('', display_id=current_display_id)
    current_display_id += 1
    
    cap = cv2.VideoCapture(video_port)
    resize_cap(cap, resize_width, resize_height)
    frame = func(cap.read()[1])  
    show_frame(frame)
    cap.release()

def show_picture(img):
    global display, current_display_id
    # setup display
    display = IPython.display.display('', display_id=current_display_id)
    current_display_id += 1
    # display image
    f = BytesIO()
    PIL.Image.fromarray(img).save(f, 'jpeg')
    display_image = IPython.display.Image(data=f.getvalue())
    display.update(display_image)
    

#############################
#### HSV Select
#############################

def withRealSenseFrames(frameProcessor, limit = None):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        if isinstance(limit, int):
            start = time.time()
            while time.time() - start < limit:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                frameProcessor(frames)
        else:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                frameProcessor(frames)
    finally:
        # Stop streaming
        pipeline.stop()

# Mask and display video
def hsv_select_live(limit = 10, fps = 5):
    global current_display_id
    display = IPython.display.display('', display_id=current_display_id)
    current_display_id += 1

    # Create sliders
    h = widgets.IntRangeSlider(value=[0, 179], min=0, max=179, description='Hue:', continuous_update=True, layout=widgets.Layout(width='100%'))
    s = widgets.IntRangeSlider(value=[0, 255], min=0, max=255, description='Saturation:', continuous_update=True, layout=widgets.Layout(width='100%'))
    v = widgets.IntRangeSlider(value=[0, 255], min=0, max=255, description='Value:', continuous_update=True, layout=widgets.Layout(width='100%'))
    display.update(h)
    display.update(s)
    display.update(v)

    def processFrame(frames):
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        # Convert image to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        hsv_min = (h.value[0], s.value[0], v.value[0])
        hsv_max = (h.value[1], s.value[1], v.value[1])
        img_hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(img_hsv, hsv_min, hsv_max)
        img_masked = cv2.bitwise_and(color_image, color_image, mask = mask)

        f = BytesIO()
        PIL.Image.fromarray(img_masked).save(f, 'jpeg')
        img = IPython.display.Image(data=f.getvalue())
        display.update(img)
        time.sleep(1.0 / fps)

    def show_masked_video():  
        withRealSenseFrames(processFrame, limit)

    # Open video on new thread (needed for slider update)
    hsv_thread = threading.Thread(target=show_masked_video)
    hsv_thread.start()

    
#############################
#### Feature Detection
#############################

def find_object(img, img_q, detected, kp_img, kp_frame, good_matches, query_columns):
    '''
    Draws an outline around a detected objects given matches and keypoints.

    If enough matches are found, extract the locations of matched keypoints in both images.
    The matched keypoints are passed to find the 3x3 perpective transformation matrix.
    Use transformation matrix to transform the corners of img to corresponding points in trainImage.
    Draw matches.
    '''
    dst = []
    if detected:
        src_pts = np.float32([ kp_img[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w,ch = img_q.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        
        if M is not None:
            dst = cv2.perspectiveTransform(pts,M)

            dst[:,:,0] += query_columns
        
            x1 = dst[:, :, 0][0]
            y1 = dst[:, :, 1][0]

            x2 = dst[:, :, 0][3]
            y2 = dst[:, :, 1][3]

            center = (x1 + abs(x1 - x2)/2, y1 - abs(y1 - y2)/2)

            return img, dst, center[0], center[1]
        else:
            matchesMask= None
            return img, dst, -1, -1
    else:
        matchesMask = None
        return img, dst, -1, -1   # if center[0] = -1 then didn't find center
    
