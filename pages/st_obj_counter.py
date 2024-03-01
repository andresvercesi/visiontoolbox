#Streamlit object counter implementation
import streamlit as st
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

det_box = []


path_base = os.path.abspath(os.path.dirname(__file__))
uploads_path = os.path.join(path_base, 'uploads')
model_path = 'yolov8n.pt'

def load_model(model_path):
    """
    Load Yolo Object detection model from specified model_path
    
    Parameters:
        model_path (str): The path to the Yolo model file

    Returns:
        A Yolo object detection model
    """
    model = YOLO(model_path)
    return model

def draw_det_box(video_file):
    """
    Draw a rectangle in first frame of video file to select
    detection area

    Parameters:
        video_file (str): The path of video file

    Returns:
        List[int] with 4 points of detection box 
    """
    det_box = []
    vidcap = cv2.VideoCapture(video_file) # load video from disk
    w, h = (int(vidcap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(w/2.8)
    h = int(h/2.8)
    success, frame = vidcap.read()
    if success:
        frame = cv2.resize(frame, (w, h))
        bg_image = cv2.imwrite('first_frame.jpg', frame)
    
    bg_color = "#eee"
    bg_image = 'first_frame.jpg'
    
    canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=3,
    stroke_color="#FB0101",
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=True,
    height= h,
    width = w,
    drawing_mode='polygon',
    display_toolbar= True,
    key="canvas",
    )
   
    if canvas_result.image_data is not None:
        if len(canvas_result.json_data['objects'])==1:
            for i in range(4):
                p = []
                for j in range(2):
                    p.append(canvas_result.json_data['objects'][0]['path'][i][j+1])
                det_box.append(p)
    os.remove(bg_image)
    return (det_box, w, h)

#Main page title
st.title("Objects count in video using YOLO")

if not os.path.exists(uploads_path):
    os.makedirs(uploads_path)

try:
    model = load_model(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)



video_file = st.file_uploader('Video File')


st_frame = st.empty()
process_button = st.button('Proccess video')

if video_file is not None:
    vid = os.path.join(uploads_path, video_file.name )
    with open(vid, mode='wb') as f:
        f.write(video_file.read()) # save video to disk
    det_box, w, h = draw_det_box(vid)



if process_button==True:
    # Init Object Counter
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=True,
                 reg_pts=det_box,
                 classes_names=model.names,
                 draw_tracks=True)
    vid_cap = cv2.VideoCapture(vid)
    
    while (vid_cap.isOpened()):
        success, image = vid_cap.read()
        if success:
            image = cv2.resize(image, (w, h))
            tracks = model.track(image, persist=True, show=False)
            frame = counter.start_counting(image, tracks)
            #res = model.predict(image)
            #result_tensor = res[0].boxes
            #res_plotted = res[0].plot()
            st_frame.image(image,
                            caption='Detected Video',
                            channels="BGR",
                            use_column_width=True
                            )
        else:
            vid_cap.release()
            break


