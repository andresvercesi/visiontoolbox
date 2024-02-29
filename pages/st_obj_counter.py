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


det_box =[]
uploads_path = 'upload_videos'
model_path = 'yolov8n.pt'

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)


st.write("Objects count in video using YOLO") #Web page title

video_file = st.file_uploader('Video File')

option = st.selectbox(
    'Select count type',
    ('Bounding Box', 'Line'))
if option == 'Bounding Box':
    detection_type = 'polygon'
    detection_range = 4
if option == 'Line':
    detection_type = 'line'
    detection_range = 2

if video_file is not None:
    vid = os.path.join(uploads_path, video_file.name )
    with open(vid, mode='wb') as f:
        f.write(video_file.read()) # save video to disk

    vidcap = cv2.VideoCapture(vid) # load video from disk
    w, h = (int(vidcap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(w/2)
    h = int(h/2)
    print(w)
    success, frame = vidcap.read()
    frame = cv2.resize(frame, (w, h))
    bg_image = cv2.imwrite('first_frame.jpg', frame)
    #pil_img = Image.fromarray(destRGB) # convert opencv frame (with type()==numpy) into PIL Image
    #st.image(pil_img, channels='BGR')
    #stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    #stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    #print(stroke_color)
    bg_color = "#eee"
    #bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    bg_image = 'first_frame.jpg'
    #realtime_update = st.sidebar.checkbox("Update in realtime", True)
    # Create a canvas component
    canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=3,
    stroke_color="#FB0101",
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=True,
    height=h,
    width = w,
    drawing_mode=detection_type,
    display_toolbar= True,
    key="full_app",
)
    # Do something interesting with the image data and paths
    #if canvas_result.image_data is not None:
    #    st.image(canvas_result.image_data)
 
    if len(canvas_result.json_data['objects'])==1:
        for i in range(detection_range):
            p = []
            for j in range(2):
                p.append(canvas_result.json_data['objects'][0]['path'][i][j+1])
            det_box.append(p)
        print(len(det_box))



process_button = st.button('Proccess video')

if process_button==True and len(det_box)==4:
    # Init Object Counter
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=True,
                 reg_pts=det_box,
                 classes_names=model.names,
                 draw_tracks=True)
    vid_cap = cv2.VideoCapture(vid)
    st_frame = st.empty()
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


