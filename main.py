import streamlit as st
import os
import time
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import requests
import gdown
# download the weights file from the link below
url = "https://drive.google.com/uc?id=1H5mn_xJsq-8fyWI3eX_YBPQpAF2paXtT"
output = "yolov8.pt"
gdown.download(url, output, quiet=False)


def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

# check if data/images folder exists 
if not os.path.exists('data/images'):
    os.makedirs('data/images')

if not os.path.exists('data/videos'):
    os.makedirs('data/videos')

model = YOLO(model='yolov8x.pt')
model.fuse()

source = ("Image detection", "Video detection")
source_index = st.sidebar.selectbox("select input", range(
        len(source)), format_func=lambda x: source[x])

if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "upload image", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='resource loading...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False
    
else:
        uploaded_file = st.sidebar.file_uploader("upload video", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='resource loading...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                source = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False

if is_valid:
        print('valid')
        if st.button('start detection'):

            model.predict(source=source,save=True,project='runs')

            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img))
                    
                    
                    

                    #st.balloons()
            else:
                with st.spinner(text='Preparing Video'):
                    
                    for vid in os.listdir(get_detection_folder()):
                        video_file=open(str(Path(f'{get_detection_folder()}') / vid), 'rb')
                        video_bytes = video_file.read()
                        st.video(video_bytes)

                    #st.balloons()

# save runs to current working directory yolov8