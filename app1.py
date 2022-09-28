import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageFilter 
from numpy import asarray
import cv2
import os

from PIL import Image, ImageFilter 
def app():
    st.markdown("<h1 style='text-align: center; color: 	#FF69B4';>Breast Cancer Classification</h1>", unsafe_allow_html=True)  
    #st.image("https://www.101stadultdentistry.com/wp-content/uploads/2018/10/breast-cancer-1080x675.jpg", width=400)
    
    #st.write("""## Breast Cancer Dataset""")
    visualisation = st.sidebar.selectbox('Types Of Cancer',('Benign','Malignant' ))
    def load_images_from_folder(folder):
        images = []
        i=0
        col1, col2 = st.columns(2)
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
                i=i+1
                if i==20:
                    break
        i=0
        while(i<len(images)):
            col1.image(images[i],use_column_width=True)
            col2.image(images[i+1],use_column_width=True)
            i=i+2
        
    def get_visualisation(visualisation):
        if visualisation == 'Benign':
            st.write("""## Benign Images""")
            load_images_from_folder(r'C:\Users\this pc\Downloads\Benign')
        elif  visualisation == 'Malignant':
            st.write("""## Malignant""")
            load_images_from_folder(r'C:\Users\this pc\Downloads\Malignant')
    get_visualisation(visualisation)
    

