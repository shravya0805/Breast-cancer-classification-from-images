import streamlit as st
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from PIL import Image
import cv2
import os
from PIL import Image, ImageFilter
#from google.colab.patches import cv2_imshow
def app():
    st.sidebar.header("Select Cancer Type")
    visualisation = st.sidebar.selectbox('Types Of Cancer',('Benign','Malignant' ))
    def Grey_Scale(folder):
        images = []
        i=0
        col1, col2 = st.columns(2)
        col1.header("Original")
        col2.header("Grey Scale")
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                gray_painting = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                col1.image(img, use_column_width=True)
                col2.image(gray_painting, use_column_width=True)
                i=i+1
                if i==15:
                    break
    def Binarized(folder):
        images = []
        i=0
        col1, col2 = st.columns(2)
        col1.header("Original")
        col2.header("Binarized")
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                gray_painting = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                (thresh, blackAndWhiteImage) = cv2.threshold(gray_painting, 127, 255, cv2.THRESH_BINARY_INV) 
                col1.image(img, use_column_width=True)
                col2.image(blackAndWhiteImage, use_column_width=True)
                i=i+1
                if i==15:
                    break
    def Noise_Removed_images(folder):
        images = []
        i=0
        col1, col2 = st.columns(2)
        col1.header("Original")
        col2.header("Noise_Removed_images")
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                gray_painting = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                (thresh, blackAndWhiteImage) = cv2.threshold(gray_painting, 127, 255, cv2.THRESH_BINARY_INV)
                cv2.imwrite(r'C:\Users\this pc\Downloads\New folder\abc.png', blackAndWhiteImage)
                binarized=cv2.imread(r'C:\Users\this pc\Downloads\New folder\abc.png')
                im1 = Image.open(r'C:\Users\this pc\Downloads\New folder\abc.png') 
                im2 = im1.filter(ImageFilter.MedianFilter(size = 3)) 
                col1.image(img, use_column_width=True)
                col2.image(im2, use_column_width=True)
                i=i+1
                if i==15:
                    break                
    def get_visualisation(visualisation):
        if visualisation == 'Benign':
            st.write("""## Benign Images""")
            ty = st.sidebar.selectbox('Preprocessed Images type',('Grey Scale','Binarized','Noise Removed images' ))
            if ty=='Grey Scale':
                Grey_Scale(r'C:\Users\this pc\Downloads\Benign')
            elif ty=='Binarized':
                Binarized(r'C:\Users\this pc\Downloads\Benign')
            elif ty=='Noise Removed images':
                Noise_Removed_images(r'C:\Users\this pc\Downloads\Benign')
        elif  visualisation == 'Malignant':
            st.write("""## Malignant Images""")
            ty = st.sidebar.selectbox('Preprocessed Images type',('Grey Scale','Binarized','Noise Removed images' ))
            if ty=='Grey Scale':
                Grey_Scale(r'C:\Users\this pc\Downloads\Malignant')
            elif ty=='Binarized':
                Binarized(r'C:\Users\this pc\Downloads\Malignant')
            elif ty=='Noise Removed images':
                Noise_Removed_images(r'C:\Users\this pc\Downloads\Malignant')

        
            
    get_visualisation(visualisation)
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
