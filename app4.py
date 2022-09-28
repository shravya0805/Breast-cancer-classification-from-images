import streamlit as st 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import re
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import sklearn.metrics as mt
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageFilter ,ImageOps
from numpy import asarray
import datetime as d
#from skimage.io import imread, imshow
from skimage.measure import label, regionprops, regionprops_table
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import os
from sklearn.naive_bayes import GaussianNB
def app():
    st.markdown("<h1 style='text-align: center; color: #FF69B4';>Breast Cancer Classification</h1>", unsafe_allow_html=True)      
    test=pd.read_csv(r"C:\Users\this pc\Downloads\test.csv")
    train=pd.read_csv(r"C:\Users\this pc\Downloads\train.csv")
    X_train=train[['area_sum','convex_area_sum','bbox_area_sum','extent_sum','solidity_sum','eccentricity_sum','orientation_sum']]
    X_test=test[['area_sum','convex_area_sum','bbox_area_sum','extent_sum','solidity_sum','eccentricity_sum','orientation_sum']]
    y_train=train['label']
    y_test = test['label']
    st.subheader("Prediction")
    file = st.file_uploader("Upload the image to be classified ", type=["jpg", "png"])
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image,width=500)    
        image = np.asarray(image)
    
        p1 = {
            "area_sum":[],"convex_area_sum":[],"bbox_area_sum":[],
          "extent_sum":[],"solidity_sum":[],"eccentricity_sum":[],
          "orientation_sum":[]
         }
        gray_painting = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        (thresh, blackAndWhiteImage) = cv2.threshold(gray_painting, 127, 255, cv2.THRESH_BINARY_INV) 
        cv2.imwrite(r'C:\Users\this pc\Downloads\New folder\abc.png', blackAndWhiteImage)
        binarized=cv2.imread(r'C:\Users\this pc\Downloads\New folder\abc.png')
        im1 = Image.open(r'C:\Users\this pc\Downloads\New folder\abc.png') 
        im2 = im1.filter(ImageFilter.MedianFilter(size = 3))
        numpydata = asarray(im2)
        label_im = label(numpydata)
        regions = regionprops(label_im)
        properties = ["area","convex_area","bbox_area", "extent", "solidity", "eccentricity", "orientation"]
        df=pd.DataFrame(regionprops_table(label_im, properties=properties))
        df=df[df['area']>1000]
        df=df[df['area']<8000]
        p1['area_sum'].append(df['area'].sum())
        p1['convex_area_sum'].append(df['convex_area'].sum())
        p1['bbox_area_sum'].append(df['bbox_area'].sum())
        p1['extent_sum'].append(df['extent'].sum())
        p1['solidity_sum'].append(df['solidity'].sum())
        p1['eccentricity_sum'].append(df['eccentricity'].sum())
        p1['orientation_sum'].append(df['orientation'].sum())
        df2=pd.DataFrame(p1)
        X_train1=df2[['area_sum','convex_area_sum','bbox_area_sum','extent_sum','solidity_sum','eccentricity_sum','orientation_sum']]

        col1, col2 = st.columns(2)
        
        #st.header('SVM')
        col1.markdown("<h6 style='color: #FF69B4';>Support Vector Machine</h6>", unsafe_allow_html=True)      
        clf = SVC()
        clf.fit(X_train, y_train)
        x=clf.predict(X_train1)
        col1.text(x[0])
        
        #st.header('KNN')
        col1.markdown("<h6 style='color: #FF69B4';>K-Nearest Neighbours</h6>", unsafe_allow_html=True)    
        clf = KNeighborsClassifier()
        clf.fit(X_train, y_train)
        x=clf.predict(X_train1)
        col1.text(x[0])
        
        #st.header('Naive Bayes')
        col1.markdown("<h6 style='color: #FF69B4';>Naive Bayes</h6>", unsafe_allow_html=True)  
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        x=clf.predict(X_train1)
        col1.text(x[0])
        
        #st.header('Logistic Regression')
        col2.markdown("<h6 style='color: #FF69B4';>Logistic Regression</h6>", unsafe_allow_html=True)  
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        x=clf.predict(X_train1)
        col2.text(x[0])
        
        #st.header('Decision Tree')
        col2.markdown("<h6 style='color: #FF69B4';>Decision Tree</h6>", unsafe_allow_html=True)  
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        x=clf.predict(X_train1)
        col2.text(x[0])

        
        col2.markdown("<h6 style='color: #FF69B4';>Neural Network</h6>", unsafe_allow_html=True)  
        clf = MLPClassifier()
        clf.fit(X_train, y_train)
        x=clf.predict(X_train1)
        col2.text(x[0])        
