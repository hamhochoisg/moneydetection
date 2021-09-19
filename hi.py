import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
# import tensorflow

st.title('Nam first streamlit app')
menu = ['Home', 'Read Data', 'Display Images', 
        'Play Videos', 'Show Webcam', 'Play Music',
        'About me']

choice = st.sidebar.selectbox('Danh mục', menu)

if choice == 'Home':
    st.title('This is Home Page')
    st.write('Xin chào, đây là 1 tấm hình')
    image_path = 'media\pic-1.jpg'
    st.image(image_path, caption='This is an image', width=200)

    st.write('This is and math function')
    st.latex(r''' e^{i\pi} + 1 = 0 ''')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('This is column 1')
        name = st.text_input('What is your pet name ?')
    with col2:
        st.write('this is column 2')
    
    with col3:
        st.write('this is column 3')

