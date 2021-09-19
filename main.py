import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image 

st.title('Banknotes Classification')
menu = ['Home','Up Load & Predict', 'Capture From Webcam']

#========================#
#==== Function #=========#
Model_Path = 'model\my_model_checkpoint.h5'
class_names = ['1000', '10000', '100000', '2000', '20000', '200000', '5000', '50000', '500000']

def get_saved_model(Model_Path):
    # Learning Rate maybe decrease so quick => start with 0.01
    restored_model = tf.keras.models.load_model(Model_Path)

    # Show the model architecture
    # restored_model.summary() #print in terminal
    return restored_model

def predict_image(image_path): #input and image show prediction label, reutrn string value of prediction
    model = get_saved_model(Model_Path)
    #Preprocess image:
    img        = image.load_img(image_path, target_size=(224, 224))
    img_array  = image.img_to_array(img)
    img_array  = np.expand_dims(img_array, axis=0) #predict nhận theo batch (1,224,224,3)

    #Prediction:
    
    prediction = model.predict(img_array)
    index = prediction.argmax()
    l = list(prediction)
    tmp_percent = l[0][index]*100

    pred = class_names[index]
    st.write('model prediction:')
    st.write(pred)
    st.write('Model Propotion:')
    st.write(tmp_percent)

def predict_image_array(img_array): #input and image array with shape = (1,224,224,3) show prediction label, reutrn string value of prediction
    model = get_saved_model(Model_Path)
  
    prediction = model.predict(img_array)
    index = prediction.argmax()
    l = list(prediction)
    tmp_percent = l[0][index]*100

    pred = class_names[index]
    st.write('model prediction:')
    st.write(pred)
    st.write('Model Propotion:')
    st.write(tmp_percent)
    
    print(l)

    return l,index

#========================#

choice = st.sidebar.selectbox('Danh mục', menu)

if choice == 'Home':
    st.title('This is Home Page')
    st.write('Xin chào, đây là ứng dụng phân loại tiền')
    
    # Get The current Path
    current_path = os.getcwd()
    st.write('current path:')
    st.write(current_path)

    #Load Model
    st.write('This is our model:')
    model = get_saved_model(Model_Path)  
    test_image_path = "media\\test\\500000\\Sự-thật-về-cách-đoán-3-số-Seri-tiền-500k-200k-100k-50k-20k-10k.jpg"
    
    #Show Image
    st.write('For Example Below Image')
    st.image(test_image_path,use_column_width='auto')
    st.write("Model Can Understand This Value")    

    #Prediction:   
    # predict_image(test_image_path)
    

elif choice == 'Up Load & Predict':
    st.title('Please Upload Your Banknotes Image, I Can Understand it:')
    photo_uploaded = st.file_uploader('Choose your banknotes photo', ['png', 'jpg', 'jpeg'])
    if photo_uploaded != None:
        
        image_np = np.asarray(bytearray(photo_uploaded.read()), dtype=np.uint8)
        # print(image_np)
        # print(image_np.shape)
        img = cv2.imdecode(image_np, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        print(img.shape)

        st.image(img)
        st.write(photo_uploaded.size)
        st.write(photo_uploaded.type)

        #Then Predict it
        img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
        img_array  = np.expand_dims(img, axis=0)
        # print(img_array.shape)
        print(type(img))

        predict_image_array(img_array)

elif choice == 'Capture From Webcam':
    st.title("Webcam Live Feed!")
    st.warning("Work on local computer ONLY")
    run = st.checkbox('Show!')
    capture = st.checkbox('Capture!')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        
        if capture == True:           
            captured_image = frame 
            break
            st.write("Stop!")   
    else:
        st.write("Stop!")
    
    camera.release()
    cv2.destroyAllWindows()

    if captured_image.shape != None:
        captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB )  
        st.write('Image That Captured')
        st.image(captured_image)
        captured_image = cv2.resize(captured_image, (224,224))

    # if captured_image.shape != None:
    #     st.write('Image That Captured')
    #     st.image(captured_image)
    #     captured_image = cv2.resize(captured_image, (224,224))
    #     print('Captured Image Shape:',captured_image.shape)
    #     print('Captured Image Type:',type(captured_image))   
    #     img_array  = np.expand_dims(captured_image, axis=0)
    #     predict_image_array(img_array)



