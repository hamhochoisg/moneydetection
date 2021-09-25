import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image 

Model_Path = '\model\my_model_checkpoint.h5'

test_image_path = "streamlit-demo\media\\test\\100000\\252.jpg"
img        = image.load_img(test_image_path, target_size=(224, 224))
img_array  = image.img_to_array(img)
img_array  = np.expand_dims(img_array, axis=0) #predict nhận theo batch (1,224,224,3)

print(img_array.shape)

model = tf.keras.models.load_model(Model_Path)

class_names = ['1000', '10000', '100000', '2000', '20000', '200000', '5000', '50000', '500000']
prediction = model.predict(img_array)
index = prediction.argmax()
pred = class_names[index]
l = list(prediction)

print('model prediction:',pred)
print('% CHính xác:',l[0][index])