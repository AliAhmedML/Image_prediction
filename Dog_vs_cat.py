# Importing the libraries
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
import requests
# loading saved models

Prediction_model = load_model('C:/Users/ali ahmed/Desktop/Dog vs Cat Predicition system/catVsdog.keras', custom_objects={'KerasLayer':hub.KerasLayer})

# Creating the sidebars
with st.sidebar:
    
    selected = option_menu('Image Prediction System', ['Dog Vs Cat prediction'], default_index = 0)
    
    
# Cat VS Dogs Prediction Page
if (selected == 'Dog Vs Cat prediction'):
    try :
            # page title
            st.title('Dog Vs Cat prediction using ML')
            
            image_url = st.text_input('Enter the image path')
            
            data = requests.get(image_url).content
            
            file = open('C:/Users/ali ahmed/Desktop/Dowloaded images/img.png','wb')
            file.write(data)
            file.close() 
            
            input_image = cv2.imread('C:/Users/ali ahmed/Desktop/Dowloaded images/img.png')
        
            #cv2.imshow('Your image',input_image)
        
            input_image_resize = cv2.resize(input_image, (224,224))
        
            input_image_scaled = input_image_resize/255
        
            image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])
        
            prediction_result = ''
            
            input_prediction = Prediction_model.predict(image_reshaped)
                
            input_pred_label = np.argmax(input_prediction)
                
            if input_pred_label == 0:
                prediction_result = 'The image represents a Cat'
        
            else:
                prediction_result = 'The image represents a Dog'
        
            st.success(prediction_result)
    except ValueError:
         pass
