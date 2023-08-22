# Importing the libraries
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import cv2

# loading saved models

Prediction_model = pickle.load(open('Dog_VS_Cat.sav', 'rb'))

# Creating the sidebars
with st.sidebar:
    
    selected = option_menu('Image Prediction System', ['Dog Vs Cat prediction'], default_index = 0)
    
    
# Cat VS Dogs Prediction Page
if (selected == 'Dog Vs Cat prediction'):
    
    # page title
    st.title('Dog Vs Cat prediction using ML')
    
    image_url = st.text_input('Enter the image path')
    
    input_image = cv2.imread(image_url)

    cv2.imshow(input_image)

    input_image_resize = cv2.resize(input_image, (224,224))

    input_image_scaled = input_image_resize/255

    image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])

    prediction_result = ''
    
    if st.button('prediction Result'):
        
        input_prediction = Prediction_model.predict(image_reshaped)
        
        input_pred_label = np.argmax(input_prediction)
        
        if input_pred_label == 0:
            prediction_result = 'The image represents a Cat'

        else:
            prediction_result = 'The image represents a Dog'

    st.success(prediction_result)