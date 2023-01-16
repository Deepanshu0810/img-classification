import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
st.title('Image Classifier Using SVM')

import pickle
model = pickle.load(open('img_model.p','rb'))

uploaded_file = st.file_uploader("Choose an image",type="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img,caption='Uploaded Image')

    if st.button("PREDICT"):
        category = ['lotus flower','rose flower','daisy flower']
        flat_data = []
        img  = np.array(img)
        img_resize = resize(img,(200,400,3))
        flat_data.append(img_resize.flatten())
        flat_data = np.array(flat_data)
        print(img.shape)
        y_out = model.predict(flat_data)
        y_out = category[y_out[0]]
        print("predicted :",y_out)
        st.write(y_out)
