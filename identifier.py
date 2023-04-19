import streamlit as st
# import pickle as pkl
import numpy as np
import tensorflow as tf
# import cv2
from PIL import Image, ImageOps

model =tf.keras.models.load_model('C:/Users/achin/PycharmProjects/PlanetIdentifier/my_model.hdf5')

# image_data = pkl.load(open('images.pkl', 'rb'))

st.title('PLANET IDENTIFIER')
file = st.file_uploader("Please upload a planet image", type=["jpeg", "jpg", "png"])


def import_and_predict(image_data, model):
    size = (144, 256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.array(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)

    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_name = ['Earth', 'Saturn', 'MakeMake', 'Mars', 'Mercury', 'Moon', 'Neptune', 'Pluto', 'Jupiter', 'Uranus',
                  'Venus']
    string = "This image most likely is: " + class_name[np.argmax(prediction)]
    st.success(string)
