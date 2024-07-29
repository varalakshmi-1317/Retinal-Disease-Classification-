import streamlit as st
import tensorflow as tf
import base64
import warnings
warnings.filterwarnings("ignore")
from PIL import Image


def load_model():
  model=tf.keras.models.load_model('./transfer_l_1.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.title("Retinal Disease Classification")
st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> Diabetic Neuropathy, Diabetic Retinopathy, Macular Hole, Normal, Optic Disc Cupping </h3>', unsafe_allow_html=True)


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('1583271548020.jpeg')



file = st.file_uploader("Please upload an eye scan file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction

class_names = ['Diabetic Neuropathy', 'Diabetic Retinopathy', 'Macular Hole', 'Normal', 'Optic Disc Cupping'] 
if file is None:
    st.text("Please upload an image file of a eye scan")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    #st.write(score)
    if(np.max(score)<0.6):
        conf = 100 * (np.max(score)+0.4)
    st.success("This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], conf)
)

