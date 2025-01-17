import streamlit as st
from fastai.vision.all import *
import plotly.express as px

import pathlib 
temp = pathlib.PosixPath

#title
st.title("Xayvonlarni Klassifikatsiya Qiluvchi Model")

#upload image
file = st.file_uploader("Upload an image: ", type = ['gif', 'jpeg', 'jpg', 'png', 'svg'])

if file:
    st.image(file)

    #pil convert
    img = PILImage.create(file)

    #model
    model = load_learner('animal_model.pkl')

    #prediction
    pred, pred_id, prob = model.predict(img)
    st.success(f"Prediction: {pred}")
    st.info(f"Probability: {prob[pred_id]*100:.1f}%")

    #plotting (Visualisation)
    fig = px.bar(x = prob*100, y = model.dls.vocab)
    st.plotly_chart(fig)
