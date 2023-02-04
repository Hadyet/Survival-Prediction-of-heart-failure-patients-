import numpy as np
import streamlit as st
import joblib as jb
from PIL import Image

# Creating title
st.title("A Survival Prediction System")
st.text("This system uses five(5) inputs to predicts the survival of heart failure patients")
image = Image.open('new.jpg')
st.image(image, width=700)



# Creating input boxes

age = st.number_input('Age (years)', min_value=0)  # min_value=40, max_value=95)
ej_fr = st.number_input('Ejection Fraction (%)', min_value=0)  # min_value=14, max_value=80)
ser_cr = st.number_input('Serum Creatinine (mg/dl)', min_value=0.5, max_value=9.4)
ser_na = st.number_input('Serum Sodium(mEq/L)', min_value=0)  # min_value=113, max_value=148)
time = st.number_input('Time (days)', min_value=0)

st.write('The user inputs are {}'.format([age, ej_fr, ser_cr, ser_na,time]))


def transform():
    age_n = float(age)
    ej_fr_n = float(ej_fr)
    ser_cr_n = float(ser_cr)
    ser_na_n = float(ser_na)
    patient = [age_n, ej_fr_n, ser_cr_n, ser_na_n, time]
    return patient


def prediction(x):
    loaded_model = jb.load('dissert_model.sav')
    patient_value = np.array(x).reshape(1, -1)
    predict = loaded_model.predict(patient_value)
    return predict


def status():
    retrieved = transform()
    predicted = prediction(retrieved)
    st.subheader("Prediction")
    if predicted == 0:
        st.write("We predict the patient to be: Alive")
    elif predicted == 1:
        st.write(" We predict the patient : Not alive")
    else:
        st.write("Please check your input")


status()
