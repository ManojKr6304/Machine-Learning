import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import streamlit as s
import pickle


#
model=pickle.load(open(r"C:\Users\Manoj\Desktop\ML (Models)\apple_model.pkl","rb")) #Deserialization
pre_p=pickle.load(open(r"C:\Users\Manoj\Desktop\ML (Models)\apple_pre_p.pkl","rb")) #Deserialization

s.title('Apple Quality Prediction')

s.subheader('Size of Apple')
size=s.slider('Enter Size', -5.0, 5.0, 0.1)

s.subheader('Weight of Apple')
weight=s.slider('Enter Weight', -2.0, 2.0, 0.1)

s.subheader('Sweetness of Apple')
sweetness=s.slider('Enter Sweetness', -5.0, 5.0, 0.1)

s.subheader('Crunchiness of Apple')
crunchiness=s.slider('Enter Crunchiness', -5.0, 5.0, 0.1)

s.subheader('Juiciness of Apple')
juciness=s.slider('Enter Juiciness', -5.0, 5.0, 0.1)

s.subheader('Ripeness of Apple')
ripeness=s.slider('Enter Ripeness', -5.0, 5.0, 0.1)

s.subheader('Acidity of Apple')
acidity=s.slider('Enter Acidity', -5.0, 5.0, 0.1)

quarry=pre_p.transform(pd.DataFrame([[size,weight,sweetness,crunchiness,juciness,ripeness,acidity]], columns = ['Size','Weight','Sweetness','Crunchiness','Juiciness','Ripeness','Acidity']))

predicted_yi=model.predict(quarry)

if predicted_yi==1:
    x='Good'
else:
    x='Bad'

if s.button('Submit'):
    s.write('Your Apple Quality is '+x)
