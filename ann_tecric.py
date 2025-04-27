import streamlit as st
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

##load the trained model
model =tf.keras.models.load_model("model.h5")

with open ('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender= pickle.load(file)
with open ('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo= pickle.load(file)
with open ('scaler.pkl','rb') as file:
    scaler= pickle.load(file)
#Streamlit
st.title("Fintech Customer Bank Exiting Prediction")

#User Input

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])

gender = st.selectbox("gender",label_encoder_gender.classes_)

age= st.slider("Age", 18,92)

balance= st.number_input('Balance')

credit_score =st.number_input("CreditScore")

estimated_salary = st.number_input("Estimated salary")


tenure= st.slider("Tenure",1,10)

num_of_product= st.slider("NumOfProducts",1,4)

has_cr_card = st.selectbox("Has Credit Card",[0,1])

is_active_member =st.selectbox("Is Act Member",[0,1])

#prepare input data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_product],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary],

})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
#combine onehot encoded columns with input data
input_data= pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)

prediction_proba = prediction[0][0]

st.write(f"Churn probability: {prediction_proba:.2f}")

if prediction_proba> 0.5:
   st.write("the customer is likely to churn")
else:
   st.write("the customer is not likely to churn")