import streamlit as st
import pandas as pd
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Load the trained model
model = joblib.load('knn_model.pkl')

# Title of the app
st.title('Heart Attack Prediction')

# User input for features
def user_input_features():
    age = st.number_input('Age', min_value=0, max_value=100, value=50)
    sex = st.selectbox('Sex', (0, 1))  
    st.write(" 0 for female, 1 for male")
    cp = st.selectbox('Chest Pain Type', (0, 1, 2, 3))
   # trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=200, value=120)
    #chol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
    #fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', (0, 1))
    #restecg = st.selectbox('Resting ECG', (0, 1, 2))
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina', (0, 1))
    oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox('Slope', (0, 1, 2))
    ca = st.selectbox('Number of Major Vessels', (0, 1, 2, 3))
    thal = st.selectbox('Thal', (0, 1, 2, 3))

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        #'trestbps': trestbps,
        #'chol': chol,
        #'fbs': fbs,
        #'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user input
st.subheader('User Input features')
st.write(input_df)

# Make predictions
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write('High risk' if prediction[0] == 1 else 'Low risk')

st.subheader('Prediction Probability')
st.write(prediction_proba)
