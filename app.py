
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
clf = joblib.load('random_forest_model.joblib')

# Load the fitted LabelEncoder
le = joblib.load('label_encoder.joblib')

st.title('Walmart Fraud Detection System')

st.header('Enter Transaction Details:')

amount = st.number_input('Amount', min_value=0.0, value=100.0)
device_id = st.selectbox('Device Type', ['mobile', 'desktop', 'tablet'])
return_count = st.number_input('Return Count', min_value=0, value=0, step=1)
coupon_used = st.radio('Coupon Used', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

if st.button('Predict Fraud'):
    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'Amount': [amount],
        'DeviceID': [device_id],
        'ReturnCount': [return_count],
        'CouponUsed': [coupon_used]
    })

    # Preprocess the 'DeviceID'
    input_data['DeviceID'] = le.transform(input_data['DeviceID'])

    # Make prediction
    prediction_proba = clf.predict_proba(input_data)
    fraud_probability = prediction_proba[0][1] # Probability of class 1 (fraud)

    # Determine the predicted class
    predicted_class = 1 if fraud_probability >= 0.5 else 0

    st.subheader('Prediction Result:')
    if predicted_class == 1:
        st.error(f'Fraudulent Transaction Predicted! Probability: {fraud_probability:.2f}')
    else:
        st.success(f'Transaction is Likely Not Fraudulent. Probability: {fraud_probability:.2f}')