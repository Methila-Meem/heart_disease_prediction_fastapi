Live Render deployment URL: https://heart-disease-prediction-fastapi.onrender.com/docs

This is an application for predicting heart disease using K-Nearest Neighbors classifire model. 
Sample data for testing the app: 
1. Input-
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}

Expected output-
{
  "heart_disease": true,
  "probability": 1
}

2. Input- 
{
  "age": 53,
  "sex": 1,
  "cp": 0,
  "trestbps": 140,
  "chol": 203,
  "fbs": 1,
  "restecg": 0,
  "thalach": 155,
  "exang": 1,
  "oldpeak": 3.1,
  "slope": 0,
  "ca": 0,
  "thal": 3
}

Expected output-
{
  "heart_disease": false,
  "probability": 0
}
