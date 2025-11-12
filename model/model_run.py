import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import os, joblib

#Load dataset
data = pd.read_csv(r'data\heart.csv')
X = data.drop('target', axis=1)
y = data['target']

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Applying feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for better readability
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train_scaled, y_train)

#Save the model to a joblib file
os.makedirs("model", exist_ok=True)

artifacts = {
    "model": model_knn,                   # your trained classifier
    "scaler": scaler,                     # fitted StandardScaler instance
    "feature_names": list(X_train.columns)  # preserve training column order
}

joblib.dump(artifacts, "model/heart_model.joblib")

print("Model trained and saved successfully.")