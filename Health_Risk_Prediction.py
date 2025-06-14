import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# 🚀 Load dataset
dataset = pd.read_csv(r"C:\Users\praha\Downloads\synthetic_medical_data.csv")

# 🧹 Drop unnecessary column
dataset = dataset.drop(columns="Patient_ID", errors='ignore')  # ignore just in case it doesn't exist

# 🧼 Fill missing values for numeric columns with mean
for col in dataset.select_dtypes(include='float64').columns:
    dataset[col].fillna(dataset[col].mean(), inplace=True)

# 🧼 Fill missing values for categorical columns with mode
for col in dataset.select_dtypes(include='object').columns:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

# 🧠 Encode categorical variables
dataset = pd.get_dummies(dataset, columns=["Gender", "Diabetes", "Smoker"], drop_first=True)

# 🗺️ Map ordinal values
dataset["Exercise_Level"] = dataset["Exercise_Level"].map({"High": 1, "Medium": 2, "Low": 3})
dataset["Previous_Conditions"] = dataset["Previous_Conditions"].map({
    'Asthma': 0,
    'Hypertension': 1,
    'Cancer': 2,
    'None': 3  # ✅ Added for people with no conditions
})
dataset["Medication"] = dataset["Medication"].map({"A": 1, "B": 2, "C": 3})

# ✅ Final features to be used
final_features = [
    'Age', 'Blood_Pressure', 'Cholesterol', 'Heart_Rate', 'Exercise_Level',
    'BMI', 'Previous_Conditions', 'Medication', 'Gender_Male', 
    'Diabetes_Yes', 'Smoker_Yes'
]

# 📊 Split features and target
x = dataset[final_features]
y = dataset['Target']

# ⚖️ Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 🎯 Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# 🤖 Train the SVM model
model = SVC(C=10, kernel='rbf', gamma='scale', probability=True, random_state=42)
model.fit(x_train, y_train)

# 💾 Save the model
with open("svc_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# 💾 Save the scaler
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("✅ Model and Scaler saved successfully!")
