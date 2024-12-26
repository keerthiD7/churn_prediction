import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
# from keras.models import Sequential
# from keras.layers import Dense
import streamlit as st

# Load dataset
dataset = pd.read_csv("C:\\Users\\keert\\OneDrive\\Desktop\\customer_churn_prediction\\churn_prediction.csv")

# Extracting features and target variable
X = dataset.iloc[:, 2:-2].values
y = dataset.iloc[:, -2].values

print("Shape of X before preprocessing:", X.shape)

# Encoding categorical data
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 3] = le.fit_transform(X[:, 3])

# Handle missing values for categorical columns
imputer_categorical = SimpleImputer(strategy='most_frequent')  # Use 'most_frequent' for categorical columns
X = imputer_categorical.fit_transform(X)

# Handle missing values for numeric columns separately if needed
# numeric_columns = X[:, numeric_column_indices]  # Specify indices for numeric columns
# imputer_numeric = SimpleImputer(strategy='mean')
# X[:, numeric_column_indices] = imputer_numeric.fit_transform(numeric_columns)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)
y_pred1 = clf.predict(X_test)
print("Accuracy of logistic regression:", accuracy_score(y_test, y_pred1))

# K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier_KNN.fit(X_train, y_train)
knn_pred = classifier_KNN.predict(X_test)
print("Accuracy of KNN:", accuracy_score(y_test, knn_pred))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier_naive_bayes = GaussianNB()
classifier_naive_bayes.fit(X_train, y_train)
gau_pred = classifier_naive_bayes.predict(X_test)
print("Accuracy of Gaussian Naive Bayes:", accuracy_score(y_test, gau_pred))

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators=100, criterion='entropy')
classifier_RF.fit(X_train, y_train)
fore_pred = classifier_RF.predict(X_test)
print("Accuracy of Random Forest:", accuracy_score(y_test, fore_pred))

# Streamlit web application
ques = st.slider("Select your age", min_value=18, max_value=90)
ques1 = st.radio("Select Gender", ("MALE", "FEMALE"))
ques1 = 1 if ques1 == "MALE" else 0
ques2 = st.slider("Select number of dependents", min_value=0, max_value=10)
ques3 = st.radio("Select your occupation", ("Self-employed", "Salaried", "Retired", "Student"))
ques3 = ["Self-employed", "Salaried", "Retired", "Student"].index(ques3)
ques4 = st.text_input("Enter the city code")
ques5 = st.text_input("Enter the customer category")
ques6 = st.text_input("Enter the branch code")
ques7 = st.text_input("Enter current_balance")
ques8 = st.text_input("Enter previous_month_end_balance")
ques9 = st.text_input("Enter average_monthly_balance_prevQ")
quesex = st.text_input("Enter average_monthly_balance_prevQ2")
ques10 = st.text_input("Enter current_month_credit")
ques11 = st.text_input("Enter previous_month_credit")
ques12 = st.text_input("Enter current_month_debit")
ques13 = st.text_input("Enter previous_month_debit")
ques14 = st.text_input("Enter current_month_balance")
ques15 = st.text_input("Enter previous_month_balance")

if st.button("Predict"):
    prd = classifier_KNN.predict([[float(ques), float(ques1), float(ques2), float(ques3),
                                    float(ques4), float(ques5), float(ques6), float(ques7),
                                    float(ques8), float(ques9), float(quesex), float(ques10),
                                    float(ques11), float(ques12), float(ques13), float(ques14), 
                                    float(ques15)]])
    if prd == 1:
        st.write("Customer churn is positive")
    else:
        st.write("Customer churn is negative")
