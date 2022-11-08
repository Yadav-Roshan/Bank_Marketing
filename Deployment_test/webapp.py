import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression

X_train = pd.read_csv("https://raw.githubusercontent.com/Yadav-Roshan/Bank_Marketing/main/Deployment_test/datatsets/model_x.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Yadav-Roshan/Bank_Marketing/main/Deployment_test/datatsets/model_y.csv")
X_train = X_train.set_index('ind')
y_train = y_train.set_index('ind')

model=LogisticRegression()
model.fit(X_train, y_train)

if st.button("predict"):
    prediciton =model.predict([[0,9,2,1,2,3.22,1,0,1,1.1,93.994]])
    st.write(prediciton[0)
