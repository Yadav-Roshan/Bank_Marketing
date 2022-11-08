import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression

X_train = pd.read_csv("datasets\model_x.csv")
y_train = pd.read_csv("datasets\model_y.csv")

model=LogisticRegression()
model.fit(X_train, y_train)

if st.button("predict"):
    prediciton =model.predict([[ 3.0,  9.0,  1.0,  5.0,0.0,  1.0,  0.0,  1.0, 6.0,  3.0,  2.95,  2.0, 6.01,  0.0,  1.0,  1.1, 9.3994, -3.64, 4.857,  5.19,0.0]])
    st.write(prediciton)
