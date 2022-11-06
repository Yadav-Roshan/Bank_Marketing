import numpy as np
import pandas as pd
import pickle

def predict(arr):
    with open('https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment/model.sav', 'rb') as f:
        model = pickle.load(f)
    
    classes = {0: 'will not subscribe', 1:'will subscribe'}
    
    preds = model.predict_proba([arr])[0]
    return (classes[np.argmax(preds)], preds)
