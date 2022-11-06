import numpy as np
import pandas as pd
import pickle

def predict(arr):
    with open('model.sav', 'rb') as f:
        model = pickle.load(f)
    
    classes = {0: 'will not subscribe', 1:'will subscribe'}
    
    preds = model.predict_proba([arr])[0]
    return (classes[np.argmax(preds)], preds)