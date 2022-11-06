import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

from model_methods import predict
classes = {0: 'will not subscribe', 1:'will subscribe'}
class_labels = list(classes.values())

st.title("Prediction of Term Subscription for TeleMarketing Campaign of a Bank")
st.markdown('**Objective** : Given some details and previous records of customer, predicting whether he will subscribe to our new schema or not.')

def predict_class():

    education_dict = {'basic.4y':0,'basic.6y':1,'basic.9y':2,'high.school':3,'illiterate':4, 'professional.course':5, 'university.degree':6}
    education = education_dict[education_input]
    
    poutcome_dict = {'Not Subscribed':0, 'Subscribed':1}
    poutcome = poutcome_dict[poutcome_input]
    
    day_of_week_dict = {'Friday':0, 'Monday':1, 'Thursday':2, 'Tuesday':3, 'Wednesday':4, 'Saturday':5, 'Sunday':6}
    day_of_week = day_of_week_dict[day_of_week_input]
    
    data = list([nr_employed, cons_conf_idx, cons_price_idx, euribor3m,poutcome, emp_var_rate,education,day_of_week,age, campaign])
    
    result, probs = predict(data)
    st.write("The predicted class is ",result)
    probs = [np.round(x,6) for x in probs]
    ax = sns.barplot(probs ,class_labels, palette="winter", orient='h')
    ax.set_yticklabels(class_labels,rotation=0)
    plt.title("Probabilities of the Data belonging to each class")
    for index, value in enumerate(probs):
        plt.text(value, index,str(value))
    st.pyplot()
    
st.markdown("**Please enter the details of the customer**")

age = st.number_input('Enter Age of Customer', step = 1, min_value = 12)

education_input = st.selectbox("Enter Highest Education of Customer", ('basic.4y', 'high.school', 'basic.6y', 'basic.9y', 'professional.course', 'university.degree', 'illiterate'))
st.write('You selected', education_input)

campaign = st.number_input('No. of Contacts Performed', step = 1,min_value=0)

poutcome_input = st.radio("Did he/she subscribe Previous Schema?", key="visibility", options=["Not Subscribed", "Subscribed"])

day_of_week_input = st.selectbox("Enter last day of Contact", ('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'))
st.write("You selected", day_of_week_input)

nr_employed = st.number_input('Number of Employees in the Bank', step =1, min_value = 0)

cons_conf_idx = st.number_input('Consumer Confidence Index', step = 0.1, min_value = 90.0, max_value = 100.0)

cons_price_idx = st.number_input('Consumer Price Index', step = 0.1, min_value = -60.0, max_value = -30.0)

euribor3m = st.number_input('EURIBOR', step = 0.001, min_value = 0.000, max_value = 5.000)

emp_var_rate = st.number_input('Employment Variation Rate', step = 0.1, min_value = -5.0, max_value = 5.0)

if st.button("Predict"):
    predict_class()