import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from streamlit_option_menu import option_menu

import base64

def set_png_as_page_bg(png_file):
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_png_as_page_bg('https://raw.githubusercontent.com/Yadav-Roshan/Bank_Marketing/main/Deployment_test/images/background.jpg')

# st.sidebar.selectbox("Navigation Bar", ("Home", "Prediction", "Contribute to Datase", "About US"))
with st.sidebar:
    selected = option_menu(None, ["Home", "Upload",  "Tasks", 'Settings'], 
        icons=['house', 'cloud-upload', "list-task", 'gear'], 
        menu_icon="cast", default_index=0, orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "green"},
        }
    )


logo, titl = st.columns([1, 4])
logo.image('https://raw.githubusercontent.com/Yadav-Roshan/Bank_Marketing/main/Deployment_test/images/bank_logo.png')
original_title = '<p style="font-family:Arial Black; color:#624F82; font-size: 45px; text-align:center">Piggy & Penny Banking Services</p>'
titl.markdown(original_title, unsafe_allow_html=True)

st.markdown("## **Prediction of Telemarketing Campaign**")
st.markdown("#### **Enter these details**")

first, middle, last = st.columns(3)
first.text_input("First Name")
middle.text_input("Middle Name")
last.text_input("Last Name")

age_col, job_col, education_col = st.columns([1.5, 3, 3])
age = age_col.number_input("Age", min_value = 12, max_value = 100, step =1)
job_input = job_col.selectbox("Job", ('', 'Housemaid' ,'Services', 'Admin.', 'Blue-collar', 'Technician', 'Retired', 'Management', 'Unemployed', 'Self-employed', 'Entrepreneur',  'Student'), format_func=lambda x: 'Select Job' if x == '' else x)
education_input = education_col.selectbox("Education", ('', 'Basic Education(4 Years)', 'Basic Education(6 Years)', 'Basic Education(9 Years)', 'High School', 'Professional Course', 'University Degree', 'Illiterate'), format_func=lambda x: 'Select Education' if x == '' else x)


contact_col1, contact_col2 = st.columns(2)
contact_input = contact_col1.selectbox("Contact Type", ("",'Telephone', 'Cellular'), format_func=lambda x: 'Select Communication Type' if x == '' else x)
contact_col2.text_input("Contact Number")

day_col, duration_col, campaign_col = st.columns([2.5,1.2,1.5])
day_of_week_input = day_col.selectbox("Last Day of Contact", ("", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"), format_func=lambda x: 'Select Day' if x == '' else x)
duration = duration_col.number_input("Duration of Call(in min)", min_value = 0.00, step = 0.01)
campaign = campaign_col.number_input("No of Calls for This Campaign", min_value = 0, step = 1)

previous_col, poutcome_col= st.columns([1,1.5])
previous = previous_col.number_input("No of Calls for Previous Campaign", min_value = 0, step = 1)
poutcome_input = poutcome_col.selectbox("Outcome of Previous Campaign", ('', 'Nonexistent','Failure','Success'), format_func=lambda x: 'Select Previous Outcome' if x == '' else x)

emp_var_rate = st.slider("Employment Variation Rate", min_value = -5.0, max_value = 5.0, step = 0.1)
cons_price_idx  = st.slider("Current Consumer Price Index", min_value = 80.00, max_value = 120.00, step = 0.01)

X_train = pd.read_csv("https://raw.githubusercontent.com/Yadav-Roshan/Bank_Marketing/main/Deployment_test/datatsets/model_x.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Yadav-Roshan/Bank_Marketing/main/Deployment_test/datatsets/model_y.csv")
X_train = X_train.set_index('ind')
y_train = y_train.set_index('ind')

model=LogisticRegression()
model.fit(X_train, y_train)

def predict_class():

    education_dict = {'Basic Education(4 Years)':0, 'Basic Education(6 Years)':1, 'Basic Education(9 Years)':2, 'High School':3, 'Professional Course':5, 'University Degree':6, 'Illiterate':4}
    education= education_dict[education_input]
    
    poutcome_dict = {'Nonexistent':1, 'Failure':0, 'Success':2}
    poutcome= poutcome_dict[poutcome_input]
    
    day_of_week_dict = {'Friday':0, 'Monday':1, 'Thursday':2, 'Tuesday':3, 'Wednesday':4, 'Saturday':5, 'Sunday':6}
    day_of_week= day_of_week_dict[day_of_week_input]

    job_dict = {'Admin.':0, 'Blue-collar':1, 'Entrepreneur':2, 'Housemaid':3, 'Management':4, 'Retired':5, 'Self-employed':6, 'Services':7, 'Student':8, 'Technician':9, 'Unemployed':10}
    job = job_dict[job_input]

    contact_dict = {'Cellular':0, 'Telephone':1}
    contact = contact_dict[contact_input]


    if age in (0,30):
        age_bins = 2
    elif age in (30, 60):
        age_bins = 0
    else:
        age_bins = 2

    

    data = list([age_bins, job, education, contact, day_of_week, duration, campaign, previous, poutcome, emp_var_rate, cons_price_idx])
    
    if model.predict([data])[0] == 0:
        result = "will not subscribe this scheme."
        st.error("This customer "+ result)
    else:
        result = "will subscribe this scheme."
        st.success("This customer "+ result)

    probas = model.predict_proba([data])
    probas = probas.T.flatten()*100

    fig1, ax1 = plt.subplots()
    ax1.pie(probas.T, explode=(0.1, 0.2), labels=['Failure', 'Success'], autopct='%1.1f%%',
            shadow=True, startangle=90, colors = ["#c05454", "#7cc95b"])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)






st.markdown("###### To predict whether this customer will be subscribing our scheme or not, click on predict.")
if st.button("Predict"):
    flag = True
    if(job_input == ""):
        job_col.error("Please select any valid job")
        flag = False
    if(education_input == ""):
        education_col.error("Please select any valid education");
        flag = False
    
    if(contact_input == ""):
        contact_col1.error("Please select any valid contact type");
        flag = False
    
    if(day_of_week_input == ""):
        day_col.error("Please select any valid day");
        flag = False
    
    if(poutcome_input == ""):
        poutcome_col.error("Please select any valid outcome");
        flag = False
    if(flag):
        predict_class()
