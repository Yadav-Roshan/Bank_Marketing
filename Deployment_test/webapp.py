import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from streamlit_option_menu import option_menu
import plotly.figure_factory as ff
import plotly.express as px
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Bank Telemarketing Project",
    page_icon="bank2",
    layout="wide"
)
styling = '''<style> div.css-ocqkz7.e1tzin5v4{
        /* background-color: #f5ebe0; */
        border-bottom: 100px;
        border-color: aqua;
        padding: 4px;
        border-radius: 5px;
        /* box-shadow: 100px; */
        /* box-shadow: blueviolet; */
    }
    
    a.nav-link.nav-link-horizontal{
        background-color: none;
        border: 5px 5px 5px 5px;
    }


    span.css-10trblm.e16nr0p30{
        background-color: #ce4257   ;
        border: 2px aquamarine;
        border-radius: 1px;
        box-shadow: 10px black;
        text-align: center;
        color: white;
        text-align: center;
        padding: 5px;
    }

    button.css-1x8cf1d.edgvbvh10 {
        background-color: crimson;
        color:cornsilk;
        border-radius:0;
        padding: 5px;
        font-size: 20px;
        width: 230px;
        font-family: Verdana, Geneva, Tahoma, sans-serif;
        -webkit-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);-moz-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);box-shadow:20px 28px 16px -26px rgba(0, 0, 0);
    }
    button.css-1x8cf1d.edgvbvh10:hover{
        background-color: #bc3908;
        color: white;
        /* width: 200px; */
        height: 150%;
    } </style>'''

st.markdown(styling, unsafe_allow_html = True)

# with open(r"https://raw.githubusercontent.com/Yadav-Roshan/Bank_Marketing/main/Deployment_test/style.css") as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

c_df = pd.read_csv(r"https://raw.githubusercontent.com/Yadav-Roshan/Bank_Marketing/main/Deployment_test/datatsets/cleaned_df.csv")
X_train = pd.read_csv("https://raw.githubusercontent.com/Yadav-Roshan/Bank_Marketing/main/Deployment_test/datatsets/model_x.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Yadav-Roshan/Bank_Marketing/main/Deployment_test/datatsets/model_y.csv")
X_train = X_train.set_index('ind')
y_train = y_train.set_index('ind')

with st.container():
    # logo, titl = st.columns([1, 4])
    # logo.image('https://raw.githubusercontent.com/Yadav-Roshan/Bank_Marketing/main/Deployment_test/images/bank_logo.png', width=140)
    # original_title = '<p style="font-family:Arial Black; color:#e76f51; font-size: 60px; text-align:left, text-decoration: solid">Piggy & Penny Banking Services</p>'
    # titl.markdown(original_title, unsafe_allow_html=True)
    st.markdown('''<center><img  align="middle" src = "https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/header3.jpg?raw=true" width = 100% height = auto></center>''', unsafe_allow_html=True)

selected = option_menu(None, ["Dashboard", "Dataset", "Prediction",  "Analysis", "Contribute", 'About Us'], 
    icons=["", 'house', 'gear-wide-connected',"bar-chart-line" ,"cloud-arrow-up", 'person-rolodex'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#9d0208", "max-width": "100%"},
        "icon": {"color": "orange", "font-size": "18px"}, 
        "nav-link": {"color":"white", "font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#780000", "border" : "2px #fb6f92"},
        "nav-link-selected": {"background-color": "#370617"},
    }
)

if(selected == "Dataset"):
    with st.container():
        st.markdown('''<center><img src = "https://raw.githubusercontent.com/Yadav-Roshan/Bank_Marketing/main/Deployment_test/images/homepage.webp" width=80% style = "border:2px #00bbf9; -webkit-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);-moz-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);box-shadow:20px 28px 16px -26px rgba(0, 0, 0)"></center>''', unsafe_allow_html = True)
#         f_img = st.image(r"https://raw.githubusercontent.com/Yadav-Roshan/Bank_Marketing/main/Deployment_test/images/homepage.webp", width = 1200)
    
    with st.container():
        st.header("About Dataset")
        _, motivation_col, _ = st.columns([4, 2, 4])
        motivation_col.image(r"https://previews.123rf.com/images/outchill/outchill2106/outchill210601273/169905108-motivation-text-written-on-red-round-vintage-rubber-stamp-.jpg", width = 200)      

        motivation = '''
        <p style = "-webkit-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);-moz-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);box-shadow:20px 28px 16px -26px rgba(0, 0, 0);font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#c9184a; border:2px #b7b7a4; padding: 90px; font-size:20px; border-radius:2px; text-align:justify">
            A retail bank has a customer call centre as one of its units, through which the bank communicates
            with potential new clients and offers term deposits. Term deposits are defined as a fixed-term
            investment that includes the deposit of money into an account at a financial institution.
            <br>
            Such an instrument would generate revenue for the bank, hence the bank records the outcomes of
            these phone calls along side other data related to the person being called, the economic indicators
            and certain parameters of the previous contact with the given person. The motivation behind the
            project is clear, by analysing previous phone calls the bank would like to improve its telemarketing
            results in two dimensions:
            <br>
                &ensp;1.&ensp;The efficiency dimension, or in other words how to reduce the number of phone calls the
                bank is performing and therefore reduce the costs &emsp;&emsp;associated with telemarketing;<br>
                &ensp;2. The effectiveness dimension, or in other words how to potentially improve the result and get
                more clients or at least the same number to &emsp;&emsp;deposit their money with our bank.
        </p>
        '''
        st.markdown(motivation, unsafe_allow_html=True)

        _, challenge_col, _ = st.columns([4, 2, 4])
        challenge_col.image(r"https://previews.123rf.com/images/ionutparvu/ionutparvu1612/ionutparvu161202091/67603294-challenge-stamp-sign-text-word-logo-red-.jpg", width = 200)

        challenge = '''
        <p style = "-webkit-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);-moz-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);box-shadow:20px 28px 16px -26px rgba(0, 0, 0);font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#9d0208; border:2px #b7b7a4; padding: 80px; font-size:20px; border-radius:2px; text-align:justify">
                Can we develop a data-driven approach to help the bank increase its success rate of telemarketing
                while incorporating the economic context? Several other questions can be raised at this point:
                <br>
                    &ensp;1.&ensp;How did the economic crisis affect consumer behaviour and how did it manifest itself in the
                    data?<br>
                    &ensp;2.&ensp;How does one’s education, marital status, job, etc. affect their economic choices?<br>
                    &ensp;3.&ensp;Do people prefer being called on the mobile phone or landline?<br>
                    &ensp;4.&ensp;Does a predictive model exist that can predict a telemarketing outcome using client and
                    economic data
        </p>
        '''
        st.markdown(challenge, unsafe_allow_html=True)

        _, desc_col, _ = st.columns([4, 2, 4])
        desc_col.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/resolve.jpg?raw=true", width = 200)

        description = '''
        <p style = "-webkit-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);-moz-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);box-shadow:20px 28px 16px -26px rgba(0, 0, 0);font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#f72585; border:2px #b7b7a4; padding: 80px; font-size:20px; border-radius:2px; text-align:justify">
                The bank marketing data set was collected by Moro, Cortez, and Rita (2014) and ordered by date
                ranging from May 2008 to November 2010. The data was ordered by date even though its year
                has to be inferred manually. The input features were categorized and described as follows:
                <br>
                <span style = "font-family: Sans-serif; font-size: 20px; background-color: #d8e2dc; color:#c1121f; text-decoration:solid; padding: 8px; border-radius: 5px">A. Bank Client Data</span><br>
                <span style = "color: #ede0d4">
                    &ensp;&emsp;&emsp;age | age <br>
                    &ensp;&emsp;&emsp;job | type of job <br>
                    &ensp;&emsp;&emsp;marital | marital status <br>
                    &ensp;&emsp;&emsp;education | education level <br>
                    &ensp;&emsp;&emsp;default | has credit in default? <br>
                    &ensp;&emsp;&emsp;housing | has housing loan? <br>
                    &ensp;&emsp;&emsp;loan | has personal loan? <br><br>
                </span>
                <span style = "font-family: Sans-serif; font-size: 20px; background-color: #d8e2dc; color:#c1121f; text-decoration:solid; padding: 8px; border-radius: 5px">B. Related with the last contact of the Current Campaign</span><br>
                <span style = "color: #ede0d4">
                    &ensp;&emsp;&emsp;contact | contact communication type<br>
                    &ensp;&emsp;&emsp;month | last contact month of year<br>
                    &ensp;&emsp;&emsp;day_of_week | last contact day of the week<br>
                    &ensp;&emsp;&emsp;duration | last contact duration, in seconds<br><br>
                </span>
                <span style = "font-family: Sans-serif; font-size: 20px; background-color: #d8e2dc; color:#c1121f; text-decoration:solid; padding: 8px; border-radius: 5px">C. Other Attributes</span><br>
                <span style = "color: #ede0d4">
                    &ensp;&emsp;&emsp;campaign | number of contacts performed during this campaign and for this client<br>
                    &ensp;&emsp;&emsp;pdays | number of days that passed by after the client was last contacted from a previous campaign<br>
                    &ensp;&emsp;&emsp;previous | number of contacts performed before this campaign and for this client<br>
                    &ensp;&emsp;&emsp;poutcome | outcome of the previous marketing campaign<br><br>
                </span>
                <span style = "font-family: Sans-serif; font-size: 20px; background-color: #d8e2dc; color:#c1121f; text-decoration:solid; padding: 8px; border-radius: 5px">D. Social & Economic Context Attributes</span><br>
                <span style = "color: #ede0d4">
                    &ensp;&emsp;&emsp;emp.var.rate | employment variation rate, quarterly indicator<br>
                    &ensp;&emsp;&emsp;cons.price.idx | consumer price index, monthly indicator<br>
                </span>
        </p>
        '''
        st.markdown(description, unsafe_allow_html=True)

if(selected == "Prediction"):
    with st.container():
        
        st.markdown('''
            <style>
            div.css-cqussq.e1tzin5v0{
            background-image: url("https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/prediction_bg.jpg?raw=true");
            background-size: cover;
            }
            </style>
            ''', unsafe_allow_html=True)
        st.markdown("## **Prediction of Telemarketing Campaign**")
        st.markdown("#### **Enter these details**")

        _, first, middle, last, _ = st.columns([2.5, 4, 4, 4, 2.5])
        first.text_input("First Name")
        middle.text_input("Middle Name")
        last.text_input("Last Name")

        _, age_col, job_col, education_col, _ = st.columns([1.5, 1.5, 3, 3, 1.5])
        age = age_col.number_input("Age", min_value = 12, max_value = 100, step =1)
        job_input = job_col.selectbox("Job", ('', 'Housemaid' ,'Services', 'Admin.', 'Blue-collar', 'Technician', 'Retired', 'Management', 'Unemployed', 'Self-employed', 'Entrepreneur',  'Student'), format_func=lambda x: 'Select Job' if x == '' else x)
        education_input = education_col.selectbox("Education", ('', 'Basic Education(4 Years)', 'Basic Education(6 Years)', 'Basic Education(9 Years)', 'High School', 'Professional Course', 'University Degree', 'Illiterate'), format_func=lambda x: 'Select Education' if x == '' else x)


        _, contact_col1, contact_col2, _ = st.columns([1.58, 4, 4, 1.56])
        contact_input = contact_col1.selectbox("Contact Type", ("",'Telephone', 'Cellular'), format_func=lambda x: 'Select Communication Type' if x == '' else x)
        contact_col2.text_input("Contact Number")

        _, day_col, duration_col, campaign_col, _ = st.columns([1.05, 2.5,1.2,1.5, 1.05])
        day_of_week_input = day_col.selectbox("Last Day of Contact", ("", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"), format_func=lambda x: 'Select Day' if x == '' else x)
        duration = duration_col.number_input("Duration of Call(in min)", min_value = 0.00, step = 0.01)
        campaign = campaign_col.number_input("No of Calls for This Campaign", min_value = 0, step = 1)

        _, previous_col, poutcome_col, _= st.columns([0.48, 1,1.5, 0.48])
        previous = previous_col.number_input("No of Calls for Previous Campaign", min_value = 0, step = 1)
        poutcome_input = poutcome_col.selectbox("Outcome of Previous Campaign", ('', 'Nonexistent','Failure','Success'), format_func=lambda x: 'Select Previous Outcome' if x == '' else x)

        _, emp, _ = st.columns([1, 5, 1])
        emp_var_rate = emp.slider("Employment Variation Rate", min_value = -5.0, max_value = 5.0, step = 0.1, value = 4.54)

        _, cons, _ = st.columns([1, 5, 1])
        cons_price_idx  = cons.slider("Current Consumer Price Index", min_value = 80.00, max_value = 120.00, step = 0.01, value = 116.74)

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
                
            categorical_cols_2 = list(c_df.select_dtypes(include = ['object', 'category']))
            categorical_dummies = pd.get_dummies(c_df[categorical_cols_2])
            df2 = pd.concat([c_df, categorical_dummies], axis =1)
            df2 = df2.drop(categorical_cols, axis = 1)
            X = df2.iloc[:, :-1]
            y = df2.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
            model=LogisticRegression()
            model.fit(X_train, y_train)

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
            _, output, _ = st.columns([2, 8, 2])
            output.pyplot(fig1)

        st.markdown("###### To predict whether this customer will be subscribing our scheme or not, click on predict.")
        _, but, _ = st.columns([3, 1 ,5])
        if but.button("Predict"):
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

if(selected == 'Analysis'):
    # df = pd.read_csv(r"https://raw.githubusercontent.com/Yadav-Roshan/Bank_Marketing/main/Deployment_test/datatsets/age_df.csv")
    _, first, second, _= st.columns([2, 3, 3, 2])
    first_menu= first.radio(
    "What would you like to analyze?",
    ("Variance in Input Features", "Distribution of Target among Input Features", "Correlation")
    )

    if (first_menu == "Variance in Input Features"):
        second_menu = second.selectbox("Select Input Feature", ("Age", "Job", "Marital", "Education", "Loan", "Previous Outcome", "Contact_Type"))

        st.markdown('''<p style = "text-align : center; font-family: Sans-serif; font-size: 40px; background-color: #ff477e; color: #d0f4de; border-radius: 20%, padding: 15px">Graph or Chart</p>''', unsafe_allow_html=True)
      
        if(second_menu == "Age"):
            # Display graph
            _, age, _ = st.columns([2, 8, 2])
            age.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/age_plot.png?raw=true")
        if(second_menu == "Job"):
            _, job3, _ = st.columns([2, 8, 2])
            job3.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/job_plot.png?raw=true")       
        if(second_menu == "Marital"):
            _, marital, _ = st.columns([2, 8, 2])
            marital.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/marital_plot.png?raw=true")        
        if(second_menu=="Education"):
            _, education, _ = st.columns([2, 8, 2])
            st.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/education_plot.png?raw=true")        
        if(second_menu == "Loan"):
            _, loan, _ = st.columns([2, 8, 2])
            loan.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/loan_plot.png?raw=true")        
        if(second_menu == "Previous Outcome"):
            _, poutcome, _ = st.columns([2, 8, 2])
            poutcome.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/poutcome_plot.png?raw=true", width = 1200)
            st.markdown('''<p style = "text-align : center; font-family: Sans-serif; font-size: 20px; background-color: #d00000; color: #d0f4de; border-radius: 5px, padding: 10px">Non-existent means customer didn't want to disclose the status of previous campaign to us.</p>''', unsafe_allow_html=True)
        if(second_menu == "Contact_Type"):
            _, ctype, _ = st.columns([2, 8, 2])
            ctype.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/communication_plot.png?raw=true")
    
    if(first_menu == "Distribution of Target among Input Features"):
        second_menu = second.selectbox("Select Input Feature", ("Age", "Job", "Marital", "Education", "Loan", "Previous Outcome", "Contact_Type", "Day of Week", "Duration", "No of Contacts Made for Previous Campaign", "Previous Outcome"))
        st.markdown('''<p style = "text-align : center; font-family: Sans-serif; font-size: 40px; background-color: #ff477e; color: #d0f4de; border-radius: 20%, padding: 15px"> Graph or Chart</p>''', unsafe_allow_html=True)
        if(second_menu == "Age"):
            _, age1, _ = st.columns([2, 8, 2])
            # second.markdown('''<span style = "font-family: Sans-serif; font-size: 20px; background-color: #2b9348; color:#ffff3f; text-decoration:solid; padding: 8px; border-radius: 5px">Inference</span><br>''', unsafe_allow_html=True)
            age1.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/age_t.png?raw=true")
            
            infer = st.button("Click for Inference")
            if infer:
                inference_age = '''
                <p style = "font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#148b07; border:2px #b7b7a4; padding: 10px; font-size:20px; border-radius:2px; text-align:justify">
                    Most of our customers are of working age but senior citizens are most likely to say yes to our campaign then young one.
                </p>
                '''
                st.markdown(inference_age, unsafe_allow_html=True)


        if(second_menu == "Job"):
            _, job2, _ = st.columns([2, 8, 2])
            # second.markdown('''<span style = "font-family: Sans-serif; font-size: 20px; background-color: #2b9348; color:#ffff3f; text-decoration:solid; padding: 8px; border-radius: 5px">Inference</span><br>''', unsafe_allow_html=True)
            
            job2.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/job_t.png?raw=true")
            infer = st.button("Click for Inference")
            if infer:
                inference_age = '''
                <p style = "font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#148b07; border:2px #b7b7a4; padding: 10px; font-size:20px; border-radius:2px; text-align:justify">
                    The graph shows the outcome count in each job group, with the light green color denoting negative outcomes and purple color denoting the positive outcomes. There is a large percentage of technicians, blue-collar workers and admins. However, it is students and retired people that are most likely to say “yes” to the long-term deposit.
                </p>
                '''
                st.markdown(inference_age, unsafe_allow_html=True)

        if(second_menu == "Marital"):
            # second.markdown('''<span style = "font-family: Sans-serif; font-size: 20px; background-color: #2b9348; color:#ffff3f; text-decoration:solid; padding: 8px; border-radius: 5px">Inference</span><br>''', unsafe_allow_html=True)
            _, marital2, _ = st.columns([2, 8, 2])
            marital2.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/marital_t.png?raw=true")
            infer = st.button("Click for Inference")
            if infer:
                inference_age = '''
                <p style = "font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#148b07; border:2px #b7b7a4; padding: 10px; font-size:20px; border-radius:2px; text-align:justify">
                    Majority of the people have marital status as married or single.People who are married have subscribed for deposits more than people with any other marital status, they are also the one’s who have turned down the deposits offered by the bank the most.                
                </p>
                '''
                st.markdown(inference_age, unsafe_allow_html=True)

        if(second_menu == "Education"):
            # second.markdown('''<span style = "font-family: Sans-serif; font-size: 20px; background-color: #2b9348; color:#ffff3f; text-decoration:solid; padding: 8px; border-radius: 5px">Inference</span><br>''', unsafe_allow_html=True)
            _, education2, _ = st.columns([2, 8, 2])
            education2.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/education_t.png?raw=true")
            infer = st.button("Click for Inference")
            if infer:
                inference_age = '''
                <p style = "font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#148b07; border:2px #b7b7a4; padding: 10px; font-size:20px; border-radius:2px; text-align:justify">
                    In terms of education, although most people in our data set have above-high-school education, the groups that are most likely to respond positively are the least(basic.4y,high-school) and the most educated(university degree, professional course) as seen in the graph.Therefore as the level of education goes up, the greater the tendency to make the deposit.                
                </p>
                '''
                st.markdown(inference_age, unsafe_allow_html=True)

        if(second_menu == "Loan"):
            # second.markdown('''<span style = "font-family: Sans-serif; font-size: 20px; background-color: #2b9348; color:#ffff3f; text-decoration:solid; padding: 8px; border-radius: 5px">Inference</span><br>''', unsafe_allow_html=True)
            _, loan, _ =st.columns([1, 7, 1])
            loan.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/loan_t.png?raw=true")
            infer = st.button("Click for Inference")
            if infer:
                inference_age = '''
                <p style = "font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#148b07; border:2px #b7b7a4; padding: 10px; font-size:20px; border-radius:2px; text-align:justify">
                    People who do not have a loan are more likely to say yes than the people who do.                
                </p>
                '''
                st.markdown(inference_age, unsafe_allow_html=True)
        
        if(second_menu == "Contact_Type"):
            # second.markdown('''<span style = "font-family: Sans-serif; font-size: 20px; background-color: #2b9348; color:#ffff3f; text-decoration:solid; padding: 8px; border-radius: 5px">Inference</span><br>''', unsafe_allow_html=True)
            _, con, _ =st.columns([1, 7, 1])
            con.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/communication_t.png?raw=true")
            infer = st.button("Click for Inference")
            if infer:
                inference_age = '''
                <p style = "font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#148b07; border:2px #b7b7a4; padding: 10px; font-size:20px; border-radius:2px; text-align:justify">
                    We see a robust positive outcome rate for cellular phone usage.                
                </p>
                '''
                st.markdown(inference_age, unsafe_allow_html=True)

        if(second_menu == "Day of Week"):
            # second.markdown('''<span style = "font-family: Sans-serif; font-size: 20px; background-color: #2b9348; color:#ffff3f; text-decoration:solid; padding: 8px; border-radius: 5px">Inference</span><br>''', unsafe_allow_html=True)
            _, dweek, _ = st.columns([2, 8, 2])
            dweek.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/day_t.png?raw=true")
            infer = st.button("Click for Inference")
            if infer:
                inference_age = '''
                <p style = "font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#148b07; border:2px #b7b7a4; padding: 10px; font-size:20px; border-radius:2px; text-align:justify">
                    Thursday is the most busy day while Friday is the least busy day of the week. <br><br>
                    As we can see clearly from the above plot that, all the days have the similar distribution for both the classes. 17–18% of the customers on any given day refuse to subscribe to a term deposit, and similarly for the customers who agree to subscribe.                
                </p>
                '''
                st.markdown(inference_age, unsafe_allow_html=True)

        if(second_menu == "Duration"):
            # second.markdown('''<span style = "font-family: Sans-serif; font-size: 20px; background-color: #2b9348; color:#ffff3f; text-decoration:solid; padding: 8px; border-radius: 5px">Inference</span><br>''', unsafe_allow_html=True)
            _, duration2, _ = st.columns([2, 8, 2])
            duration2.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/duration_t.png?raw=true")
            infer = st.button("Click for Inference")
            if infer:
                inference_age = '''
                <p style = "font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#148b07; border:2px #b7b7a4; padding: 10px; font-size:20px; border-radius:2px; text-align:justify">
                    The duration of calls averages between 3 and 10 minutes.
                </p>
                '''
                st.markdown(inference_age, unsafe_allow_html=True)

        if(second_menu == "No of Contacts Made for Previous Campaign"):
            # second.markdown('''<span style = "font-family: Sans-serif; font-size: 20px; background-color: #2b9348; color:#ffff3f; text-decoration:solid; padding: 8px; border-radius: 5px">Inference</span><br>''', unsafe_allow_html=True)
            _, nc, _ = st.columns([1, 8, 1])
            nc.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/pre_contact_t.png?raw=true")
            infer = st.button("Click for Inference")
            if infer:
                inference_age = '''
                <p style = "font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#148b07; border:2px #b7b7a4; padding: 10px; font-size:20px; border-radius:2px; text-align:justify">
                    It looks like, during the campaign, most of the clients are new. Similarly, most of the subscribers obtained during the campaign have not been contacted before. 85% of subscribers have been contacted twice or less before the current campaign. It seems like if a client has not subscribed in the past 3 campaigns, including the current one, they are more likely to not subscribe during consecutive campaigns.
                </p>
                '''
                st.markdown(inference_age, unsafe_allow_html=True)

        if(second_menu == "Previous Outcome"):
            # second.markdown('''<span style = "font-family: Sans-serif; font-size: 20px; background-color: #2b9348; color:#ffff3f; text-decoration:solid; padding: 8px; border-radius: 5px">Inference</span><br>''', unsafe_allow_html=True)
            _, poutcome2, _ = st.columns([2, 8, 2])
            poutcome2.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/poutcome_t.png?raw=true")
            infer = st.button("Click for Inference")
            if infer:
                inference_age = '''
                <p style = "font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#148b07; border:2px #b7b7a4; padding: 10px; font-size:20px; border-radius:2px; text-align:justify">
                    For most of the customers, the previous marketing campaign outcome does not exists. It means that most of the customers are new customers who have not been contacted earlier. Also one thing to note here that, for the customers who had a successful outcome from the previous campaign, majority of those customers did subscribe for a term deposit. <b>Therefore, a success in the previous campaign indicates that these people are very likely to buy a product once again.</b>
                </p>
                '''
                st.markdown(inference_age, unsafe_allow_html=True)

    if(first_menu == "Correlation"):
        st.markdown('''<p style = "text-align : center; font-family: Sans-serif; font-size: 40px; background-color: #ff477e; color: #d0f4de; border-radius: 20%, padding: 15px"> Graph or Chart</p>''', unsafe_allow_html=True)

        _, corr, _ = st.columns([2, 8, 2])
        corr.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/correlation.png?raw=true")
        infer = st.button("Click for Inference")
        if infer:
                inference_age = '''
                <p style = "font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#148b07; border:2px #b7b7a4; padding: 10px; font-size:20px; border-radius:2px; text-align:justify">
                    From the above heatmap we can see that there are some numerical features which share a high correlation between them, e.g nr.employed and euribor3m these features share a correlation value of 0.95, and euribor3m and emp.var.rate share a correlation of 0.97, which is very high compared to the other features that we see in the heatmap.
                </p>
                '''
                st.markdown(inference_age, unsafe_allow_html=True)

if(selected == "Contribute"):
    _, uploader, _ = st.columns([2, 8, 2])
    uploader = st.file_uploader("Upload CSV")
    a_df = pd.DataFrame()
    if uploader is not None:
        a_df = pd.read_csv(uploader)
    
    d_df = pd.concat([c_df, a_df])
    c_df = d_df.reset_index(drop = True)
    
    
#     _, contr, _ = st.columns([2, 8, 2])
#     contr.image(r"https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/contribute.jpg?raw=true", width = 1200)

if(selected == "About Us"):
    first, second, third = st.columns([1.5, 2, 1.5])
    sec = '''
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css"/>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css"/>
        <p style = "-webkit-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);-moz-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);box-shadow:20px 28px 16px -26px rgba(0, 0, 0);font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#da2c38; border:2px #b7b7a4; padding: 80px; font-size:20px; border-radius:2px; text-align:center">
            <span><img src = "https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/roshan.png?raw=true" width = 300 height= 300></span><br>
            <span style = "font-size: 25px"><b>Roshan Kumar Yadav</b></span><br>
            <span>2nd Year AIML, SIT, Pune</span><br>
            <span><b>Developer</b></span><br>
            <span style = "font-size: 40px">
                <a href = "https://www.linkedin.com/in/roshan-yadav-942536143"><i class="bi bi-linkedin"></i></a> | 
                <a href = "https://github.com/Yadav-Roshan"><i class="bi bi-github"></i></a> |
                <a href = "https://www.instagram.com/yadavroshan276/"><i class="bi bi-instagram"></i></a>
            </span>
        </p>    
    '''
    second.markdown(sec, unsafe_allow_html=True)

    fir = '''
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css"/>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css"/>
        <p style = "-webkit-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);-moz-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);box-shadow:20px 28px 16px -26px rgba(0, 0, 0);font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#da2c38; border:2px #b7b7a4; padding: 80px; font-size:20px; border-radius:2px; text-align:center">
            <span><img src = "https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/dhvani.png?raw=true" width = 250 height= 250></span><br>
            <span style = "font-size: 25px"><b>Dhvani Parekh</b></span><br>
            <span>2nd Year AIML, SIT, Pune</span><br>
            <span><b>Developer</b></span><br>
            <span style = "font-size: 40px">
                <a href = "https://www.linkedin.com/in/dhvaniparekh2204/"><i class="bi bi-linkedin"></i></a> | 
                <a href = "https://github.com/dhvani2204"><i class="bi bi-github"></i></a> |
                <a href = "https://www.instagram.com/dhvani224/"><i class="bi bi-instagram"></i></a>
            </span>
        </p>    
    '''
    first.markdown(fir, unsafe_allow_html = True)

    thir = '''
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css"/>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css"/>
        <p style = "-webkit-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);-moz-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);box-shadow:20px 28px 16px -26px rgba(0, 0, 0);font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#da2c38; border:2px #b7b7a4; padding: 80px; font-size:20px; border-radius:2px; text-align:center">
            <span><img src = "https://github.com/Yadav-Roshan/Bank_Marketing/blob/main/Deployment_test/images/lob.png?raw=true" width = 250 height= 250></span><br>
            <span style = "font-size: 25px"><b>Joshua Lobo</b></span><br>
            <span>2nd Year AIML, SIT, Pune</span><br>
            <span><b>Developer</b></span><br>
            <span style = "font-size: 40px">
                <a href = "https://www.linkedin.com/in/joshua-lobo-4a7642230"><i class="bi bi-linkedin"></i></a> | 
                <a href = https://github.com/Yadav-Roshan/Bank_Marketing/"><i class="bi bi-github"></i></a> |
                <a href = "https://instagram.com/forever_faster133?igshid=NTA5ZTk1NTc="><i class="bi bi-instagram"></i></a>
            </span>
        </p>    
    '''
    third.markdown(thir, unsafe_allow_html=True)

footer = '''
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css"/>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css"/>
        <p style = "-webkit-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);-moz-box-shadow: 0 28px 16px -26px rgba(0, 0, 0);box-shadow:20px 28px 16px -26px rgba(0, 0, 0);font-family: Sans-serif;color:white; text-decoration:solid; box-shadow:5px;background-color:#3c1642; border:2px #b7b7a4; padding: 20px; font-size:15px; border-radius:2px; text-align:center">
            <span style = "font-size: 15px">
            <b>
             <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-envelope-at" viewBox="0 0 16 16">
                <path d="M2 2a2 2 0 0 0-2 2v8.01A2 2 0 0 0 2 14h5.5a.5.5 0 0 0 0-1H2a1 1 0 0 1-.966-.741l5.64-3.471L8 9.583l7-4.2V8.5a.5.5 0 0 0 1 0V4a2 2 0 0 0-2-2H2Zm3.708 6.208L1 11.105V5.383l4.708 2.825ZM1 4.217V4a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v.217l-7 4.2-7-4.2Z"/>
                <path d="M14.247 14.269c1.01 0 1.587-.857 1.587-2.025v-.21C15.834 10.43 14.64 9 12.52 9h-.035C10.42 9 9 10.36 9 12.432v.214C9 14.82 10.438 16 12.358 16h.044c.594 0 1.018-.074 1.237-.175v-.73c-.245.11-.673.18-1.18.18h-.044c-1.334 0-2.571-.788-2.571-2.655v-.157c0-1.657 1.058-2.724 2.64-2.724h.04c1.535 0 2.484 1.05 2.484 2.326v.118c0 .975-.324 1.39-.639 1.39-.232 0-.41-.148-.41-.42v-2.19h-.906v.569h-.03c-.084-.298-.368-.63-.954-.63-.778 0-1.259.555-1.259 1.4v.528c0 .892.49 1.434 1.26 1.434.471 0 .896-.227 1.014-.643h.043c.118.42.617.648 1.12.648Zm-2.453-1.588v-.227c0-.546.227-.791.573-.791.297 0 .572.192.572.708v.367c0 .573-.253.744-.564.744-.354 0-.581-.215-.581-.8Z"/>
             </svg>
             <a href="mailto:yadavroshan276@gmail.com"> Roshan Kumar Yadav</a>
            </b></span><br>
            <span style = "font-size: 15px">
            <b>
             <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-envelope-at" viewBox="0 0 16 16">
                <path d="M2 2a2 2 0 0 0-2 2v8.01A2 2 0 0 0 2 14h5.5a.5.5 0 0 0 0-1H2a1 1 0 0 1-.966-.741l5.64-3.471L8 9.583l7-4.2V8.5a.5.5 0 0 0 1 0V4a2 2 0 0 0-2-2H2Zm3.708 6.208L1 11.105V5.383l4.708 2.825ZM1 4.217V4a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v.217l-7 4.2-7-4.2Z"/>
                <path d="M14.247 14.269c1.01 0 1.587-.857 1.587-2.025v-.21C15.834 10.43 14.64 9 12.52 9h-.035C10.42 9 9 10.36 9 12.432v.214C9 14.82 10.438 16 12.358 16h.044c.594 0 1.018-.074 1.237-.175v-.73c-.245.11-.673.18-1.18.18h-.044c-1.334 0-2.571-.788-2.571-2.655v-.157c0-1.657 1.058-2.724 2.64-2.724h.04c1.535 0 2.484 1.05 2.484 2.326v.118c0 .975-.324 1.39-.639 1.39-.232 0-.41-.148-.41-.42v-2.19h-.906v.569h-.03c-.084-.298-.368-.63-.954-.63-.778 0-1.259.555-1.259 1.4v.528c0 .892.49 1.434 1.26 1.434.471 0 .896-.227 1.014-.643h.043c.118.42.617.648 1.12.648Zm-2.453-1.588v-.227c0-.546.227-.791.573-.791.297 0 .572.192.572.708v.367c0 .573-.253.744-.564.744-.354 0-.581-.215-.581-.8Z"/>
             </svg>
             <a href="mailto:yadavroshan276@gmail.com"> Dhvani Parekh</a>
            </b></span><br>
            <span style = "font-size: 15px">
            <b>
             <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-envelope-at" viewBox="0 0 16 16">
                <path d="M2 2a2 2 0 0 0-2 2v8.01A2 2 0 0 0 2 14h5.5a.5.5 0 0 0 0-1H2a1 1 0 0 1-.966-.741l5.64-3.471L8 9.583l7-4.2V8.5a.5.5 0 0 0 1 0V4a2 2 0 0 0-2-2H2Zm3.708 6.208L1 11.105V5.383l4.708 2.825ZM1 4.217V4a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v.217l-7 4.2-7-4.2Z"/>
                <path d="M14.247 14.269c1.01 0 1.587-.857 1.587-2.025v-.21C15.834 10.43 14.64 9 12.52 9h-.035C10.42 9 9 10.36 9 12.432v.214C9 14.82 10.438 16 12.358 16h.044c.594 0 1.018-.074 1.237-.175v-.73c-.245.11-.673.18-1.18.18h-.044c-1.334 0-2.571-.788-2.571-2.655v-.157c0-1.657 1.058-2.724 2.64-2.724h.04c1.535 0 2.484 1.05 2.484 2.326v.118c0 .975-.324 1.39-.639 1.39-.232 0-.41-.148-.41-.42v-2.19h-.906v.569h-.03c-.084-.298-.368-.63-.954-.63-.778 0-1.259.555-1.259 1.4v.528c0 .892.49 1.434 1.26 1.434.471 0 .896-.227 1.014-.643h.043c.118.42.617.648 1.12.648Zm-2.453-1.588v-.227c0-.546.227-.791.573-.791.297 0 .572.192.572.708v.367c0 .573-.253.744-.564.744-.354 0-.581-.215-.581-.8Z"/>
             </svg>
             <a href="mailto:yadavroshan276@gmail.com"> Joshua Lobo</a>
            </b></span><br>
            <span style = "font-size: 40px">
                <a href = "https://github.com/Yadav-Roshan/Bank_Marketing/"><i class="bi bi-github"></i></a> |
                <a href = "https://www.instagram.com/yadavroshan276/"><i class="bi bi-instagram"></i></a>
            </span><br>
            <span style = "font-size: 15px"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-c-circle" viewBox="0 0 16 16">
  <path d="M1 8a7 7 0 1 0 14 0A7 7 0 0 0 1 8Zm15 0A8 8 0 1 1 0 8a8 8 0 0 1 16 0ZM8.146 4.992c-1.212 0-1.927.92-1.927 2.502v1.06c0 1.571.703 2.462 1.927 2.462.979 0 1.641-.586 1.729-1.418h1.295v.093c-.1 1.448-1.354 2.467-3.03 2.467-2.091 0-3.269-1.336-3.269-3.603V7.482c0-2.261 1.201-3.638 3.27-3.638 1.681 0 2.935 1.054 3.029 2.572v.088H9.875c-.088-.879-.768-1.512-1.729-1.512Z"/>
</svg> Copyright 2022</span><br>
        </p>    
    '''

st.markdown(footer, unsafe_allow_html = True)
