"""
0 Attrition_Flag
1 Customer_Age
2 Gender
3 Dependent_count
4 Education_Level
5 Marital_Status
6 Income_Category
7 Card_Category
8 Months_on_book
9 Total_Relationship_Count
10 Months_Inactive_12_mon
11 Contacts_Count_12_mon
12 Credit_Limit
13 Total_Revolving_Bal
14 Avg_Open_To_Buy
15 Total_Amt_Chng_Q4_Q1
16 Total_Trans_Amt
17 Total_Trans_Ct
18 Total_Ct_Chng_Q4_Q1
19 Avg_Utilization_Ratio
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import streamlit as st
import plotly.express as px

# st.set_page_config(layout="wide")
@st.cache # Even if it works, the effect is miniscule
def load_data():
    data_raw = pd.read_csv('data/BankChurners.csv')
    dataset = data_raw.copy()
    unused_columns = ['CLIENTNUM',
                    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']
    dataset = dataset.drop(unused_columns,axis=1)
    return dataset
dataset = load_data()

#labels sorted to be easily pickable as Continuous variables for y-axis and Discrete or Categorical for x-axis
x_labels = ['Total_Trans_Ct', 'Card_Category', 'Customer_Age', 'Dependent_count', 'Education_Level', 'Gender', 'Income_Category', 'Marital_Status',  'Months_on_book', 'Months_Inactive_12_mon',  'Contacts_Count_12_mon', 'Total_Revolving_Bal', 'Total_Trans_Amt',  'Credit_Limit', 'Total_Relationship_Count', 'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1', 'Avg_Open_To_Buy', 'Avg_Utilization_Ratio']

y_labels = ['Total_Trans_Amt', 'Total_Revolving_Bal', 'Credit_Limit',  'Total_Trans_Ct', 'Total_Relationship_Count', 'Avg_Open_To_Buy', 'Avg_Utilization_Ratio', 'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1', 'Card_Category', 'Customer_Age', 'Dependent_count', 'Education_Level', 'Gender', 'Income_Category', 'Marital_Status',  'Months_on_book', 'Months_Inactive_12_mon',  'Contacts_Count_12_mon']

st.title('Babuchka churn prediction model')
st.header('Exploratory data analysis')
# left, right = st.columns(2) # columns really are not working well without tuning
# right1, right2 = right.columns(2) # no nested columns => SAD
st.markdown('Here you may see an original dataset and explore it by choosing features to plot. Labels are sorted to be easily pickable as Continuous variables for the y-axis and Discrete or Categorical for the x-axis. Below you may check the data in a table view.')
x_select = st.selectbox('Select a feature for X axis', x_labels, key='x_select')
y_select = st.selectbox('Select a feature for Y axis', y_labels, key='y_select')
fig_plotpy = px.scatter(x=dataset[x_select],y=dataset[y_select],color=dataset['Attrition_Flag'],labels={'x':x_select, 'y':y_select})
st.plotly_chart(fig_plotpy)

if st.checkbox('Show the whole dataframe'):
    data_load_state = st.text('Loading data...')
    st.dataframe(dataset)
    # data_load_state.text('Loading data...done!')
    data_load_state.text('')


#expander = right.expander("FAQ")
#expander.write("Here you could put in some really, really long explanations...")

