import streamlit as st
import pickle
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import plotly.graph_objs as go

lr = pickle.load(open('model_attrition','rb'))
ss = pickle.load(open('standard_scaler','rb'))

def feature_engineering(df):
    df['satisfaction_level_lt_0.47'] = df['satisfaction_level'].copy()
    df['satisfaction_level_btw_0.47_0.72'] = df['satisfaction_level'].copy()
    df['satisfaction_level_gt_0.72'] = df['satisfaction_level'].copy()
    df.loc[df['satisfaction_level_lt_0.47'] >= 0.47, 'satisfaction_level_lt_0.47'] = 0
    df.loc[((df['satisfaction_level_btw_0.47_0.72'] < 0.47) | (df['satisfaction_level_btw_0.47_0.72'] >= 0.72)),
           'satisfaction_level_btw_0.47_0.72'] = 0
    df.loc[df['satisfaction_level_gt_0.72'] < 0.72, 'satisfaction_level_gt_0.72'] = 0
    df.drop(columns=['satisfaction_level'], inplace=True)

    df['last_evaluation_lt_0.57'] = df['last_evaluation'].copy()
    df['last_evaluation_btw_0.57_0.76'] = df['last_evaluation'].copy()
    df['last_evaluation_gt_0.76'] = df['last_evaluation'].copy()
    df.loc[df['last_evaluation_lt_0.57'] >= 0.57, 'last_evaluation_lt_0.57'] = 0
    df.loc[((df['last_evaluation_btw_0.57_0.76'] < 0.57) | (df['last_evaluation_btw_0.57_0.76'] >= 0.72)),
           'last_evaluation_btw_0.57_0.76'] = 0
    df.loc[df['last_evaluation_gt_0.76'] <= 0.76, 'last_evaluation_gt_0.76'] = 0
    df.drop(columns=['last_evaluation'], inplace=True)

    df['number_project_lt_3'] = df['number_project'].copy()
    df['number_project_btw_3_4'] = df['number_project'].copy()
    df['number_project_gt_3'] = df['number_project'].copy()
    df.loc[df['number_project_lt_3'] >= 3, 'number_project_lt_3'] = 0
    df.loc[((df['number_project_btw_3_4'] < 3) | (df['number_project_btw_3_4'] >= 4)),
           'number_project_btw_3_4'] = 0
    df.loc[df['number_project_gt_3'] < 4, 'number_project_gt_3'] = 0
    df.drop(columns=['number_project'], inplace=True)

    df['average_montly_hours_lt_160.5'] = df['average_montly_hours'].copy()
    df['average_montly_hours_btw_160_204'] = df['average_montly_hours'].copy()
    df['average_montly_hours_gt_204'] = df['average_montly_hours'].copy()
    df.loc[df['average_montly_hours_lt_160.5'] >= 160.5, 'average_montly_hours_lt_160.5'] = 0
    df.loc[((df['average_montly_hours_btw_160_204'] < 160.5) | (df['average_montly_hours_btw_160_204'] >= 204.5)),
           'average_montly_hours_btw_160_204'] = 0
    df.loc[df['average_montly_hours_gt_204'] < 204.5, 'average_montly_hours_gt_204'] = 0
    df.drop(columns=['average_montly_hours'], inplace=True)

    df['time_spend_company_lt_3'] = df['time_spend_company'].copy()
    df['time_spend_company_btw_3_6'] = df['time_spend_company'].copy()
    df['time_spend_company_gt_6'] = df['time_spend_company'].copy()
    df.loc[df['time_spend_company_lt_3'] >= 3, 'time_spend_company_lt_3'] = 0
    df.loc[((df['time_spend_company_btw_3_6'] < 3) | (df['time_spend_company_btw_3_6'] >= 6)),
           'time_spend_company_btw_3_6'] = 0
    df.loc[df['time_spend_company_gt_6'] < 6, 'time_spend_company_gt_6'] = 0
    df.drop(columns=['time_spend_company'], inplace=True)

    df.drop(columns=['sales'], inplace=True)
    encoded_data_new = pd.get_dummies(df)
    
    missing_columns = list(set(model_features)-set(list(df.columns)))
    for column in missing_columns:
        df.loc[:, column] = 0
    
    return df[model_features]

model_features = ['Work_accident', 'promotion_last_5years', 'satisfaction_level_lt_0.47',
       'satisfaction_level_btw_0.47_0.72', 'satisfaction_level_gt_0.72',
       'last_evaluation_lt_0.57', 'last_evaluation_btw_0.57_0.76',
       'last_evaluation_gt_0.76', 'number_project_lt_3',
       'number_project_btw_3_4', 'number_project_gt_3',
       'average_montly_hours_lt_160.5', 'average_montly_hours_btw_160_204',
       'average_montly_hours_gt_204', 'time_spend_company_lt_3',
       'time_spend_company_btw_3_6', 'time_spend_company_gt_6', 'salary_high',
       'salary_low', 'salary_medium']


def predict(employee_data):
    df = pd.json_normalize(employee_data)
    df = feature_engineering(df)
    scaled = ss.transform(df)
    y_pred = lr.predict_proba(scaled)
    explainability = pd.DataFrame()
    explainability['feature'] = list(df.columns)
    explainability['importance'] = lr.coef_.reshape(20) * scaled[0]
    return y_pred, explainability, df

    
st.title('Employee Attrition Prediction')
satisfaction_level = st.slider('Satisfaction Level', min_value=0.00, max_value=1.00, step=0.01)
last_evaluation = st.slider('Last Evaluation', min_value=0.00, max_value=1.00, step=0.01)
number_project = st.slider('Number of Projects', min_value=0, max_value=10, step=1)
average_montly_hours = st.slider('Average Monthly Hours', min_value=90, max_value=320, step=1)
time_spend_company = st.slider('Time Spent at Company', min_value=1, max_value=10, step=1)
Work_accident = st.checkbox('Work Accident')
promotion_last_5years = st.checkbox('Promotion in Last 5 years')
sales = st.selectbox('Sales', ['sales', 'accounting', 'hr', 'technical', 'support', 'management',
       'IT', 'product_mng', 'marketing', 'RandD'])
salary = st.radio('Salary', ['low', 'medium', 'high'])
submit = st.button('Predict')
if submit:
    prediction_data = {
       "satisfaction_level":satisfaction_level,
       "last_evaluation":last_evaluation,
       "number_project":number_project,
       "average_montly_hours":average_montly_hours,
       "time_spend_company":time_spend_company,
       "Work_accident":Work_accident,
       "promotion_last_5years":promotion_last_5years,
       "sales":sales,
       "salary":salary
    }
    print(prediction_data)
    y_pred, explainability, processed_df = predict(prediction_data)
    st.text('Attrition Score :' + str(np.round(y_pred[0][1], 2)))
    explainability["Color"] = np.where(explainability['importance'] > 0, 'red', 'green')
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name='Net',
               x=explainability['feature'],
               y=explainability['importance'],
               marker_color=explainability['Color']))
    fig.update_layout(barmode='stack')
    st.plotly_chart(fig)