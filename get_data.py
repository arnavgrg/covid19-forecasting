import pandas as pd
import datetime

# Original training dataset
train_df = pd.read_csv('./data/train_round2.csv')
train_df.drop(columns='ID', inplace=True)

# Set date range
date = datetime.date(2020, 11, 23)
today = datetime.date(2020, 12, 3)
delta = datetime.timedelta(days=1)

# Append new training data
while date <= today:
    repo = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/'
    path = 'master/csse_covid_19_data/csse_covid_19_daily_reports_us/'
    df = pd.read_csv(repo + path + date.strftime('%m-%d-%Y') + '.csv')
    namer = {'Case_Fatality_Ratio': 'Mortality_Rate', 'Total_Test_Results': 'People_Tested'}
    df.rename(columns=namer, inplace=True)
    df['Date'] = date.strftime('%m-%d-%Y')
    df = df[train_df.columns]
    train_df = train_df.append(df)
    date += delta

# Fix IDs and output
train_df.reset_index(drop=True, inplace=True)
train_df.index.names = ['ID']
train_df.to_csv('./data/train_round2_updated.csv')