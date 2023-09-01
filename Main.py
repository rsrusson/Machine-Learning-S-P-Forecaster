import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import pandas_datareader.data as web
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

current_time = datetime.now()
turnover_time = current_time.replace(hour=16, minute=0, second=0, microsecond=0)

start = datetime(2000, 1, 1)
end = datetime.today()
start_training = datetime(2000,1,3)
end_one_day = end - timedelta(days=2)
end_one_month = end - timedelta(days=31)
end_three_month = end - timedelta(days=93)
end_six_month = end - timedelta(days=184)
end_twelve_month = end - timedelta(days=366)

ind_column_names = ['GDP', 'UNRATE', 'CPIAUCNS_x', 'FEDFUNDS', 'CPIAUCNS_y', 'PPIACO',
       'RSAFS', 'Open', 'Close', 'Volume']
corr_column_names = ['GDP', 'CPIAUCNS_y', 'PPIACO', 'RSAFS', 'Open', 'Close']

gdp = web.DataReader("GDP", "fred", start, end)
gdp_df = pd.DataFrame(gdp)
unemployment = web.DataReader("UNRATE", "fred", start, end)
unemployment_df = pd.DataFrame(unemployment)
inflation = web.DataReader("CPIAUCNS", "fred", start, end)
inflation_df = pd.DataFrame(inflation)
interest_rates = web.DataReader("FEDFUNDS", "fred", start, end)
interest_rates_df = pd.DataFrame(interest_rates)
cpi = web.DataReader("CPIAUCNS", "fred", start, end)
cpi_df = pd.DataFrame(cpi)
ppi = web.DataReader("PPIACO", "fred", start, end)
ppi_df = pd.DataFrame(ppi)
retail_sales = web.DataReader("RSAFS", "fred", start, end)
retail_sales_df = pd.DataFrame(retail_sales)

ticker = "SPY"
spy_data = yf.download(ticker, start=start, end=end)
spy_df = pd.DataFrame(spy_data)

all_df = pd.merge(gdp_df, unemployment_df, left_index=True, right_index=True, how='outer')
all_df = pd.merge(all_df, inflation_df, left_index=True, right_index=True, how='outer')
all_df = pd.merge(all_df, interest_rates_df, left_index=True, right_index=True, how='outer')
all_df = pd.merge(all_df, cpi_df, left_index=True, right_index=True, how='outer')
all_df = pd.merge(all_df, ppi_df, left_index=True, right_index=True, how='outer')
all_df = pd.merge(all_df, retail_sales_df, left_index=True, right_index=True, how='outer')
all_df = pd.merge(all_df, spy_df, left_index=True, right_index=True, how='outer')
all_df.drop('High', axis=1, inplace=True)
all_df.drop('Low', axis=1, inplace=True)
all_df.drop('Adj Close', axis=1, inplace=True)
all_df.fillna(method='ffill', inplace=True)
all_df['Next Day\'s Close'] = all_df['Close'].shift(-1)
all_df['Next Month\'s Close'] = all_df['Close'].shift(-22)
all_df['Three Month\'s Close'] = all_df['Close'].shift(-64)
all_df['Six Month\'s Close'] = all_df['Close'].shift(-129)
all_df['Twelve Month\'s Close'] = all_df['Close'].shift(-255)
all_df.index.name = 'Date'

dep_column_names = ['Next Day\'s Close', 'Next Month\'s Close', 'Three Month\'s Close', 'Six Month\'s Close', 'Twelve Month\'s Close']

ind_df = all_df[ind_column_names]
ind_df = ind_df[start_training:end_one_day]
corr_ind_df = all_df[corr_column_names]
corr_ind_df = corr_ind_df[start_training:end_one_day]

today = corr_ind_df.iloc[-1].values.reshape(1, -1)

st.sidebar.header('User Input for S&P Forecast')

with st.sidebar.form(key='Input'):
    today_data = all_df.iloc[-1]
    today_gdp = today_data['GDP']
    today_cpi = today_data['CPIAUCNS_y']
    today_ppi = today_data['PPIACO']
    today_retail_sales = today_data['RSAFS']
    today_open = today_data['Open']
    today_close = today_data['Close']
    gdp_input = st.sidebar.number_input(f'US Gross Domestic Product Ex: {today_gdp}')
    #unemployment_input = st.sidebar.number_input('US Unemployment Rate')
    cpi_input = st.sidebar.number_input(f'US Consumer Price Index Ex: {today_cpi}')
    #fed_funds_input = st.sidebar.number_input('US Federal Fund Interest Rate')
    ppi_input = st.sidebar.number_input(f'Producer Price Index Ex: {today_ppi}')
    retail_sales_input = st.sidebar.number_input(f'US Retail Sales Ex: {today_retail_sales}')
    prev_open_input = st.sidebar.number_input(f'Previous S&P 500 Open Ex: {today_open}')
    prev_close_input = st.sidebar.number_input(f'Previous S&P 500 Close Ex: {today_close}')
    #volume_input = st.sidebar.number_input('Previous S&P 500 Volume')
    forecast_button = st.form_submit_button(label='Forecast')


def user_input():
    data = {
        'GDP': [gdp_input],
        # 'UNRATE': [unemployment_input],
        # 'CPIAUCNS_x': [cpi_input],
        # 'FEDFUNDS': [fed_funds_input],
        'CPIAUCNS_y': [cpi_input],
        'PPIACO': [ppi_input],
        'RSAFS': [retail_sales_input],
        'Open': [prev_open_input],
        'Close': [prev_close_input],
        # 'Volume': [volume_input]
    }
    data_df = pd.DataFrame(data)
    return data_df


input_df = user_input()


def separate_dfs(start, end, df, new_y_column):
    this_df = df[start:end]
    all_columns = corr_column_names + [new_y_column]
    this_df = this_df[all_columns]
    this_df_X = this_df[corr_column_names]
    this_df_y = this_df[[new_y_column]]
    return this_df_X, this_df_y, this_df


one_day_X, one_day_y, one_day_df = separate_dfs(start_training, end_one_day, all_df, 'Next Day\'s Close')
one_month_X, one_month_y, one_month_df = separate_dfs(start_training, end_one_month, all_df, 'Next Month\'s Close')
three_month_X, three_month_y, three_month_df = separate_dfs(start_training, end_three_month, all_df, 'Three Month\'s Close')
six_month_X, six_month_y, six_month_df = separate_dfs(start_training, end_six_month, all_df, 'Six Month\'s Close')
twelve_month_X, twelve_month_y, twelve_month_df = separate_dfs(start_training, end_twelve_month, all_df, 'Twelve Month\'s Close')

model_one_day = RandomForestRegressor(n_estimators=100, random_state=42)
X_day_train, X_day_test, y_day_train, y_day_test = train_test_split(one_day_X, one_day_y, test_size=0.2, random_state=0)
model_one_day.fit(X_day_train, y_day_train)

model_one_month = RandomForestRegressor(n_estimators=100, random_state=42)
X_month_train, X_month_test, y_month_train, y_month_test = train_test_split(one_month_X, one_month_y, test_size=0.2, random_state=0)
model_one_month.fit(X_month_train, y_month_train)

model_three_month = RandomForestRegressor(n_estimators=100, random_state=42)
X_three_train, X_three_test, y_three_train, y_three_test = train_test_split(three_month_X, three_month_y, test_size=0.2, random_state=0)
model_three_month.fit(X_three_train, y_three_train)

model_six_month = RandomForestRegressor(n_estimators=100, random_state=42)
X_six_train, X_six_test, y_six_train, y_six_test = train_test_split(six_month_X, six_month_y, test_size=0.2, random_state=0)
model_six_month.fit(X_six_train, y_six_train)

model_twelve_month = RandomForestRegressor(n_estimators=100, random_state=42)
X_twelve_train, X_twelve_test, y_twelve_train, y_twelve_test = train_test_split(twelve_month_X, twelve_month_y, test_size=0.2, random_state=0)
model_twelve_month.fit(X_twelve_train, y_twelve_train)

model_list = [model_one_day, model_one_month, model_three_month, model_six_month, model_twelve_month]


def prediction(model, test_data):
    return model.predict(test_data)


def rmse(y_test, y_predicted):
    y_test = y_test.to_numpy().flatten()
    return math.sqrt(mean_squared_error(y_test, y_predicted))


def mae(y_test, y_predicted):
    y_test = y_test.to_numpy().flatten()
    return np.mean(np.abs(y_test - y_predicted))


def mape(y_test, y_predicted):
    y_test = y_test.to_numpy().flatten()
    return np.mean(np.abs((y_test - y_predicted) / y_test)) * 100


def r_squared(y_test, y_predicted):
    return r2_score(y_test, y_predicted)


y_day_predict = prediction(model_one_day, X_day_test)
rmse_day = rmse(y_day_test, y_day_predict)
mae_day = mae(y_day_test, y_day_predict)
mape_day = mape(y_day_test, y_day_predict)
r2_day = r_squared(y_day_test, y_day_predict)

y_month_predict = prediction(model_one_month, X_month_test)
rmse_month = rmse(y_month_test, y_month_predict)
mae_month = mae(y_month_test, y_month_predict)
mape_month = mape(y_month_test, y_month_predict)
r2_month = r_squared(y_month_test, y_month_predict)

y_three_predict = prediction(model_three_month, X_three_test)
rmse_three = rmse(y_three_test, y_three_predict)
mae_three = mae(y_three_test, y_three_predict)
mape_three = mape(y_three_test, y_three_predict)
r2_three = r_squared(y_three_test, y_three_predict)

y_six_predict = prediction(model_six_month, X_six_test)
rmse_six = rmse(y_six_test, y_six_predict)
mae_six = mae(y_six_test, y_six_predict)
mape_six = mape(y_six_test, y_six_predict)
r2_six = r_squared(y_six_test, y_six_predict)

y_twelve_predict = prediction(model_twelve_month, X_twelve_test)
rmse_twelve = rmse(y_twelve_test, y_twelve_predict)
mae_twelve = mae(y_twelve_test, y_twelve_predict)
mape_twelve = mape(y_twelve_test, y_twelve_predict)
r2_twelve = r_squared(y_twelve_test, y_twelve_predict)

day_prediction = model_one_day.predict(today)
month_prediction = model_one_month.predict(today)
three_prediction = model_three_month.predict(today)
six_prediction = model_six_month.predict(today)
twelve_prediction = model_twelve_month.predict(today)

prediction_df = pd.DataFrame({'One Day Prediction': [day_prediction],
                              'One Month Prediction': [month_prediction],
                              'Three Month Prediction': [three_prediction],
                              'Six Month Prediction': [six_prediction],
                              'Twelve Month Prediction': [twelve_prediction]
                              })
prediction_df['Label'] = ['Current Market Prediction: ']
prediction_df.set_index('Label', inplace=True)

input_df = input_df.values.reshape(1, -1)
user_day_prediction = model_one_day.predict(input_df)
user_month_prediction = model_one_month.predict(input_df)
user_three_prediction = model_three_month.predict(input_df)
user_six_prediction = model_six_month.predict(input_df)
user_twelve_prediction = model_twelve_month.predict(input_df)

user_prediction_df = pd.DataFrame({'One Day Prediction': [user_day_prediction],
                                   'One Month Prediction': [user_month_prediction],
                                   'Three Month Prediction': [user_three_prediction],
                                   'Six Month Prediction': [user_six_prediction],
                                   'Twelve Month Prediction': [user_twelve_prediction]
                                   })
user_prediction_df['Label'] = ['User Input Prediction: ']
user_prediction_df.set_index('Label', inplace=True)

error_df = pd.DataFrame({'R Squared Error': [r2_day, r2_month, r2_three, r2_six, r2_twelve],
                         'Mean Absolute Percentage Error': [mape_day, mape_month, mape_three, mape_six, mape_twelve],
                         'Root Mean Squared Error': [rmse_day, rmse_month, rmse_three, rmse_six, rmse_twelve],
                         'Mean Absolute Error': [mae_day, mae_month, mae_three, mae_six, mae_twelve]
                         })
error_df['Prediction Timeline'] = ['One Day Prediction Errors', 'One Month Prediction Errors', 'Three Month Prediction Errors', 'Six Month Prediction Errors', 'Twelve Month Prediction Errors']
error_df.set_index('Prediction Timeline', inplace=True)

importance_2d_array = np.concatenate((model_one_day.feature_importances_.reshape(1, -1),
                                     model_one_month.feature_importances_.reshape(1, -1),
                                     model_three_month.feature_importances_.reshape(1, -1),
                                     model_six_month.feature_importances_.reshape(1, -1),
                                     model_twelve_month.feature_importances_.reshape(1, -1)), axis=0)
importance_df = pd.DataFrame(importance_2d_array)
importance_df['Model Time Frame'] = ['1D', '1M', '3M', '6M', '12M']
importance_df.set_index('Model Time Frame', inplace=True)
importance_df.columns = corr_column_names

fig1, ax = plt.subplots()
for idx, model in enumerate(importance_df.index):
    ax.barh(y=[f"{feature} of {model}" for feature in importance_df.columns],
            width=importance_df.loc[model],
            height=0.4,
            label=model)
ax.set_xlabel('Degree of Importance')
ax.set_title('Independent Variable Importance by Model')
fig1.legend(title='Different Models')


def plot_creator(end_time, day_change, timed_df, prediction_col_name, model, title_name):
    fig_num, sub_name = plt.subplots(figsize=(10, 6))
    short_start = end_time - timedelta(days=day_change)
    short_start = short_start.replace(hour=0, minute=0, second=0, microsecond=0)
    this_df = timed_df[short_start:end_time]
    this_df_y = this_df[prediction_col_name]
    this_df_X = this_df.drop(columns=prediction_col_name)
    sub_name.plot(this_df.index, this_df_y, label='Actual', linewidth=2)
    sub_name.plot(this_df.index, prediction(model, this_df_X), label='Prediction', linewidth=2, linestyle='--')
    sub_name.set_xlabel('Time')
    sub_name.set_ylabel('S&P 500 Price')
    sub_name.set_title(title_name)
    fig_num.legend(title='Actual vs Prediction')
    return fig_num, sub_name


fig2, bx = plot_creator(end_twelve_month, 365, twelve_month_df, 'Twelve Month\'s Close', model_twelve_month, '12 Month Actual vs 12 Month Prediction')
fig3, cx = plot_creator(end_six_month, 365, six_month_df, 'Six Month\'s Close', model_six_month, '6 Month Actual vs 6 Month Prediction')
fig4, dx = plot_creator(end_three_month, 365, three_month_df, 'Three Month\'s Close', model_three_month, '3 Month Actual vs 3 Month Prediction')
fig5, ex = plot_creator(end_one_month, 184, one_month_df, 'Next Month\'s Close', model_one_month, '1 Month Actual vs 1 Month Prediction')
fig6, fx = plot_creator(end_one_day, 92, one_day_df, 'Next Day\'s Close', model_one_day, '1 Day Actual vs 1 Day Prediction')

correlation_twelve = twelve_month_df.corr().loc['Twelve Month\'s Close']
correlation_twelve.drop('Twelve Month\'s Close', inplace=True)
importance_twelve = importance_df.loc['12M']
n_vars = len(corr_column_names)
bar_width = 0.35
index = np.arange(n_vars)
fig7, gx = plt.subplots(figsize=(10, 6))
bar1 = plt.bar(index, correlation_twelve, bar_width, label='Actual Correlation', color='b')
bar2 = plt.bar(index + bar_width, importance_twelve, bar_width, label='Model Importance', color='orange')
gx.set_xlabel('Independent Variables')
gx.set_ylabel('Magnitude')
gx.set_title('Twelve Month Model')
gx.text(0.5, 1.05, 'Actual Correlation vs. Prediction Importance', transform=gx.transAxes, ha='center')
gx.set_xticks(index + bar_width / 2)
gx.set_xticklabels(corr_column_names)
fig7.legend()
fig7.tight_layout()

corr_matrix = all_df.corr()
corr_matrix.drop(dep_column_names, axis=1, inplace=True)
corr_matrix.drop(ind_column_names, axis=0, inplace=True)

fig8, hx = plt.subplots(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
hx.set_title('Correlation Heatmap')
hx.set_xlabel('Independent Variables')
hx.set_ylabel('Dependent Variables')

fig9, ix = plt.subplots(figsize=(12, 6))
sns.heatmap(importance_df, annot=True, cmap='coolwarm', fmt=".2f")
ix.set_title('Importance Heatmap')
ix.set_xlabel('Correlated Independent Variables')
ix.set_ylabel('Dependent Variables')

st.write("""
    # S&P 500 (SPY) Forecaster

    This application predicts the S&P 500 via historical SPY data and USA economic data
""")

st.write(
    '''
    ## Predictions by the Models
    ### User Input Prediction: 
    '''
)
st.write(user_prediction_df)
st.write(
    '### Predictions Based off Current S&P and US Economic Numbers',
    prediction_df,
    '### Errors Rates of Forecasted Variable vs Actual Variable',
    error_df,
    '### Actual Correlations between Independent Variables and Forecasted Variable',
    corr_matrix,
    '### How Much Importance The Model Has Placed on Each Independent Variable',
    importance_df,
 )
st.pyplot(fig8)
st.pyplot(fig9)
st.pyplot(fig7)
#st.pyplot(fig1)
st.pyplot(fig2)
st.pyplot(fig3)
st.pyplot(fig4)
st.pyplot(fig5)
st.pyplot(fig6)
st.write('All Data')
st.write(all_df)
