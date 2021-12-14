from autotimeseries.core import AutoTS
import streamlit as st


'''
# Time Series Forecasting at Scale by Nixtla

:fire::fire::fire: With this App you can  easly build you own forecasts at scale in under 5 minutes leveragin the power of [Nixtla](https://github.com/Nixtla/nixtla) :fire::fire::fire:

You can use our fully hosted version as a service through this App or through [python SDK](https://github.com/Nixtla/nixtla/tree/main/sdk/) ([autotimeseries](https://pypi.org/project/autotimeseries/)). 
To consume the APIs on our own infrastructure just request tokens by sending an email to federico@nixtla.io or opening a GitHub issue. 
**We currently have free resources available for anyone interested.**

'''



BUCKET_NAME = st.sidebar.text_input('Enter bucket name', type='password', value="TEST", help="Type your bla bla bla")
API_ID = st.sidebar.text_input('Enter API_ID', type='password')
API_KEY = st.sidebar.text_input('Enter API_KEY', type='password')
AWS_ACCESS_KEY_ID = st.sidebar.text_input('Enter AWS_ACCESS_KEY_ID', type='password')
AWS_SECRET_ACCESS_KEY = st.sidebar.text_input('Enter AWS_SECRET_ACCESS_KEY', type='password')

st.subheader('Select what service you want to use')



service = st.selectbox(
    "Services",
    ("Calendar features", "Forecasting")
)

autotimeseries = AutoTS(bucket_name=BUCKET_NAME,
                        api_id=API_ID, 
                        api_key=API_KEY,
                        aws_access_key_id=AWS_ACCESS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

if 'Calendar' in service:
    st.subheader('Add calendar variables to your data')
    
    filename_temporal = st.text_input('Enter temporal file')

    unique_id_column = st.text_input('Enter unique_id column', value='item_id')
    ds_column = st.text_input('Enter date column', value='timestamp')
    y_column = st.text_input('Enter target column', value='demand')

    country = st.text_input('Enter country holidays', value='USA')

    columns = dict(unique_id_column=unique_id_column,
                   ds_column=ds_column,
                   y_column=y_column)

    if st.button('Generate calendar variables'):
        filename_temporal = autotimeseries.upload_to_s3(filename_temporal)
        filename_calendar_holidays = autotimeseries.upload_to_s3(filename_calendar_holidays)
        response_calendar = autotimeseries.calendartsfeatures(filename=filename_temporal,
                                                          country=country,
                                                          events=filename_calendar_holidays,
                                                          **columns)
        st.write(response_calendar)

    st.subheader('Get status')
    
    id_job = st.text_input('Enter id Job')

    if st.button('Get status'):
        status = autotimeseries.get_status(id_job)
        st.write(status)
    
    st.subheader('Download file')
    filename = st.text_input('Enter filename')
    filename_output = st.text_input('Enter filename output')
    
    if st.button('Download data'):
        autotimeseries.download_from_s3(filename=filename,
                                        filename_output=filename_output)
        st.write(f'Data downloaded at {filename_output}')

elif 'Forecast' in service:
    st.subheader('Forecast your data')

    filename_target = st.text_input('Enter target file')
    filename_temporal = st.text_input('Enter temporal file')
    filename_static = st.text_input('Enter static file')

    unique_id_column = st.text_input('Enter unique_id column', value='item_id')
    ds_column = st.text_input('Enter date column', value='timestamp')
    y_column = st.text_input('Enter target column', value='demand')

    columns = dict(unique_id_column=unique_id_column,
                   ds_column=ds_column,
                   y_column=y_column)
    
    freq = st.text_input('Enter frequency of your data', value='D')
    horizon = st.text_input('Enter horizon to forecast', value=28)

    if st.button('Forecast'):
        filename_target = autotimeseries.upload_to_s3(filename_target)
        filename_temporal = autotimeseries.upload_to_s3(filename_temporal)
        filename_static = autotimeseries.upload_to_s3(filename_static)
        response_forecast = autotimeseries.tsforecast(filename_target=filename_target,
                                                      freq=freq,
                                                      horizon=horizon, 
                                                      filename_static=filename_static,
                                                      filename_temporal=filename_temporal,
                                                      objective='tweedie',
                                                      metric='rmse',
                                                      n_estimators=170,
                                                      **columns)
        st.write(response_forecast)

    st.subheader('Get status')
    id_job = st.text_input('Enter id Job')

    if st.button('Get status'):
        status = autotimeseries.get_status(id_job)
        st.write(status)
    
    st.subheader('Download file')
    filename = st.text_input('Enter filename')
    filename_output = st.text_input('Enter filename output')
    
    if st.button('Download data'):
        autotimeseries.download_from_s3(filename=filename,
                                        filename_output=filename_output)
        st.write(f'Data downloaded at {filename_output}')





