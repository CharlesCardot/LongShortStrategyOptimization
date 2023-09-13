import json
import time
import requests

import pandas as pd
from datetime import datetime

with open('secrets.json', 'r') as secrets_file:
    secrets = json.load(secrets_file)

# Define the API endpoint and parameters
endpoint = 'https://www.alphavantage.co/query'
params = {
    'function': 'TIME_SERIES_INTRADAY',
    'symbol': 'SPY',
    'interval': '1min',
    'apikey': secrets["alphavantage_apikey"][0],
    'extended_hours': 'true',
    'outputsize': 'full',
    'month': '2023-07'
}

print(secrets["alphavantage_apikey"][0])

# Initialize an empty DataFrame to store the data
full_df = pd.DataFrame()

# Define the date range for data retrieval
# start_date = datetime(2015, 7, 1)
# end_date = datetime(2023, 7, 31)
start_date = datetime(2013, 5, 1)
end_date = datetime(2016, 6, 30)

counter = 0
# Loop through each month within the date range
current_date = start_date
while current_date <= end_date:
    month = current_date.strftime('%Y-%m')
    params['month'] = month

    print("Requesting data for", month)
    
    # Make the API request
    response = requests.get(endpoint, params=params)
    data = response.json()
    
    # Extract the time series data from the response
    time_series = data.get('Time Series (1min)')
    
    if time_series:
        counter = 0

        # Convert the data to a pandas DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Filter data between market hours (9:30 AM to 4:00 PM)
        df = df.between_time('09:30:00', '16:00:00')
        
        # Append the current month's data to the full DataFrame
        full_df = pd.concat([full_df, df])
    
        # Move to the next month
        current_date = current_date.replace(day=1) + pd.DateOffset(months=1)
    
    else:
        counter += 1
        print("API request failed, waiting and retrying")
        time.sleep(40) # Maximum API requests per min have been reached

        if counter > 4:
            try:
                secrets["alphavantage_apikey"] = secrets["alphavantage_apikey"][1:]
                params["apikey"] = secrets["alphavantage_apikey"][0]
            except Exception as e:
                print(e)
                print("Used up all API keys")
                quit()


# Sort the full DataFrame by index (date)
full_df.sort_index(inplace=True)

# Save the full DataFrame to a CSV file
# full_df.to_csv('full_data_07-2015_to_07-2023.csv')
full_df.to_csv('full_data_07-2010_to_06-2016.csv')

# Print the first and last few rows of the full DataFrame
print(full_df.head())
print(full_df.tail())