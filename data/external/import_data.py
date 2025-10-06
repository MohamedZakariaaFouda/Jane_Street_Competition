
#You have two methods to import the data from Kaggle.

'''
The first is to run the code below, which uses 
the kagglehub package to download the data directly into your environment.

The second is to manually download the data from 
the Kaggle website using the provided link.
'''

# RUN THIS CELL IN ORDER TO IMPORT KAGGLE DATA SOURCES.

import kagglehub
kagglehub.login()

jane_street_real_time_market_data_forecasting_path = kagglehub.competition_download('jane-street-real-time-market-data-forecasting')
print('Data source import complete.')


# If the above code does not work, you can download manually from
'''
https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data

'''
