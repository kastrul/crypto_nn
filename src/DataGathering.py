import json
import os
import urllib.request

import numpy as np
import pandas as pd

from PastSampler import PastSampler


def json_to_df(json_dict):
    """
    Converts a dictionary created from json.loads to a pandas dataframe
    json_dict:      The dictionary
    """
    data_points = len(json_dict)
    columns = []
    if data_points > 0:  # Place the column in sorted order
        columns = sorted(list(json_dict[0].keys()))
    data_frame = pd.DataFrame(columns=columns, index=range(data_points))
    for index in range(data_points):
        for column in columns:
            data_frame.at[index, column] = json_dict[index][column]
    return data_frame


def get_api_url(currency):
    """
    Makes a URL for querying historical prices of a cyrpto from Poloniex
    cur:    3 letter abbreviation for cryptocurrency (BTC, LTC, etc)
    """
    api_site = 'https://poloniex.com/public'
    command = 'returnChartData'
    currency_pair = 'USDT_' + currency
    period = '7200'  # candlestick period in seconds (valid values are 300, 900, 1800, 7200, 14400, and 86400)
    start = '1420070400'
    api_url = '{0}?command={1}&currencyPair={2}&start={3}&end=9999999999&period={4}'.format(api_site, command,
                                                                                            currency_pair, start,
                                                                                            period)
    print('API url: ', api_url)
    return api_url


def get_currency_data_file(currency, file_path):
    """
    currency:    3 letter abbreviation for cryptocurrency (BTC, LTC, etc)
    file_path:     File path (to save price data to CSV)
    """
    open_url = urllib.request.urlopen(get_api_url(currency))
    data_bytes = open_url.read()
    data = json.loads(data_bytes.decode())  # list of price data dictionaries
    data_frame = json_to_df(data)
    data_frame.to_csv(file_path, sep=',')
    return data_frame


def stacked_data_scaling(stacked_data, holdout_period=16):
    # holdout period i.e. number of time units in period
    stacked_data_holdout = stacked_data[0:-holdout_period]
    scale_vector = stacked_data_holdout.mean(axis=0)
    return stacked_data / scale_vector


def get_currency_data(past_samples_n=256, future_samples_n=16, use_old=True, holdout_period=16, ):
    # Path to store cached currency data
    data_path = 'currency_data/'
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # Different cryptocurrency types
    currency_types = ['BTC', 'LTC', 'ETH']
    # Columns of currency's data to use
    price_data = ['close', 'high', 'low', 'open', 'volume']

    # Store data frames for each of above types
    currency_df_list = []
    for currency in currency_types:
        data_file_path = os.path.join('{0}{1}.csv'.format(data_path, currency))
        if use_old:
            try:
                currency_data_frame = pd.read_csv(data_file_path, sep=',')
            except FileNotFoundError:
                currency_data_frame = get_currency_data_file(currency, data_file_path)
        else:
            currency_data_frame = get_currency_data_file(currency, data_file_path)
        currency_df_list.append(currency_data_frame)

    # use the lowest number of data points
    data_rows = min(currency_df.shape[0] for currency_df in currency_df_list)
    for i in range(len(currency_types)):
        index_from = (currency_df_list[i].shape[0] - data_rows)
        currency_df_list[i] = currency_df_list[i][index_from:]

    # Features are channels
    stacked_data = np.hstack((currency_df[price_data] for currency_df in currency_df_list))[:, None, :]

    # -------- SCALING OF DATA --------------
    stacked_data_holdout = stacked_data[0:-holdout_period]
    scale_vector = stacked_data_holdout.mean(axis=0)
    stacked_data /= scale_vector

    # Make samples of temporal sequences of pricing data (channel)
    past_sample = PastSampler(past_samples_n, future_samples_n)
    sample_matrix, target_matrix = past_sample.transform(stacked_data_holdout)

    return sample_matrix, target_matrix, stacked_data_holdout, stacked_data, scale_vector


def main():
    get_currency_data()
    # print(A)


if __name__ == '__main__':
    main()
