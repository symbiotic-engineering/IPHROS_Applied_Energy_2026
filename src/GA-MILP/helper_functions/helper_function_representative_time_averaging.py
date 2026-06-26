# -*- coding: utf-8 -*-
"""
Matt Haefner (mwh85)
SEA Lab
Setting up representative days
2/17/25
"""

import pandas as pd
from bs4 import BeautifulSoup
import json
import numpy as np


def rep_weeks_electricity(month_list,hours_idx,days_in_season):
    # NOTE: THIS FUNCTION CALCULATES ELECTRICITY PRICES IN UNITS OF EUROS/MWH (UNIT CONVERSION TO EUROS/KWH HAPPENS IN SIMULATION FUNCTION)
    
    grid_data_df = pd.DataFrame()
    for month in month_list:
        # Read the .txt file with BeautifulSoup
        txt_string = 'grid_data/2023/' + month + '_2023.txt'
        with open(txt_string, 'r') as file:
            soup = BeautifulSoup(file, 'html.parser')
        
        # Extract the text content (assuming the file contains valid JSON data)
        json_text = soup.get_text()
        
        # Parse the JSON content
        data = json.loads(json_text)
        
        # Convert the JSON data into a Pandas DataFrame
        df = pd.DataFrame(data['included'][0]['attributes']['values']) # This is for real-time energy prices. I believe changing the 0 index to a 1 will get spot prices
        
        grid_data_df[month] = df['value']
        
        # For adding the first column, add one nan element to the end so october is ok
        if month == 'january':
            grid_data_df.loc[len(grid_data_df), 'january'] = np.nan
    
    grid_data_np_flat = grid_data_df.to_numpy().transpose().flatten()
    grid_data_np_flat = grid_data_np_flat[~np.isnan(grid_data_np_flat)]
    
    # Going season by season
    prices_winter_end = grid_data_np_flat[hours_idx[0]:hours_idx[1]].reshape((days_in_season[0],24)).transpose() # this flipped indices and transpose business is to get each column to be one day of data
    prices_spring = grid_data_np_flat[hours_idx[1]:hours_idx[2]].reshape((days_in_season[1],24)).transpose()
    prices_summer = grid_data_np_flat[hours_idx[2]:hours_idx[3]].reshape((days_in_season[2],24)).transpose()
    prices_fall = grid_data_np_flat[hours_idx[3]:hours_idx[4]].reshape((days_in_season[3],24)).transpose()
    prices_winter_start = grid_data_np_flat[hours_idx[4]:].reshape((days_in_season[-1],24)).transpose()
    
    prices_winter = np.hstack((prices_winter_start,prices_winter_end))
    
    '''
    # Representative Day Averaging
    prices_spring_rep_day = np.mean(prices_spring,1)
    prices_summer_rep_day = np.mean(prices_summer,1)
    prices_fall_rep_day = np.mean(prices_fall,1)
    prices_winter_rep_day = np.mean(prices_winter,1)
    '''
    
    # Representative Week Averaging
    prices_spring_flat = prices_spring.transpose().flatten()
    prices_summer_flat = prices_summer.transpose().flatten()
    prices_fall_flat = prices_fall.transpose().flatten()
    prices_winter_flat = prices_winter.transpose().flatten()
    
    num_nan_elements_spring_week = (24*7*14) - len(prices_spring_flat)
    num_nan_elements_summer_week = (24*7*14) - len(prices_summer_flat)
    num_nan_elements_fall_week = (24*7*14) - len(prices_fall_flat)
    num_nan_elements_winter_week = (24*7*14) - len(prices_winter_flat)
    
    prices_spring_flat_week_nan = prices_spring_flat
    prices_summer_flat_week_nan = prices_summer_flat
    prices_fall_flat_week_nan = prices_fall_flat
    prices_winter_flat_week_nan = prices_winter_flat
    
    for i in range(num_nan_elements_spring_week):
        prices_spring_flat_week_nan = np.append(prices_spring_flat_week_nan, [np.nan])
    for i in range(num_nan_elements_summer_week):
        prices_summer_flat_week_nan = np.append(prices_summer_flat_week_nan, [np.nan])
    for i in range(num_nan_elements_fall_week):
        prices_fall_flat_week_nan = np.append(prices_fall_flat_week_nan, [np.nan])
    for i in range(num_nan_elements_winter_week):
        prices_winter_flat_week_nan = np.append(prices_winter_flat_week_nan, [np.nan])
        
    prices_spring_rep_week_nan = prices_spring_flat_week_nan.reshape((14,24*7)).transpose()
    prices_summer_rep_week_nan = prices_summer_flat_week_nan.reshape((14,24*7)).transpose()
    prices_fall_rep_week_nan = prices_fall_flat_week_nan.reshape((14,24*7)).transpose()
    prices_winter_rep_week_nan = prices_winter_flat_week_nan.reshape((14,24*7)).transpose()
    
    prices_spring_rep_week = np.nanmean(prices_spring_rep_week_nan, axis=1)
    prices_summer_rep_week = np.nanmean(prices_summer_rep_week_nan, axis=1)
    prices_fall_rep_week = np.nanmean(prices_fall_rep_week_nan, axis=1)
    prices_winter_rep_week = np.nanmean(prices_winter_rep_week_nan, axis=1)
    
    electricity_prices_rep_week = {"spring":prices_spring_rep_week,
                                   "summer":prices_summer_rep_week,
                                   "fall":prices_fall_rep_week,
                                   "winter":prices_winter_rep_week}
    
    '''
    # Representative Month Averaging
    num_nan_elements_spring_month = (24*30*4) - len(prices_spring_flat)
    num_nan_elements_summer_month = (24*30*4) - len(prices_summer_flat)
    num_nan_elements_fall_month = (24*30*3) - len(prices_fall_flat)
    num_nan_elements_winter_month = (24*30*3) - len(prices_winter_flat)
    
    prices_spring_flat_month_nan = prices_spring_flat
    prices_summer_flat_month_nan = prices_summer_flat
    prices_fall_flat_month_nan = prices_fall_flat
    prices_winter_flat_month_nan = prices_winter_flat
    
    for i in range(num_nan_elements_spring_month):
        prices_spring_flat_month_nan = np.append(prices_spring_flat_month_nan, [np.nan])
    for i in range(num_nan_elements_summer_month):
        prices_summer_flat_month_nan = np.append(prices_summer_flat_month_nan, [np.nan])
    for i in range(num_nan_elements_fall_month):
        prices_fall_flat_month_nan = np.append(prices_fall_flat_month_nan, [np.nan])
    for i in range(num_nan_elements_winter_month):
        prices_winter_flat_month_nan = np.append(prices_winter_flat_month_nan, [np.nan])
        
    prices_spring_rep_month_nan = prices_spring_flat_month_nan.reshape((4,24*30)).transpose()
    prices_summer_rep_month_nan = prices_summer_flat_month_nan.reshape((4,24*30)).transpose()
    prices_fall_rep_month_nan = prices_fall_flat_month_nan.reshape((3,24*30)).transpose()
    prices_winter_rep_month_nan = prices_winter_flat_month_nan.reshape((3,24*30)).transpose()
    
    prices_spring_rep_month = np.nanmean(prices_spring_rep_month_nan, axis=1)
    prices_summer_rep_month = np.nanmean(prices_summer_rep_month_nan, axis=1)
    prices_fall_rep_month = np.nanmean(prices_fall_rep_month_nan, axis=1)
    prices_winter_rep_month = np.nanmean(prices_winter_rep_month_nan, axis=1)
    '''
    
    return electricity_prices_rep_week


# Taking this out for a test spin (need to change line 21 to make it work here)
"""
month_list = ['january','february','march','april','may','june','july','august','september','october','november','december']
days_idx = np.array([0,78,(78+93),(78+93+94),(78+93+94+89)])
hours_idx = days_idx*24 
days_in_season = np.array([78,93,94,89,11]) # End of Winter, Spring, Summer, Fall, Start of Winter
electricity_prices_rep_week = rep_weeks_electricity(month_list,hours_idx,days_in_season)
"""