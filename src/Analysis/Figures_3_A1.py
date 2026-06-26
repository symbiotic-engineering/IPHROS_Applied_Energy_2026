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
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import skew
from scipy.stats import f_oneway
from scipy.stats import ks_2samp
from matplotlib.lines import Line2D


#----------------------------------------------------------------------
# Read in data, divy up by season
#---------------------------------------------------------------------- 

def rep_weeks_electricity(month_list,hours_idx,days_in_season):
    # NOTE: THIS FUNCTION CALCULATES ELECTRICITY PRICES IN UNITS OF EUROS/MWH (UNIT CONVERSION TO EUROS/KWH HAPPENS IN SIMULATION FUNCTION)
    
    grid_data_df = pd.DataFrame()
    for month in month_list:
        # Read the .txt file with BeautifulSoup
        txt_string = '../GA-MILP/grid_data/2023/' + month + '_2023.txt'
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
    
    prices_all = np.hstack((prices_spring,prices_summer,prices_fall,prices_winter))
    
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
    
    return [grid_data_np_flat,prices_spring,prices_summer,prices_fall,prices_winter,prices_all,
            prices_spring_rep_week_nan,prices_summer_rep_week_nan,prices_fall_rep_week_nan,prices_winter_rep_week_nan,electricity_prices_rep_week]


# Taking this out for a test spin (need to change line 21 to make it work here)
month_list = ['january','february','march','april','may','june','july','august','september','october','november','december']
days_idx = np.array([0,78,(78+93),(78+93+94),(78+93+94+89)])
hours_idx = days_idx*24 
days_in_season = np.array([78,93,94,89,11]) # End of Winter, Spring, Summer, Fall, Start of Winter
[grid_data_np_flat,prices_spring,prices_summer,prices_fall,prices_winter,
 prices_all,prices_spring_rep_week_nan,prices_summer_rep_week_nan,
 prices_fall_rep_week_nan,prices_winter_rep_week_nan,
 electricity_prices_rep_week] = rep_weeks_electricity(month_list,hours_idx,days_in_season)


#----------------------------------------------------------------------
# Autocorrelation
#----------------------------------------------------------------------

big_lag = 0 # 1 is long, 0 is short
save_fig = 0 # 1 is save, 0 is don't save

fig, ax = plt.subplots(figsize=(8,4))

if big_lag == 1:
    plot_acf(grid_data_np_flat, lags=8760/4, ax=ax, title=None)
elif big_lag == 0:
    plot_acf(grid_data_np_flat, lags=700, ax=ax, title=None)
    
# Mark daily and weekly cycles
if big_lag == 0:
    ax.axvline(24, color='red', linestyle='--', label='24 hr (daily cycle)')
    ax.axvline(168, color='blue', linestyle='--', label='168 hr (weekly cycle)')
    ax.axvline(168*2, color='green', linestyle='--', label='336 hr (bi-weekly cycle)')
    ax.axvline(168*3, color='black', linestyle='--', label='504 hr (tri-weekly cycle)')
    ax.axvline(168*4, color='orange', linestyle='--', label='672 hr (monthly cycle)')
    
    ax.legend()

ax.set_xlabel(r"$Lag \; (hours)$", fontsize=14)
ax.set_ylabel(r"$\lambda_e \; Autocorrelation$", fontsize=14)

plt.tight_layout()
if big_lag == 1 and save_fig == 1:
    a = 1 # placeholder
elif big_lag == 0 and save_fig == 1:
    plt.savefig("Figure_3.pdf")
    a = 1 # placeholder
plt.show()


#----------------------------------------------------------------------
# Plot average electricity price by week of the year (I think I also want weekly standard deviation)
#----------------------------------------------------------------------
x = np.linspace(0,51,52)
week_avgs = np.ones(52)*(-69)
week_stds = np.ones(52)*(-69)
for i in range(52):
    if i == 51:
        prices_week = prices_all[:,i*7:]
    else:
        prices_week = prices_all[:,i*7:(i+1)*7]
    
    week_avgs[i] = np.mean(prices_week)
    week_stds[i] = np.std(prices_week)

# Plot 1: Generic averages and stds
fig1, ax1 = plt.subplots()

# First y-axis
ax1.plot(x, week_avgs, 'b-', label='Average')
ax1.set_xlabel('x')
ax1.set_ylabel('Weekly Average Price', color='b')
ax1.tick_params(axis='y', labelcolor='b')

for xx in [days_idx[1]/7,days_idx[2]/7,days_idx[3]/7,days_idx[4]/7]:
    ax1.axvline(xx, linestyle="--", color="gray")

# Second y-axis
ax2 = ax1.twinx()
ax2.plot(x, week_stds, 'r-', label='STD')
ax2.set_ylabel('Weekly STD Price', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.show()


# Plot 2: Rolling averages and stds
x_rolling3 = np.linspace(1,50,50)
week_avgs_rolling3_with_nans = pd.Series(week_avgs).rolling(window=3,center=True).mean().to_numpy()
week_stds_rolling3_with_nans = pd.Series(week_stds).rolling(window=3,center=True).mean().to_numpy()

week_avgs_rolling3 = week_avgs_rolling3_with_nans[~np.isnan(week_avgs_rolling3_with_nans)]
week_stds_rolling3 = week_stds_rolling3_with_nans[~np.isnan(week_stds_rolling3_with_nans)]

# First y-axis
fig2,ax3 = plt.subplots()
ax3.plot(x_rolling3, week_avgs_rolling3, 'b-', label='Average')
ax3.set_xlabel('x')
ax3.set_ylabel('Weekly Average Price', color='b')
ax3.tick_params(axis='y', labelcolor='b')

for xx in [days_idx[1]/7,days_idx[2]/7,days_idx[3]/7,days_idx[4]/7]:
    ax3.axvline(xx, linestyle="--", color="gray")

# Second y-axis
ax4 = ax3.twinx()
ax4.plot(x_rolling3, week_stds_rolling3, 'r-', label='STD')
ax4.set_ylabel('Weekly STD Price', color='r')
ax4.tick_params(axis='y', labelcolor='r')

plt.show()


#----------------------------------------------------------------------
# Plot weekly drift (change) in mean and volatility
#----------------------------------------------------------------------

mean_diff = np.diff(week_avgs)
std_diff = np.diff(week_stds)

plt.figure(figsize=(10,4))
plt.plot(mean_diff, label="Mean drift")
plt.plot(std_diff, label="Volatility drift")

for xx in [days_idx[1]/7,days_idx[2]/7,days_idx[3]/7,days_idx[4]/7]:
    plt.axvline(xx, linestyle="--", color="gray")

plt.axhline(0, color="black")
plt.xlabel("Week")
plt.ylabel("Drift")
plt.legend()
plt.show()


#----------------------------------------------------------------------
# Plot combined drift metric
#----------------------------------------------------------------------

norm_mean = mean_diff / mean_diff.std()
norm_std = std_diff / std_diff.std()

drift_metric = np.sqrt(norm_mean**2 + norm_std**2)

plt.figure(figsize=(10,4))
plt.plot(drift_metric, linewidth=2)

for xx in [days_idx[1]/7,days_idx[2]/7,days_idx[3]/7,days_idx[4]/7]:
    plt.axvline(xx, linestyle="--", color="gray")

plt.xlabel("Week")
plt.ylabel("Seasonal Drift Metric")
plt.title("Combined Drift of Price Mean and Volatility")

plt.show()


#----------------------------------------------------------------------
# Rolling average combined drift metric
#----------------------------------------------------------------------

norm_mean = mean_diff / mean_diff.std()
norm_std = std_diff / std_diff.std()

drift_metric = np.sqrt(norm_mean**2 + norm_std**2)


x_rolling3 = np.linspace(2,50,49)
drift_metric_rolling3_with_nans = pd.Series(drift_metric).rolling(window=3,center=True).mean().to_numpy()

drift_metric_rolling3 = drift_metric_rolling3_with_nans[~np.isnan(drift_metric_rolling3_with_nans)]


plt.figure(figsize=(10,4))
plt.plot(x_rolling3,drift_metric_rolling3, linewidth=2)

for xx in [days_idx[1]/7,days_idx[2]/7,days_idx[3]/7,days_idx[4]/7]:
    plt.axvline(xx, linestyle="--", color="gray")

plt.xlabel("Week")
plt.ylabel("Seasonal Drift Metric")
plt.title("Rolling Combined Drift of Price Mean and Volatility")

plt.show()


#----------------------------------------------------------------------
# Run Autocorrelation on the Weekly Average Prices
#----------------------------------------------------------------------

"""
fig, ax = plt.subplots(figsize=(8,4))

plot_acf(week_avgs, lags=51, ax=ax, title=None)

ax.set_xlabel(r"$Lag \; (hours)$", fontsize=14)
ax.set_ylabel(r"$Weekly \; Autocorrelation$", fontsize=14)
ax.legend()

plt.tight_layout()
plt.show()
"""


#----------------------------------------------------------------------
# Show that electricity prices differ between seasons
#----------------------------------------------------------------------
season_names = ["Spring", "Summer", "Fall","Winter"]

means = np.ones(4)*-69
stds = np.ones(4)*-69
peaks = np.ones(4)*-69
skewnesses = np.ones(4)*-69


means[0] = np.mean(prices_spring)
means[1] = np.mean(prices_summer)
means[2] = np.mean(prices_fall)
means[3] = np.mean(prices_winter)

stds[0] = np.std(prices_spring)
stds[1] = np.std(prices_summer)
stds[2] = np.std(prices_fall)
stds[3] = np.std(prices_winter)

peaks[0] = np.max(prices_spring)
peaks[1] = np.max(prices_summer)
peaks[2] = np.max(prices_fall)
peaks[3] = np.max(prices_winter)

skewnesses[0] = skew(prices_spring.flatten())
skewnesses[1] = skew(prices_summer.flatten())
skewnesses[2] = skew(prices_fall.flatten())
skewnesses[3] = skew(prices_winter.flatten())


print("SEASON METRICS")
for i, name in enumerate(season_names):
    print(f"{name:8s} | Mean = {means[i]:6.5f} | STD = {stds[i]:6.5f} | Max = {peaks[i]:6.5f} | Skew = {skewnesses[i]:6.5f}")



#----------------------------------------------------------------------
# Some last second stats --> don't end up using
#----------------------------------------------------------------------

print(f_oneway(prices_spring.flatten(), prices_summer.flatten(), prices_fall.flatten(), prices_winter.flatten()))
print(ks_2samp(prices_spring.flatten(), prices_summer.flatten()))
print(ks_2samp(prices_summer.flatten(), prices_fall.flatten()))
print(ks_2samp(prices_fall.flatten(), prices_winter.flatten()))
print(ks_2samp(prices_winter.flatten(), prices_spring.flatten()))


#----------------------------------------------------------------------
# Plot the representative weeks on top of the other weeks
#----------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

x = np.linspace(0,167,168)
xticks = [0, 24, 48, 72, 96, 120, 144, 168]

for col in range(14):
    axes[0,0].plot(x, prices_spring_rep_week_nan[:,col]/1000, color='gray', alpha=0.4, linewidth=1)
    axes[0,1].plot(x, prices_summer_rep_week_nan[:,col]/1000, color='gray', alpha=0.4, linewidth=1)
    axes[1,0].plot(x, prices_fall_rep_week_nan[:,col]/1000, color='gray', alpha=0.4, linewidth=1)
    axes[1,1].plot(x, prices_winter_rep_week_nan[:,col]/1000, color='gray', alpha=0.4, linewidth=1)
    if col == 13:
        axes[0,0].plot(x, electricity_prices_rep_week['spring']/1000, linewidth=2.5)
        axes[0,1].plot(x, electricity_prices_rep_week['summer']/1000, linewidth=2.5)
        axes[1,0].plot(x, electricity_prices_rep_week['fall']/1000, linewidth=2.5)
        axes[1,1].plot(x, electricity_prices_rep_week['winter']/1000, linewidth=2.5)
        
        axes[0,0].set_xticks(xticks)
        axes[0,1].set_xticks(xticks)
        axes[1,0].set_xticks(xticks)
        axes[1,1].set_xticks(xticks)
        
axes[0,0].set_title(r'$Spring$',fontsize=14)
axes[0,1].set_title(r'$Summer$',fontsize=14)
axes[1,0].set_title(r'$Fall$',fontsize=14)
axes[1,1].set_title(r'$Winter$',fontsize=14)

# Leave space at bottom manually
fig.subplots_adjust(bottom=0.18)

# Shared axis labels
fig.supxlabel(r"$Time \; [hr]$",fontsize=14, y=0.09)
fig.supylabel(r"$\lambda_e \; [€/kW]$",fontsize=14)

# Legend
legend_elements = [
    Line2D([0], [0], color='gray', lw=2, alpha=0.3, label='Individual Week Profiles'),
    Line2D([0], [0], color='#1f77b4', lw=2.5, label='Representative Week Profile')
]

fig.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.01),  # move legend below xlabel
    ncol=1,
    frameon=True
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("Figure_A1.pdf")
plt.show()


#----------------------------------------------------------------------
# Assessing the quality of the averages
#----------------------------------------------------------------------

def compute_rep_metrics(season_data, rep_profile):
    """
    Compute RMSE, correlation, and R^2 between each original time series
    and a representative (average) profile.

    Parameters
    ----------
    season_data : np.ndarray
        Array containing all original weekly time series for one season.
        Shape is:
            - (168, n_series)

    rep_profile : np.ndarray
        Representative/average time series for the season, length 168.

    Returns
    -------
    df_metrics : pd.DataFrame
        DataFrame with one row per original time series and columns:
        ['series_id', 'RMSE', 'Correlation', 'R2']

    summary : pd.Series
        Summary statistics (mean/min/max) across all original series.
    """
    
    T, n_series = season_data.shape
    n_series = n_series-1 # Not going to deal with any partial weeks

    rmse_list = []
    corr_list = []
    r2_list = []

    for i in range(n_series):
        x = season_data[:,i]
        y = rep_profile

        # RMSE
        rmse = np.sqrt(np.mean((x - y)**2))

        # Correlation
        # protect against zero-variance cases
        if np.std(x) == 0 or np.std(y) == 0:
            corr = np.nan
        else:
            corr = np.corrcoef(x, y)[0, 1]

        # R^2 relative to the mean of the original series
        ss_res = np.sum((x - y)**2)
        ss_tot = np.sum((x - np.mean(x))**2)

        if ss_tot == 0:
            r2 = np.nan
        else:
            r2 = 1 - ss_res / ss_tot

        rmse_list.append(rmse)
        corr_list.append(corr)
        r2_list.append(r2)

    df_metrics = pd.DataFrame({
        "series_id": np.arange(n_series),
        "RMSE": rmse_list,
        "Correlation": corr_list,
        "R2": r2_list
    })

    summary = df_metrics[["RMSE", "Correlation", "R2"]].agg(["mean", "min", "max", "std"])

    return df_metrics, summary

df_metrics_spring, summary_spring = compute_rep_metrics(prices_spring_rep_week_nan, electricity_prices_rep_week['spring'])
df_metrics_summer, summary_summer = compute_rep_metrics(prices_summer_rep_week_nan, electricity_prices_rep_week['summer'])
df_metrics_fall, summary_fall = compute_rep_metrics(prices_fall_rep_week_nan, electricity_prices_rep_week['fall'])
df_metrics_winter, summary_winter = compute_rep_metrics(prices_winter_rep_week_nan, electricity_prices_rep_week['winter'])





