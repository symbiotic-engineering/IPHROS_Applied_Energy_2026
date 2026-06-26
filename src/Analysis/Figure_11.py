# -*- coding: utf-8 -*-
"""
NOTE: This file makes the dictionaries used in _operational_state_analysis.py

It also makes the main operational state plot
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#%% Load in Data
#----------------------------------------------
# Load in Raw Data
#----------------------------------------------
with open("../../data/Optimal IPHROS/AWS_turbo_dicts.pkl", "rb") as f:
    dict_turbo_RO_convergence, dict_LP_efficiencies_convergence, dict_MILP_post_processing, dict_big_convergence = pickle.load(f)

x_MILP_season0 = pd.read_csv('../../data/Optimal IPHROS/x_MILP_season0.csv',header=None).to_numpy()
x_MILP_season1 = pd.read_csv('../../data/Optimal IPHROS/x_MILP_season1.csv',header=None).to_numpy()
x_MILP_season2 = pd.read_csv('../../data/Optimal IPHROS/x_MILP_season2.csv',header=None).to_numpy()
x_MILP_season3 = pd.read_csv('../../data/Optimal IPHROS/x_MILP_season3.csv',header=None).to_numpy()

V_res_max = 28635496
V_res_min = 0.3*V_res_max
V_res_init = 0.65*V_res_max


#%% Load in energy prices and salinity profile
with open("../../data/RO and Miscellaneous/energy_prices_dicts.pkl", "rb") as f:
    energy_prices_dict = pickle.load(f)
    
energy_prices_spring = energy_prices_dict['spring']
energy_prices_summer = energy_prices_dict['summer']
energy_prices_fall = energy_prices_dict['fall']
energy_prices_winter = energy_prices_dict['winter']

with open("../../data/RO and Miscellaneous/energy_prices_norm_dicts.pkl", "rb") as f:
    energy_prices_norm_dict = pickle.load(f)
    
energy_prices_spring_norm = energy_prices_norm_dict['spring']
energy_prices_summer_norm = energy_prices_norm_dict['summer']
energy_prices_fall_norm = energy_prices_norm_dict['fall']
energy_prices_winter_norm = energy_prices_norm_dict['winter']

with open("../../data/RO and Miscellaneous/salinity_profiles_dict.pkl", "rb") as f:
    salinity_profile_dict = pickle.load(f)
    
S_ht_max_array_spring = salinity_profile_dict['spring']
S_ht_max_array_summer = salinity_profile_dict['summer']
S_ht_max_array_fall = salinity_profile_dict['fall']
S_ht_max_array_winter = salinity_profile_dict['winter']


#%% Process data --> Raw data to variables

last_big_iter_spring = list(dict_MILP_post_processing['season0'].keys())[-1]
last_big_iter_summer = list(dict_MILP_post_processing['season1'].keys())[-1]
last_big_iter_fall = list(dict_MILP_post_processing['season2'].keys())[-1]
last_big_iter_winter = list(dict_MILP_post_processing['season3'].keys())[-1]

second_last_LP_eff_iter_spring = list(dict_LP_efficiencies_convergence['season0'][last_big_iter_spring].keys())[-2]
second_last_LP_eff_iter_summer = list(dict_LP_efficiencies_convergence['season1'][last_big_iter_summer].keys())[-2]
second_last_LP_eff_iter_fall = list(dict_LP_efficiencies_convergence['season2'][last_big_iter_fall].keys())[-2]
second_last_LP_eff_iter_winter = list(dict_LP_efficiencies_convergence['season3'][last_big_iter_winter].keys())[-2]

    
P_in_PSH1_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['P_in_PSH1_array']
P_in_PSH1_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['P_in_PSH1_array']
P_in_PSH1_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['P_in_PSH1_array']
P_in_PSH1_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['P_in_PSH1_array']

P_in_PSH2_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['P_in_PSH2_array']
P_in_PSH2_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['P_in_PSH2_array']
P_in_PSH2_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['P_in_PSH2_array']
P_in_PSH2_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['P_in_PSH2_array']

P_in_PSH3_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['P_in_PSH3_array']
P_in_PSH3_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['P_in_PSH3_array']
P_in_PSH3_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['P_in_PSH3_array']
P_in_PSH3_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['P_in_PSH3_array']

P_in_PSH4_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['P_in_PSH4_array']
P_in_PSH4_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['P_in_PSH4_array']
P_in_PSH4_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['P_in_PSH4_array']
P_in_PSH4_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['P_in_PSH4_array']

P_in_PSH5_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['P_in_PSH5_array']
P_in_PSH5_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['P_in_PSH5_array']
P_in_PSH5_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['P_in_PSH5_array']
P_in_PSH5_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['P_in_PSH5_array']

P_in_PSH6_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['P_in_PSH6_array']
P_in_PSH6_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['P_in_PSH6_array']
P_in_PSH6_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['P_in_PSH6_array']
P_in_PSH6_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['P_in_PSH6_array']

P_in_PSH7_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['P_in_PSH7_array']
P_in_PSH7_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['P_in_PSH7_array']
P_in_PSH7_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['P_in_PSH7_array']
P_in_PSH7_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['P_in_PSH7_array']


P_out_PSH1_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['P_out_PSH1_array']
P_out_PSH1_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['P_out_PSH1_array']
P_out_PSH1_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['P_out_PSH1_array']
P_out_PSH1_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['P_out_PSH1_array']

P_out_PSH2_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['P_out_PSH2_array']
P_out_PSH2_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['P_out_PSH2_array']
P_out_PSH2_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['P_out_PSH2_array']
P_out_PSH2_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['P_out_PSH2_array']

P_out_PSH3_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['P_out_PSH3_array']
P_out_PSH3_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['P_out_PSH3_array']
P_out_PSH3_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['P_out_PSH3_array']
P_out_PSH3_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['P_out_PSH3_array']

P_out_PSH4_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['P_out_PSH4_array']
P_out_PSH4_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['P_out_PSH4_array']
P_out_PSH4_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['P_out_PSH4_array']
P_out_PSH4_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['P_out_PSH4_array']

P_out_PSH5_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['P_out_PSH5_array']
P_out_PSH5_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['P_out_PSH5_array']
P_out_PSH5_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['P_out_PSH5_array']
P_out_PSH5_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['P_out_PSH5_array']

P_out_PSH6_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['P_out_PSH6_array']
P_out_PSH6_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['P_out_PSH6_array']
P_out_PSH6_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['P_out_PSH6_array']
P_out_PSH6_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['P_out_PSH6_array']

P_out_PSH7_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['P_out_PSH7_array']
P_out_PSH7_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['P_out_PSH7_array']
P_out_PSH7_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['P_out_PSH7_array']
P_out_PSH7_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['P_out_PSH7_array']


V_dot_fwRO_spring = dict_MILP_post_processing['season0'][last_big_iter_spring]['V_dot_fwRO_array']
V_dot_fwRO_summer = dict_MILP_post_processing['season1'][last_big_iter_summer]['V_dot_fwRO_array']
V_dot_fwRO_fall = dict_MILP_post_processing['season2'][last_big_iter_fall]['V_dot_fwRO_array']
V_dot_fwRO_winter = dict_MILP_post_processing['season3'][last_big_iter_winter]['V_dot_fwRO_array']


eta_ht_PSH1_spring = dict_LP_efficiencies_convergence['season0'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine1_array']
eta_ht_PSH1_summer = dict_LP_efficiencies_convergence['season1'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine1_array']
eta_ht_PSH1_fall = dict_LP_efficiencies_convergence['season2'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine1_array']
eta_ht_PSH1_winter = dict_LP_efficiencies_convergence['season3'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine1_array']

eta_ht_PSH2_spring = dict_LP_efficiencies_convergence['season0'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine2_array']
eta_ht_PSH2_summer = dict_LP_efficiencies_convergence['season1'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine2_array']
eta_ht_PSH2_fall = dict_LP_efficiencies_convergence['season2'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine2_array']
eta_ht_PSH2_winter = dict_LP_efficiencies_convergence['season3'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine2_array']

eta_ht_PSH3_spring = dict_LP_efficiencies_convergence['season0'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine3_array']
eta_ht_PSH3_summer = dict_LP_efficiencies_convergence['season1'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine3_array']
eta_ht_PSH3_fall = dict_LP_efficiencies_convergence['season2'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine3_array']
eta_ht_PSH3_winter = dict_LP_efficiencies_convergence['season3'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine3_array']

eta_ht_PSH4_spring = dict_LP_efficiencies_convergence['season0'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine4_array']
eta_ht_PSH4_summer = dict_LP_efficiencies_convergence['season1'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine4_array']
eta_ht_PSH4_fall = dict_LP_efficiencies_convergence['season2'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine4_array']
eta_ht_PSH4_winter = dict_LP_efficiencies_convergence['season3'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine4_array']

eta_ht_PSH5_spring = dict_LP_efficiencies_convergence['season0'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine5_array']
eta_ht_PSH5_summer = dict_LP_efficiencies_convergence['season1'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine5_array']
eta_ht_PSH5_fall = dict_LP_efficiencies_convergence['season2'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine5_array']
eta_ht_PSH5_winter = dict_LP_efficiencies_convergence['season3'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine5_array']

eta_ht_PSH6_spring = dict_LP_efficiencies_convergence['season0'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine6_array']
eta_ht_PSH6_summer = dict_LP_efficiencies_convergence['season1'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine6_array']
eta_ht_PSH6_fall = dict_LP_efficiencies_convergence['season2'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine6_array']
eta_ht_PSH6_winter = dict_LP_efficiencies_convergence['season3'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine6_array']

eta_ht_PSH7_spring = dict_LP_efficiencies_convergence['season0'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine7_array']
eta_ht_PSH7_summer = dict_LP_efficiencies_convergence['season1'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine7_array']
eta_ht_PSH7_fall = dict_LP_efficiencies_convergence['season2'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine7_array']
eta_ht_PSH7_winter = dict_LP_efficiencies_convergence['season3'][last_big_iter_spring][second_last_LP_eff_iter_spring]['efficiencies_outputs']['eta_ht_turbine7_array']


V_dot_wp1_spring = x_MILP_season0[:,0]
V_dot_wp2_spring = x_MILP_season0[:,1]
V_dot_wp3_spring = x_MILP_season0[:,2]
V_dot_wp4_spring = x_MILP_season0[:,3]
V_dot_wp5_spring = x_MILP_season0[:,4]
V_dot_wp6_spring = x_MILP_season0[:,5]
V_dot_wp7_spring = x_MILP_season0[:,6]

V_dot_wp1_summer = x_MILP_season1[:,0]
V_dot_wp2_summer = x_MILP_season1[:,1]
V_dot_wp3_summer = x_MILP_season1[:,2]
V_dot_wp4_summer = x_MILP_season1[:,3]
V_dot_wp5_summer = x_MILP_season1[:,4]
V_dot_wp6_summer = x_MILP_season1[:,5]
V_dot_wp7_summer = x_MILP_season1[:,6]

V_dot_wp1_fall = x_MILP_season2[:,0]
V_dot_wp2_fall = x_MILP_season2[:,1]
V_dot_wp3_fall = x_MILP_season2[:,2]
V_dot_wp4_fall = x_MILP_season2[:,3]
V_dot_wp5_fall = x_MILP_season2[:,4]
V_dot_wp6_fall = x_MILP_season2[:,5]
V_dot_wp7_fall = x_MILP_season2[:,6]

V_dot_wp1_winter = x_MILP_season3[:,0]
V_dot_wp2_winter = x_MILP_season3[:,1]
V_dot_wp3_winter = x_MILP_season3[:,2]
V_dot_wp4_winter = x_MILP_season3[:,3]
V_dot_wp5_winter = x_MILP_season3[:,4]
V_dot_wp6_winter = x_MILP_season3[:,5]
V_dot_wp7_winter = x_MILP_season3[:,6]


V_dot_swht1_spring = x_MILP_season0[:,7]
V_dot_swht2_spring = x_MILP_season0[:,8]
V_dot_swht3_spring = x_MILP_season0[:,9]
V_dot_swht4_spring = x_MILP_season0[:,10]
V_dot_swht5_spring = x_MILP_season0[:,11]
V_dot_swht6_spring = x_MILP_season0[:,12]
V_dot_swht7_spring = x_MILP_season0[:,13]

V_dot_swht1_summer = x_MILP_season1[:,7]
V_dot_swht2_summer = x_MILP_season1[:,8]
V_dot_swht3_summer = x_MILP_season1[:,9]
V_dot_swht4_summer = x_MILP_season1[:,10]
V_dot_swht5_summer = x_MILP_season1[:,11]
V_dot_swht6_summer = x_MILP_season1[:,12]
V_dot_swht7_summer = x_MILP_season1[:,13]

V_dot_swht1_fall = x_MILP_season2[:,7]
V_dot_swht2_fall = x_MILP_season2[:,8]
V_dot_swht3_fall = x_MILP_season2[:,9]
V_dot_swht4_fall = x_MILP_season2[:,10]
V_dot_swht5_fall = x_MILP_season2[:,11]
V_dot_swht6_fall = x_MILP_season2[:,12]
V_dot_swht7_fall = x_MILP_season2[:,13]

V_dot_swht1_winter = x_MILP_season3[:,7]
V_dot_swht2_winter = x_MILP_season3[:,8]
V_dot_swht3_winter = x_MILP_season3[:,9]
V_dot_swht4_winter = x_MILP_season3[:,10]
V_dot_swht5_winter = x_MILP_season3[:,11]
V_dot_swht6_winter = x_MILP_season3[:,12]
V_dot_swht7_winter = x_MILP_season3[:,13]


V_res_spring = x_MILP_season0[:,15]
V_res_summer = x_MILP_season1[:,15]
V_res_fall = x_MILP_season2[:,15]
V_res_winter = x_MILP_season3[:,15]

RO_on_off_spring = x_MILP_season0[:,16]
RO_on_off_summer = x_MILP_season1[:,16]
RO_on_off_fall = x_MILP_season2[:,16]
RO_on_off_winter = x_MILP_season3[:,16]

v_PSH_matrix_spring = x_MILP_season0[:,31:38]
v_PSH_matrix_summer = x_MILP_season1[:,31:38]
v_PSH_matrix_fall = x_MILP_season2[:,31:38]
v_PSH_matrix_winter = x_MILP_season3[:,31:38]

w_PSH_matrix_spring = x_MILP_season0[:,38:]
w_PSH_matrix_summer = x_MILP_season1[:,38:]
w_PSH_matrix_fall = x_MILP_season2[:,38:]
w_PSH_matrix_winter = x_MILP_season3[:,38:]


#%% Process data --> variables to usable variables

# Summating PSH flowrates
V_dot_wp_spring = V_dot_wp1_spring+V_dot_wp2_spring+V_dot_wp3_spring+V_dot_wp4_spring+V_dot_wp5_spring+V_dot_wp6_spring+V_dot_wp7_spring
V_dot_wp_summer = V_dot_wp1_summer+V_dot_wp2_summer+V_dot_wp3_summer+V_dot_wp4_summer+V_dot_wp5_summer+V_dot_wp6_summer+V_dot_wp7_summer
V_dot_wp_fall = V_dot_wp1_fall+V_dot_wp2_fall+V_dot_wp3_fall+V_dot_wp4_fall+V_dot_wp5_fall+V_dot_wp6_fall+V_dot_wp7_fall
V_dot_wp_winter = V_dot_wp1_winter+V_dot_wp2_winter+V_dot_wp3_winter+V_dot_wp4_winter+V_dot_wp5_winter+V_dot_wp6_winter+V_dot_wp7_winter

V_dot_swht_spring = V_dot_swht1_spring+V_dot_swht2_spring+V_dot_swht3_spring+V_dot_swht4_spring+V_dot_swht5_spring+V_dot_swht6_spring+V_dot_swht7_spring
V_dot_swht_summer = V_dot_swht1_summer+V_dot_swht2_summer+V_dot_swht3_summer+V_dot_swht4_summer+V_dot_swht5_summer+V_dot_swht6_summer+V_dot_swht7_summer
V_dot_swht_fall = V_dot_swht1_fall+V_dot_swht2_fall+V_dot_swht3_fall+V_dot_swht4_fall+V_dot_swht5_fall+V_dot_swht6_fall+V_dot_swht7_fall
V_dot_swht_winter = V_dot_swht1_winter+V_dot_swht2_winter+V_dot_swht3_winter+V_dot_swht4_winter+V_dot_swht5_winter+V_dot_swht6_winter+V_dot_swht7_winter

# Powers in and out matrices
P_out_PSH_spring = P_out_PSH1_spring+P_out_PSH2_spring+P_out_PSH3_spring+P_out_PSH4_spring+P_out_PSH5_spring+P_out_PSH6_spring+P_out_PSH7_spring
P_out_PSH_summer = P_out_PSH1_summer+P_out_PSH2_summer+P_out_PSH3_summer+P_out_PSH4_summer+P_out_PSH5_summer+P_out_PSH6_summer+P_out_PSH7_summer
P_out_PSH_fall = P_out_PSH1_fall+P_out_PSH2_fall+P_out_PSH3_fall+P_out_PSH4_fall+P_out_PSH5_fall+P_out_PSH6_fall+P_out_PSH7_fall
P_out_PSH_winter = P_out_PSH1_winter+P_out_PSH2_winter+P_out_PSH3_winter+P_out_PSH4_winter+P_out_PSH5_winter+P_out_PSH6_winter+P_out_PSH7_winter

P_in_PSH_spring = P_in_PSH1_spring+P_in_PSH2_spring+P_in_PSH3_spring+P_in_PSH4_spring+P_in_PSH5_spring+P_in_PSH6_spring+P_in_PSH7_spring
P_in_PSH_summer = P_in_PSH1_summer+P_in_PSH2_summer+P_in_PSH3_summer+P_in_PSH4_summer+P_in_PSH5_summer+P_in_PSH6_summer+P_in_PSH7_summer
P_in_PSH_fall = P_in_PSH1_fall+P_in_PSH2_fall+P_in_PSH3_fall+P_in_PSH4_fall+P_in_PSH5_fall+P_in_PSH6_fall+P_in_PSH7_fall
P_in_PSH_winter = P_in_PSH1_winter+P_in_PSH2_winter+P_in_PSH3_winter+P_in_PSH4_winter+P_in_PSH5_winter+P_in_PSH6_winter+P_in_PSH7_winter


P_in_PSH_spring_matrix = np.ones((168,7))*(-69)
P_in_PSH_summer_matrix = np.ones((168,7))*(-69)
P_in_PSH_fall_matrix = np.ones((168,7))*(-69)
P_in_PSH_winter_matrix = np.ones((168,7))*(-69)

P_in_PSH_spring_matrix[:,0] = P_in_PSH1_spring
P_in_PSH_spring_matrix[:,1] = P_in_PSH2_spring
P_in_PSH_spring_matrix[:,2] = P_in_PSH3_spring
P_in_PSH_spring_matrix[:,3] = P_in_PSH4_spring
P_in_PSH_spring_matrix[:,4] = P_in_PSH5_spring
P_in_PSH_spring_matrix[:,5] = P_in_PSH6_spring
P_in_PSH_spring_matrix[:,6] = P_in_PSH7_spring

P_in_PSH_summer_matrix[:,0] = P_in_PSH1_summer
P_in_PSH_summer_matrix[:,1] = P_in_PSH2_summer
P_in_PSH_summer_matrix[:,2] = P_in_PSH3_summer
P_in_PSH_summer_matrix[:,3] = P_in_PSH4_summer
P_in_PSH_summer_matrix[:,4] = P_in_PSH5_summer
P_in_PSH_summer_matrix[:,5] = P_in_PSH6_summer
P_in_PSH_summer_matrix[:,6] = P_in_PSH7_summer

P_in_PSH_fall_matrix[:,0] = P_in_PSH1_fall
P_in_PSH_fall_matrix[:,1] = P_in_PSH2_fall
P_in_PSH_fall_matrix[:,2] = P_in_PSH3_fall
P_in_PSH_fall_matrix[:,3] = P_in_PSH4_fall
P_in_PSH_fall_matrix[:,4] = P_in_PSH5_fall
P_in_PSH_fall_matrix[:,5] = P_in_PSH6_fall
P_in_PSH_fall_matrix[:,6] = P_in_PSH7_fall

P_in_PSH_winter_matrix[:,0] = P_in_PSH1_winter
P_in_PSH_winter_matrix[:,1] = P_in_PSH2_winter
P_in_PSH_winter_matrix[:,2] = P_in_PSH3_winter
P_in_PSH_winter_matrix[:,3] = P_in_PSH4_winter
P_in_PSH_winter_matrix[:,4] = P_in_PSH5_winter
P_in_PSH_winter_matrix[:,5] = P_in_PSH6_winter
P_in_PSH_winter_matrix[:,6] = P_in_PSH7_winter

P_out_PSH_spring_matrix = np.ones((168,7))*(-69)
P_out_PSH_summer_matrix = np.ones((168,7))*(-69)
P_out_PSH_fall_matrix = np.ones((168,7))*(-69)
P_out_PSH_winter_matrix = np.ones((168,7))*(-69)

P_out_PSH_spring_matrix[:,0] = P_out_PSH1_spring
P_out_PSH_spring_matrix[:,1] = P_out_PSH2_spring
P_out_PSH_spring_matrix[:,2] = P_out_PSH3_spring
P_out_PSH_spring_matrix[:,3] = P_out_PSH4_spring
P_out_PSH_spring_matrix[:,4] = P_out_PSH5_spring
P_out_PSH_spring_matrix[:,5] = P_out_PSH6_spring
P_out_PSH_spring_matrix[:,6] = P_out_PSH7_spring

P_out_PSH_summer_matrix[:,0] = P_out_PSH1_summer
P_out_PSH_summer_matrix[:,1] = P_out_PSH2_summer
P_out_PSH_summer_matrix[:,2] = P_out_PSH3_summer
P_out_PSH_summer_matrix[:,3] = P_out_PSH4_summer
P_out_PSH_summer_matrix[:,4] = P_out_PSH5_summer
P_out_PSH_summer_matrix[:,5] = P_out_PSH6_summer
P_out_PSH_summer_matrix[:,6] = P_out_PSH7_summer

P_out_PSH_fall_matrix[:,0] = P_out_PSH1_fall
P_out_PSH_fall_matrix[:,1] = P_out_PSH2_fall
P_out_PSH_fall_matrix[:,2] = P_out_PSH3_fall
P_out_PSH_fall_matrix[:,3] = P_out_PSH4_fall
P_out_PSH_fall_matrix[:,4] = P_out_PSH5_fall
P_out_PSH_fall_matrix[:,5] = P_out_PSH6_fall
P_out_PSH_fall_matrix[:,6] = P_out_PSH7_fall

P_out_PSH_winter_matrix[:,0] = P_out_PSH1_winter
P_out_PSH_winter_matrix[:,1] = P_out_PSH2_winter
P_out_PSH_winter_matrix[:,2] = P_out_PSH3_winter
P_out_PSH_winter_matrix[:,3] = P_out_PSH4_winter
P_out_PSH_winter_matrix[:,4] = P_out_PSH5_winter
P_out_PSH_winter_matrix[:,5] = P_out_PSH6_winter
P_out_PSH_winter_matrix[:,6] = P_out_PSH7_winter


# Normalized cycle counting
cycle_counting_frac_PSH1_spring = np.cumsum(v_PSH_matrix_spring[1:,0] + w_PSH_matrix_spring[1:,0])/28
cycle_counting_frac_PSH2_spring = np.cumsum(v_PSH_matrix_spring[1:,1] + w_PSH_matrix_spring[1:,1])/28
cycle_counting_frac_PSH3_spring = np.cumsum(v_PSH_matrix_spring[1:,2] + w_PSH_matrix_spring[1:,2])/28
cycle_counting_frac_PSH4_spring = np.cumsum(v_PSH_matrix_spring[1:,3] + w_PSH_matrix_spring[1:,3])/28
cycle_counting_frac_PSH5_spring = np.cumsum(v_PSH_matrix_spring[1:,4] + w_PSH_matrix_spring[1:,4])/28
cycle_counting_frac_PSH6_spring = np.cumsum(v_PSH_matrix_spring[1:,5] + w_PSH_matrix_spring[1:,5])/28
cycle_counting_frac_PSH7_spring = np.cumsum(v_PSH_matrix_spring[1:,6] + w_PSH_matrix_spring[1:,6])/28

cycle_counting_frac_PSH1_summer = np.cumsum(v_PSH_matrix_summer[1:,0] + w_PSH_matrix_summer[1:,0])/28
cycle_counting_frac_PSH2_summer = np.cumsum(v_PSH_matrix_summer[1:,1] + w_PSH_matrix_summer[1:,1])/28
cycle_counting_frac_PSH3_summer = np.cumsum(v_PSH_matrix_summer[1:,2] + w_PSH_matrix_summer[1:,2])/28
cycle_counting_frac_PSH4_summer = np.cumsum(v_PSH_matrix_summer[1:,3] + w_PSH_matrix_summer[1:,3])/28
cycle_counting_frac_PSH5_summer = np.cumsum(v_PSH_matrix_summer[1:,4] + w_PSH_matrix_summer[1:,4])/28
cycle_counting_frac_PSH6_summer = np.cumsum(v_PSH_matrix_summer[1:,5] + w_PSH_matrix_summer[1:,5])/28
cycle_counting_frac_PSH7_summer = np.cumsum(v_PSH_matrix_summer[1:,6] + w_PSH_matrix_summer[1:,6])/28

cycle_counting_frac_PSH1_fall = np.cumsum(v_PSH_matrix_fall[1:,0] + w_PSH_matrix_fall[1:,0])/28
cycle_counting_frac_PSH2_fall = np.cumsum(v_PSH_matrix_fall[1:,1] + w_PSH_matrix_fall[1:,1])/28
cycle_counting_frac_PSH3_fall = np.cumsum(v_PSH_matrix_fall[1:,2] + w_PSH_matrix_fall[1:,2])/28
cycle_counting_frac_PSH4_fall = np.cumsum(v_PSH_matrix_fall[1:,3] + w_PSH_matrix_fall[1:,3])/28
cycle_counting_frac_PSH5_fall = np.cumsum(v_PSH_matrix_fall[1:,4] + w_PSH_matrix_fall[1:,4])/28
cycle_counting_frac_PSH6_fall = np.cumsum(v_PSH_matrix_fall[1:,5] + w_PSH_matrix_fall[1:,5])/28
cycle_counting_frac_PSH7_fall = np.cumsum(v_PSH_matrix_fall[1:,6] + w_PSH_matrix_fall[1:,6])/28

cycle_counting_frac_PSH1_winter = np.cumsum(v_PSH_matrix_winter[1:,0] + w_PSH_matrix_winter[1:,0])/28
cycle_counting_frac_PSH2_winter = np.cumsum(v_PSH_matrix_winter[1:,1] + w_PSH_matrix_winter[1:,1])/28
cycle_counting_frac_PSH3_winter = np.cumsum(v_PSH_matrix_winter[1:,2] + w_PSH_matrix_winter[1:,2])/28
cycle_counting_frac_PSH4_winter = np.cumsum(v_PSH_matrix_winter[1:,3] + w_PSH_matrix_winter[1:,3])/28
cycle_counting_frac_PSH5_winter = np.cumsum(v_PSH_matrix_winter[1:,4] + w_PSH_matrix_winter[1:,4])/28
cycle_counting_frac_PSH6_winter = np.cumsum(v_PSH_matrix_winter[1:,5] + w_PSH_matrix_winter[1:,5])/28
cycle_counting_frac_PSH7_winter = np.cumsum(v_PSH_matrix_winter[1:,6] + w_PSH_matrix_winter[1:,6])/28


# Slack variables
V_lower_frac_spring_slack = (V_res_spring - (0.3*V_res_max))/(V_res_max - V_res_min)
V_lower_frac_summer_slack = (V_res_summer - (0.3*V_res_max))/(V_res_max - V_res_min)
V_lower_frac_fall_slack = (V_res_fall - (0.3*V_res_max))/(V_res_max - V_res_min)
V_lower_frac_winter_slack = (V_res_winter - (0.3*V_res_max))/(V_res_max - V_res_min)

V_upper_frac_spring_slack =  (V_res_max - V_res_spring)/(V_res_max - V_res_min)
V_upper_frac_summer_slack =  (V_res_max - V_res_summer)/(V_res_max - V_res_min)
V_upper_frac_fall_slack =  (V_res_max - V_res_fall)/(V_res_max - V_res_min)
V_upper_frac_winter_slack =  (V_res_max - V_res_winter)/(V_res_max - V_res_min)

cycles_frac_PSH1_spring_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH1_spring))
cycles_frac_PSH2_spring_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH2_spring))
cycles_frac_PSH3_spring_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH3_spring))
cycles_frac_PSH4_spring_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH4_spring))
cycles_frac_PSH5_spring_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH5_spring))
cycles_frac_PSH6_spring_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH6_spring))
cycles_frac_PSH7_spring_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH7_spring))

cycles_frac_PSH1_summer_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH1_summer))
cycles_frac_PSH2_summer_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH2_summer))
cycles_frac_PSH3_summer_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH3_summer))
cycles_frac_PSH4_summer_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH4_summer))
cycles_frac_PSH5_summer_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH5_summer))
cycles_frac_PSH6_summer_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH6_summer))
cycles_frac_PSH7_summer_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH7_summer))

cycles_frac_PSH1_fall_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH1_fall))
cycles_frac_PSH2_fall_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH2_fall))
cycles_frac_PSH3_fall_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH3_fall))
cycles_frac_PSH4_fall_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH4_fall))
cycles_frac_PSH5_fall_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH5_fall))
cycles_frac_PSH6_fall_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH6_fall))
cycles_frac_PSH7_fall_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH7_fall))

cycles_frac_PSH1_winter_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH1_winter))
cycles_frac_PSH2_winter_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH2_winter))
cycles_frac_PSH3_winter_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH3_winter))
cycles_frac_PSH4_winter_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH4_winter))
cycles_frac_PSH5_winter_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH5_winter))
cycles_frac_PSH6_winter_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH6_winter))
cycles_frac_PSH7_winter_slack = 1 - np.concatenate(([0], cycle_counting_frac_PSH7_winter))

t_array = np.linspace(0,167,168)
end_of_week_frac_spring_slack = abs(((V_res_init - V_res_spring)/(0.5*(V_res_max - V_res_min))) / (168 - t_array))
end_of_week_frac_summer_slack = abs(((V_res_init - V_res_summer)/(0.5*(V_res_max - V_res_min))) / (168 - t_array))
end_of_week_frac_fall_slack = abs(((V_res_init - V_res_fall)/(0.5*(V_res_max - V_res_min))) / (168 - t_array))
end_of_week_frac_winter_slack = abs(((V_res_init - V_res_winter)/(0.5*(V_res_max - V_res_min))) / (168 - t_array))


#%% IPHROS States Matrix Creation

nonzero_counts_P_in_spring = np.sum(P_in_PSH_spring_matrix != 0, axis=1)
nonzero_counts_P_in_summer = np.sum(P_in_PSH_summer_matrix != 0, axis=1)
nonzero_counts_P_in_fall = np.sum(P_in_PSH_fall_matrix != 0, axis=1)
nonzero_counts_P_in_winter = np.sum(P_in_PSH_winter_matrix != 0, axis=1)

nonzero_counts_P_out_spring = np.sum(P_out_PSH_spring_matrix != 0, axis=1)
nonzero_counts_P_out_summer = np.sum(P_out_PSH_summer_matrix != 0, axis=1)
nonzero_counts_P_out_fall = np.sum(P_out_PSH_fall_matrix != 0, axis=1)
nonzero_counts_P_out_winter = np.sum(P_out_PSH_winter_matrix != 0, axis=1)

nonzero_counts_spring = np.vstack((nonzero_counts_P_in_spring,nonzero_counts_P_out_spring,RO_on_off_spring)).transpose()
nonzero_counts_summer = np.vstack((nonzero_counts_P_in_summer,nonzero_counts_P_out_summer,RO_on_off_summer)).transpose()
nonzero_counts_fall = np.vstack((nonzero_counts_P_in_fall,nonzero_counts_P_out_fall,RO_on_off_fall)).transpose()
nonzero_counts_winter = np.vstack((nonzero_counts_P_in_winter,nonzero_counts_P_out_winter,RO_on_off_winter)).transpose()


unique_states_spring, counts_spring = np.unique(nonzero_counts_spring, axis=0, return_counts=True)
unique_states_summer, counts_summer = np.unique(nonzero_counts_summer, axis=0, return_counts=True)
unique_states_fall, counts_fall = np.unique(nonzero_counts_fall, axis=0, return_counts=True)
unique_states_winter, counts_winter = np.unique(nonzero_counts_winter, axis=0, return_counts=True)

freqs_spring = counts_spring/168
freqs_summer = counts_summer/168
freqs_fall = counts_fall/168
freqs_winter = counts_winter/168

unique_states_spring = np.hstack((unique_states_spring, freqs_spring.reshape(-1, 1)))
unique_states_summer = np.hstack((unique_states_summer, freqs_summer.reshape(-1, 1)))
unique_states_fall = np.hstack((unique_states_fall, freqs_fall.reshape(-1, 1)))
unique_states_winter = np.hstack((unique_states_winter, freqs_winter.reshape(-1, 1)))

unique_states_spring_sorted = unique_states_spring[unique_states_spring[:, 3].argsort()]
unique_states_summer_sorted = unique_states_summer[unique_states_summer[:, 3].argsort()]
unique_states_fall_sorted = unique_states_fall[unique_states_fall[:, 3].argsort()]
unique_states_winter_sorted = unique_states_winter[unique_states_winter[:, 3].argsort()]


#%% Visualization of turbine states

for i in range(4):
    if i == 0:
        col1 = unique_states_spring_sorted[:, 0]
        col2 = unique_states_spring_sorted[:, 1]
        col3 = unique_states_spring_sorted[:, 2]
        col4 = unique_states_spring_sorted[:, 3]
        
        # Isolating where the RO system is on and off
        RO_off_bars = np.where(unique_states_spring_sorted[:, 2] == 0)[0] # For spring: [1,6,8,10]
        RO_on_bars = np.where(unique_states_spring_sorted[:, 2] == 1)[0] # For summer: [0,2,3,4,5,7,9,11,12,13]
    elif i == 1:
        col1 = unique_states_summer_sorted[:, 0]
        col2 = unique_states_summer_sorted[:, 1]
        col3 = unique_states_summer_sorted[:, 2]
        col4 = unique_states_summer_sorted[:, 3]
        
        # Isolating where the RO system is on and off
        RO_off_bars = np.where(unique_states_summer_sorted[:, 2] == 0)[0] 
        RO_on_bars = np.where(unique_states_summer_sorted[:, 2] == 1)[0]
    elif i == 2:
        col1 = unique_states_fall_sorted[:, 0]
        col2 = unique_states_fall_sorted[:, 1]
        col3 = unique_states_fall_sorted[:, 2]
        col4 = unique_states_fall_sorted[:, 3]
        
        # Isolating where the RO system is on and off
        RO_off_bars = np.where(unique_states_fall_sorted[:, 2] == 0)[0] 
        RO_on_bars = np.where(unique_states_fall_sorted[:, 2] == 1)[0]
    elif i == 3:
        col1 = unique_states_winter_sorted[:, 0]
        col2 = unique_states_winter_sorted[:, 1]
        col3 = unique_states_winter_sorted[:, 2]
        col4 = unique_states_winter_sorted[:, 3]
        
        # Isolating where the RO system is on and off
        RO_off_bars = np.where(unique_states_winter_sorted[:, 2] == 0)[0] 
        RO_on_bars = np.where(unique_states_winter_sorted[:, 2] == 1)[0]
    
    # Create labels like [col1, col2]
    labels = [f"[# con={int(c1)}, # gen={int(c2)}]" for c1, c2 in zip(col1, col2)]

    # Plot
    plt.figure(figsize=(12, 6))
    bars_RO_on = plt.barh(RO_on_bars, col4[RO_on_bars], color='tab:blue', label = r'$RO \; On$')
    bars_RO_off = plt.barh(RO_off_bars, col4[RO_off_bars], color='tab:orange', label = r'$RO \; Off$')
    
    # X-axis formatting
    plt.yticks(range(len(labels)), labels, rotation=45, ha='right')
    
    # Labels
    plt.ylabel(r"$State \; [\# \; PSH \; Units \; Consuming, \# \; PSH \; Units \; Generating]$",fontsize=14)
    plt.xlabel(r"$Frequency$",fontsize=14)
    if i == 0:
        plt.title(r"$Spring$",fontsize=14)
    elif i == 1:
        plt.title(r"$Summer$",fontsize=14)
    elif i == 2:
        plt.title(r"$Fall$",fontsize=14)
    elif i == 3:
        plt.title(r"$Winter$",fontsize=14)
        
    plt.bar_label(bars_RO_on, fmt='%.3f', padding=3)
    plt.bar_label(bars_RO_off, fmt='%.3f', padding=3)
    
    plt.legend()
    plt.tight_layout()
    if i == 0:
        plt.savefig('Figure_11.pdf')
        a = 1 # Placeholder
    plt.show()
    
