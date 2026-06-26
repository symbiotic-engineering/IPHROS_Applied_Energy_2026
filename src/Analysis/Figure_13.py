# -*- coding: utf-8 -*-
"""
Answering the question "How sensitive is IPHROS to the price of water?"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


#%% General parameters, NPV values
lambda_w = 1.97 # Euros/m^3
lambda_w_multipliers = np.array([0.94,0.97,0.99,1.00,1.01,1.03,1.06])

lambda_w_array = lambda_w*lambda_w_multipliers

# These values were just copied and pasted from log files
NPV_array = np.array([860556097.4480016,946655556.9417396,1030520694.7028049,1006041068.0574907,1060076282.3499115,1089134762.7132971,1157205617.0802355])

# These values were obtained from running helper_operational_state_frequency_varying_water.py for different lambda_w values
freq_7_0_0_094_array = np.array([0.101,0.113,0.089,0.155])
freq_7_0_0_097_array = np.array([0.077,0.054,0.060,0.155])
freq_7_0_0_099_array = np.array([0.071,0.054,0.060,0.155])
freq_7_0_0_100_array = np.array([0.071,0.054,0.060,0.155])
freq_7_0_0_101_array = np.array([0.048,0.054,0.030,0.155])
freq_7_0_0_103_array = np.array([0.048,0.054,0.030,0.107])
freq_7_0_0_106_array = np.array([0.048,0.054,0.030,0.101])

freq_6_1_1_094_array = np.array([0.196,0.131,0.173,0.143])
freq_6_1_1_097_array = np.array([0.220,0.220,0.256,0.143])
freq_6_1_1_099_array = np.array([0.232,0.220,0.250,0.131])
freq_6_1_1_100_array = np.array([0.208,0.220,0.256,0.125])
freq_6_1_1_101_array = np.array([0.256,0.226,0.292,0.125])
freq_6_1_1_103_array = np.array([0.256,0.232,0.280,0.196])
freq_6_1_1_106_array = np.array([0.256,0.238,0.292,0.202])


#%% Load in Data
with open("../../data/Varying Lambda_w/lambda_w_094_turbo_dicts.pkl", "rb") as f_094:
    dict_turbo_RO_convergence_094, dict_LP_efficiencies_convergence_094, dict_MILP_post_processing_094, dict_big_convergence_094 = pickle.load(f_094) 
x_MILP_season0_094 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season0_094_lambda_w.csv',header=None).to_numpy()
x_MILP_season1_094 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season1_094_lambda_w.csv',header=None).to_numpy()
x_MILP_season2_094 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season2_094_lambda_w.csv',header=None).to_numpy()
x_MILP_season3_094 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season3_094_lambda_w.csv',header=None).to_numpy()
    
with open("../../data/Varying Lambda_w/lambda_w_097_turbo_dicts.pkl", "rb") as f_097:
    dict_turbo_RO_convergence_097, dict_LP_efficiencies_convergence_097, dict_MILP_post_processing_097, dict_big_convergence_097 = pickle.load(f_097)
x_MILP_season0_097 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season0_097_lambda_w.csv',header=None).to_numpy()
x_MILP_season1_097 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season1_097_lambda_w.csv',header=None).to_numpy()
x_MILP_season2_097 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season2_097_lambda_w.csv',header=None).to_numpy()
x_MILP_season3_097 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season3_097_lambda_w.csv',header=None).to_numpy()
    
with open("../../data/Varying Lambda_w/lambda_w_099_turbo_dicts.pkl", "rb") as f_099:
    dict_turbo_RO_convergence_099, dict_LP_efficiencies_convergence_099, dict_MILP_post_processing_099, dict_big_convergence_099 = pickle.load(f_099)
x_MILP_season0_099 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season0_099_lambda_w.csv',header=None).to_numpy()
x_MILP_season1_099 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season1_099_lambda_w.csv',header=None).to_numpy()
x_MILP_season2_099 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season2_099_lambda_w.csv',header=None).to_numpy()
x_MILP_season3_099 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season3_099_lambda_w.csv',header=None).to_numpy()
    
with open("../../data/Optimal IPHROS/AWS_turbo_dicts.pkl", "rb") as f_100:
    dict_turbo_RO_convergence_100, dict_LP_efficiencies_convergence_100, dict_MILP_post_processing_100, dict_big_convergence_100 = pickle.load(f_100)
x_MILP_season0_100 = pd.read_csv('../../data/Optimal IPHROS/x_MILP_season0.csv',header=None).to_numpy()
x_MILP_season1_100 = pd.read_csv('../../data/Optimal IPHROS/x_MILP_season1.csv',header=None).to_numpy()
x_MILP_season2_100 = pd.read_csv('../../data/Optimal IPHROS/x_MILP_season2.csv',header=None).to_numpy()
x_MILP_season3_100 = pd.read_csv('../../data/Optimal IPHROS/x_MILP_season3.csv',header=None).to_numpy()

with open("../../data/Varying Lambda_w/lambda_w_101_turbo_dicts.pkl", "rb") as f_101:
    dict_turbo_RO_convergence_101, dict_LP_efficiencies_convergence_101, dict_MILP_post_processing_101, dict_big_convergence_101 = pickle.load(f_101)
x_MILP_season0_101 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season0_101_lambda_w.csv',header=None).to_numpy()
x_MILP_season1_101 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season1_101_lambda_w.csv',header=None).to_numpy()
x_MILP_season2_101 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season2_101_lambda_w.csv',header=None).to_numpy()
x_MILP_season3_101 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season3_101_lambda_w.csv',header=None).to_numpy()

with open("../../data/Varying Lambda_w/lambda_w_103_turbo_dicts.pkl", "rb") as f_103:
    dict_turbo_RO_convergence_103, dict_LP_efficiencies_convergence_103, dict_MILP_post_processing_103, dict_big_convergence_103 = pickle.load(f_103)
x_MILP_season0_103 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season0_103_lambda_w.csv',header=None).to_numpy()
x_MILP_season1_103 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season1_103_lambda_w.csv',header=None).to_numpy()
x_MILP_season2_103 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season2_103_lambda_w.csv',header=None).to_numpy()
x_MILP_season3_103 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season3_103_lambda_w.csv',header=None).to_numpy()

with open("../../data/Varying Lambda_w/lambda_w_106_turbo_dicts.pkl", "rb") as f_106:
    dict_turbo_RO_convergence_106, dict_LP_efficiencies_convergence_106, dict_MILP_post_processing_106, dict_big_convergence_106 = pickle.load(f_106)
x_MILP_season0_106 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season0_106_lambda_w.csv',header=None).to_numpy()
x_MILP_season1_106 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season1_106_lambda_w.csv',header=None).to_numpy()
x_MILP_season2_106 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season2_106_lambda_w.csv',header=None).to_numpy()
x_MILP_season3_106 = pd.read_csv('../../data/Varying Lambda_w/x_MILP_season3_106_lambda_w.csv',header=None).to_numpy()
    

#%% Load in energy prices
with open("../../data/RO and Miscellaneous/energy_prices_dicts.pkl", "rb") as f:
    energy_prices_dict = pickle.load(f)


#%% RO statuses

def RO_statuses(x_MILP_spring,x_MILP_summer,x_MILP_fall,x_MILP_winter):
    RO_on_off_spring = x_MILP_spring[:,16]
    RO_on_off_summer = x_MILP_summer[:,16]
    RO_on_off_fall = x_MILP_fall[:,16]
    RO_on_off_winter = x_MILP_winter[:,16]
    
    freq_RO_on_off_spring = np.sum(RO_on_off_spring)/168
    freq_RO_on_off_summer = np.sum(RO_on_off_summer)/168
    freq_RO_on_off_fall = np.sum(RO_on_off_fall)/168
    freq_RO_on_off_winter = np.sum(RO_on_off_winter)/168
    
    freq_RO_on_off_array = np.array([freq_RO_on_off_spring,freq_RO_on_off_summer,freq_RO_on_off_fall,freq_RO_on_off_winter])
    
    return freq_RO_on_off_array

freq_RO_on_off_array_094 = RO_statuses(x_MILP_season0_094,x_MILP_season1_094,x_MILP_season2_094,x_MILP_season3_094)
freq_RO_on_off_array_097 = RO_statuses(x_MILP_season0_097,x_MILP_season1_097,x_MILP_season2_097,x_MILP_season3_097)
freq_RO_on_off_array_099 = RO_statuses(x_MILP_season0_099,x_MILP_season1_099,x_MILP_season2_099,x_MILP_season3_099)
freq_RO_on_off_array_100 = RO_statuses(x_MILP_season0_100,x_MILP_season1_100,x_MILP_season2_100,x_MILP_season3_100)
freq_RO_on_off_array_101 = RO_statuses(x_MILP_season0_101,x_MILP_season1_101,x_MILP_season2_101,x_MILP_season3_101)
freq_RO_on_off_array_103 = RO_statuses(x_MILP_season0_103,x_MILP_season1_103,x_MILP_season2_103,x_MILP_season3_103)
freq_RO_on_off_array_106 = RO_statuses(x_MILP_season0_106,x_MILP_season1_106,x_MILP_season2_106,x_MILP_season3_106)

freq_RO_on_off_spring_array = np.array([freq_RO_on_off_array_094[0],freq_RO_on_off_array_097[0],freq_RO_on_off_array_099[0],freq_RO_on_off_array_100[0],
                                        freq_RO_on_off_array_101[0],freq_RO_on_off_array_103[0],freq_RO_on_off_array_106[0]])
freq_RO_on_off_summer_array = np.array([freq_RO_on_off_array_094[1],freq_RO_on_off_array_097[1],freq_RO_on_off_array_099[1],freq_RO_on_off_array_100[1],
                                        freq_RO_on_off_array_101[1],freq_RO_on_off_array_103[1],freq_RO_on_off_array_106[1]])
freq_RO_on_off_fall_array = np.array([freq_RO_on_off_array_094[2],freq_RO_on_off_array_097[2],freq_RO_on_off_array_099[2],freq_RO_on_off_array_100[2],
                                        freq_RO_on_off_array_101[2],freq_RO_on_off_array_103[2],freq_RO_on_off_array_106[2]])
freq_RO_on_off_winter_array = np.array([freq_RO_on_off_array_094[3],freq_RO_on_off_array_097[3],freq_RO_on_off_array_099[3],freq_RO_on_off_array_100[3],
                                        freq_RO_on_off_array_101[3],freq_RO_on_off_array_103[3],freq_RO_on_off_array_106[3]])

freq_RO_on_off_avg_array = np.mean([freq_RO_on_off_spring_array, freq_RO_on_off_summer_array, freq_RO_on_off_fall_array, freq_RO_on_off_winter_array], axis=0)


#%% Profit Contributions

def profit_contributions(LP_efficiencies_convergence_dict,MILP_post_processing_dict,energy_prices_dict,lambda_w):
    
    # Key MILP dictionary indices
    last_big_iter_spring = list(MILP_post_processing_dict['season0'].keys())[-1]
    last_big_iter_summer = list(MILP_post_processing_dict['season1'].keys())[-1]
    last_big_iter_fall = list(MILP_post_processing_dict['season2'].keys())[-1]
    last_big_iter_winter = list(MILP_post_processing_dict['season3'].keys())[-1]
    
    # Average operational profit
    LP_efficiencies_spring_dict = LP_efficiencies_convergence_dict['season0']
    LP_efficiencies_spring_last_big_iter_dict = LP_efficiencies_spring_dict[list(LP_efficiencies_spring_dict.keys())[-1]]
    operational_profit_spring = LP_efficiencies_spring_last_big_iter_dict[list(LP_efficiencies_spring_last_big_iter_dict.keys())[-1]]['MILP_objective']
    
    LP_efficiencies_summer_dict = LP_efficiencies_convergence_dict['season1']
    LP_efficiencies_summer_last_big_iter_dict = LP_efficiencies_summer_dict[list(LP_efficiencies_summer_dict.keys())[-1]]
    operational_profit_summer = LP_efficiencies_summer_last_big_iter_dict[list(LP_efficiencies_summer_last_big_iter_dict.keys())[-1]]['MILP_objective']
    
    LP_efficiencies_fall_dict = LP_efficiencies_convergence_dict['season2']
    LP_efficiencies_fall_last_big_iter_dict = LP_efficiencies_fall_dict[list(LP_efficiencies_fall_dict.keys())[-1]]
    operational_profit_fall = LP_efficiencies_fall_last_big_iter_dict[list(LP_efficiencies_fall_last_big_iter_dict.keys())[-1]]['MILP_objective']
    
    LP_efficiencies_winter_dict = LP_efficiencies_convergence_dict['season3']
    LP_efficiencies_winter_last_big_iter_dict = LP_efficiencies_winter_dict[list(LP_efficiencies_winter_dict.keys())[-1]]
    operational_profit_winter = LP_efficiencies_winter_last_big_iter_dict[list(LP_efficiencies_winter_last_big_iter_dict.keys())[-1]]['MILP_objective']
    
    average_operational_profit = -np.mean([operational_profit_spring,operational_profit_summer,operational_profit_fall,operational_profit_winter])
    
    
    # Average weekly water profit
    
    V_dot_fwRO_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['V_dot_fwRO_array']
    V_dot_fwRO_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['V_dot_fwRO_array']
    V_dot_fwRO_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['V_dot_fwRO_array']
    V_dot_fwRO_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['V_dot_fwRO_array']
    
    V_fwRO_spring_week = np.sum(V_dot_fwRO_array_spring)
    V_fwRO_summer_week = np.sum(V_dot_fwRO_array_summer)
    V_fwRO_fall_week = np.sum(V_dot_fwRO_array_fall)
    V_fwRO_winter_week = np.sum(V_dot_fwRO_array_winter)
    
    water_revenue_spring = lambda_w*V_fwRO_spring_week
    water_revenue_summer = lambda_w*V_fwRO_summer_week
    water_revenue_fall = lambda_w*V_fwRO_fall_week
    water_revenue_winter = lambda_w*V_fwRO_winter_week
    
    average_weekly_water_revenue = np.mean([water_revenue_spring,water_revenue_summer,water_revenue_fall,water_revenue_winter])
    
    # Average weeekly electricity profit
    
    P_in_PSH1_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['P_in_PSH1_array']
    P_in_PSH1_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['P_in_PSH1_array']
    P_in_PSH1_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['P_in_PSH1_array']
    P_in_PSH1_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['P_in_PSH1_array']
    
    P_in_PSH2_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['P_in_PSH2_array']
    P_in_PSH2_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['P_in_PSH2_array']
    P_in_PSH2_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['P_in_PSH2_array']
    P_in_PSH2_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['P_in_PSH2_array']
    
    P_in_PSH3_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['P_in_PSH3_array']
    P_in_PSH3_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['P_in_PSH3_array']
    P_in_PSH3_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['P_in_PSH3_array']
    P_in_PSH3_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['P_in_PSH3_array']
    
    P_in_PSH4_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['P_in_PSH4_array']
    P_in_PSH4_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['P_in_PSH4_array']
    P_in_PSH4_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['P_in_PSH4_array']
    P_in_PSH4_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['P_in_PSH4_array']
    
    P_in_PSH5_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['P_in_PSH5_array']
    P_in_PSH5_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['P_in_PSH5_array']
    P_in_PSH5_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['P_in_PSH5_array']
    P_in_PSH5_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['P_in_PSH5_array']
    
    P_in_PSH6_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['P_in_PSH6_array']
    P_in_PSH6_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['P_in_PSH6_array']
    P_in_PSH6_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['P_in_PSH6_array']
    P_in_PSH6_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['P_in_PSH6_array']
    
    P_in_PSH7_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['P_in_PSH7_array']
    P_in_PSH7_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['P_in_PSH7_array']
    P_in_PSH7_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['P_in_PSH7_array']
    P_in_PSH7_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['P_in_PSH7_array']
    
    P_out_PSH1_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['P_out_PSH1_array']
    P_out_PSH1_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['P_out_PSH1_array']
    P_out_PSH1_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['P_out_PSH1_array']
    P_out_PSH1_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['P_out_PSH1_array']
    
    P_out_PSH2_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['P_out_PSH2_array']
    P_out_PSH2_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['P_out_PSH2_array']
    P_out_PSH2_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['P_out_PSH2_array']
    P_out_PSH2_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['P_out_PSH2_array']
    
    P_out_PSH3_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['P_out_PSH3_array']
    P_out_PSH3_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['P_out_PSH3_array']
    P_out_PSH3_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['P_out_PSH3_array']
    P_out_PSH3_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['P_out_PSH3_array']
    
    P_out_PSH4_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['P_out_PSH4_array']
    P_out_PSH4_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['P_out_PSH4_array']
    P_out_PSH4_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['P_out_PSH4_array']
    P_out_PSH4_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['P_out_PSH4_array']
    
    P_out_PSH5_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['P_out_PSH5_array']
    P_out_PSH5_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['P_out_PSH5_array']
    P_out_PSH5_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['P_out_PSH5_array']
    P_out_PSH5_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['P_out_PSH5_array']
    
    P_out_PSH6_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['P_out_PSH6_array']
    P_out_PSH6_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['P_out_PSH6_array']
    P_out_PSH6_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['P_out_PSH6_array']
    P_out_PSH6_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['P_out_PSH6_array']
    
    P_out_PSH7_array_spring = MILP_post_processing_dict['season0'][last_big_iter_spring]['P_out_PSH7_array']
    P_out_PSH7_array_summer = MILP_post_processing_dict['season1'][last_big_iter_summer]['P_out_PSH7_array']
    P_out_PSH7_array_fall = MILP_post_processing_dict['season2'][last_big_iter_fall]['P_out_PSH7_array']
    P_out_PSH7_array_winter = MILP_post_processing_dict['season3'][last_big_iter_winter]['P_out_PSH7_array']
    
    P_out_PSH_array_spring = P_out_PSH1_array_spring+P_out_PSH2_array_spring+P_out_PSH3_array_spring+P_out_PSH4_array_spring+P_out_PSH5_array_spring+P_out_PSH6_array_spring+P_out_PSH7_array_spring
    P_out_PSH_array_summer = P_out_PSH1_array_summer+P_out_PSH2_array_summer+P_out_PSH3_array_summer+P_out_PSH4_array_summer+P_out_PSH5_array_summer+P_out_PSH6_array_summer+P_out_PSH7_array_summer
    P_out_PSH_array_fall = P_out_PSH1_array_fall+P_out_PSH2_array_fall+P_out_PSH3_array_fall+P_out_PSH4_array_fall+P_out_PSH5_array_fall+P_out_PSH6_array_fall+P_out_PSH7_array_fall
    P_out_PSH_array_winter = P_out_PSH1_array_winter+P_out_PSH2_array_winter+P_out_PSH3_array_winter+P_out_PSH4_array_winter+P_out_PSH5_array_winter+P_out_PSH6_array_winter+P_out_PSH7_array_winter
    
    P_in_PSH_array_spring = P_in_PSH1_array_spring+P_in_PSH2_array_spring+P_in_PSH3_array_spring+P_in_PSH4_array_spring+P_in_PSH5_array_spring+P_in_PSH6_array_spring+P_in_PSH7_array_spring
    P_in_PSH_array_summer = P_in_PSH1_array_summer+P_in_PSH2_array_summer+P_in_PSH3_array_summer+P_in_PSH4_array_summer+P_in_PSH5_array_summer+P_in_PSH6_array_summer+P_in_PSH7_array_summer
    P_in_PSH_array_fall = P_in_PSH1_array_fall+P_in_PSH2_array_fall+P_in_PSH3_array_fall+P_in_PSH4_array_fall+P_in_PSH5_array_fall+P_in_PSH6_array_fall+P_in_PSH7_array_fall
    P_in_PSH_array_winter = P_in_PSH1_array_winter+P_in_PSH2_array_winter+P_in_PSH3_array_winter+P_in_PSH4_array_winter+P_in_PSH5_array_winter+P_in_PSH6_array_winter+P_in_PSH7_array_winter
    
    elec_cost_spring = np.sum(energy_prices_dict['spring']*P_in_PSH_array_spring)
    elec_cost_summer = np.sum(energy_prices_dict['summer']*P_in_PSH_array_summer)
    elec_cost_fall = np.sum(energy_prices_dict['fall']*P_in_PSH_array_fall)
    elec_cost_winter = np.sum(energy_prices_dict['winter']*P_in_PSH_array_winter)
    
    average_weekly_electricity_cost = np.mean([elec_cost_spring,elec_cost_summer,elec_cost_fall,elec_cost_winter])
    
    elec_revenue_spring = np.sum(energy_prices_dict['spring']*P_out_PSH_array_spring)
    elec_revenue_summer = np.sum(energy_prices_dict['summer']*P_out_PSH_array_summer)
    elec_revenue_fall = np.sum(energy_prices_dict['fall']*P_out_PSH_array_fall)
    elec_revenue_winter = np.sum(energy_prices_dict['winter']*P_out_PSH_array_winter)
    
    average_weekly_electricity_revenue = np.mean([elec_revenue_spring,elec_revenue_summer,elec_revenue_fall,elec_revenue_winter])
    
    P_net_PSH_array_spring = P_out_PSH_array_spring-P_in_PSH_array_spring
    P_net_PSH_array_summer = P_out_PSH_array_summer-P_in_PSH_array_summer
    P_net_PSH_array_fall = P_out_PSH_array_fall-P_in_PSH_array_fall
    P_net_PSH_array_winter = P_out_PSH_array_winter-P_in_PSH_array_winter
    
    elec_net_profit_spring = np.sum(energy_prices_dict['spring']*P_net_PSH_array_spring)
    elec_net_profit_summer = np.sum(energy_prices_dict['summer']*P_net_PSH_array_summer)
    elec_net_profit_fall = np.sum(energy_prices_dict['fall']*P_net_PSH_array_fall)
    elec_net_profit_winter = np.sum(energy_prices_dict['winter']*P_net_PSH_array_winter)
    
    average_weekly_electricity_net_profit = np.mean([elec_net_profit_spring,elec_net_profit_summer,elec_net_profit_fall,elec_net_profit_winter])
    
    return [average_weekly_electricity_cost,average_weekly_electricity_revenue,average_weekly_electricity_net_profit,
            average_weekly_water_revenue,average_operational_profit]

[average_weekly_electricity_cost_094,average_weekly_electricity_revenue_094,average_weekly_electricity_net_profit_094,
        average_weekly_water_revenue_094,average_operational_profit_094] = profit_contributions(dict_LP_efficiencies_convergence_094,dict_MILP_post_processing_094,energy_prices_dict,lambda_w)
[average_weekly_electricity_cost_097,average_weekly_electricity_revenue_097,average_weekly_electricity_net_profit_097,
        average_weekly_water_revenue_097,average_operational_profit_097] = profit_contributions(dict_LP_efficiencies_convergence_097,dict_MILP_post_processing_097,energy_prices_dict,lambda_w)
[average_weekly_electricity_cost_099,average_weekly_electricity_revenue_099,average_weekly_electricity_net_profit_099,
        average_weekly_water_revenue_099,average_operational_profit_099] = profit_contributions(dict_LP_efficiencies_convergence_099,dict_MILP_post_processing_099,energy_prices_dict,lambda_w)
[average_weekly_electricity_cost_100,average_weekly_electricity_revenue_100,average_weekly_electricity_net_profit_100,
        average_weekly_water_revenue_100,average_operational_profit_100] = profit_contributions(dict_LP_efficiencies_convergence_100,dict_MILP_post_processing_100,energy_prices_dict,lambda_w)
[average_weekly_electricity_cost_101,average_weekly_electricity_revenue_101,average_weekly_electricity_net_profit_101,
        average_weekly_water_revenue_101,average_operational_profit_101] = profit_contributions(dict_LP_efficiencies_convergence_101,dict_MILP_post_processing_101,energy_prices_dict,lambda_w)
[average_weekly_electricity_cost_103,average_weekly_electricity_revenue_103,average_weekly_electricity_net_profit_103,
        average_weekly_water_revenue_103,average_operational_profit_103] = profit_contributions(dict_LP_efficiencies_convergence_103,dict_MILP_post_processing_103,energy_prices_dict,lambda_w)
[average_weekly_electricity_cost_106,average_weekly_electricity_revenue_106,average_weekly_electricity_net_profit_106,
        average_weekly_water_revenue_106,average_operational_profit_106] = profit_contributions(dict_LP_efficiencies_convergence_106,dict_MILP_post_processing_106,energy_prices_dict,lambda_w)

average_weekly_electricity_net_profit_array = np.array([average_weekly_electricity_net_profit_094,average_weekly_electricity_net_profit_097,
                                                    average_weekly_electricity_net_profit_099,average_weekly_electricity_net_profit_100,
                                                    average_weekly_electricity_net_profit_101,average_weekly_electricity_net_profit_103,
                                                    average_weekly_electricity_net_profit_106])
average_weekly_electricity_cost_array = np.array([average_weekly_electricity_cost_094,average_weekly_electricity_cost_097,
                                                    average_weekly_electricity_cost_099,average_weekly_electricity_cost_100,
                                                    average_weekly_electricity_cost_101,average_weekly_electricity_cost_103,
                                                    average_weekly_electricity_cost_106])
average_weekly_electricity_revenue_array = np.array([average_weekly_electricity_revenue_094,average_weekly_electricity_revenue_097,
                                                    average_weekly_electricity_revenue_099,average_weekly_electricity_revenue_100,
                                                    average_weekly_electricity_revenue_101,average_weekly_electricity_revenue_103,
                                                    average_weekly_electricity_revenue_106])
average_weekly_water_revenue_array = np.array([average_weekly_water_revenue_094,average_weekly_water_revenue_097,
                                                    average_weekly_water_revenue_099,average_weekly_water_revenue_100,
                                                    average_weekly_water_revenue_101,average_weekly_water_revenue_103,
                                                    average_weekly_water_revenue_106])
average_operational_profit_array = np.array([average_operational_profit_094,average_operational_profit_097,
                                                    average_operational_profit_099,average_operational_profit_100,
                                                    average_operational_profit_101,average_operational_profit_103,
                                                    average_operational_profit_106])


#%% Frequency of time that IPHROS forgoes water production for full energy consumption
freq_all_in_094_array = freq_7_0_0_094_array/(freq_7_0_0_094_array+freq_6_1_1_094_array)
freq_all_in_097_array = freq_7_0_0_097_array/(freq_7_0_0_097_array+freq_6_1_1_097_array)
freq_all_in_099_array = freq_7_0_0_099_array/(freq_7_0_0_099_array+freq_6_1_1_099_array)
freq_all_in_100_array = freq_7_0_0_100_array/(freq_7_0_0_100_array+freq_6_1_1_100_array)
freq_all_in_101_array = freq_7_0_0_101_array/(freq_7_0_0_101_array+freq_6_1_1_101_array)
freq_all_in_103_array = freq_7_0_0_103_array/(freq_7_0_0_103_array+freq_6_1_1_103_array)
freq_all_in_106_array = freq_7_0_0_106_array/(freq_7_0_0_106_array+freq_6_1_1_106_array)

freq_all_in_spring_array = np.array([freq_all_in_094_array[0],freq_all_in_097_array[0],freq_all_in_099_array[0],freq_all_in_100_array[0],freq_all_in_101_array[0],freq_all_in_103_array[0],freq_all_in_106_array[0]])
freq_all_in_summer_array = np.array([freq_all_in_094_array[1],freq_all_in_097_array[1],freq_all_in_099_array[1],freq_all_in_100_array[1],freq_all_in_101_array[1],freq_all_in_103_array[1],freq_all_in_106_array[1]])
freq_all_in_fall_array = np.array([freq_all_in_094_array[2],freq_all_in_097_array[2],freq_all_in_099_array[2],freq_all_in_100_array[2],freq_all_in_101_array[2],freq_all_in_103_array[2],freq_all_in_106_array[2]])
freq_all_in_winter_array = np.array([freq_all_in_094_array[3],freq_all_in_097_array[3],freq_all_in_099_array[3],freq_all_in_100_array[3],freq_all_in_101_array[3],freq_all_in_103_array[3],freq_all_in_106_array[3]])

freq_all_in_avg_array = np.mean([freq_all_in_spring_array, freq_all_in_summer_array, freq_all_in_fall_array, freq_all_in_winter_array], axis=0)


#%% Plotting Time!!!

#====================================
# Plot 1: NPV and frequencies (of RO on/off and states [6,1,1] vs. [7,0,0]
#====================================

fig, ax1 = plt.subplots()

# Left axis
ax1.plot(lambda_w_array, NPV_array, linewidth=2, label=r'$NPV$')
ax1.set_xlabel(r"$\lambda_w \; [2023\;€/m^3]$",fontsize=14,y=0.2)
ax1.set_ylabel(r"$NPV \; [2023\;€]$",fontsize=14)
ax1.tick_params(axis='both', labelsize=11)

# Right axis
ax2 = ax1.twinx()
ax2.plot(lambda_w_array, freq_RO_on_off_avg_array, color='tab:orange', linewidth=2, linestyle='--', label=r'$Avg. \; Freq. \; RO \; On$')
ax2.plot(lambda_w_array, freq_all_in_avg_array, color='tab:green', linewidth=2, linestyle='--', label=r'$Avg. \frac{[7,0,0]}{[6,1,1]&[7,0,0]}$')
ax2.set_ylabel(r"$Frequency$",fontsize=14)
ax2.tick_params(axis='both', labelsize=11)

handles_part1, labels_part1 = ax1.get_legend_handles_labels()
handles_part2, labels_part2 = ax2.get_legend_handles_labels()
handles = handles_part1+handles_part2
labels = labels_part1+labels_part2
fig.legend(handles, labels,bbox_to_anchor=(1.32, 0.72),frameon=True,fontsize=11)

plt.tight_layout()#rect=[0, 0, 0.90, 1])  # leave space for legend  
plt.savefig('Figure_13.pdf', bbox_inches='tight')        
plt.show()


#====================================
# Plot 2: Operational Profit Breakdown
# I want to basically show the change in bar size, rather than just show the absolute values, because I'm not getting the best resolution with 
# that method (I mean just look at the graph, all the bars look the same)
#====================================

x = np.arange(len(lambda_w_array))

fig, axes = plt.subplots(1,2,figsize=(10,5))#(figsize=(8, 5))

#-------------------
# Left Plot: absolute values
#-------------------
# --- Negative bar (cost) ---
ax1a = axes[0]
ax1a.bar(x, -average_weekly_electricity_cost_array, label=r'$Electricity \; Cost$', alpha=0.8)

# --- Positive stacked bars (revenues) ---
ax1a.bar(x, average_weekly_electricity_revenue_array, label=r'$Electricity \; Profit$', alpha=0.8)
ax1a.bar(x, average_weekly_water_revenue_array, bottom=average_weekly_electricity_revenue_array, label=r'$Freshwater \; Profit$', alpha=0.8)

# --- Formatting ---
ax1a.axhline(0, color='black', linewidth=1)

ax1a.set_xticks(x)
ax1a.set_xticklabels(np.round(lambda_w_array,2))
ax1a.set_ylabel(r"$Absolute \; Value \; [2023\;€]$",fontsize=14)

# Right axis
ax1b = ax1a.twinx()
ax1b.plot(x, average_operational_profit_array, color='tab:red', linewidth=2, label=r'$Operational \; Profit$')
ax1b.tick_params(axis='both', labelsize=11)
ax1b.set_title(r"$(a)$",fontsize=14)

#-------------------
# Left Plot: absolute values
#-------------------
nom_elec_cost = average_weekly_electricity_cost_array[len(average_weekly_electricity_cost_array)//2]
nom_elec_rev = average_weekly_electricity_revenue_array[len(average_weekly_electricity_revenue_array)//2]
nom_water_rev = average_weekly_water_revenue_array[len(average_weekly_water_revenue_array)//2]
nom_profit = average_operational_profit_array[len(average_operational_profit_array)//2]

elec_cost_delta = average_weekly_electricity_cost_array - nom_elec_cost
elec_rev_delta = average_weekly_electricity_revenue_array - nom_elec_rev
water_rev_delta = average_weekly_water_revenue_array - nom_water_rev
profit_delta = average_operational_profit_array - nom_profit

elec_cost_percent_change = ((average_weekly_electricity_cost_array - nom_elec_cost)/nom_elec_cost)*100
elec_rev_percent_change = ((average_weekly_electricity_revenue_array - nom_elec_rev)/nom_elec_rev)*100
water_rev_percent_change = ((average_weekly_water_revenue_array - nom_water_rev)/nom_water_rev)*100
profit_percent_change = ((average_operational_profit_array - nom_profit)/nom_profit)*100


#-------------------
# Right Plot: percent changes
#-------------------

ax2a = axes[1]
ax2a.plot(lambda_w_array,elec_cost_percent_change,linewidth=2)
ax2a.plot(lambda_w_array,elec_rev_percent_change,linewidth=2)
ax2a.plot(lambda_w_array,water_rev_percent_change,linewidth=2)
ax2a.plot(lambda_w_array,profit_percent_change,linewidth=2)
ax2a.set_ylabel(r'$\% \; \Delta \; From \; \lambda_w=1.97 \; [2023\;€]$',fontsize=14)
ax2a.set_title(r"$(b)$",fontsize=14)


"""
# Negative bar
ax2a.bar(x, -elec_cost_delta, label=r'$Electricity \; Cost$', alpha=0.8)
# Positive bars
ax2a.bar(x, elec_rev_delta, label=r'$Electricity \; Profit$', alpha=0.8)
ax2a.bar(x, water_rev_delta, bottom=elec_rev_delta, label=r'$Freshwater Profit$', alpha=0.8)

ax2a.axhline(0, color='black', linewidth=1)

ax2a.set_xticks(x)
ax2a.set_xticklabels(np.round(lambda_w_array,2))
ax2a.set_ylim([-200000,200000])

ax2b = ax2a.twinx()
ax2b.plot(x, profit_delta, color='tab:red', linewidth=2, label=r'$Operational \; Profit$')
#ax2b.set_ylabel(r"$2023\;€$",fontsize=14)
ax2b.tick_params(axis='both', labelsize=11)

ax2b.set_ylim([-200000,200000])
"""


# Shared x and y labels
fig.supxlabel(r'$\lambda_w \; [2023\;€/m^3]$',fontsize=14, y=0.1)
#fig.supylabel(r'$2023\;€$',fontsize=14)

# Getting the legend all set up
handles_part1, labels_part1 = ax1a.get_legend_handles_labels()
handles_part2, labels_part2 = ax1b.get_legend_handles_labels()
handles = handles_part1+handles_part2
labels = labels_part1+labels_part2
fig.legend(handles, labels, loc="lower center", ncol=4,bbox_to_anchor=(0.5, 0.01),frameon=True)

#plt.title('IPHROS Cost and Revenue for Varying Water Price')
plt.tight_layout(rect=[0, 0.06, 0.97, 1])
plt.show()


