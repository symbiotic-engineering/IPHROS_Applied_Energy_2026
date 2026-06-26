"""
Matt Haefner (mwh85)
SEA Lab
Desalination Paper Optimization Price Taker - End-of-year Values Module
LAST UPDATED: 10/29/25     CREATED: 9/10/24
"""

import numpy as np

def year_summator(N_pv1,
                  P_in_PSH1_arrays,P_in_PSH2_arrays,P_in_PSH3_arrays,P_in_PSH4_arrays,P_in_PSH5_arrays,P_in_PSH6_arrays,P_in_PSH7_arrays,
                  P_out_PSH1_arrays,P_out_PSH2_arrays,P_out_PSH3_arrays,P_out_PSH4_arrays,P_out_PSH5_arrays,P_out_PSH6_arrays,P_out_PSH7_arrays,
                  eta_ht_PSH1_arrays,eta_ht_PSH2_arrays,eta_ht_PSH3_arrays,eta_ht_PSH4_arrays,eta_ht_PSH5_arrays,eta_ht_PSH6_arrays,eta_ht_PSH7_arrays,
                  eta_RO_arrays,neg_buy_sell_profit_array,days_list,print_statements):
    
    # Adjust format of days_list
    days_list = np.array(days_list)
    
    # Energy totals year
    P_in_PSH1_week_array = np.sum(P_in_PSH1_arrays,0)
    P_in_PSH1_year = np.sum(P_in_PSH1_week_array*(days_list/7)) # kWh/year
    
    P_in_PSH2_week_array = np.sum(P_in_PSH2_arrays,0)
    P_in_PSH2_year = np.sum(P_in_PSH2_week_array*(days_list/7)) # kWh/year
    
    P_in_PSH3_week_array = np.sum(P_in_PSH3_arrays,0)
    P_in_PSH3_year = np.sum(P_in_PSH3_week_array*(days_list/7)) # kWh/year
    
    P_in_PSH4_week_array = np.sum(P_in_PSH4_arrays,0)
    P_in_PSH4_year = np.sum(P_in_PSH4_week_array*(days_list/7)) # kWh/year
    
    P_in_PSH5_week_array = np.sum(P_in_PSH5_arrays,0)
    P_in_PSH5_year = np.sum(P_in_PSH5_week_array*(days_list/7)) # kWh/year
    
    P_in_PSH6_week_array = np.sum(P_in_PSH6_arrays,0)
    P_in_PSH6_year = np.sum(P_in_PSH6_week_array*(days_list/7)) # kWh/year
    
    P_in_PSH7_week_array = np.sum(P_in_PSH7_arrays,0)
    P_in_PSH7_year = np.sum(P_in_PSH7_week_array*(days_list/7)) # kWh/year
    
    P_in_PSH_year = P_in_PSH1_year + P_in_PSH2_year + P_in_PSH3_year + P_in_PSH4_year + P_in_PSH5_year + P_in_PSH6_year + P_in_PSH7_year
    
    P_out_PSH1_week_array = np.sum(P_out_PSH1_arrays,0)
    P_out_PSH1_year = np.sum(P_out_PSH1_week_array*(days_list/7)) # kWh/year
    
    P_out_PSH2_week_array = np.sum(P_out_PSH2_arrays,0)
    P_out_PSH2_year = np.sum(P_out_PSH2_week_array*(days_list/7)) # kWh/year
    
    P_out_PSH3_week_array = np.sum(P_out_PSH3_arrays,0)
    P_out_PSH3_year = np.sum(P_out_PSH3_week_array*(days_list/7)) # kWh/year
    
    P_out_PSH4_week_array = np.sum(P_out_PSH4_arrays,0)
    P_out_PSH4_year = np.sum(P_out_PSH4_week_array*(days_list/7)) # kWh/year
    
    P_out_PSH5_week_array = np.sum(P_out_PSH5_arrays,0)
    P_out_PSH5_year = np.sum(P_out_PSH5_week_array*(days_list/7)) # kWh/year
    
    P_out_PSH6_week_array = np.sum(P_out_PSH6_arrays,0)
    P_out_PSH6_year = np.sum(P_out_PSH6_week_array*(days_list/7)) # kWh/year
    
    P_out_PSH7_week_array = np.sum(P_out_PSH7_arrays,0)
    P_out_PSH7_year = np.sum(P_out_PSH7_week_array*(days_list/7)) # kWh/year
    
    P_out_PSH_year = P_out_PSH1_year + P_out_PSH2_year + P_out_PSH3_year + P_out_PSH4_year + P_out_PSH5_year + P_out_PSH6_year + P_out_PSH7_year
    
    """
    P_out_pelton_week_array = np.sum(P_out_pelton_arrays,0)
    P_out_pelton_year = np.sum(P_out_pelton_week_array*(days_list/7)) # kWh/year
    """
    
    # Freshwater total year
    """
    V_dot_fwRO_stage1_arrays = Q_fwRO_stage1_arrays*N_pv1
    V_dot_fwRO_stage2_arrays = Q_fwRO_stage2_arrays*N_pv2
    V_dot_fwRO_arrays = V_dot_fwRO_stage1_arrays+V_dot_fwRO_stage2_arrays
        
    V_fwRO_week_array = np.sum(V_dot_fwRO_arrays,0)
    V_fwRO_year = np.sum(V_fwRO_week_array*(days_list/7)) # m^3/year
    """
    
    # Average eta_RO
    eta_RO_week_array = np.mean(eta_RO_arrays,0)
    eta_RO_avg_year = np.sum((eta_RO_week_array*(days_list/365)))
    
    # Buy/Sell Profit year
    neg_buy_sell_profit_year = np.sum(neg_buy_sell_profit_array*(days_list/7))
    
    # Average eta_ht
    eta_ht_PSH1_week_array = np.mean(eta_ht_PSH1_arrays,0)
    eta_ht_PSH1_avg_year = np.sum((eta_ht_PSH1_week_array*(days_list/365)))
    
    eta_ht_PSH2_week_array = np.mean(eta_ht_PSH2_arrays,0)
    eta_ht_PSH2_avg_year = np.sum((eta_ht_PSH2_week_array*(days_list/365)))
    
    eta_ht_PSH3_week_array = np.mean(eta_ht_PSH3_arrays,0)
    eta_ht_PSH3_avg_year = np.sum((eta_ht_PSH3_week_array*(days_list/365)))
    
    eta_ht_PSH4_week_array = np.mean(eta_ht_PSH4_arrays,0)
    eta_ht_PSH4_avg_year = np.sum((eta_ht_PSH4_week_array*(days_list/365)))
    
    eta_ht_PSH5_week_array = np.mean(eta_ht_PSH5_arrays,0)
    eta_ht_PSH5_avg_year = np.sum((eta_ht_PSH5_week_array*(days_list/365)))
    
    eta_ht_PSH6_week_array = np.mean(eta_ht_PSH6_arrays,0)
    eta_ht_PSH6_avg_year = np.sum((eta_ht_PSH6_week_array*(days_list/365)))
    
    eta_ht_PSH7_week_array = np.mean(eta_ht_PSH7_arrays,0)
    eta_ht_PSH7_avg_year = np.sum((eta_ht_PSH7_week_array*(days_list/365)))
    
    eta_ht_avg_year = np.mean([eta_ht_PSH1_avg_year,eta_ht_PSH2_avg_year,eta_ht_PSH3_avg_year,eta_ht_PSH4_avg_year,
                               eta_ht_PSH5_avg_year,eta_ht_PSH6_avg_year,eta_ht_PSH7_avg_year])
    
    if print_statements == 1:
        print('Start of Year Summator Module Output')
        print('P_in_year: ' + str(P_in_PSH_year))
        print('P_out_PSH_year: ' + str(P_out_PSH_year))
        print('eta_ht_avg_year: ' + str(eta_ht_avg_year))
        print('eta_RO_avg_year: ' + str(eta_RO_avg_year))
        print('neg_buy_sell_profit_year: ' + str(neg_buy_sell_profit_year))
        print('End of Year Summator Module Output')
    
    return [P_in_PSH_year,P_out_PSH_year,eta_ht_avg_year,eta_RO_avg_year,neg_buy_sell_profit_year]




