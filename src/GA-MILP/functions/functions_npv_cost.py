"""
Matt Haefner (mwh85)
SEA Lab
Desalination Paper Optimization Price Taker - End-of-year Values Module
LAST UPDATED: 10/29/25     CREATED: 9/14/24
"""

import numpy as np

# Import Helper Functions
from helper_functions.helper_functions_voutchkov_ro_cost_model import opex,capex

def npv_cost(B_PSH,B_RO,P_in_PSH_year,P_out_PSH_year,eta_ht_avg_year,neg_buy_sell_profit_year,eta_RO_avg_year,
             p_financial,p_assumed,p_RO_capex_coeffs,p_state,p_RO_capex_project,p_RO_capex_total,p_RO_opex_coeffs,print_statements):
    """
    Calculate the Net Present Value (NPV) of an investment.

    :param initial_investment: The initial investment cost (C_0)
    :param cash_flows: List of expected cash flows for each period (C_t)
    :param discount_rate: The discount rate (r) as a decimal (e.g., 0.05 for 5%)
    :return: Net Present Value (NPV)
    
    VERY KEY NOTE: Energy Recovery Device Cost is embodied in the Voutchkov RO cost model!!!
    """
    
    # Real quick, make the operational profit from the LP positive
    buy_sell_profit_year = -neg_buy_sell_profit_year
    #print('positive_buy_sell_profit_year = ' + str(buy_sell_profit_year))
    
    # Financial parameters
    r = p_financial['r']
    n = p_financial['n']
    eta_capex_RO = p_financial['eta_capex_RO']
    eta_opex_RO = p_financial['eta_opex_RO']
    avg_e_price = p_financial['avg_e_price']

    # State parameters
    S_sw_g_kg = p_state['S_sw_g_kg']
    rho_sw = p_state['rho_sw']
    S_sw_mg_L = S_sw_g_kg*rho_sw
    
    # Assumed parameters
    pipe_length = p_assumed['pipe_length']
    plant_sec = p_assumed['plant_sec']
    
    ############################
    #
    # Helper terms
    #
    ############################
    
    # CAPEX_RO
    if eta_RO_avg_year > 0:
        perm = B_RO
        feed = B_RO/eta_RO_avg_year
        CAPEX_RO = max(0,capex(feed,perm,p_RO_capex_coeffs,S_sw_mg_L,pipe_length,avg_e_price,plant_sec,p_RO_capex_project,p_RO_capex_total))
        CAPEX_RO_IPHROS = CAPEX_RO*eta_capex_RO
        #print('CAPEX_RO_IPHROS = ' + str(CAPEX_RO_IPHROS))
    else:
        CAPEX_RO_IPHROS = 0
    

    # CAPEX_PSH (NOTE: the cost of any energy recovery device is assumed to be embodied in the cost of the RO system!)
    CAPEX2_PSH = 1.2e9 # 2014 euros
    CAPEX2_PSH = CAPEX2_PSH*1.27 # 2023 euros
    CAPACITY2_PSH = 1500e3 # kW
    CAPACITY1_PSH = B_PSH # kW
    CAPEX_PSH = CAPEX2_PSH*((CAPACITY1_PSH/CAPACITY2_PSH)**1.1)
    #print('CAPEX_PSH = ' + str(CAPEX_PSH))
    
    CAPEX_REV = CAPEX_PSH
    # CAPEX_SEP = (2*0.322*CAPEX_PSH) + ((1-0.322)*CAPEX_PSH)
    
    # OPEX_RO
    if eta_RO_avg_year > 0:
        #print(feed)
        #print(perm)
        OPEX_RO = max(0,opex(feed,perm,p_RO_opex_coeffs,S_sw_mg_L,pipe_length,avg_e_price,plant_sec))
        OPEX_RO_IPHROS = eta_opex_RO*OPEX_RO
        #print('OPEX_RO_IPHROS = ' + str(OPEX_RO_IPHROS))
    else:
        OPEX_RO_IPHROS = 0
    
    # OPEX_BRINE
    '''
    P_brine = B_pelton/1000 # MW
    AE_brine = E_dot_brine_year/eta_ht # kWh
    AE_brine = AE_brine/1000 # MWh (per year)
    OPEX_brine = 34730*(P_brine**0.32)*(AE_brine**0.33) # 2016 USD
    OPEX_brine = OPEX_brine*1.31 # 2023 USD
    OPEX_brine = OPEX_brine/1.0813 # 2023 euros
    #print('OPEX_brine = ' + str(OPEX_brine))
    '''
    
    # OPEX_PSH
    P_PSH_kW = B_PSH # kW
    P_PSH_MW = P_PSH_kW/1000 # MW
    AE_PSH_kWh = P_in_PSH_year+(P_out_PSH_year/eta_ht_avg_year) # kWh
    AE_PSH_MWh = AE_PSH_kWh/1000 # MWh
    AE_PSH_GWh = AE_PSH_MWh/1000 # GWh
    OPEX_PSH = 34730*(P_PSH_MW**0.32)*(AE_PSH_GWh**0.33) # 1987 USD
    OPEX_PSH = OPEX_PSH*2.68 # 2023 USD
    OPEX_PSH = OPEX_PSH/1.0813 # 2023 euros
    #print('OPEX_PSH = ' + str(OPEX_PSH))
    
    OPEX_REV = OPEX_PSH
    #OPEX_SEP = (2*0.382*OPEX_PSH) + ((1-0.382)*OPEX_PSH)
    
    ############################
    #
    # Get into the actual calculation
    #
    ############################
    
    # Total initial investment cost
    initial_investment = CAPEX_RO_IPHROS + CAPEX_REV 
    #print('initial_investment = ' + str(initial_investment))
    
    # Calculate cash flow
    cash_flow = buy_sell_profit_year - OPEX_RO_IPHROS - OPEX_REV# - OPEX_brine
    cash_flows = np.ones(n)*cash_flow
    #print('cash_flow = ' + str(cash_flow))
    
    # Initialize NPV
    npv = -initial_investment  # Initial investment is an outflow (negative)
    #print('initial_npv = ' + str(npv))
    
    # Discount rate
    discount_rate = r
    
    for t, cash_flow in enumerate(cash_flows, start=1):
        npv += cash_flow / (1 + discount_rate) ** t
        #print('The NPV after year ' + str(t) + ' is ' + str(npv))
    #print('npv = ' + str(npv))
    
    if print_statements == 1:
        print('Start of NPV Module Outputs')
        print('CAPEX_RO_IPHROS = ' + str(CAPEX_RO_IPHROS))
        print('CAPEX_PSH = ' + str(CAPEX_PSH))
        print('OPEX_RO_IPHROS = ' + str(OPEX_RO_IPHROS))
        print('OPEX_PSH = ' + str(OPEX_PSH))
        print('initial_investment = ' + str(initial_investment))
        print('cash_flow = ' + str(cash_flow))
        print('End of NPV Module Outputs')
    
    return npv
