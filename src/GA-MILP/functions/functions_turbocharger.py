# -*- coding: utf-8 -*-
"""
Matthew Haefner (mwh85)
SEA Lab
Desalination Paper Optimization Price Taker - RO Inlet Ajuster Helper Function
LAST UPDATED: 10/21/25     CREATED: 3/10/25
"""

import numpy as np

def turbocharger(P_c_array,S_c_array,rho_c_array,h_turbo_inlet_array,p_state,p_francis_turbo_pelton_coeffs,print_statements,in_flow_adjuster):
    # Unpack state parameters
    rho_sw = p_state['rho_sw']
    rho_fw = p_state['rho_fw']
    
    # Unpack regression parameters
    a = p_francis_turbo_pelton_coeffs['a_turbo']
    b = p_francis_turbo_pelton_coeffs['b_turbo']
    c = p_francis_turbo_pelton_coeffs['c_turbo']
    d = p_francis_turbo_pelton_coeffs['d_turbo']
      
    # Initialize h_turbo_outlet_array
    h_turbo_outlet_array = np.ones(len(P_c_array))*69
    
    # Create a dictionary of unique (h_turbo_outlet, V_dot_wRO) pairs with their indices
    unique_quad_dict = {
        quad: np.where((P_c_array == quad[0]) & (S_c_array == quad[1]) & (rho_c_array == quad[2]) & (h_turbo_inlet_array == quad[3]))[0]
        for quad in set(zip(P_c_array, S_c_array, rho_c_array, h_turbo_inlet_array))
    }

    for i in range(len(unique_quad_dict)):

        # Get a unique RO inlet flowrate, and all the indices that it is associated with
        P_c = list(unique_quad_dict.keys())[i][0]
        S_c = list(unique_quad_dict.keys())[i][1]
        rho_c = list(unique_quad_dict.keys())[i][2]
        h_turbo_inlet = list(unique_quad_dict.keys())[i][3]
        idxs = unique_quad_dict[(P_c,S_c,rho_c,h_turbo_inlet)]
        
        # First need to convert units
        P_inlet_bar = 0.0981*h_turbo_inlet*(rho_sw/rho_fw) # bar 
        P_c_bar = P_c/14.504 # bar (from psi)
        S_c_mg_L = S_c*rho_c # mg/L
        
        # Use regression to get the outlet pressure
        P_f_bar = a + (b*P_c_bar) + (c*P_inlet_bar) + (d*P_c_bar*S_c_mg_L)
        
        # Convert back to head
        h_turbo_outlet = P_f_bar/(0.0981*(rho_sw/rho_fw))
        
        # Store the head height
        h_turbo_outlet_array[idxs] = np.ones(len(idxs))*h_turbo_outlet
        
    if print_statements == 1 and in_flow_adjuster != 1:
        print('Start of Turbocharger Module Output')
        print('h_turbo_outlet_array: ' + str(h_turbo_outlet_array))
        print('End of Turbocharger Module Output')
    
    return h_turbo_outlet_array