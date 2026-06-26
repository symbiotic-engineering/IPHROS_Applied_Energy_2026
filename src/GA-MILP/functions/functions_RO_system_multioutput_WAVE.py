# -*- coding: utf-8 -*-
"""
Matt Haefner (mwh85)
SEA Lab
Desalination Paper Optimization Price Taker - Multi-Output WAVE RO Module
LAST UPDATED: 10/30/25     CREATED: 4/14/25
"""

import numpy as np

from helper_functions.helper_functions_RO_helper_function import RO_helper_function


def ro_system_WAVE_multioutput(N_pv1,h_RO_outlet,V_dot_wRO_array,WAVE_model_bundle,p_state,p_RO_sys,print_statements):
    # Unpack state parameters
    rho_sw = p_state['rho_sw']
    rho_fw = p_state['rho_fw']
    S_sw_mg_kg = p_state['S_sw_mg_kg']
    
    # Unpack RO parameters (note: Q_c only needs to be squared)
    N_pv_rat = p_RO_sys['N_pv_rat']
    convex_polygon_eqns_coeffs = p_RO_sys['convex_polygon_eqns_coeffs']

    # Initialize storage arrays
    V_dot_fwRO_array = np.ones(len(V_dot_wRO_array))*69
    S_fwRO_array = np.ones(len(V_dot_wRO_array))*69
    V_dot_oRO_array = np.ones(len(V_dot_wRO_array))*69
    S_oRO_stage2_array = np.ones(len(V_dot_wRO_array))*69
    rho_oRO_stage2_array = np.ones(len(V_dot_wRO_array))*69
    P_oRO_stage2_array = np.ones(len(V_dot_wRO_array))*69
    h_oRO_stage2_array = np.ones(len(V_dot_wRO_array))*69
    #eta_ROio_array = np.ones(len(V_dot_wRO_array))*69
    eta_RO_array = np.ones(len(V_dot_wRO_array))*69
    eta_RO_stage1_array = np.ones(len(V_dot_wRO_array))*69
    eta_RO_stage2_array = np.ones(len(V_dot_wRO_array))*69
    RO_feasible_space_con_array = np.ones(len(V_dot_wRO_array))*69

    # Create a dictionary of unique (h_RO, V_dot_wRO) pairs with their indices
    unique_pair_dict = {
        pair: np.where((h_RO_outlet == pair[0]) & (V_dot_wRO_array == pair[1]))[0]
        for pair in set(zip(h_RO_outlet, V_dot_wRO_array))
    }

    for i in range(len(unique_pair_dict)):

        # Get a unique RO inlet flowrate, and all the indices that it is associated with
        h_RO = list(unique_pair_dict.keys())[i][0]
        V_dot_wRO = list(unique_pair_dict.keys())[i][1]
        idxs = unique_pair_dict[(h_RO,V_dot_wRO)]

        # Convert head (m) to psi
        P_f_stage1 = (0.0981*h_RO*(rho_sw/rho_fw))*14.5038 # psi (from bar)
        
        # Get the feed flowrate going into a pressure vessel
        Q_f_stage1 = V_dot_wRO/N_pv1 # m^3/hr
        
        # Get the salinity into the proper units for the joblib file
        S_f_stage1 = S_sw_mg_kg*rho_sw/1000 # mg/L
        
        # Stage number
        stage_number_minus_1 = 0
        
        # Pre-process inputs
        X_phys_np_stage1 = np.array([[P_f_stage1, S_f_stage1, Q_f_stage1, stage_number_minus_1]])
        
        # Calculate the first stage output values
        output_stage1 = RO_helper_function(X_phys_np_stage1,WAVE_model_bundle) # [P_c, TDS_c, Q_c, TDS_p, Q_p], but in dictionary form!
        
        # Set up the inputs for the second stage, call the joblib file for the second stage
        P_f_stage2 = output_stage1['P_c'][0] # psi
        S_f_stage2 = output_stage1['TDS_c'][0] # mg/L
        
        V_dot_intermediate = output_stage1['Q_c'][0]*N_pv1 # m^3/hr
        N_pv2 = N_pv1/N_pv_rat
        Q_f_stage2 = V_dot_intermediate/N_pv2
        
        # Next stage
        stage_number_minus_1 = 1
        
        # Pre-process inputs
        X_phys_np_stage2 = np.array([[P_f_stage2, S_f_stage2, Q_f_stage2, stage_number_minus_1]])
        
        output_stage2 = RO_helper_function(X_phys_np_stage2,WAVE_model_bundle) # [P_c, TDS_c, Q_c, TDS_p, Q_p], but in dictionary form!
        
        # Compile RO outputs
        P_c_stage2 = output_stage2['P_c'][0]
        S_c_stage2 = output_stage2['TDS_c'][0]
        Q_c_stage2 = output_stage2['Q_c'][0]
        S_p_stage1 = output_stage1['TDS_p'][0]
        S_p_stage2 = output_stage2['TDS_p'][0]
        Q_p_stage1 = output_stage1['Q_p'][0]
        Q_p_stage2 = output_stage2['Q_p'][0]
        
        V_dot_fwRO = (Q_p_stage1*N_pv1) + (Q_p_stage2*N_pv2) # m^3/hr
        V_dot_fwRO_stage1 = (Q_p_stage1*N_pv1) # m^3/hr
        V_dot_fwRO_stage2 = (Q_p_stage2*N_pv2) # m^3/hr
        
        S_p = ((S_p_stage1*Q_p_stage1)+(S_p_stage2*Q_p_stage2))/(Q_p_stage1+Q_p_stage2) # mg/L
        S_p_g_L = S_p/1000 # g/L
        S_p_kg_kg = max(np.roots([756, 995, (-S_p_g_L)])) # kg/kg, from Bortholomew paper
        S_fwRO_g_kg = S_p_kg_kg*1000 # g/kg
        
        V_dot_oRO = Q_c_stage2*N_pv2 # m^3/hr
        
        #S_f_stage2_g_L = S_f_stage2/1000 # g/L
        #S_f_stage2_kg_kg = max(np.roots([756, 995, (-S_f_stage2_g_L)])) # kg/kg, from Bortholomew paper
        #S_f_stage2_g_kg = S_f_stage2_kg_kg*1000 # g/kg 
        
        #rho_oRO_stage1 = (756*S_f_stage2_kg_kg) + 995 # kg/m^3, from Bortholomew paper
        
        S_c_stage2_g_L = S_c_stage2/1000 # g/L
        S_c_stage2_kg_kg = max(np.roots([756, 995, (-S_c_stage2_g_L)])) # kg/kg, from Bortholomew paper
        S_oRO_stage2_g_kg = S_c_stage2_kg_kg*1000 # g/kg 
        
        rho_oRO_stage2 = (756*S_c_stage2_kg_kg) + 995 # kg/m^3, from Bortholomew paper
        
        #P_oRO_stage1 = P_f_stage2 # psi
        
        #h_oRO_stage1 = (P_oRO_stage1/14.5038)/(0.0981*(rho_oRO_stage1/rho_fw)) # m (pressure was converted to bar from psi)
        
        P_oRO_stage2 = P_c_stage2 # psi
        
        h_oRO_stage2 = (P_oRO_stage2/14.5038)/(0.0981*(rho_oRO_stage2/rho_fw)) # m (pressure was converted to bar from psi)
        #eta_ROio = 1-((P_f_stage1-P_c_stage2)/P_f_stage1) # -
        
        
        if V_dot_wRO > 1:
            eta_RO = V_dot_fwRO/V_dot_wRO # -
            
            eta_RO_stage1 = V_dot_fwRO_stage1/V_dot_wRO # -
            eta_RO_stage2 = V_dot_fwRO_stage2/V_dot_intermediate # -
            
            # Output, specifically WAVE feasibility constraint - STAGE 1 (note: with convex polygon, don't need to do this for stage 2)
            max_violation = 0
            RO_design_point_stage1 = np.array([P_f_stage1, Q_f_stage1]).reshape((1,2))  # Inlet conditions into RO system
            for eq in convex_polygon_eqns_coeffs:
                a, b, c = eq
                signed_dist = (a*RO_design_point_stage1[0,0] + b*RO_design_point_stage1[0,1] + c) / np.linalg.norm([a, b])
                max_violation = max(max_violation, signed_dist)
            RO_feasible_space_con = max_violation
            
        else:
            eta_RO = 0
            eta_RO_stage1 = 0
            eta_RO_stage2 = 0
            RO_feasible_space_con = 0
            
        
        # Store all the calculated values
        V_dot_fwRO_array[idxs] = V_dot_fwRO
        S_fwRO_array[idxs] = S_fwRO_g_kg
        V_dot_oRO_array[idxs] = V_dot_oRO
        S_oRO_stage2_array[idxs] = S_oRO_stage2_g_kg
        rho_oRO_stage2_array[idxs] = rho_oRO_stage2
        P_oRO_stage2_array[idxs] = P_oRO_stage2
        h_oRO_stage2_array[idxs] = h_oRO_stage2
        #eta_ROio_array[idxs] = eta_ROio
        eta_RO_array[idxs] = eta_RO
        eta_RO_stage1_array[idxs] = eta_RO_stage1
        eta_RO_stage2_array[idxs] = eta_RO_stage2
        RO_feasible_space_con_array[idxs] = RO_feasible_space_con
    
    if print_statements == 1:
        print('Start of RO Module Output')
        print('V_dot_fwRO_array: ' + str(V_dot_fwRO_array))
        print('S_fwRO_array: ' + str(S_fwRO_array))
        print('V_dot_oRO_array: ' + str(V_dot_oRO_array))
        print('S_oRO_stage2_array: ' + str(S_oRO_stage2_array))
        print('rho_oRO_stage2_array: ' + str(rho_oRO_stage2_array))
        print('P_oRO_stage2_array: ' + str(P_oRO_stage2_array))
        print('h_oRO_stage2_array: ' + str(h_oRO_stage2_array))
        print('eta_RO_array: ' + str(eta_RO_array))
        print('eta_RO_stage1_array: ' + str(eta_RO_stage1_array))
        print('eta_RO_stage2_array: ' + str(eta_RO_stage2_array))
        print('RO_feasible_space_con_array: ' + str(RO_feasible_space_con_array))
        print('End of RO Module Output')
              
        

    return [V_dot_fwRO_array,S_fwRO_array,V_dot_oRO_array,S_oRO_stage2_array,rho_oRO_stage2_array,P_oRO_stage2_array,h_oRO_stage2_array,
            eta_RO_array,eta_RO_stage1_array,eta_RO_stage2_array,RO_feasible_space_con_array]

