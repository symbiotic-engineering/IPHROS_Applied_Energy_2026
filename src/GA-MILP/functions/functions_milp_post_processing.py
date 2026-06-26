# -*- coding: utf-8 -*-
"""
Matt Haefner (mwh85)
SEA Lab
Post-MILP Conversion Helper Function
1/21/26
"""

import numpy as np

def milp_post_processing(N_pv1,x_LP,eta_RO_stage1_array,eta_RO_stage2_array,
                         rho_oRO_stage2_array,h_oRO_stage2_array,
                         eta_hp_matrix,eta_ht_matrix,eta_pelton_array,
                         p_state,p_reservoir,p_universal,print_statements):
    
    # Unpack parameters
    rho_sw = p_state['rho_sw'] # kg/m^3
    h_res = p_reservoir['h_res'] # m
    g = p_universal['g'] # m/s^2
    
    # To start off, need to unpack x_LP
    V_dot_wp1_array = x_LP[:,0]
    V_dot_wp2_array = x_LP[:,1]
    V_dot_wp3_array = x_LP[:,2]
    V_dot_wp4_array = x_LP[:,3]
    V_dot_wp5_array = x_LP[:,4]
    V_dot_wp6_array = x_LP[:,5]
    V_dot_wp7_array = x_LP[:,6]
    
    V_dot_swht1_array = x_LP[:,7]
    V_dot_swht2_array = x_LP[:,8]
    V_dot_swht3_array = x_LP[:,9]
    V_dot_swht4_array = x_LP[:,10]
    V_dot_swht5_array = x_LP[:,11]
    V_dot_swht6_array = x_LP[:,12]
    V_dot_swht7_array = x_LP[:,13]
    
    Q_wRO_stage1_array = x_LP[:,14]
    
    # And unpack efficiency matrices
    eta_hp1_array = eta_hp_matrix[:,0]
    eta_hp2_array = eta_hp_matrix[:,1]
    eta_hp3_array = eta_hp_matrix[:,2]
    eta_hp4_array = eta_hp_matrix[:,3]
    eta_hp5_array = eta_hp_matrix[:,4]
    eta_hp6_array = eta_hp_matrix[:,5]
    eta_hp7_array = eta_hp_matrix[:,6]
    
    eta_ht1_array = eta_ht_matrix[:,0]
    eta_ht2_array = eta_ht_matrix[:,1]
    eta_ht3_array = eta_ht_matrix[:,2]
    eta_ht4_array = eta_ht_matrix[:,3]
    eta_ht5_array = eta_ht_matrix[:,4]
    eta_ht6_array = eta_ht_matrix[:,5]
    eta_ht7_array = eta_ht_matrix[:,6]
    
    
    # This first chunk will be getting RO flowrates downstream from the RO inlet
    V_dot_wRO_stage1_array = Q_wRO_stage1_array*N_pv1 # m^3/hr
    V_dot_fwRO_stage1_array = eta_RO_stage1_array*V_dot_wRO_stage1_array
    V_dot_oRO_stage1_array = (1-eta_RO_stage1_array)*V_dot_wRO_stage1_array
    
    V_dot_wRO_stage2_array = V_dot_oRO_stage1_array
    V_dot_fwRO_stage2_array = eta_RO_stage2_array*V_dot_wRO_stage2_array
    V_dot_oRO_stage2_array = (1-eta_RO_stage2_array)*V_dot_wRO_stage2_array
    
    V_dot_fwRO_array = V_dot_fwRO_stage1_array + V_dot_fwRO_stage2_array
    
    # Now we'll get some power values
    P_in_PSH1_array = np.ones(168)*(-69)
    P_in_PSH2_array = np.ones(168)*(-69)
    P_in_PSH3_array = np.ones(168)*(-69)
    P_in_PSH4_array = np.ones(168)*(-69)
    P_in_PSH5_array = np.ones(168)*(-69)
    P_in_PSH6_array = np.ones(168)*(-69)
    P_in_PSH7_array = np.ones(168)*(-69)
    
    for i in range(168): # Need to protect myself against divide by 0 issues
        if eta_hp1_array[i] == 0:
            P_in_PSH1_array[i] = 0
        else:
            P_in_PSH1_array[i] = (rho_sw*V_dot_wp1_array[i]*g*h_res)/(eta_hp1_array[i]*(3.6e6)) # kW
        if eta_hp2_array[i] == 0:
            P_in_PSH2_array[i] = 0
        else:
            P_in_PSH2_array[i] = (rho_sw*V_dot_wp2_array[i]*g*h_res)/(eta_hp2_array[i]*(3.6e6)) # kW
        if eta_hp3_array[i] == 0:
            P_in_PSH3_array[i] = 0
        else:
            P_in_PSH3_array[i] = (rho_sw*V_dot_wp3_array[i]*g*h_res)/(eta_hp3_array[i]*(3.6e6)) # kW
        if eta_hp4_array[i] == 0:
            P_in_PSH4_array[i] = 0
        else:
            P_in_PSH4_array[i] = (rho_sw*V_dot_wp4_array[i]*g*h_res)/(eta_hp4_array[i]*(3.6e6)) # kW
        if eta_hp5_array[i] == 0:
            P_in_PSH5_array[i] = 0
        else:
            P_in_PSH5_array[i] = (rho_sw*V_dot_wp5_array[i]*g*h_res)/(eta_hp5_array[i]*(3.6e6)) # kW
        if eta_hp6_array[i] == 0:
            P_in_PSH6_array[i] = 0
        else:
            P_in_PSH6_array[i] = (rho_sw*V_dot_wp6_array[i]*g*h_res)/(eta_hp6_array[i]*(3.6e6)) # kW
        if eta_hp7_array[i] == 0:
            P_in_PSH7_array[i] = 0
        else:
            P_in_PSH7_array[i] = (rho_sw*V_dot_wp7_array[i]*g*h_res)/(eta_hp7_array[i]*(3.6e6)) # kW
    
    #P_in_PSH_matrix = np.vstack((P_in_PSH1_array,P_in_PSH2_array,P_in_PSH3_array,P_in_PSH4_array,
    #                         P_in_PSH5_array,P_in_PSH6_array,P_in_PSH7_array))
    
    P_out_PSH1_array = (eta_ht1_array*rho_sw*V_dot_swht1_array*g*h_res)/(3.6e6) # kW
    P_out_PSH2_array = (eta_ht2_array*rho_sw*V_dot_swht2_array*g*h_res)/(3.6e6) # kW
    P_out_PSH3_array = (eta_ht3_array*rho_sw*V_dot_swht3_array*g*h_res)/(3.6e6) # kW
    P_out_PSH4_array = (eta_ht4_array*rho_sw*V_dot_swht4_array*g*h_res)/(3.6e6) # kW
    P_out_PSH5_array = (eta_ht5_array*rho_sw*V_dot_swht5_array*g*h_res)/(3.6e6) # kW
    P_out_PSH6_array = (eta_ht6_array*rho_sw*V_dot_swht6_array*g*h_res)/(3.6e6) # kW
    P_out_PSH7_array = (eta_ht7_array*rho_sw*V_dot_swht7_array*g*h_res)/(3.6e6) # kW
    
    #P_out_PSH_matrix = np.vstack((P_out_PSH1_array,P_out_PSH2_array,P_out_PSH3_array,P_out_PSH4_array,
    #                              P_out_PSH5_array,P_out_PSH6_array,P_out_PSH7_array))
    
    P_out_pelton_array = (eta_pelton_array*rho_oRO_stage2_array*V_dot_oRO_stage2_array*g*h_oRO_stage2_array)/(3.6e6) # kW
    
    if print_statements == 1:
        print('Start of MILP Post-Processing Module Output')
        print('V_dot_fwRO_array: ' + str(V_dot_fwRO_array))
        print('V_dot_oRO_stage2_array: ' + str(V_dot_oRO_stage2_array))
        print('P_in_PSH1_array: ' + str(P_in_PSH1_array))
        print('P_in_PSH2_array: ' + str(P_in_PSH2_array))
        print('P_in_PSH3_array: ' + str(P_in_PSH3_array))
        print('P_in_PSH4_array: ' + str(P_in_PSH4_array))
        print('P_in_PSH5_array: ' + str(P_in_PSH5_array))
        print('P_in_PSH6_array: ' + str(P_in_PSH6_array))
        print('P_in_PSH7_array: ' + str(P_in_PSH7_array))
        print('P_out_PSH1_array: ' + str(P_out_PSH1_array))
        print('P_out_PSH2_array: ' + str(P_out_PSH2_array))
        print('P_out_PSH3_array: ' + str(P_out_PSH3_array))
        print('P_out_PSH4_array: ' + str(P_out_PSH4_array))
        print('P_out_PSH5_array: ' + str(P_out_PSH5_array))
        print('P_out_PSH6_array: ' + str(P_out_PSH6_array))
        print('P_out_PSH7_array: ' + str(P_out_PSH7_array))
        print('P_out_pelton_array: ' + str(P_out_pelton_array))
        print('End of MILP Post-Processing Module Output')
    
    
    return [Q_wRO_stage1_array,V_dot_fwRO_array,
            P_in_PSH1_array,P_in_PSH2_array,P_in_PSH3_array,P_in_PSH4_array,P_in_PSH5_array,P_in_PSH6_array,P_in_PSH7_array,
            P_out_PSH1_array,P_out_PSH2_array,P_out_PSH3_array,P_out_PSH4_array,P_out_PSH5_array,P_out_PSH6_array,P_out_PSH7_array,
            P_out_pelton_array]



