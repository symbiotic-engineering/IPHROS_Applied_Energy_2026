"""
Matt Haefner (mwh85)
SEA Lab
Desalination Paper Optimization Price Taker - MILP Inner Loop (SIMPLIFIED)
CREATED: 12/16/25
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
import time


#-----------------------------------------------------------------------------
# NOTE: On 2/15/26, I modified my logic here relative to the "parent" of this 
# code where if a flow for a given hour is decided to be zero in a previous 
# iteration, then it is upper bounded by zero. What was revealed after working 
# with this simplified MILP version for a bit is that removing the equality 
# constraints relating power and flow quantities removed the logic I had 
# indirectly implied where if an efficiency was set to zero in the 
# efficiencies() function due to a zero flow from a previous MILP iteration 
# then the corresponding flow for the next iteration was also zero. 
# Implementing this upper bound of 0 logic should restore that reasoning
#-----------------------------------------------------------------------------


def individual_hour_maker(n_vars,n_pump_turbines,prev,single):
    '''

    Parameters
    ----------
    n_vars : int
        number of variables in an hour
    n_pump_turbines : int
        number of turbines
    prev : int
        Equals 1 if the constraint depends on a value from a previous hour
        Equals 0 otherwise
    single : int
        Equals 1 if the constraint does not pertain to the turbines
        Equals 0 otherwise

    Returns
    -------
    individual_hour_dict : dictionary
        DESCRIPTION.

    '''
    
    individual_hour_dict = {}
    if single == 0: 
        if prev == 0:
            for i in range(n_pump_turbines):
                individual_hour_dict[str(i+1)] = list(np.zeros(n_vars))
        else:
            for i in range(n_pump_turbines):
                individual_hour_dict[str(i+1)+'_h'] = list(np.zeros(n_vars))
                individual_hour_dict[str(i+1)+'_h_minus_1'] = list(np.zeros(n_vars))
    else:
        if prev == 0:
            individual_hour_dict['0'] = list(np.zeros(n_vars))
        else:
            individual_hour_dict['0_h'] = list(np.zeros(n_vars))
            individual_hour_dict['0_h_minus_1'] = list(np.zeros(n_vars))
    
    return individual_hour_dict


def my_milp_reversible_simplified(energy_prices,water_price,B_PSH,N_pv1,ERD_choice,
                                  eta_RO_stage1_array,eta_RO_stage2_array,h_oRO_stage2_array,
                                  rho_oRO_stage2_array,S_oRO_stage2_array,h_RO_array,
                                  S_ht_array,eta_hp_matrix,eta_ht_matrix,eta_pelton_array,solve_status,
                                  p_state,p_RO_sys,p_universal,p_reservoir,p_assumed,epsilon,print_statements,
                                  iteration):
    
    """
    print('energy_prices: ' +str(energy_prices))
    print('water_price: ' +str(water_price))
    print('B_PSH: ' +str(B_PSH))
    print('N_pv1: ' +str(N_pv1))
    print('ERD_choice: ' +str(ERD_choice))
    print('eta_RO_stage1_array: ' +str(eta_RO_stage1_array))
    print('eta_RO_stage2_array: ' +str(eta_RO_stage2_array))
    print('h_oRO_stage2_array: ' +str(h_oRO_stage2_array))
    print('rho_oRO_stage2_array: ' +str(rho_oRO_stage2_array))
    print('S_oRO_stage2_array: ' +str(S_oRO_stage2_array))
    print('h_RO_array: ' +str(h_RO_array))
    print('S_ht_array: ' +str(S_ht_array))
    print('eta_hp_matrix: ' +str(eta_hp_matrix))
    print('eta_ht_matrix: ' +str(eta_ht_matrix))
    print('eta_pelton_array: ' +str(eta_pelton_array))
    print('solve_status: ' +str(solve_status))
    
    time.sleep(60)
    """
    
    ########################### Parameters ###################################
    
    # State parameters
    rho_sw = p_state['rho_sw']
    rho_fw = p_state['rho_fw']
    S_sw_g_kg = p_state['S_sw_g_kg']
    
    # Universal constants
    g = p_universal['g']
    
    # Reservoir parameters
    h_res = p_reservoir['h_res']
    V_res_init = p_reservoir['res_init']
    V_res_cap = p_reservoir['res_cap']
    
    # RO parameters
    N_pv_rat = p_RO_sys['N_pv_rat']
    N_pv2 = N_pv1/N_pv_rat
    reduced_convex_polygon_eqns_coeffs = p_RO_sys['reduced_convex_polygon_eqns_coeffs']
    n_polygon_eqns = np.shape(reduced_convex_polygon_eqns_coeffs)[0]
    
    # Assumed parameters
    n_pump_turbines = p_assumed['n_pump_turbines']
    eta_francis_max_flow = p_assumed['eta_francis_max_flow']
    # eta_pelton_max_flow = p_assumed['eta_pelton_max_flow'] # Never used (this is ok though, it's only relevant in the efficiencies function)
      
    # Parameters specifically for LP 
    n_vars = 45
    n_hours = 24*7
    n_vars_total = n_vars*n_hours
    M_RO_con = 1e4
    n_cycle_var_fams = 2
    
    # Unequal nameplates (Will need to change this later)
    nominal_nameplate = B_PSH/7
    nameplates = [nominal_nameplate*(1+(3*epsilon)),
                  nominal_nameplate*(1+(2*epsilon)),
                  nominal_nameplate*(1+(1*epsilon)),
                  nominal_nameplate,
                  nominal_nameplate*(1-(1*epsilon)),
                  nominal_nameplate*(1-(2*epsilon)),
                  nominal_nameplate*(1-(3*epsilon))
                  ]
    if print_statements == 1:
        print('epsilon: ' + str(epsilon))
        print('nameplates:' + str(nameplates))
    
    # Reshape eta_hp and eta_ht matrices to make them more compatible with this framework
    eta_hp_turbine1_array = eta_hp_matrix[:,0]
    eta_hp_turbine2_array = eta_hp_matrix[:,1]
    eta_hp_turbine3_array = eta_hp_matrix[:,2]
    eta_hp_turbine4_array = eta_hp_matrix[:,3]
    eta_hp_turbine5_array = eta_hp_matrix[:,4]
    eta_hp_turbine6_array = eta_hp_matrix[:,5]
    eta_hp_turbine7_array = eta_hp_matrix[:,6]
    
    eta_ht_turbine1_array = eta_ht_matrix[:,0]
    eta_ht_turbine2_array = eta_ht_matrix[:,1]
    eta_ht_turbine3_array = eta_ht_matrix[:,2]
    eta_ht_turbine4_array = eta_ht_matrix[:,3]
    eta_ht_turbine5_array = eta_ht_matrix[:,4]
    eta_ht_turbine6_array = eta_ht_matrix[:,5]
    eta_ht_turbine7_array = eta_ht_matrix[:,6]

    
    # Define the indices of my variables
    my_idxs = {'V_dot_wp_idxs':[[i] for i in range(0, 7)],
               'V_dot_swht_idxs':[[i] for i in range(7, 14)],
               'Q_wRO_stage1_idx':[14],
               'V_res_idx':[15],
               'r_idx':[16],
               'b_con_idxs':[[i] for i in range(17, 24)],
               'b_gen_idxs':[[i] for i in range(24, 31)],
               'v_idxs':[[i] for i in range(31, 38)],
               'w_idxs':[[i] for i in range(38, 45)]
    }
    
    
    #################### Defining Inequality Constraints ####################
    # Initialize inequality constraint matrices
    A_ineq_Reversible_uphills_lb = np.zeros((n_hours,n_vars_total,n_pump_turbines))
    A_ineq_Reversible_uphills_ub = np.zeros((n_hours,n_vars_total,n_pump_turbines))
    A_ineq_Reversible_downhills_lb = np.zeros((n_hours,n_vars_total,n_pump_turbines))
    A_ineq_Reversible_downhills_ub = np.zeros((n_hours,n_vars_total,n_pump_turbines))
    A_ineq_Mixing = np.zeros((n_hours,n_vars_total))
    A_ineq_RO_on_off_lb = np.zeros((n_hours,n_vars_total)) # Constraints that dictate if RO system is on or off
    A_ineq_RO_on_off_ub = np.zeros((n_hours,n_vars_total)) # Constraints that dictate if RO system is on or off
    A_ineq_RO_feasibility = np.zeros((n_hours*n_polygon_eqns,n_vars_total)) # RO feasibility
    A_ineq_Siva_eqn = np.zeros((n_hours,n_vars_total,n_pump_turbines))
    A_ineq_Upshifts = np.zeros((n_hours-1,n_vars_total,n_pump_turbines)) # pump/turbine upshift counter
    A_ineq_Downshifts = np.zeros((n_hours-1,n_vars_total,n_pump_turbines)) # pump/turbine downshift counter
    A_ineq_Cycles = np.zeros((1,n_vars_total,n_pump_turbines)) # pump/turbine total cycle count
                
    
    # RO on/off constraints
    A_ineq_RO_on_off_lb_individual_hour_dict = individual_hour_maker(n_vars,n_pump_turbines,0,1)
    A_ineq_RO_on_off_ub_individual_hour_dict = individual_hour_maker(n_vars,n_pump_turbines,0,1)
    for var in range(n_vars):
        if var in my_idxs['Q_wRO_stage1_idx']:
            A_ineq_RO_on_off_lb_individual_hour_dict['0'][var] = -1
            A_ineq_RO_on_off_ub_individual_hour_dict['0'][var] = 1
        elif var in my_idxs['r_idx']:
            A_ineq_RO_on_off_lb_individual_hour_dict['0'][var] = 3.41
            A_ineq_RO_on_off_ub_individual_hour_dict['0'][var] = -15.5
            
            
    # The "Siva Equation" 
    A_ineq_Siva_eqn_individual_hour_dict = individual_hour_maker(n_vars,n_pump_turbines,0,0)
    for pump_turbine in range(n_pump_turbines):
        for var in range(n_vars):
            if var in my_idxs['b_con_idxs'][pump_turbine]:
                A_ineq_Siva_eqn_individual_hour_dict[str(pump_turbine+1)][var] = 1
            elif var in my_idxs['b_gen_idxs'][pump_turbine]:
                A_ineq_Siva_eqn_individual_hour_dict[str(pump_turbine+1)][var] = 1
    
    
    # Upshift constraints
    A_ineq_Upshifts_individual_hours_dict = individual_hour_maker(n_vars,n_pump_turbines,1,0)
    for pump_turbine in range(n_pump_turbines):
        for var in range(n_vars):
            if var in my_idxs['b_con_idxs'][pump_turbine]:
                A_ineq_Upshifts_individual_hours_dict[str(pump_turbine+1)+'_h'][var] = -1
                A_ineq_Upshifts_individual_hours_dict[str(pump_turbine+1)+'_h_minus_1'][var] = 1
            elif var in my_idxs['b_gen_idxs'][pump_turbine]:
                A_ineq_Upshifts_individual_hours_dict[str(pump_turbine+1)+'_h'][var] = 1
                A_ineq_Upshifts_individual_hours_dict[str(pump_turbine+1)+'_h_minus_1'][var] = -1
            elif var in my_idxs['v_idxs'][pump_turbine]:
                A_ineq_Upshifts_individual_hours_dict[str(pump_turbine+1)+'_h'][var] = -1

    
    # Downshift constraints
    A_ineq_Downshifts_individual_hours_dict = individual_hour_maker(n_vars,n_pump_turbines,1,0)
    for pump_turbine in range(n_pump_turbines):
        for var in range(n_vars):
            if var in my_idxs['b_con_idxs'][pump_turbine]:
                A_ineq_Downshifts_individual_hours_dict[str(pump_turbine+1)+'_h'][var] = 1
                A_ineq_Downshifts_individual_hours_dict[str(pump_turbine+1)+'_h_minus_1'][var] = -1
            elif var in my_idxs['b_gen_idxs'][pump_turbine]:
                A_ineq_Downshifts_individual_hours_dict[str(pump_turbine+1)+'_h'][var] = -1
                A_ineq_Downshifts_individual_hours_dict[str(pump_turbine+1)+'_h_minus_1'][var] = 1
            elif var in my_idxs['w_idxs'][pump_turbine]:
                A_ineq_Downshifts_individual_hours_dict[str(pump_turbine+1)+'_h'][var] = -1
    

    # Cycle counting constraints
    A_ineq_Cycles_individual_hour_dict = individual_hour_maker(n_vars,n_pump_turbines,0,0)
    for pump_turbine in range(n_pump_turbines):
        for var in range(n_vars):
            if var in my_idxs['v_idxs'][pump_turbine]:
                A_ineq_Cycles_individual_hour_dict[str(pump_turbine+1)][var] = 1
            elif var in my_idxs['w_idxs'][pump_turbine]:
                A_ineq_Cycles_individual_hour_dict[str(pump_turbine+1)][var] = 1

    
    '''
    All other inequality constraints (mixing) have to be defined in a loop since they depend on array-based values
    '''     
    
    row_insert = 0
    for n in range(n_hours):
        
        rho_oRO_stage2 = rho_oRO_stage2_array[n]
        S_oRO_stage2 = S_oRO_stage2_array[n]
        S_ht = S_ht_array[n]
        h_RO = h_RO_array[n]
        eta_RO_stage1 = eta_RO_stage1_array[n]
        eta_RO_stage2 = eta_RO_stage2_array[n]
        
        
        # V_dot_wps constraints
        A_ineq_Reversible_uphills_lb_individual_hour_dict = individual_hour_maker(n_vars,n_pump_turbines,0,0)
        A_ineq_Reversible_uphills_ub_individual_hour_dict = individual_hour_maker(n_vars,n_pump_turbines,0,0)
        for pump_turbine in range(n_pump_turbines):
            for var in range(n_vars):
                if var in my_idxs['V_dot_wp_idxs'][pump_turbine]:
                    A_ineq_Reversible_uphills_lb_individual_hour_dict[str(pump_turbine+1)][var] = -1
                    A_ineq_Reversible_uphills_ub_individual_hour_dict[str(pump_turbine+1)][var] = 1
                elif var in my_idxs['b_con_idxs'][pump_turbine]:
                    if pump_turbine == 0: 
                        V_dot_wp_max = eta_francis_max_flow*nameplates[pump_turbine]*(3.6e6)/(rho_sw*g*h_res)
                    elif pump_turbine == 1: 
                        V_dot_wp_max = eta_francis_max_flow*nameplates[pump_turbine]*(3.6e6)/(rho_sw*g*h_res)
                    elif pump_turbine == 2:  
                        V_dot_wp_max = eta_francis_max_flow*nameplates[pump_turbine]*(3.6e6)/(rho_sw*g*h_res)
                    elif pump_turbine == 3: 
                        V_dot_wp_max = eta_francis_max_flow*nameplates[pump_turbine]*(3.6e6)/(rho_sw*g*h_res)
                    elif pump_turbine == 4:  
                        V_dot_wp_max = eta_francis_max_flow*nameplates[pump_turbine]*(3.6e6)/(rho_sw*g*h_res)
                    elif pump_turbine == 5: 
                        V_dot_wp_max = eta_francis_max_flow*nameplates[pump_turbine]*(3.6e6)/(rho_sw*g*h_res)
                    elif pump_turbine == 6: 
                        V_dot_wp_max = eta_francis_max_flow*nameplates[pump_turbine]*(3.6e6)/(rho_sw*g*h_res)
                    
                    A_ineq_Reversible_uphills_lb_individual_hour_dict[str(pump_turbine+1)][var] = 0.212*V_dot_wp_max
                    A_ineq_Reversible_uphills_ub_individual_hour_dict[str(pump_turbine+1)][var] = -V_dot_wp_max


        # V_dot_swhts constraints
        A_ineq_Reversible_downhills_lb_individual_hour_dict = individual_hour_maker(n_vars,n_pump_turbines,0,0)
        A_ineq_Reversible_downhills_ub_individual_hour_dict = individual_hour_maker(n_vars,n_pump_turbines,0,0)
        for pump_turbine in range(n_pump_turbines):
            for var in range(n_vars):
                if var in my_idxs['V_dot_swht_idxs'][pump_turbine]:
                    A_ineq_Reversible_downhills_lb_individual_hour_dict[str(pump_turbine+1)][var] = -1
                    A_ineq_Reversible_downhills_ub_individual_hour_dict[str(pump_turbine+1)][var] = 1
                elif var in my_idxs['b_gen_idxs'][pump_turbine]:
                    if pump_turbine == 0: 
                        V_dot_swht_max = nameplates[pump_turbine]*(3.6e6)/(eta_francis_max_flow*rho_sw*g*h_res)
                    elif pump_turbine == 1: 
                        V_dot_swht_max = nameplates[pump_turbine]*(3.6e6)/(eta_francis_max_flow*rho_sw*g*h_res)
                    elif pump_turbine == 2:  
                        V_dot_swht_max = nameplates[pump_turbine]*(3.6e6)/(eta_francis_max_flow*rho_sw*g*h_res)
                    elif pump_turbine == 3: 
                        V_dot_swht_max = nameplates[pump_turbine]*(3.6e6)/(eta_francis_max_flow*rho_sw*g*h_res)
                    elif pump_turbine == 4:  
                        V_dot_swht_max = nameplates[pump_turbine]*(3.6e6)/(eta_francis_max_flow*rho_sw*g*h_res)
                    elif pump_turbine == 5: 
                        V_dot_swht_max = nameplates[pump_turbine]*(3.6e6)/(eta_francis_max_flow*rho_sw*g*h_res)
                    elif pump_turbine == 6: 
                        V_dot_swht_max = nameplates[pump_turbine]*(3.6e6)/(eta_francis_max_flow*rho_sw*g*h_res)
                        
                    A_ineq_Reversible_downhills_lb_individual_hour_dict[str(pump_turbine+1)][var] = 0.212*V_dot_swht_max
                    A_ineq_Reversible_downhills_ub_individual_hour_dict[str(pump_turbine+1)][var] = -V_dot_swht_max
        
        
        # RO feasibility constraints
        A_ineq_RO_feasibility_individual_hour = np.zeros((n_polygon_eqns,n_vars))
        for var in range(n_vars):
            if var in my_idxs['Q_wRO_stage1_idx']:
                A_ineq_RO_feasibility_individual_hour[:,var] = reduced_convex_polygon_eqns_coeffs[:,1]
            elif var in my_idxs['r_idx']:
                A_ineq_RO_feasibility_individual_hour[:,var] = M_RO_con
                
        
        # Mixing constraint
        A_ineq_Mixing_individual_hour = individual_hour_maker(n_vars,n_pump_turbines,0,1)
        C_mix1a = (S_oRO_stage2*rho_oRO_stage2*N_pv2)-(S_ht*rho_oRO_stage2*N_pv2)
        C_mix1b = (1-eta_RO_stage2)*(N_pv1/N_pv2)*(1-eta_RO_stage1)
        C_mix1 = C_mix1a*C_mix1b
        C_mix2 = (S_sw_g_kg*rho_sw)-(S_ht*rho_sw)
        for pump_turbine in range(n_pump_turbines):
            for var in range(n_vars):
                if var in my_idxs['V_dot_swht_idxs'][pump_turbine]:
                    A_ineq_Mixing_individual_hour['0'][var] = C_mix2
                elif var in my_idxs['Q_wRO_stage1_idx']:
                    A_ineq_Mixing_individual_hour['0'][var] = C_mix1
        
        if row_insert == n:
            idx = row_insert*n_vars
            for pump_turbine in range(n_pump_turbines):
                A_ineq_Reversible_uphills_lb[n,idx:idx+n_vars,pump_turbine] = A_ineq_Reversible_uphills_lb_individual_hour_dict[str(pump_turbine+1)]
                A_ineq_Reversible_uphills_ub[n,idx:idx+n_vars,pump_turbine] = A_ineq_Reversible_uphills_ub_individual_hour_dict[str(pump_turbine+1)]
                A_ineq_Reversible_downhills_lb[n,idx:idx+n_vars,pump_turbine] = A_ineq_Reversible_downhills_lb_individual_hour_dict[str(pump_turbine+1)]
                A_ineq_Reversible_downhills_ub[n,idx:idx+n_vars,pump_turbine] = A_ineq_Reversible_downhills_ub_individual_hour_dict[str(pump_turbine+1)]
                A_ineq_Siva_eqn[n,idx:idx+n_vars,pump_turbine] = A_ineq_Siva_eqn_individual_hour_dict[str(pump_turbine+1)]
            
            A_ineq_RO_on_off_lb[n,idx:idx+n_vars] = A_ineq_RO_on_off_lb_individual_hour_dict['0']
            A_ineq_RO_on_off_ub[n,idx:idx+n_vars] = A_ineq_RO_on_off_ub_individual_hour_dict['0']
            A_ineq_RO_feasibility[n*n_polygon_eqns:(n+1)*n_polygon_eqns,idx:idx+n_vars] = A_ineq_RO_feasibility_individual_hour
            A_ineq_Mixing[n,idx:idx+n_vars] = A_ineq_Mixing_individual_hour['0']
            
            if row_insert != 0: # Skip the first row, since there is no constraint for the first hour
                for pump_turbine in range(n_pump_turbines):
                    A_ineq_Upshifts[(n-1),idx-n_vars:idx,pump_turbine] = A_ineq_Upshifts_individual_hours_dict[str(pump_turbine+1)+'_h_minus_1']
                    A_ineq_Upshifts[(n-1),idx:idx+n_vars,pump_turbine] = A_ineq_Upshifts_individual_hours_dict[str(pump_turbine+1)+'_h']
                    A_ineq_Downshifts[(n-1),idx-n_vars:idx,pump_turbine] = A_ineq_Downshifts_individual_hours_dict[str(pump_turbine+1)+'_h_minus_1']
                    A_ineq_Downshifts[(n-1),idx:idx+n_vars,pump_turbine] = A_ineq_Downshifts_individual_hours_dict[str(pump_turbine+1)+'_h']
                
            row_insert += 1

    for h in range(1,n_hours):
        idx = h*n_vars # Need to skip the first hour
        idx1 = (h+1)*n_vars
        
        for pump_turbine in range(n_pump_turbines):
            A_ineq_Cycles[0,idx:idx1,pump_turbine] = A_ineq_Cycles_individual_hour_dict[str(pump_turbine+1)]
    
    
    # Compile A_ineq
    A_ineq_Reversible_uphills_lb_2D = A_ineq_Reversible_uphills_lb.transpose(2,0,1).reshape(n_pump_turbines*n_hours,n_vars_total)
    A_ineq_Reversible_uphills_ub_2D = A_ineq_Reversible_uphills_ub.transpose(2,0,1).reshape(n_pump_turbines*n_hours,n_vars_total)
    A_ineq_Reversible_downhills_lb_2D = A_ineq_Reversible_downhills_lb.transpose(2,0,1).reshape(n_pump_turbines*n_hours,n_vars_total)
    A_ineq_Reversible_downhills_ub_2D = A_ineq_Reversible_downhills_ub.transpose(2,0,1).reshape(n_pump_turbines*n_hours,n_vars_total)
    A_ineq_Siva_eqn_2D = A_ineq_Siva_eqn.transpose(2,0,1).reshape(n_pump_turbines*n_hours,n_vars_total)
    A_ineq_Upshifts_2D = A_ineq_Upshifts.transpose(2,0,1).reshape((n_pump_turbines*n_hours-(1*n_pump_turbines)),n_vars_total) # the -(1*n_pump_turbines) is to get rid of the first hour of these constraints for each turbine, since the counting starts on the second hour 
    A_ineq_Downshifts_2D = A_ineq_Downshifts.transpose(2,0,1).reshape((n_pump_turbines*n_hours-(1*n_pump_turbines)),n_vars_total)
    A_ineq_Cycles_2D = A_ineq_Cycles.transpose(2,0,1).reshape(n_pump_turbines*1,n_vars_total)
    
    A_ineq = np.vstack((A_ineq_Reversible_uphills_lb_2D,
                        A_ineq_Reversible_uphills_ub_2D,
                        A_ineq_Reversible_downhills_lb_2D,
                        A_ineq_Reversible_downhills_ub_2D,
                        A_ineq_Mixing,
                        A_ineq_RO_on_off_lb,
                        A_ineq_RO_on_off_ub,
                        A_ineq_RO_feasibility,
                        A_ineq_Siva_eqn_2D,
                        A_ineq_Upshifts_2D,
                        A_ineq_Downshifts_2D,
                        A_ineq_Cycles_2D))    

    
    n_ineq_eqns = np.shape(A_ineq)[0]


    # Declare inequality constraint vector values
    b_ineq_Reversible_uphills_lb_2D = np.zeros(n_pump_turbines*n_hours)
    b_ineq_Reversible_uphills_ub_2D = np.zeros(n_pump_turbines*n_hours)
    b_ineq_Reversible_downhills_lb_2D = np.zeros(n_pump_turbines*n_hours)
    b_ineq_Reversible_downhills_ub_2D = np.zeros(n_pump_turbines*n_hours)
    b_ineq_Mixing = np.zeros(n_hours)
    b_ineq_RO_on_off_lb = np.zeros(n_hours)
    b_ineq_RO_on_off_ub = np.zeros(n_hours)
    b_ineq_RO_feasibility = np.ones(n_hours*n_polygon_eqns)*69 # just being initialized
    b_ineq_Siva_eqn_2D = np.ones(n_pump_turbines*n_hours)
    b_ineq_Upshifts_2D = np.zeros(n_pump_turbines*n_hours-(1*n_pump_turbines)) # the -(1*n_pump_turbines) at the end is from taking off the first hour constraint for each turbine
    b_ineq_Downshifts_2D = np.zeros(n_pump_turbines*n_hours-(1*n_pump_turbines))
    b_ineq_Cycles_2D = np.ones(n_pump_turbines)*(4*7)
 
    
    # Set specific values for RO feasibility constraints
    for h in range(n_hours):
        idx1 = h*n_polygon_eqns
        idx2 = (h+1)*n_polygon_eqns
        
        h_RO = h_RO_array[h]
        
        P_f = (0.0981*h_RO*(rho_sw/rho_fw))*14.5038 # psi (from bar)
        
        b_ineq_RO_feasibility[idx1:idx2] = -(reduced_convex_polygon_eqns_coeffs[:,0]*P_f)-reduced_convex_polygon_eqns_coeffs[:,2] + M_RO_con
        
    
    # Compile b_ineq
    b_ineq = np.concatenate((b_ineq_Reversible_uphills_lb_2D,
                             b_ineq_Reversible_uphills_ub_2D,
                             b_ineq_Reversible_downhills_lb_2D,
                             b_ineq_Reversible_downhills_ub_2D,
                             b_ineq_Mixing,
                             b_ineq_RO_on_off_lb,
                             b_ineq_RO_on_off_ub,
                             b_ineq_RO_feasibility,
                             b_ineq_Siva_eqn_2D,
                             b_ineq_Upshifts_2D,
                             b_ineq_Downshifts_2D,
                             b_ineq_Cycles_2D))
    
    
    ################### Defining Design Variable Bounds ######################
    # Initialize variable bound list of lists (to be flattened eventually)
    my_bounds_list_of_lists_of_lists = [[[] for _ in range(24)] for _ in range(7)] # One empty list for each hour, 7 days x 24 hours = 1 week total
    
    my_var_types_list_of_lists_of_lists = [[[] for _ in range(24)] for _ in range(7)] # One empty list for each hour, 7 days x 24 hours = 1 week total
    
    # Decision variable bounds (and then store in list)
    for var in range(n_vars): # number of design variables in an hour
        eta_hour_idx = 0 # This is for determining if eta is zero, from which the upper bound of the corresponding flow is zero
        for d in range(7): # number of days in the week
            for h in range(24): # number of hours in a day
                # Define bounds of variables
                if [var] in my_idxs['V_dot_wp_idxs']: # PSH capacity for each pump(/turbine)
                    nameplate_idx = var - 0
                    V_dot_wp_lb = 0
                    
                    eta_for_flow = eta_hp_matrix[eta_hour_idx,nameplate_idx]
                    if eta_for_flow == 0:
                        V_dot_wp_ub = 0
                    else:
                        V_dot_wp_ub = (eta_francis_max_flow*nameplates[nameplate_idx]*(3.6e6))/(rho_sw*g*h_res)
                        
                    V_dot_wp_bounds = (V_dot_wp_lb, V_dot_wp_ub)
                    
                    my_bounds_list_of_lists_of_lists[d][h].append(V_dot_wp_bounds)
                    my_var_types_list_of_lists_of_lists[d][h].append(0)
                elif [var] in my_idxs['V_dot_swht_idxs']: # PSH capacity for each pump(/turbine)
                    nameplate_idx = var - 7
                    V_dot_swht_lb = 0
                    
                    eta_for_flow = eta_ht_matrix[eta_hour_idx,nameplate_idx]
                    if eta_for_flow == 0:
                        V_dot_swht_ub = 0
                    else:
                        V_dot_swht_ub = (nameplates[nameplate_idx]*(3.6e6))/(eta_francis_max_flow*rho_sw*g*h_res)
                        
                    V_dot_swht_bounds = (V_dot_swht_lb, V_dot_swht_ub)
                    
                    my_bounds_list_of_lists_of_lists[d][h].append(V_dot_swht_bounds)
                    my_var_types_list_of_lists_of_lists[d][h].append(0)
                elif var in my_idxs['Q_wRO_stage1_idx']:
                    Q_wRO_stage1_lb = 0
                    
                    eta_for_flow = eta_pelton_array[eta_hour_idx]
                    if eta_for_flow == 0 and ERD_choice == 0: # This upper bounding with 0 thing only applies to pelton turbine designs
                        Q_wRO_stage1_ub = 0
                    else:
                        Q_wRO_stage1_ub = 15.5
                        
                    Q_wRO_stage1_bounds = (Q_wRO_stage1_lb, Q_wRO_stage1_ub)
                    
                    my_bounds_list_of_lists_of_lists[d][h].append(Q_wRO_stage1_bounds)
                    my_var_types_list_of_lists_of_lists[d][h].append(0)
                elif var in my_idxs['V_res_idx']: # Reservoir capacity bound
                    V_res_lb = 0.3*V_res_cap
                    V_res_ub = V_res_cap
                    V_res_bounds = (V_res_lb, V_res_ub)
                    
                    my_bounds_list_of_lists_of_lists[d][h].append(V_res_bounds)
                    my_var_types_list_of_lists_of_lists[d][h].append(0)
                elif var in my_idxs['r_idx']: # Binary variable dictating RO being on or off (0: off, 1: on)
                    r_lb = 0
                    r_ub = 1
                    r_bounds = (r_lb,r_ub)
                    
                    my_bounds_list_of_lists_of_lists[d][h].append(r_bounds)
                    my_var_types_list_of_lists_of_lists[d][h].append(1) 
                elif [var] in my_idxs['b_con_idxs']: # Binary variable for whether turbine is in consumption mode or not
                    b_charge_lb = 0
                    b_charge_ub = 1
                    b_charge_bounds = (b_charge_lb,b_charge_ub)
                    
                    my_bounds_list_of_lists_of_lists[d][h].append(b_charge_bounds)
                    my_var_types_list_of_lists_of_lists[d][h].append(1) 
                elif [var] in my_idxs['b_gen_idxs']: # Binary variable for whether turbine is in generation mode or not
                    b_gen_lb = 0
                    b_gen_ub = 1
                    b_gen_bounds = (b_gen_lb,b_gen_ub)
                    
                    my_bounds_list_of_lists_of_lists[d][h].append(b_gen_bounds)
                    my_var_types_list_of_lists_of_lists[d][h].append(1) 
                elif [var] in my_idxs['v_idxs']: # Auxilliary binary variable for counting upshifts (1: half upshift; 2: full upshift)
                    v_lb = 0
                    v_ub = 2
                    v_bounds = (v_lb,v_ub)
                    
                    my_bounds_list_of_lists_of_lists[d][h].append(v_bounds)
                    my_var_types_list_of_lists_of_lists[d][h].append(1)
                elif [var] in my_idxs['w_idxs']: # Auxilliary binary variable for counting downshifts (1: half downshift; 2: full downshift)
                    w_lb = 0
                    w_ub = 2
                    w_bounds = (w_lb,w_ub)
                    
                    my_bounds_list_of_lists_of_lists[d][h].append(w_bounds)
                    my_var_types_list_of_lists_of_lists[d][h].append(1)
                else: # This is a catch-all for the variables that have no specific bounds
                    basic_bounds = (0,None)
                    
                    my_bounds_list_of_lists_of_lists[d][h].append(basic_bounds)
                    my_var_types_list_of_lists_of_lists[d][h].append(0)
                    
                # Don't forget to increaseeta_hour_idx by 1
                eta_hour_idx += 1
           
    # Then flatten the list of list of tuples
    my_bounds = [item for sublist1 in my_bounds_list_of_lists_of_lists for sublist2 in sublist1 for item in sublist2]
    my_var_types = [item for sublist1 in my_var_types_list_of_lists_of_lists for sublist2 in sublist1 for item in sublist2]
    my_var_types = np.array(my_var_types)
    
    if solve_status == 0: # add additional variables and bounds
        # for inequality constraints (s_ineq)
        for i in range(n_ineq_eqns): 
            my_bounds.append((0,None))
        my_var_types = np.hstack((my_var_types,np.zeros(n_ineq_eqns)))
        
    
    ###################### Defining Objective Equation ######################
    # Initialize objective function coefficient list
    c_list_of_lists = []
    
    # Declare objective coefficients list values (first make list of lists, then flatten)
    lambda_e_idx = 0
    for d in range(7):
        for h in range(24):
            lambda_e = energy_prices[lambda_e_idx]
            lambda_w = water_price
            
            eta_hp_turbine1 = eta_hp_turbine1_array[lambda_e_idx]
            eta_hp_turbine2 = eta_hp_turbine2_array[lambda_e_idx]
            eta_hp_turbine3 = eta_hp_turbine3_array[lambda_e_idx]
            eta_hp_turbine4 = eta_hp_turbine4_array[lambda_e_idx]
            eta_hp_turbine5 = eta_hp_turbine5_array[lambda_e_idx]
            eta_hp_turbine6 = eta_hp_turbine6_array[lambda_e_idx]
            eta_hp_turbine7 = eta_hp_turbine7_array[lambda_e_idx]
            
            eta_hps_hour_n = [eta_hp_turbine1,eta_hp_turbine2,eta_hp_turbine3,eta_hp_turbine4,
                              eta_hp_turbine5,eta_hp_turbine6,eta_hp_turbine7]
            
            eta_ht_turbine1 = eta_ht_turbine1_array[lambda_e_idx]
            eta_ht_turbine2 = eta_ht_turbine2_array[lambda_e_idx]
            eta_ht_turbine3 = eta_ht_turbine3_array[lambda_e_idx]
            eta_ht_turbine4 = eta_ht_turbine4_array[lambda_e_idx]
            eta_ht_turbine5 = eta_ht_turbine5_array[lambda_e_idx]
            eta_ht_turbine6 = eta_ht_turbine6_array[lambda_e_idx]
            eta_ht_turbine7 = eta_ht_turbine7_array[lambda_e_idx]
            
            eta_hts_hour_n = [eta_ht_turbine1,eta_ht_turbine2,eta_ht_turbine3,eta_ht_turbine4,
                              eta_ht_turbine5,eta_ht_turbine6,eta_ht_turbine7]
            
            eta_pelton = eta_pelton_array[lambda_e_idx]
            rho_oRO_stage2 = rho_oRO_stage2_array[lambda_e_idx]
            eta_RO_stage1 = eta_RO_stage1_array[lambda_e_idx]
            eta_RO_stage2 = eta_RO_stage2_array[lambda_e_idx]
            h_oRO_stage2 = h_oRO_stage2_array[lambda_e_idx]
            
            c_individual_hour = []
            for var in range(n_vars):
                if [var] in my_idxs['V_dot_wp_idxs']: # PSH in consumption mode
                    pump_idx = var - 0
                    eta_hp = eta_hps_hour_n[pump_idx]
                    if eta_hp == 0: # This is to avoid divide by zero errors
                        V_dot_wp_coeff = 0
                    else:    
                        V_dot_wp_coeff = (rho_sw*g*h_res)/(eta_hp*(3.6e6))
                    c_individual_hour.append(lambda_e*V_dot_wp_coeff) # not negative due to minimization of algorithm (we're maximizing profit)
                elif [var] in my_idxs['V_dot_swht_idxs']: # PSH in generation mode
                    turbine_idx = var-7
                    eta_ht = eta_hts_hour_n[turbine_idx]
                    V_dot_swht_coeff = (eta_ht*rho_sw*g*h_res)/(3.6e6)
                    c_individual_hour.append(-lambda_e*V_dot_swht_coeff) # negative due to minimization of algorithm (we're maximizing profit)
                elif var in my_idxs['Q_wRO_stage1_idx']: # Brine turbine in generation mode
                    if ERD_choice == 0: # denotes pelton
                        Q_wRO_stage1_coeff_pelton = (eta_pelton*rho_oRO_stage2*(1-eta_RO_stage2)*N_pv1*(1-eta_RO_stage1)*g*h_oRO_stage2)/(3.6e6)
                    elif ERD_choice == 1: # denotes turbocharger
                        Q_wRO_stage1_coeff_pelton = 0
                    Q_wRO_stage1_coeff_fw = (N_pv1*eta_RO_stage1)+(eta_RO_stage2*N_pv1*(1-eta_RO_stage1))
                    c_individual_hour.append((-lambda_e*Q_wRO_stage1_coeff_pelton)+(-lambda_w*Q_wRO_stage1_coeff_fw)) # negative due to minimization of algorithm (we're maximizing profit)
                else: # Variable does not contribute to cost function
                    c_individual_hour.append(0)
            
            c_list_of_lists.append(c_individual_hour)
            
            lambda_e_idx += 1
    c = np.array([item for sublist in c_list_of_lists for item in sublist])
        
    
    ##################### Defining Equality Constraints #####################
    # Initialize equality constraint matrices
    A_eq_EoWRes = np.zeros(n_vars_total) # End-of-rep-period reservoir constraint
    A_eq_CoM = np.zeros((n_hours,n_vars_total)) # Conservation of mass constraint
    
    
    # EoWRes constraint (don't need to do individual hour schenanigans)
    A_eq_EoWRes[-(n_vars-my_idxs['V_res_idx'][-1])] = 1 # This implementation is good!
    #A_eq_EoWRes[-((7*n_pump_turbines)+1+1)] = 1 # 7 refers to the number of families of binary variables, the first +1 is for the single binary variable (r), the second +1 is python shenanigans
    
    
    # Conservation of mass constraints
    A_eq_CoM_first_hour_individual_hour_dict = individual_hour_maker(n_vars,n_pump_turbines,0,1)
    A_eq_CoM_other_hours_individual_hours_dict = individual_hour_maker(n_vars,n_pump_turbines,1,1)
    for pump_turbine in range(n_pump_turbines):
        for var in range(n_vars):
            if var in my_idxs['V_dot_wp_idxs'][pump_turbine]:
                A_eq_CoM_first_hour_individual_hour_dict['0'][var] = 1
                A_eq_CoM_other_hours_individual_hours_dict['0_h'][var] = 1
            elif var in my_idxs['V_dot_swht_idxs'][pump_turbine]:
                A_eq_CoM_first_hour_individual_hour_dict['0'][var] = -1
                A_eq_CoM_other_hours_individual_hours_dict['0_h'][var] = -1
            elif var in my_idxs['Q_wRO_stage1_idx']:
                A_eq_CoM_first_hour_individual_hour_dict['0'][var] = -1*N_pv1
                A_eq_CoM_other_hours_individual_hours_dict['0_h'][var] = -1*N_pv1
            elif var in my_idxs['V_res_idx']:
                A_eq_CoM_first_hour_individual_hour_dict['0'][var] = -1
                A_eq_CoM_other_hours_individual_hours_dict['0_h_minus_1'][var] = 1
                A_eq_CoM_other_hours_individual_hours_dict['0_h'][var] = -1
                
                
    '''
    All other equality constraints have to be defined in a loop since they depend on array-based values
    '''                                   
                              
    row_insert = 0
    for n in range(n_hours): # Good for all constraints except CoM constraint
        
        rho_oRO_stage2 = rho_oRO_stage2_array[n]
        h_oRO_stage2 = h_oRO_stage2_array[n]
        eta_RO_stage1 = eta_RO_stage1_array[n]
        eta_RO_stage2 = eta_RO_stage2_array[n]
        eta_hp_turbine1 = eta_hp_turbine1_array[n]
        eta_hp_turbine2 = eta_hp_turbine2_array[n]
        eta_hp_turbine3 = eta_hp_turbine3_array[n]
        eta_hp_turbine4 = eta_hp_turbine4_array[n]
        eta_hp_turbine5 = eta_hp_turbine5_array[n]
        eta_hp_turbine6 = eta_hp_turbine6_array[n]
        eta_hp_turbine7 = eta_hp_turbine7_array[n]
        eta_ht_turbine1 = eta_ht_turbine1_array[n]
        eta_ht_turbine2 = eta_ht_turbine2_array[n]
        eta_ht_turbine3 = eta_ht_turbine3_array[n]
        eta_ht_turbine4 = eta_ht_turbine4_array[n]
        eta_ht_turbine5 = eta_ht_turbine5_array[n]
        eta_ht_turbine6 = eta_ht_turbine6_array[n]
        eta_ht_turbine7 = eta_ht_turbine7_array[n]
        eta_pelton = eta_pelton_array[n]
                
    
    row_insert = 0
    for n in range(n_hours): # For CoM constraint
        if row_insert == n:
            idx = row_insert*n_vars
            if row_insert == 0:
                A_eq_CoM[n,idx:idx+n_vars] = A_eq_CoM_first_hour_individual_hour_dict['0']
            else:
                A_eq_CoM[n,idx-n_vars:idx] = A_eq_CoM_other_hours_individual_hours_dict['0_h_minus_1']
                A_eq_CoM[n,idx:idx+n_vars] = A_eq_CoM_other_hours_individual_hours_dict['0_h']
            row_insert += 1
    
    
    # Compile A_eq
    A_eq = np.vstack((A_eq_EoWRes,
                      A_eq_CoM))
    
    
    # Initialize equality constraint vector
    b_eq_EoWRes = np.ones(1)*V_res_init
    b_eq_CoM = np.zeros(n_hours) # just initializing
    
    # Declare inequality constraint vector values (if not zero)
    b_eq_CoM[0] = -V_res_init

    # Compile b_eq
    b_eq = np.concatenate((b_eq_EoWRes,
                           b_eq_CoM))
    
    
    ########################### NEED TO REMOVE TEMPORARY VARIABLES ####################
    # These "temp" variables are for the cycles constraints, since there are less of them than there are for the other variables, but I needed a similar 
    # structure so I could define my variables consistently
    
    del my_bounds[(n_vars-(n_cycle_var_fams*n_pump_turbines)):n_vars]
    my_var_types = np.delete(my_var_types, np.arange((n_vars-(n_cycle_var_fams*n_pump_turbines)),n_vars)) 
    c = np.delete(c, np.arange((n_vars-(n_cycle_var_fams*n_pump_turbines)),n_vars))
    A_eq = np.delete(A_eq, np.arange((n_vars-(n_cycle_var_fams*n_pump_turbines)), n_vars), axis=1)
    A_ineq = np.delete(A_ineq, np.arange((n_vars-(n_cycle_var_fams*n_pump_turbines)), n_vars), axis=1)
    
    if solve_status == 0: # solve the relaxed problem
        A_ineq_aug = np.hstack([A_ineq, -np.eye(n_ineq_eqns)])
        A_eq_aug = np.hstack([A_eq, np.zeros((np.shape(A_eq)[0],n_ineq_eqns))])
        c_aug = np.hstack([np.zeros(n_vars_total), np.ones(n_ineq_eqns)]) # need to update this later
    
    #print("Done setting up problem")
    ########################### Solving the LP ###########################
    # solve 
    my_options = {
        'disp':False, # True
        #'maxiter':int(200000), # 12e6 --> this is actually not for the total number of iterations, but for the number of iterations to solve a LP for ONE branch and bound
        "time_limit": int(360), #int(7200),
        "mip_max_nodes": int(7897), # int(27000),
        "mip_rel_gap":0.0003
        }
    #print(my_options)
    #tic = time.time()
    if solve_status == 1: # solve normally
        results = linprog(c=c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq = b_eq, bounds=my_bounds, method='highs', integrality=my_var_types, options=my_options) # highs # there is also highs-ds and highs-ipm, normal highs uses the simplex method
    else: # Solve the relaxed problem
        results = linprog(c=c_aug, A_ub=A_ineq_aug, b_ub=b_ineq, A_eq=A_eq_aug, b_eq = b_eq, bounds=my_bounds, method='highs', integrality=my_var_types, options=my_options) # highs # there is also highs-ds and highs-ipm, normal highs uses the simplex method
    #toc = time.time()
    #print(toc-tic)
    #time.sleep(5)
    #print(results.status)
    #print(results.message)
    #print(results.success)
    #print(results.nit)

    # Assess the quality of the solve
    LP_feasibility = results.status
    # print('LP_feasibility: ' + str(LP_feasibility))
    
    if LP_feasibility == 0 or LP_feasibility == 1 or LP_feasibility == 4: # all is good in the hood (with HiGHs 1.8, turns out 4 is acceptable now too (max # nodes being reached raises a flag of 4))
        
        #if LP_feasibility == 1:
        #    print('did not quite work')
    
        # Save things as normal
        x_LP_reduced = results.x
        for i in range((n_vars-(n_cycle_var_fams*n_pump_turbines)),n_vars):
            x_LP_reduced = np.insert(x_LP_reduced, i, -69) 
        if solve_status == 1: # solve normally
            x_LP = x_LP_reduced.reshape(n_hours,n_vars)
        else: # solve the relaxed problem
            #x_LP_slack = x_LP_reduced[(n_vars*n_hours):]
            x_LP_reduced = x_LP_reduced[:(n_vars*n_hours)]
            x_LP = x_LP_reduced.reshape(n_hours,n_vars)
        
        neg_buy_sell_profit = results.fun
        
        '''
        slack_array = results.slack
        df = pd.DataFrame(slack_array)
        df.to_csv("inequality_cons.csv", index=False, header=False)
        con_array = results.con
        df = pd.DataFrame(con_array)
        df.to_csv("equality_cons.csv", index=False, header=False)
        df = pd.DataFrame(x_LP)
        df.to_csv("x_LP.csv", index=False, header=False)
        '''
        
        #df = pd.DataFrame(x_LP)
        #csv_string = "x_LP_" + str(iteration) + ".csv"
        #df.to_csv(csv_string, index=False, header=False)
        
    else:
        x_LP = np.zeros((n_hours,n_vars))
        neg_buy_sell_profit = 0
        
    
    if print_statements == 1:
        print('Start of LP Module Output')
        
        print('V_dot_wp1_array: ' + str(x_LP[:,0]))
        print('V_dot_wp2_array: ' + str(x_LP[:,1]))
        print('V_dot_wp3_array: ' + str(x_LP[:,2]))
        print('V_dot_wp4_array: ' + str(x_LP[:,3]))
        print('V_dot_wp5_array: ' + str(x_LP[:,4]))
        print('V_dot_wp6_array: ' + str(x_LP[:,5]))
        print('V_dot_wp7_array: ' + str(x_LP[:,6]))
        
        print('V_dot_swht1_array: ' + str(x_LP[:,7]))
        print('V_dot_swht2_array: ' + str(x_LP[:,8]))
        print('V_dot_swht3_array: ' + str(x_LP[:,9]))
        print('V_dot_swht4_array: ' + str(x_LP[:,10]))
        print('V_dot_swht5_array: ' + str(x_LP[:,11]))
        print('V_dot_swht6_array: ' + str(x_LP[:,12]))
        print('V_dot_swht7_array: ' + str(x_LP[:,13]))
        
        print('Q_wRO_stage1_array: ' + str(x_LP[:,14]))
        print('V_res_array: ' + str(x_LP[:,15]))
        
        print('End of LP Module Output')
        
        
        
    return x_LP,neg_buy_sell_profit,LP_feasibility


# SEE LP_driver.py FOR TAKING THIS OUT FOR A TEST SPIN!


