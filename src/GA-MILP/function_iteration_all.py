# -*- coding: utf-8 -*-
"""
Matt Haefner (mwh85)
SEA Lab
Desalination Paper Optimization Price Taker - Outer Iteration Loop
LAST UPDATED: 10/30/25     CREATED: ?
"""

import numpy as np
import pandas as pd
import joblib
import time

from function_iteration_turbocharger_RO import turbocharger_RO_iteration
from function_iteration_LP_efficiencies import LP_efficiencies_iteration



def iteration_combined(B_PSH,N_pv1,ERD_choice,S_ht_array,epsilon,WAVE_model_bundle,energy_prices,water_price,
                       p_state,p_reservoir,p_RO_sys,p_francis_turbo_pelton_coeffs,p_universal,p_assumed,p_iteration,print_statements):

    ####################################################
    #
    # THINGS I NEED FOR TURBO-RO (AA) ITERATION LOOP
    #
    ####################################################
    
    # Initialize h_turbo_inlet and V_dot_wRO_array
    h_res = p_reservoir['h_res']
    h_loss_pretreatment = p_RO_sys['h_loss_pretreatment']
    h_inlet_array = np.ones(24*7)*(h_res - h_loss_pretreatment)
    rho_sw = p_state['rho_sw']
    rho_fw = p_state['rho_fw']
    V_dot_wRO_array = np.ones(24*7)*N_pv1*11.6 # m^3/hr
    
    '''
    # Calculate feed salinity in mg/L
    S_sw_mg_kg = p_state['S_sw_mg_kg']
    S_f_mg_L = S_sw_mg_kg*rho_sw/1000 # mg/L
    '''
    
    # Some convergence stuff
    rtol_turbo_RO = p_iteration['rtol_turbo_RO']
    atol_turbo_RO_P_oRO = p_iteration['atol_turbo_RO_P_oRO']
    atol_turbo_RO_S_oRO = p_iteration['atol_turbo_RO_S_oRO']
    atol_turbo_RO_rho_oRO = p_iteration['atol_turbo_RO_rho_oRO']
    max_iters_turbo_RO = p_iteration['max_iters_turbo_RO']
    converged_turbo_RO = p_iteration['converged_turbo_RO']
    
    ################ Initializing variables for finding a feasible initial point ####################
    # Get the convex hull equations, break up terms into A and b
    eqns_reduced = p_RO_sys['reduced_convex_polygon_eqns_coeffs']  # Each row: [a, b, c]
    A_reduced = eqns_reduced[:, :2]       # shape (27, 2)
    b_reduced = -eqns_reduced[:, 2]       # Negate the b term → Ax <= b (with tolerance)
    
    # Define constraints --> Equality constraints have to be defined in the iterative loop
    ineq_constraints_reduced_convex_polygon = [{'type': 'ineq', 'fun': lambda x, A=A_reduced, b=b_reduced, i=i: b[i] - np.dot(A_reduced[i], x)}  # Note: scipy's minimize() function likes inequality constraints to be defined as eqn >= 0     
                                               for i in range(len(b_reduced))]
    
    # Define an initial guess for a feasible point in the convex_hull
    x0_convex_polygon = np.array([550,10.1])
    
    ################ Initialized inputs for turbocharger model #######################
    # Brine density
    rho_oRO_stage2_array = np.ones(24*7)*1045 # [kg/m^3]
    
    # Brine salinity
    S_oRO_stage2_array = np.ones(24*7)*50 # g/kg
    
    # Brine pressure
    h_RO_stage1_array_sub_calc = h_inlet_array+150
    h_brine_stage1_array_sub_calc = h_RO_stage1_array_sub_calc*0.9 # [m]
    h_brine_stage2_array_sub_calc = h_brine_stage1_array_sub_calc*0.9 # [m]
    P_oRO_stage2_array = (0.0981*h_brine_stage2_array_sub_calc*(rho_oRO_stage2_array/rho_fw))*14.5038 # psi (from bar)
    
    
    ####################################################
    #
    # THINGS I NEED FOR LP-EFFICIENCIES (BB) ITERATION LOOP
    #
    ####################################################  
    
    # Some convergence stuff
    rtol_LP_efficiencies = p_iteration['rtol_LP_efficiencies']
    atol_LP_efficiencies = p_iteration['atol_LP_efficiencies']
    max_iters_LP_efficiencies = p_iteration['max_iters_LP_efficiencies']
    converged_LP_efficiencies = p_iteration['converged_LP_efficiencies']
    
    # I don't think I need this anymore, but what the heck, let's keep it in there
    solve_status = 1 # 1 means solve normally, 0 means solve the relaxed version of the problem
    
    
    ################ Initialized inputs for LP model #######################                                                                                                
    #eta_hp_matrix = np.random.uniform(low=0.75, high=0.95, size=(24*7,7))
    #eta_ht_matrix = np.random.uniform(low=0.75, high=0.95, size=(24*7,7))
    eta_hp_matrix = np.ones((24*7,7))*0.894
    eta_ht_matrix = np.ones((24*7,7))*0.894
    eta_pelton_array = np.ones(24*7)*0.894
    
    
    ####################################################
    #
    # THINGS I NEED FOR AA-BB ITERATION LOOP
    #
    #################################################### 
    
    max_iters_all = p_iteration['max_iters_all']
    rtol_all = p_iteration['rtol_all']
    atol_all = p_iteration['atol_all']
    converged_all = p_iteration['converged_all']
    
    neg_buy_sell_profits = [] # initializing for convergence assessment
    
    for big_iteration in range(max_iters_all):
        
        if print_statements == 1:
            print('Big iteration ' + str(big_iteration))
        
        # This if statement is here because different things are returned depending on the ERD chosen. They take the same inputs, but return different things
        if N_pv1 > 0: # i.e. if there is an RO system
            [S_fwRO_array,S_oRO_stage2_array,rho_oRO_stage2_array,
             P_oRO_stage2_array,h_inlet_array,h_RO_array,h_oRO_stage2_array,
             eta_RO_array,eta_RO_stage1_array,eta_RO_stage2_array,V_dot_wRO_array,
             RO_feasible_space_con_array,get_rid_of_design] = turbocharger_RO_iteration(N_pv1,ERD_choice,
                                                                                        h_inlet_array,P_oRO_stage2_array,S_oRO_stage2_array,rho_oRO_stage2_array,V_dot_wRO_array,
                                                                                        WAVE_model_bundle,ineq_constraints_reduced_convex_polygon,x0_convex_polygon,
                                                                                        max_iters_turbo_RO,rtol_turbo_RO,atol_turbo_RO_P_oRO,atol_turbo_RO_S_oRO,
                                                                                        atol_turbo_RO_rho_oRO,converged_turbo_RO,
                                                                                        p_state,p_reservoir,p_RO_sys,p_francis_turbo_pelton_coeffs,
                                                                                        rho_sw,rho_fw,big_iteration,print_statements)
                                                                                        
        else: # There is no RO system, so no need to do Turbocharger-RO iteration
            S_fwRO_array = np.zeros(168)
            S_oRO_stage2_array = np.ones(168)*p_state['S_sw_g_kg']
            rho_oRO_stage2_array = np.ones(168)*p_state['rho_sw']
            P_oRO_stage2_array = np.zeros(168)
            # h_inlet_array = h_inlet_array
            h_RO_array = np.zeros(168)
            h_oRO_stage2_array = np.zeros(168)
            eta_RO_array = np.zeros(168)
            eta_RO_stage1_array = np.zeros(168)
            eta_RO_stage2_array = np.zeros(168)
            # V_dot_wRO_array = V_dot_wRO_array
            RO_feasible_space_con_array = np.zeros(168)
            get_rid_of_design = 0
            
        
        if print_statements == 1:
            print('Am I getting rid of this design...? ' + str(get_rid_of_design))
        

        if get_rid_of_design == 1:
            break
                                                                                                      
        #print('Done with turbo-RO loop')
        Q_wRO_array_turbo_RO = V_dot_wRO_array/N_pv1
        if print_statements == 1:
            print('Start of RO feed flowrate status update, Turbo-RO')
            print('Q_wRO_array_turbo_RO out of turbo-RO: ' + str(Q_wRO_array_turbo_RO) + ' m^3/hr')
            print('End of RO feed flowrate status update, Turbo-RO')
        
        
    
        #print('Solving LP-Head_Loss loop')
        [Q_wRO_stage1_array,V_dot_fwRO_array,P_in_PSH1_array,P_in_PSH2_array,
         P_in_PSH3_array,P_in_PSH4_array,P_in_PSH5_array,P_in_PSH6_array,
         P_in_PSH7_array,P_out_PSH1_array,P_out_PSH2_array,P_out_PSH3_array,
         P_out_PSH4_array,P_out_PSH5_array,P_out_PSH6_array,P_out_PSH7_array,
         P_out_pelton_array,neg_buy_sell_profit,LP_feasibility,eta_hp_matrix,
         eta_ht_matrix,eta_pelton_array,get_rid_of_design] = LP_efficiencies_iteration(B_PSH,N_pv1,ERD_choice,solve_status,
                                                                                       h_RO_array,eta_RO_stage1_array,eta_RO_stage2_array,h_oRO_stage2_array,rho_oRO_stage2_array,
                                                                                       S_oRO_stage2_array,eta_hp_matrix,eta_ht_matrix,eta_pelton_array,S_ht_array,
                                                                                       rtol_LP_efficiencies,atol_LP_efficiencies,max_iters_LP_efficiencies,converged_LP_efficiencies,
                                                                                       epsilon,energy_prices,water_price,
                                                                                       p_state,p_reservoir,p_RO_sys,p_universal,p_assumed,p_francis_turbo_pelton_coeffs,print_statements)
                                 
                                                                         
        if get_rid_of_design == 1:
            break                                                                 
                                                            
        if LP_feasibility == 2 or LP_feasibility == 3: # The code should never get in here...delete later if I want to
            # Do whatever shit I need to do to make sure that I end up returning the same thing as the case where LP_feasibility = 0 or 1
            P_in_PSH1_array = np.zeros(24*7)
            P_in_PSH2_array = np.zeros(24*7)
            P_in_PSH3_array = np.zeros(24*7)
            P_in_PSH4_array = np.zeros(24*7)
            P_in_PSH5_array = np.zeros(24*7)
            P_in_PSH6_array = np.zeros(24*7)
            P_in_PSH7_array = np.zeros(24*7)
            
            P_out_PSH1_array = np.zeros(24*7)
            P_out_PSH2_array = np.zeros(24*7)
            P_out_PSH3_array = np.zeros(24*7)
            P_out_PSH4_array = np.zeros(24*7)
            P_out_PSH5_array = np.zeros(24*7)
            P_out_PSH6_array = np.zeros(24*7)
            P_out_PSH7_array = np.zeros(24*7)
            
            eta_ht_PSH1_array = np.zeros(24*7)
            eta_ht_PSH2_array = np.zeros(24*7)
            eta_ht_PSH3_array = np.zeros(24*7)
            eta_ht_PSH4_array = np.zeros(24*7)
            eta_ht_PSH5_array = np.zeros(24*7)
            eta_ht_PSH6_array = np.zeros(24*7)
            eta_ht_PSH7_array = np.zeros(24*7)
            
            P_out_pelton_array = np.zeros(24*7)
            V_dot_fwRO_array = np.zeros(24*7)
            neg_buy_sell_profit = 0
            
            return [eta_RO_array,S_fwRO_array,RO_feasible_space_con_array,
                    P_in_PSH1_array,P_in_PSH2_array,P_in_PSH3_array,P_in_PSH4_array,P_in_PSH5_array,P_in_PSH6_array,P_in_PSH7_array,
                    P_out_PSH1_array,P_out_PSH2_array,P_out_PSH3_array,P_out_PSH4_array,P_out_PSH5_array,P_out_PSH6_array,P_out_PSH7_array,
                    eta_ht_PSH1_array,eta_ht_PSH2_array,eta_ht_PSH3_array,eta_ht_PSH4_array,eta_ht_PSH5_array,eta_ht_PSH6_array,eta_ht_PSH7_array,
                    P_out_pelton_array,V_dot_fwRO_array,
                    neg_buy_sell_profit,LP_feasibility,get_rid_of_design]
                                                                                
                                                                       
        
        #print('Done with LP-Efficiencies loop')
        if print_statements == 1:
            print('Start of RO feed flowrate status update, MILP-Efficiencies')
            print('Q_wRO_stage1_array out of LP-efficiencies: ' + str(Q_wRO_stage1_array) + ' m^3/hr')
            print('End of RO feed flowrate status update, MILP-Efficiencies')
        Q_wRO_array_LP_efficiencies = Q_wRO_stage1_array
        #print(V_dot_wRO_array_LP_efficiencies)
        
        #print('V_dot_wRO out of LP')
        #print(V_dot_wRO_array)  
        
        neg_buy_sell_profits.append(neg_buy_sell_profit)
        
        converged1a = np.allclose(Q_wRO_array_LP_efficiencies, Q_wRO_array_turbo_RO, rtol=rtol_all, atol=atol_all)
        
        if big_iteration < 1:
            converged1b = False
        else:
            Q_wRO_array_LP_efficiencies_sorted = np.sort(Q_wRO_array_LP_efficiencies)[::-1]  # Sort descending
            Q_wRO_array_turbo_RO_sorted = np.sort(Q_wRO_array_turbo_RO)[::-1]  # Sort descending
            LP_obj_diff = abs(neg_buy_sell_profits[big_iteration] - neg_buy_sell_profits[big_iteration-1])
            
            subcon0b = np.allclose(Q_wRO_array_LP_efficiencies_sorted, Q_wRO_array_turbo_RO_sorted, rtol=rtol_all, atol=atol_all)
            subcon1b = LP_obj_diff < 2100
            
            if subcon0b and subcon1b:
                converged1b = True
            else:
                converged1b = False
        
        if big_iteration < 3:
            converged1c = False
        else:
            subcon0c = abs(neg_buy_sell_profits[big_iteration] - neg_buy_sell_profits[big_iteration-1]) < 2100
            subcon1c = abs(neg_buy_sell_profits[big_iteration-1] - neg_buy_sell_profits[big_iteration-2]) < 2100
            subcon2c = abs(neg_buy_sell_profits[big_iteration-2] - neg_buy_sell_profits[big_iteration-3]) < 2100
            
            if subcon0c and subcon1c and subcon2c:
                converged1c = True
            else:
                converged1c = False
                
        if print_statements == 1:
            print('Start of Big Iteration Loop Status Update')
            print('converged1a: ' + str(converged1a))
            print('converged1b: ' + str(converged1b))
            print('converged1c: ' + str(converged1c))
            print('End of Big Iteration Loop Status Update')
            
        if converged1a or converged1b or converged1c:
            converged_all = True
            break

        
        '''
        if np.allclose(Q_wRO_array_LP_efficiencies, Q_wRO_array_turbo_RO, rtol=rtol_all, atol=atol_all):
            converged_all = True
            break  
        else:
            Q_wRO_array_LP_efficiencies_sorted = np.sort(Q_wRO_array_LP_efficiencies)[::-1]  # Sort descending
            Q_wRO_array_turbo_RO_sorted = np.sort(Q_wRO_array_turbo_RO)[::-1]  # Sort descending
            LP_obj_diff = abs(neg_buy_sell_profit - previous_LP_obj_val)
            if (np.allclose(Q_wRO_array_LP_efficiencies_sorted, Q_wRO_array_turbo_RO_sorted, rtol=rtol_all, atol=atol_all) or LP_obj_diff < 700) and big_iteration > 4:
                converged_all = True
                break
            else:
                previous_LP_obj_val = neg_buy_sell_profit
        '''
        
        V_dot_wRO_array = Q_wRO_array_LP_efficiencies*N_pv1
    
    
    if converged_all:
        if print_statements == 1:
            print("Big Loop converged in", big_iteration + 1, "iterations.")
        
        # Post process efficiency variables
        eta_ht_PSH1_array = eta_ht_matrix[:,0]
        eta_ht_PSH2_array = eta_ht_matrix[:,1]
        eta_ht_PSH3_array = eta_ht_matrix[:,2]
        eta_ht_PSH4_array = eta_ht_matrix[:,3]
        eta_ht_PSH5_array = eta_ht_matrix[:,4]
        eta_ht_PSH6_array = eta_ht_matrix[:,5]
        eta_ht_PSH7_array = eta_ht_matrix[:,6]
        
        
        return [eta_RO_array,S_fwRO_array,RO_feasible_space_con_array,
                P_in_PSH1_array,P_in_PSH2_array,P_in_PSH3_array,P_in_PSH4_array,P_in_PSH5_array,P_in_PSH6_array,P_in_PSH7_array,
                P_out_PSH1_array,P_out_PSH2_array,P_out_PSH3_array,P_out_PSH4_array,P_out_PSH5_array,P_out_PSH6_array,P_out_PSH7_array,
                eta_ht_PSH1_array,eta_ht_PSH2_array,eta_ht_PSH3_array,eta_ht_PSH4_array,eta_ht_PSH5_array,eta_ht_PSH6_array,eta_ht_PSH7_array,
                P_out_pelton_array,V_dot_fwRO_array,
                neg_buy_sell_profit,LP_feasibility,get_rid_of_design]
    
    else: 
        if print_statements == 1:
            if get_rid_of_design == 1:
                print("This was a bad design")
            else:
                print("Big iteration did not converge after", max_iters_all, "iterations.")
        
        if get_rid_of_design == 1:
            # Do whatever I need to do to be able to return something 
            P_in_PSH1_array = np.zeros(24*7)
            P_in_PSH2_array = np.zeros(24*7)
            P_in_PSH3_array = np.zeros(24*7)
            P_in_PSH4_array = np.zeros(24*7)
            P_in_PSH5_array = np.zeros(24*7)
            P_in_PSH6_array = np.zeros(24*7)
            P_in_PSH7_array = np.zeros(24*7)
            
            P_out_PSH1_array = np.zeros(24*7)
            P_out_PSH2_array = np.zeros(24*7)
            P_out_PSH3_array = np.zeros(24*7)
            P_out_PSH4_array = np.zeros(24*7)
            P_out_PSH5_array = np.zeros(24*7)
            P_out_PSH6_array = np.zeros(24*7)
            P_out_PSH7_array = np.zeros(24*7)
            
            eta_ht_PSH1_array = np.zeros(24*7)
            eta_ht_PSH2_array = np.zeros(24*7)
            eta_ht_PSH3_array = np.zeros(24*7)
            eta_ht_PSH4_array = np.zeros(24*7)
            eta_ht_PSH5_array = np.zeros(24*7)
            eta_ht_PSH6_array = np.zeros(24*7)
            eta_ht_PSH7_array = np.zeros(24*7)
            
            V_dot_fwRO_array = np.zeros(24*7)
            neg_buy_sell_profit = 0
            LP_feasibility = 69
            
            return [eta_RO_array,S_fwRO_array,RO_feasible_space_con_array,
                    P_in_PSH1_array,P_in_PSH2_array,P_in_PSH3_array,P_in_PSH4_array,P_in_PSH5_array,P_in_PSH6_array,P_in_PSH7_array,
                    P_out_PSH1_array,P_out_PSH2_array,P_out_PSH3_array,P_out_PSH4_array,P_out_PSH5_array,P_out_PSH6_array,P_out_PSH7_array,
                    eta_ht_PSH1_array,eta_ht_PSH2_array,eta_ht_PSH3_array,eta_ht_PSH4_array,eta_ht_PSH5_array,eta_ht_PSH6_array,eta_ht_PSH7_array,
                    P_out_pelton_array,V_dot_fwRO_array,
                    neg_buy_sell_profit,LP_feasibility,get_rid_of_design]
        else: # Return things as-is
            
            # Post process efficiency variables
            eta_ht_PSH1_array = eta_ht_matrix[:,0]
            eta_ht_PSH2_array = eta_ht_matrix[:,1]
            eta_ht_PSH3_array = eta_ht_matrix[:,2]
            eta_ht_PSH4_array = eta_ht_matrix[:,3]
            eta_ht_PSH5_array = eta_ht_matrix[:,4]
            eta_ht_PSH6_array = eta_ht_matrix[:,5]
            eta_ht_PSH7_array = eta_ht_matrix[:,6]
            
            
            return [eta_RO_array,S_fwRO_array,RO_feasible_space_con_array,
                    P_in_PSH1_array,P_in_PSH2_array,P_in_PSH3_array,P_in_PSH4_array,P_in_PSH5_array,P_in_PSH6_array,P_in_PSH7_array,
                    P_out_PSH1_array,P_out_PSH2_array,P_out_PSH3_array,P_out_PSH4_array,P_out_PSH5_array,P_out_PSH6_array,P_out_PSH7_array,
                    eta_ht_PSH1_array,eta_ht_PSH2_array,eta_ht_PSH3_array,eta_ht_PSH4_array,eta_ht_PSH5_array,eta_ht_PSH6_array,eta_ht_PSH7_array,
                    P_out_pelton_array,V_dot_fwRO_array,
                    neg_buy_sell_profit,LP_feasibility,get_rid_of_design]
    
    
        
     
      