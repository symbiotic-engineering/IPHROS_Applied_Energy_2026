# -*- coding: utf-8 -*-
"""
Matt Haefner (mwh85)
SEA Lab
Desalination Paper Optimization Price Taker - Fixed point iteration LP / Efficiencies
LAST UPDATED: 10/21/25     CREATED: ?
"""

import numpy as np
import joblib
import pandas as pd
import time

from functions.functions_milp_reversible_unequal_turbines_simplified import my_milp_reversible_simplified
from functions.functions_milp_post_processing import milp_post_processing
from functions.functions_pump_turbine_pelton_efficiencies import francis_pelton_efficiencies


def LP_efficiencies_iteration(B_PSH,N_pv1,ERD_choice,solve_status,
                              h_RO_array,eta_RO_stage1_array,eta_RO_stage2_array,h_oRO_stage2_array,rho_oRO_stage2_array,
                              S_oRO_stage2_array,eta_hp_matrix,eta_ht_matrix,eta_pelton_array,S_ht_array,
                              rtol_LP_efficiencies,atol_LP_efficiencies,max_iters_LP_efficiencies,converged_LP_efficiencies,
                              epsilon,energy_prices,water_price,
                              p_state,p_reservoir,p_RO_sys,p_universal,p_assumed,p_francis_turbo_pelton_coeffs,print_statements):
    
    x_A_input_LP_efficiencies = [eta_hp_matrix.flatten(),eta_ht_matrix.flatten(),eta_pelton_array]
    neg_buy_sell_profits = [] # initializing for convergence assessment
    
    for iteration in range(max_iters_LP_efficiencies):
        
        if print_statements == 1:
            print('LP-head_loss iteration ' + str(iteration))
            
        # Solve model A (Linear Program)
        x_LP,neg_buy_sell_profit,LP_feasibility = my_milp_reversible_simplified(energy_prices,water_price,B_PSH,N_pv1,ERD_choice,
                                                                                eta_RO_stage1_array,eta_RO_stage2_array,h_oRO_stage2_array,
                                                                                rho_oRO_stage2_array,S_oRO_stage2_array,h_RO_array,
                                                                                S_ht_array,eta_hp_matrix,eta_ht_matrix,eta_pelton_array,solve_status,
                                                                                p_state,p_RO_sys,p_universal,p_reservoir,p_assumed,epsilon,print_statements,
                                                                                iteration)
                                                                                              
        if print_statements == 1:
            print('LP feasibility: ' + str(LP_feasibility))
            if LP_feasibility == 0:
                print('woohoo!')
            elif LP_feasibility == 1:
                print('woohoo?? 1')
            elif LP_feasibility == 4:
                print('woohoo?? 4')
            else:
                print('rats')
            print('Negative LP operational profit: $' + str(neg_buy_sell_profit))
        
        if LP_feasibility == 2 or LP_feasibility == 3:
            get_rid_of_design = 1
            
            Q_wRO_stage1_array = np.zeros(168)
            V_dot_fwRO_array = np.zeros(168)
            P_in_PSH1_array = np.zeros(168)
            P_in_PSH2_array = np.zeros(168)
            P_in_PSH3_array = np.zeros(168)
            P_in_PSH4_array = np.zeros(168)
            P_in_PSH5_array = np.zeros(168)
            P_in_PSH6_array = np.zeros(168)
            P_in_PSH7_array = np.zeros(168)
            P_out_PSH1_array = np.zeros(168)
            P_out_PSH2_array = np.zeros(168)
            P_out_PSH3_array = np.zeros(168)
            P_out_PSH4_array = np.zeros(168)
            P_out_PSH5_array = np.zeros(168)
            P_out_PSH6_array = np.zeros(168)
            P_out_PSH7_array = np.zeros(168)
            P_out_pelton_array = np.zeros(168)
            
            return [Q_wRO_stage1_array,V_dot_fwRO_array,P_in_PSH1_array,P_in_PSH2_array,P_in_PSH3_array,P_in_PSH4_array,P_in_PSH5_array,P_in_PSH6_array,P_in_PSH7_array,
                    P_out_PSH1_array,P_out_PSH2_array,P_out_PSH3_array,P_out_PSH4_array,P_out_PSH5_array,P_out_PSH6_array,P_out_PSH7_array,P_out_pelton_array,
                    neg_buy_sell_profit,LP_feasibility,eta_hp_matrix,eta_ht_matrix,eta_pelton_array,get_rid_of_design]
        else:
            
            # Solve model B, passing the output from A
            [eta_hp_matrix,eta_ht_matrix,eta_pelton_array] = francis_pelton_efficiencies(B_PSH,N_pv1,ERD_choice,x_LP,eta_pelton_array,h_oRO_stage2_array,rho_oRO_stage2_array,
                                                                                         eta_RO_stage1_array,eta_RO_stage2_array,epsilon,
                                                                                         p_state,p_reservoir,p_assumed,p_universal,p_francis_turbo_pelton_coeffs,print_statements)
            
            
            # Define variables to match up with what Chat has
            new_x_A_input_LP_efficiencies = [eta_hp_matrix.flatten(),eta_ht_matrix.flatten(),eta_pelton_array]
            neg_buy_sell_profits.append(neg_buy_sell_profit)

            converged1a = np.allclose(new_x_A_input_LP_efficiencies[0], x_A_input_LP_efficiencies[0], rtol=rtol_LP_efficiencies, atol=atol_LP_efficiencies)
            converged2a = np.allclose(new_x_A_input_LP_efficiencies[1], x_A_input_LP_efficiencies[1], rtol=rtol_LP_efficiencies, atol=atol_LP_efficiencies)
            converged3a = np.allclose(new_x_A_input_LP_efficiencies[2], x_A_input_LP_efficiencies[2], rtol=rtol_LP_efficiencies, atol=atol_LP_efficiencies)
            if iteration < 4:
                converged1b = False
            else:
                subcon0 = abs(neg_buy_sell_profits[iteration] - neg_buy_sell_profits[iteration-1]) < 2100
                subcon1 = abs(neg_buy_sell_profits[iteration-1] - neg_buy_sell_profits[iteration-2]) < 2100
                subcon2 = abs(neg_buy_sell_profits[iteration-2] - neg_buy_sell_profits[iteration-3]) < 2100
                if subcon0 and subcon1 and subcon2:
                    converged1b = True
                else:
                    converged1b = False
                    
            if print_statements == 1:
                print('Start of LP-Efficiencies Status Update')
                print('converged1a: ' + str(converged1a))
                print('converged2a: ' + str(converged2a))
                print('converged3a: ' + str(converged3a))
                print('converged1b: ' + str(converged1b))
                print('neg_buy_sell_profits: ' + str(neg_buy_sell_profits))
                print('End of LP-Efficiencies Status Update')
                    
            if (converged1a and converged2a and converged3a) or converged1b:
                converged_LP_efficiencies = True
                break
            
            # Step 4: Update input to Model A for next iteration
            x_A_input_LP_efficiencies = new_x_A_input_LP_efficiencies
            eta_hp_matrix = x_A_input_LP_efficiencies[0].reshape(eta_hp_matrix.shape)
            eta_ht_matrix = x_A_input_LP_efficiencies[1].reshape(eta_ht_matrix.shape)
            eta_pelton_array = x_A_input_LP_efficiencies[2]
    
    # Post-process the MILP results
    [Q_wRO_stage1_array,V_dot_fwRO_array,
     P_in_PSH1_array,P_in_PSH2_array,P_in_PSH3_array,P_in_PSH4_array,
     P_in_PSH5_array,P_in_PSH6_array,P_in_PSH7_array,
     P_out_PSH1_array,P_out_PSH2_array,P_out_PSH3_array,P_out_PSH4_array,
     P_out_PSH5_array,P_out_PSH6_array,P_out_PSH7_array,P_out_pelton_array] = milp_post_processing(N_pv1,x_LP,eta_RO_stage1_array,eta_RO_stage2_array,
                                                                                                   rho_oRO_stage2_array,h_oRO_stage2_array,
                                                                                                   eta_hp_matrix,eta_ht_matrix,eta_pelton_array,
                                                                                                   p_state,p_reservoir,p_universal,print_statements)    
    
    if converged_LP_efficiencies:
        get_rid_of_design = 0
        if print_statements == 1:
            print("LP-efficiencies converged in", iteration + 1, "iterations.")
        return [Q_wRO_stage1_array,V_dot_fwRO_array,P_in_PSH1_array,P_in_PSH2_array,P_in_PSH3_array,P_in_PSH4_array,P_in_PSH5_array,P_in_PSH6_array,P_in_PSH7_array,
                P_out_PSH1_array,P_out_PSH2_array,P_out_PSH3_array,P_out_PSH4_array,P_out_PSH5_array,P_out_PSH6_array,P_out_PSH7_array,P_out_pelton_array,
                neg_buy_sell_profit,LP_feasibility,eta_hp_matrix,eta_ht_matrix,eta_pelton_array,get_rid_of_design]
    else:
        # get_rid_of_design = 1 # No longer getting rid of the design under this condition!!! Just returning whatever I have
        get_rid_of_design = 0
        if print_statements == 1:
            print("LP-efficiencies did not converge after", max_iters_LP_efficiencies, "iterations.")
            
            '''
            print(B_PSH)
            print(N_pv1)
            print(ERD_choice)
            print(h_RO_array)
            print(eta_RO_stage1_array)
            print(eta_RO_stage2_array)
            print(h_oRO_stage2_array)
            print(rho_oRO_stage2_array)
            print(S_oRO_stage2_array)
            print(eta_hp_matrix)
            print(eta_ht_matrix)
            print(eta_pelton_array)
            '''
            
        return [Q_wRO_stage1_array,V_dot_fwRO_array,P_in_PSH1_array,P_in_PSH2_array,P_in_PSH3_array,P_in_PSH4_array,P_in_PSH5_array,P_in_PSH6_array,P_in_PSH7_array,
                P_out_PSH1_array,P_out_PSH2_array,P_out_PSH3_array,P_out_PSH4_array,P_out_PSH5_array,P_out_PSH6_array,P_out_PSH7_array,P_out_pelton_array,
                neg_buy_sell_profit,LP_feasibility,eta_hp_matrix,eta_ht_matrix,eta_pelton_array,get_rid_of_design]
        