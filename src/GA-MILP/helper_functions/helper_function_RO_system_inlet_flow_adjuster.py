# -*- coding: utf-8 -*-
"""
Matthew Haefner (mwh85)
SEA Lab
Desalination Paper Optimization Price Taker - RO Inlet Ajuster Helper Function
LAST UPDATED: 10/21/25     CREATED: 6/3/25
"""

import numpy as np
from scipy.optimize import minimize, root_scalar

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions')))
from functions_turbocharger import turbocharger

def RO_system_inlet_flow_adjuster(ERD_choice,RO_feasible_space_con_array,P_f_RO_array,V_dot_wRO_array,N_pv1,
                                  P_oRO_stage2_array,S_oRO_stage2_array,rho_oRO_stage2_array,h_inlet_array,h_RO_array,
                                  big_iteration,ineq_constraints_reduced_convex_polygon,x0_convex_polygon,
                                  rho_sw,rho_fw,
                                  p_state,p_francis_turbo_pelton_coeffs,print_statements):
    
    # NOTE: IF ERD_choice == 0 (i.e. pelton), then h_inlet_array and h_RO_array are the same!
    
    # Find the troublesome RO indices
    infeasible_indices = np.where(RO_feasible_space_con_array > 0)[0]
    
    for idx in infeasible_indices: 
        # Get the feed conditions (for an individual hour) for stage 1
        P_f_RO_stage1 = P_f_RO_array[idx]
        Q_f_stage1 = V_dot_wRO_array[idx]/N_pv1
        x_outside = np.array([P_f_RO_stage1,Q_f_stage1])
        
        #print('Here are the inputs that resulted in RO infeasibility (for an hour)')
        #print(str(P_f_RO_stage1) + ' psi')
        #print(str(Q_f_stage1) + ' m^3/hr')
        
        ######################### Define constraints for flow adjustment #######################################
        # This constraint is saying that the feed pressure can't increase due to this adjustment process
        ineq_constraint_other = [{'type': 'ineq', 'fun': lambda x: P_f_RO_stage1 - x[0]}] # Note: scipy's minimize() function likes inequality constraints to be defined as eqn >= 0
        if big_iteration > 0: # NOTE: As recently as 5/19/25, I had this condition being if big_iteration > -1...don't remember why, but if something funky happens it might be from here
            eq_constraints_convex_polygon = [{'type': 'eq', 'fun': lambda x: x[1] - Q_f_stage1}]
            all_constraints_convex_polygon = ineq_constraints_reduced_convex_polygon + ineq_constraint_other + eq_constraints_convex_polygon
        else:
            all_constraints_convex_polygon = ineq_constraints_reduced_convex_polygon + ineq_constraint_other
        
        # Define objective for finding closest feasible point
        def objective(x):
            return np.sum((x - x_outside) ** 2)
        
        # Determine what (nearby) feed configuration results in a feasible design (if its even possible)
        result = minimize(objective, x0_convex_polygon, constraints=all_constraints_convex_polygon, tol=1e-9)
        if result.success == False:
            get_rid_of_design = 1
        elif result.success == True:
            get_rid_of_design = 0
        x_proj = result.x # If nothing is possible, it will return the closest result
        
        # Update initial guess stuff (V_dot_wRO is only updated on the first iteration)
        if big_iteration == 0:
            P_f_RO_array[idx] = x_proj[0] # psi
            V_dot_wRO_array[idx] = x_proj[1]*N_pv1 # m^3/hr
        else:
            P_f_RO_array[idx] = x_proj[0] # psi
        
        # This gives us a target head height for the turbocharger
        h_target = (P_f_RO_array[idx]/14.5038)/(0.0981*(rho_sw/rho_fw)) # m
        #print('This is the head height we want coming out of the turbocharger')
        #print(str(h_target) + ' m')
        #print(str(P_f_RO_array[idx]) + ' psi')
        
        if ERD_choice == 1: # Denotes Turbocharger
            # Isolate key variables for later determining updated h_turbo_inlet
            P_oRO_stage2 = P_oRO_stage2_array[idx]
            S_oRO_stage2 = S_oRO_stage2_array[idx]
            rho_oRO_stage2 = rho_oRO_stage2_array[idx]
            h_turbo_inlet = h_inlet_array[idx]
        
            # Solve for a value for h_turbo_inlet that results in h_target
            def f_scalar(h_turbo_inlet_guess):
                P_oRO_stage2_array_temp = np.ones(1)*P_oRO_stage2
                S_oRO_stage2_array_temp = np.ones(1)*S_oRO_stage2
                rho_oRO_stage2_array_temp = np.ones(1)*rho_oRO_stage2
                h_turbo_inlet_array_temp = np.ones(1)*h_turbo_inlet_guess
                
                in_flow_adjuster = 1
                new_h_turbo_inlet_array = turbocharger(P_oRO_stage2_array_temp,S_oRO_stage2_array_temp,rho_oRO_stage2_array_temp,h_turbo_inlet_array_temp,
                                                       p_state,p_francis_turbo_pelton_coeffs,print_statements,in_flow_adjuster)
                return new_h_turbo_inlet_array[0] - h_target # The [0] index for new_h_turbo_inlet_array is just because turbocharger returns an array. In this case, the array is size 1
            
            result = root_scalar(f_scalar,method='newton',x0=h_turbo_inlet,xtol=1e-9)
            h_inlet_array[idx] = result.root # m
            h_RO_array[idx] = h_target
        
        elif ERD_choice == 0: # Denotes pelton turbine
            h_inlet_array[idx] = h_target
            h_RO_array[idx] = h_target
        
    return h_inlet_array,h_RO_array,V_dot_wRO_array,get_rid_of_design