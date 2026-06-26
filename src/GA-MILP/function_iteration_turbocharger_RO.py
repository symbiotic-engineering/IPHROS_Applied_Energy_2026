# -*- coding: utf-8 -*-
"""
Matt Haefner (mwh85)
SEA Lab
Desalination Paper Optimization Price Taker - Fixed point iteration turbocharger & RO
LAST UPDATED: 10/30/25     CREATED: ?
"""

import numpy as np
#import joblib
#import time

from functions.functions_turbocharger import turbocharger
from functions.functions_RO_system_multioutput_WAVE import ro_system_WAVE_multioutput
from helper_functions.helper_function_RO_system_inlet_flow_adjuster import RO_system_inlet_flow_adjuster



################ This is the actual iterative loop #######################
def turbocharger_RO_iteration(N_pv1,ERD_choice,
                              h_inlet_array,P_oRO_stage2_array,S_oRO_stage2_array,rho_oRO_stage2_array,V_dot_wRO_array,
                              WAVE_model_bundle,ineq_constraints_reduced_convex_polygon,x0_convex_polygon,
                              max_iters_turbo_RO,rtol_turbo_RO,atol_turbo_RO_P_oRO,atol_turbo_RO_S_oRO,
                              atol_turbo_RO_rho_oRO,converged_turbo_RO,
                              p_state,p_reservoir,p_RO_sys,p_francis_turbo_pelton_coeffs,
                              rho_sw,rho_fw,big_iteration,print_statements):
    
    if ERD_choice == 0: # Pelton wheel
        # Set the RO head height equal to the head from reservoir minus pretreatment
        h_RO_array = h_inlet_array
        
        # This is what's going into the RO module
        if print_statements == 1:
            print('Original head height into RO model: ' + str(h_RO_array) + ' m')
            print('Original flowrate into RO model: ' + str(V_dot_wRO_array/N_pv1) + ' m^3/hr')
        
        # Just solve the RO module
        [_,S_fwRO_array,_,S_oRO_stage2_array,
         rho_oRO_stage2_array,P_oRO_stage2_array,h_oRO_stage2_array,eta_RO_array,
         eta_RO_stage1_array,eta_RO_stage2_array,RO_feasible_space_con_array] = ro_system_WAVE_multioutput(N_pv1,h_RO_array,V_dot_wRO_array,WAVE_model_bundle,
                                                                                                           p_state,p_RO_sys,print_statements)
        
        # Assess the feasibility. If bad, re-adjust the feed flow
        if print_statements == 1:
            print('RO feasibility: ' + str(RO_feasible_space_con_array))
            
        if sum(RO_feasible_space_con_array) > 0:
            if print_statements == 1:
                print('Need to adjust the flow into RO system (pelton)')
            
            # Define the feed pressure (needed for flow adjuster function)
            P_f_RO_array = 0.0981*h_RO_array*(rho_sw/rho_fw)*14.5038 # psi (from bar)
            
            # Adjust the feed flow going into the RO system to try and find feasibility
            [h_inlet_array,h_RO_array,
             V_dot_wRO_array,get_rid_of_design] = RO_system_inlet_flow_adjuster(ERD_choice,RO_feasible_space_con_array,P_f_RO_array,V_dot_wRO_array,N_pv1,
                                                                                P_oRO_stage2_array,S_oRO_stage2_array,rho_oRO_stage2_array,h_inlet_array,h_RO_array,
                                                                                big_iteration,ineq_constraints_reduced_convex_polygon,x0_convex_polygon,
                                                                                rho_sw,rho_fw,
                                                                                p_state,p_francis_turbo_pelton_coeffs,print_statements)
                             
            if get_rid_of_design == 0:
                # Solve the RO module one last time
                [_,S_fwRO_array,_,S_oRO_stage2_array,
                 rho_oRO_stage2_array,P_oRO_stage2_array,h_oRO_stage2_array,eta_RO_array,
                 eta_RO_stage1_array,eta_RO_stage2_array,RO_feasible_space_con_array] = ro_system_WAVE_multioutput(N_pv1,h_RO_array,V_dot_wRO_array,WAVE_model_bundle,
                                                                                                                   p_state,p_RO_sys,print_statements)
                
                if print_statements == 1:
                    print('Flow was successfully adjusted (pelton)!')
                    print('New head height into RO model: ' + str(h_RO_array) + ' m')
                    print('(Maybe) New flowrate into RO model: ' + str(V_dot_wRO_array/N_pv1) + ' m^3/hr')
                    print('Post-adjustment RO feasibility from RO model: ' + str(RO_feasible_space_con_array))
            
            elif get_rid_of_design == 1:
                # RIP to this design
                if print_statements == 1:
                    print('Flow could not be adjusted :(')
                S_fwRO_array,S_oRO_stage2_array,h_RO_array = np.ones(24*7)*100,np.ones(24*7)*100,np.zeros(24*7)
                rho_oRO_stage2_array,P_oRO_stage2_array,h_oRO_stage2_array,eta_RO_array = np.ones(24*7)*2000,np.zeros(24*7),np.zeros(24*7),np.zeros(24*7)
                eta_RO_stage1_array,eta_RO_stage2_array,RO_feasible_space_con_array = np.zeros(24*7),np.zeros(24*7),np.ones(24*7)*69
                V_dot_wRO_array = np.zeros(24*7)
                # Note: h_inlet_array still stays the same
        
        # If everything looks good, don't get rid of design, and return RO solution
        else:
            if print_statements == 1:
                print('Nothing else needed (pelton)!')
            get_rid_of_design = 0
            
        
    elif ERD_choice == 1: # Turbocharger
        
        # Compile convergence variables
        x_A_input_turbo_RO = [P_oRO_stage2_array,S_oRO_stage2_array,rho_oRO_stage2_array]
        
        for iteration in range(max_iters_turbo_RO):
            
            if print_statements == 1:
                print('turbo-RO iteration ' + str(iteration))
            
            # SELF-REFERENCE ONLY: Inlet pressure
            if print_statements == 1:
                #print('Inlet turbocharger head: ' + str(h_inlet_array) + ' m')
                P_inlet_array = 0.0981*h_inlet_array*(rho_sw/rho_fw)*14.5038 # psi (from bar)
                print('Inlet turbocharger pressure (self-reference): ' + str(P_inlet_array) + ' psi')
            
            # Solve model A (turbocharger)
            in_flow_adjuster = 0
            h_turbo_outlet_array = turbocharger(P_oRO_stage2_array,S_oRO_stage2_array,rho_oRO_stage2_array,h_inlet_array,
                                                p_state,p_francis_turbo_pelton_coeffs,print_statements,in_flow_adjuster)

            # Set the RO head height equal to the head coming out of the turbocharger array
            h_RO_array = h_turbo_outlet_array # m
            P_f_RO_array = 0.0981*h_RO_array*(rho_sw/rho_fw)*14.5038 # psi (from bar)
            if print_statements == 1:
                #print('Outlet turbocharger head: ' + str(h_RO_array) + ' m')
                print('Outlet turbocharger pressure: ' + str(P_f_RO_array) + ' psi')
            
            # SELF-REFERENCE ONLY: Flowrates going into the RO system 
            if print_statements == 1:
                Q_wRO_array = V_dot_wRO_array/N_pv1
                print('Feed flowrate before any potential flow adjustment is (self-reference): ' + str(Q_wRO_array) + ' m^3/hr')
            
            # Solve model B, passing the output from A
            [_,S_fwRO_array,_,S_oRO_stage2_array,
             rho_oRO_stage2_array,P_oRO_stage2_array,h_oRO_stage2_array,eta_RO_array,
             eta_RO_stage1_array,eta_RO_stage2_array,RO_feasible_space_con_array] = ro_system_WAVE_multioutput(N_pv1,h_RO_array,V_dot_wRO_array,WAVE_model_bundle,
                                                                                                               p_state,p_RO_sys,print_statements)                                                                                                            
                          
            
            if print_statements == 1:
                print('RO feasibility: ' + str(RO_feasible_space_con_array))  
            
            if sum(RO_feasible_space_con_array) == 0:
                # PROCEED AS NORMAL
                if print_statements == 1:
                    print('No flow adjustment needed (turbo)!')
                
                # Define variables to match up with what Chat has
                #x_B_input = h_turbo_outlet_array
                new_x_A_input_turbo_RO = [P_oRO_stage2_array, S_oRO_stage2_array, rho_oRO_stage2_array]
                
                converged1 = np.allclose(new_x_A_input_turbo_RO[0], x_A_input_turbo_RO[0], rtol=rtol_turbo_RO, atol=atol_turbo_RO_P_oRO)
                converged2 = np.allclose(new_x_A_input_turbo_RO[1], x_A_input_turbo_RO[1], rtol=rtol_turbo_RO, atol=atol_turbo_RO_S_oRO)
                converged3 = np.allclose(new_x_A_input_turbo_RO[2], x_A_input_turbo_RO[2], rtol=rtol_turbo_RO, atol=atol_turbo_RO_rho_oRO)
                if converged1 and converged2 and converged3:
                    converged_turbo_RO = True
                    break
                
                # Step 4: Update input to Model A for next iteration
                x_A_input_turbo_RO = new_x_A_input_turbo_RO
                P_oRO_stage2_array = x_A_input_turbo_RO[0]
                S_oRO_stage2_array = x_A_input_turbo_RO[1]
                rho_oRO_stage2_array = x_A_input_turbo_RO[2]
                
            else:
                # Update my head height guesses, and initial flowrate if I am in the first outer-loop iteration
                if print_statements == 1:
                    print('Need to adjust the feed flow')
                    
                [h_inlet_array,h_RO_array,
                 V_dot_wRO_array,get_rid_of_design] = RO_system_inlet_flow_adjuster(ERD_choice,RO_feasible_space_con_array,P_f_RO_array,V_dot_wRO_array,N_pv1,
                                                                                    P_oRO_stage2_array,S_oRO_stage2_array,rho_oRO_stage2_array,h_inlet_array,h_RO_array,
                                                                                    big_iteration,ineq_constraints_reduced_convex_polygon,x0_convex_polygon,
                                                                                    rho_sw,rho_fw,
                                                                                    p_state,p_francis_turbo_pelton_coeffs,print_statements)
                
                if get_rid_of_design == 0:
                    
                    if print_statements == 1:
                        print('Flow was successfully adjusted (turbo)!')
                        print('New head height into turbocharger model: ' + str(h_inlet_array) + ' m')
                        this_pressure = 0.0981*h_inlet_array*(rho_sw/rho_fw)*14.5038 # psi (from bar)
                        print('New head pressure into turbocharger model: ' + str(this_pressure) + ' psi')
                        print('New head height into RO model: ' + str(h_RO_array) + ' m')
                        that_pressure = 0.0981*h_RO_array*(rho_sw/rho_fw)*14.5038 # psi (from bar)
                        print('New head pressure into turbocharger model: ' + str(that_pressure) + ' psi')
                        print('New flowrate into turbocharger model: ' + str(V_dot_wRO_array/N_pv1) + ' m^3/hr')
                    
                    # Solve the RO module one last time before updating iteration count                    
                    [_,S_fwRO_array,_,S_oRO_stage2_array,
                     rho_oRO_stage2_array,P_oRO_stage2_array,h_oRO_stage2_array,eta_RO_array,
                     eta_RO_stage1_array,eta_RO_stage2_array,RO_feasible_space_con_array] = ro_system_WAVE_multioutput(N_pv1,h_RO_array,V_dot_wRO_array,WAVE_model_bundle,
                                                                                                                       p_state,p_RO_sys,print_statements)
                    
                    
                    if print_statements == 1:
                        print('This is the adjusted RO feasibility: ' + str(RO_feasible_space_con_array))
                    
                    new_x_A_input_turbo_RO = [P_oRO_stage2_array, S_oRO_stage2_array, rho_oRO_stage2_array]
                    
                    converged1 = np.allclose(new_x_A_input_turbo_RO[0], x_A_input_turbo_RO[0], rtol=rtol_turbo_RO, atol=atol_turbo_RO_P_oRO)
                    converged2 = np.allclose(new_x_A_input_turbo_RO[1], x_A_input_turbo_RO[1], rtol=rtol_turbo_RO, atol=atol_turbo_RO_S_oRO)
                    converged3 = np.allclose(new_x_A_input_turbo_RO[2], x_A_input_turbo_RO[2], rtol=rtol_turbo_RO, atol=atol_turbo_RO_rho_oRO)
                    if converged1 and converged2 and converged3:
                        converged_turbo_RO = True
                        break
                    
                    # Step 4: Update input to Model A for next iteration
                    x_A_input_turbo_RO = new_x_A_input_turbo_RO
                    P_oRO_stage2_array = x_A_input_turbo_RO[0]
                    S_oRO_stage2_array = x_A_input_turbo_RO[1]
                    rho_oRO_stage2_array = x_A_input_turbo_RO[2]
                    
                              
                elif get_rid_of_design == 1:
                    # RIP to this design
                    if print_statements == 1:
                        print('Flow could not be adjusted :( (turbo)')
                        print(N_pv1)
                        print(ERD_choice)
                    S_fwRO_array,S_oRO_stage2_array,h_RO_array = np.ones(24*7)*100,np.ones(24*7)*100,np.zeros(24*7)
                    rho_oRO_stage2_array,P_oRO_stage2_array,h_oRO_stage2_array,eta_RO_array = np.ones(24*7)*2000,np.zeros(24*7),np.zeros(24*7),np.zeros(24*7)
                    eta_RO_stage1_array,eta_RO_stage2_array,RO_feasible_space_con_array = np.zeros(24*7),np.zeros(24*7),np.ones(24*7)*69
                    V_dot_wRO_array = np.zeros(24*7)
                    # Note: h_inlet_array still stays the same
                    break
                
                
        if converged_turbo_RO:
            get_rid_of_design = 0
            if print_statements == 1:
                print("Turbo-RO converged in", iteration + 1, "iterations.")
    
        else:
            # get_rid_of_design = 1 # Was just for good measure, now no longer terminating my code based on whether convergence was reached or not 
            get_rid_of_design = 0
            if print_statements == 1:
                #print("Turbo-RO did not converge after", max_iters_turbo_RO + 1, "iterations.")
                print("Turbo-RO didn't work. It stopped after iteration: " + str(iteration))
            
    return [S_fwRO_array,S_oRO_stage2_array,rho_oRO_stage2_array,P_oRO_stage2_array,
            h_inlet_array,h_RO_array,h_oRO_stage2_array,eta_RO_array,eta_RO_stage1_array,eta_RO_stage2_array,
            V_dot_wRO_array,RO_feasible_space_con_array,get_rid_of_design]
        
        


'''
# TAKE THIS THING OUT FOR A TEST SPIN
# Bring in WAVE RO model
WAVE_model = joblib.load('helper_functions/WAVE_outputs_autoML_2_hr_transform.joblib')

# Initialize h_turbo_inlet and V_dot_wRO_array
h_res = p_reservoir['h_res']
h_loss_pretreatment = p_RO_sys['h_loss_pretreatment']
#h_turbo_inlet_array = np.ones(24)*(h_res - h_loss_pretreatment)
h_inlet_array = np.ones(n_time_steps)*100#(h_res - h_loss_pretreatment)
rho_sw = p_state['rho_sw']
rho_fw = p_state['rho_fw']
N_pv1 = 900
ERD_choice = 1
V_dot_wRO_array = np.ones(n_time_steps)*N_pv1*10.5 # m^3/hr
#V_dot_wRO_array = np.random.uniform(8.5, 11.6, size=n_time_steps)*N_pv1

# Some convergence stuff
rtol_turbo_RO = p_iteration['rtol_turbo_RO']
atol_turbo_RO = p_iteration['atol_turbo_RO']
max_iters_turbo_RO = p_iteration['max_iters_turbo_RO']
converged_turbo_RO = p_iteration['converged_turbo_RO']

################ Initializing variables for finding a feasible initial point ####################
# Get the convex hull equations, break up terms into A and b
eqns = p_RO_sys['convex_polygon_eqns_coeffs']  # Each row: [a, b, c]
A = eqns[:, :2]       # shape (27, 2)
b = -eqns[:, 2]-1e-6       # Negate the b term → Ax <= b (with tolerance)

# Define constraints --> Equality constraints have to be defined in the iterative loop
ineq_constraints_convex_polygon = [{'type': 'ineq', 'fun': lambda x, A=A, b=b, i=i: b[i] - np.dot(A[i], x)}
                                       for i in range(len(b))]

# Define an initial guess for a feasible point in the convex_hull
x0_convex_polygon = np.array([550,10.1])


################ Initialized inputs for turbocharger model #######################
# Brine density
rho_oRO_stage2_array = np.ones(n_time_steps)*1045 # [kg/m^3]

# Brine salinity
S_oRO_stage2_array = np.ones(n_time_steps)*50 # g/kg

# Brine pressure
h_RO_stage1_array_sub_calc = h_inlet_array+150
h_brine_stage1_array_sub_calc = h_RO_stage1_array_sub_calc*0.9 # [m]
h_brine_stage2_array_sub_calc = h_brine_stage1_array_sub_calc*0.9 # [m]
P_oRO_stage2_array = (0.0981*h_brine_stage2_array_sub_calc*(rho_oRO_stage2_array/rho_fw))*14.5038 # psi (from bar)

# Compile
x_A_input_turbo_RO = np.concatenate((P_oRO_stage2_array,S_oRO_stage2_array,rho_oRO_stage2_array))


big_iteration = 0


if ERD_choice == 1:
    [h_turbo_inlet_array,h_turbo_outlet_array,eta_RO_array,eta_RO_stage1_array,
     eta_RO_stage2_array,P_oRO_stage2_array,h_oRO_stage2_array,rho_oRO_stage2_array,
     S_oRO_stage2_array,V_dot_wRO_array,S_fwRO_array,
     plotting_turbo_RO,RO_feasible_space_con_array,get_rid_of_design] = turbocharger_RO_iteration(N_pv1,ERD_choice,
                                                                                h_inlet_array,P_oRO_stage2_array,S_oRO_stage2_array,rho_oRO_stage2_array,V_dot_wRO_array,
                                                                                WAVE_model,ineq_constraints_convex_polygon,x0_convex_polygon,
                                                                                x_A_input_turbo_RO,max_iters_turbo_RO,rtol_turbo_RO,atol_turbo_RO,converged_turbo_RO,
                                                                                p_state,p_reservoir,p_RO_sys,p_pump_turbine_turbo_pelton_coeffs,
                                                                                rho_sw,rho_fw,big_iteration,n_time_steps)
elif ERD_choice == 0:
    [S_fwRO_array,S_oRO_stage2_array,h_RO_array,rho_oRO_stage2_array,
     P_oRO_stage2_array,h_oRO_stage2_array,eta_RO_array,eta_RO_stage1_array,
     eta_RO_stage2_array,RO_feasible_space_con_array,get_rid_of_design] = turbocharger_RO_iteration(N_pv1,ERD_choice,
                                                                                                    h_inlet_array,P_oRO_stage2_array,S_oRO_stage2_array,rho_oRO_stage2_array,V_dot_wRO_array,
                                                                                                    WAVE_model,ineq_constraints_convex_polygon,x0_convex_polygon,
                                                                                                    x_A_input_turbo_RO,max_iters_turbo_RO,rtol_turbo_RO,atol_turbo_RO,converged_turbo_RO,
                                                                                                    p_state,p_reservoir,p_RO_sys,p_pump_turbine_turbo_pelton_coeffs,
                                                                                                    rho_sw,rho_fw,big_iteration,n_time_steps)
                                                    

#print(plotting_turbo_RO)
                                                                            
# Save the plotting output as a csv
if ERD_choice == 1:
    df = pd.DataFrame(plotting_turbo_RO)
    df.to_csv("turbo_RO_convergence_900_1_0_4_5.csv") # (N_pv1,ERD_choice,big_iteration)
'''