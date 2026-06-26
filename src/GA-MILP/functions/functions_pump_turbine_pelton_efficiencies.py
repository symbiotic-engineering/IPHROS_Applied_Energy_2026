# -*- coding: utf-8 -*-
"""
Matt Haefner (mwh85)
SEA Lab
Desalination Paper Optimization Price Taker - pump/turbine efficiency module
LAST UPDATED: 11/11/25     CREATED: 3/10/25
"""

import numpy as np

def francis_pelton_efficiencies(B_PSH,N_pv1,ERD_choice,x_LP,eta_pelton_array,h_oRO_stage2_array,rho_oRO_stage2_array,eta_RO_stage1_array,eta_RO_stage2_array,epsilon,
                                p_state,p_reservoir,p_assumed,p_universal,p_francis_turbo_pelton_coeffs,print_statements):
    
    # Unpack parameters
    g = p_universal['g'] # m/s^2
    
    # Start off by unpacking and lightly post-processing x_LP
    V_dot_wp_matrix = x_LP[:,:7]
    V_dot_swht_matrix = x_LP[:,7:14]
    Q_wRO_stage1_array = x_LP[:,14]
    
    V_dot_wRO_stage1_array = Q_wRO_stage1_array*N_pv1 # m^3/hr
    V_dot_oRO_stage1_array = (1-eta_RO_stage1_array)*V_dot_wRO_stage1_array
    V_dot_wRO_stage2_array = V_dot_oRO_stage1_array
    V_dot_oRO_stage2_array = (1-eta_RO_stage2_array)*V_dot_wRO_stage2_array
    
    P_out_pelton_array = (eta_pelton_array*rho_oRO_stage2_array*V_dot_oRO_stage2_array*g*h_oRO_stage2_array)/(3.6e6) # kW
    
    
    # Initialize unequal nameplates
    nominal_nameplate = B_PSH/7
    nameplates = [nominal_nameplate*(1+(3*epsilon)),
                  nominal_nameplate*(1+(2*epsilon)),
                  nominal_nameplate*(1+(1*epsilon)),
                  nominal_nameplate,
                  nominal_nameplate*(1-(1*epsilon)),
                  nominal_nameplate*(1-(2*epsilon)),
                  nominal_nameplate*(1-(3*epsilon))
                  ]
    
    # Unpack state parameters
    rho_sw = p_state['rho_sw']
    
    # Unpack reservoir parameters
    h_res = p_reservoir['h_res']
    
    # Unpack universal parameters
    g = p_universal['g']
    
    # Unpack assumed parameters
    n_turbines = p_assumed['n_pump_turbines']
    eta_francis_max_flow = p_assumed['eta_francis_max_flow']
    eta_pelton_max_flow = p_assumed['eta_pelton_max_flow'] 
    
    # Unpack francis pump/turbine coefficients
    a_francis = p_francis_turbo_pelton_coeffs['a_francis']
    b_francis = p_francis_turbo_pelton_coeffs['b_francis']
    c_francis = p_francis_turbo_pelton_coeffs['c_francis']
    d_francis = p_francis_turbo_pelton_coeffs['d_francis']
    e_francis = p_francis_turbo_pelton_coeffs['e_francis']
    f_francis = p_francis_turbo_pelton_coeffs['f_francis']
    
    # Unpack pelton coefficients
    a_pelton = p_francis_turbo_pelton_coeffs['a_pelton']
    b_pelton = p_francis_turbo_pelton_coeffs['b_pelton']
    c_pelton = p_francis_turbo_pelton_coeffs['c_pelton']
    d_pelton = p_francis_turbo_pelton_coeffs['d_pelton']
    e_pelton = p_francis_turbo_pelton_coeffs['e_pelton']
    
    # Initialize output matrices
    eta_hp_matrix = np.ones((24*7,7))*69
    eta_ht_matrix = np.ones((24*7,7))*69
    eta_pelton_array = np.ones(24*7)*69
    
    for n in range(n_turbines):
        # Deal with one turbine at a time
        V_dot_wp_array = V_dot_wp_matrix[:,n]
        V_dot_swht_array = V_dot_swht_matrix[:,n]
    
        # Initialize eta arrays
        eta_hp_array = np.ones(len(V_dot_wp_array))*69
        eta_ht_array = np.ones(len(V_dot_swht_array))*69
        
        # Create a dictionary of unique V_dot_wp values with their indices
        unique_value_dict = {
                            val: np.where(V_dot_wp_array == val)[0]
                            for val in set(V_dot_wp_array)
                            }
    
        for i in range(len(unique_value_dict)):
            
            # Get a unique pelton turbine inflow configuration, and all the indices that it is associated with
            V_dot_wp = list(unique_value_dict.keys())[i]
            idxs = unique_value_dict[V_dot_wp]
            
            # Calculate what the maximum flowrate is on the pump side
            V_dot_wp_max = (eta_francis_max_flow*(3.6e6)*(nameplates[n]))/(rho_sw*g*h_res) # m^3/hr
            
            # Get the efficiency associated with that flowrate
            if V_dot_wp < 1:
                eta_francis = 0
            else:
                x = V_dot_wp/V_dot_wp_max
                eta_francis_expression = ((a_francis*(x**4))+(b_francis*(x**3))+(c_francis*(x**2))+(d_francis*x)+e_francis)/(x-f_francis)
                eta_francis = max(0,eta_francis_expression)
            
            #print('This is for a unique uphill flowrate')
            #print('V_dot_wp = ' + str(V_dot_wp) + ' m^3/hr')
            #print('V_dot_wp_max = ' + str(V_dot_wp_max) + ' m^3/hr')
            #print('eta_hp = ' + str(eta_francis))
    
            # Store the efficiency
            eta_hp_array[idxs] = np.ones(len(idxs))*eta_francis
            
        # Same thing but now for V_dot_swht_array
        unique_value_dict = {
                            val: np.where(V_dot_swht_array == val)[0]
                            for val in set(V_dot_swht_array)
                            }
    
        for i in range(len(unique_value_dict)):
            
            # Get a unique pelton turbine inflow configuration, and all the indices that it is associated with
            V_dot_swht = list(unique_value_dict.keys())[i]
            idxs = unique_value_dict[V_dot_swht]
            
            # Calculate what the maximum flowrate is on the turbine side
            V_dot_swht_max = ((3.6e6)*(nameplates[n]))/(eta_francis_max_flow*rho_sw*g*h_res) # m^3/hr
            
            # Get the efficiency associated with that flowrate
            if V_dot_swht < 1:
                eta_francis = 0
            else:
                x = V_dot_swht/V_dot_swht_max
                eta_francis_expression = ((a_francis*(x**4))+(b_francis*(x**3))+(c_francis*(x**2))+(d_francis*x)+e_francis)/(x-f_francis)
                eta_francis = max(0,eta_francis_expression)
    
            # Store the efficiency
            eta_ht_array[idxs] = np.ones(len(idxs))*eta_francis
            
            #print('This is for a unique downhill flowrate')
            #print('V_dot_swht = ' + str(V_dot_swht) + ' m^3/hr')
            #print('V_dot_swht_max = ' + str(V_dot_swht_max) + ' m^3/hr')
            #print('eta_ht = ' + str(eta_francis))
            
        # Store resulting efficiency arrays
        eta_hp_matrix[:,n] = eta_hp_array
        eta_ht_matrix[:,n] = eta_ht_array
        
    
    if ERD_choice == 0: # This indicates pelton turbine is activated
        
        # First calculate all the possible nameplate capacities
        B_pelton = max(P_out_pelton_array) # kW
    
        # Create a dictionary of unique (V_dot_oRO, h_oRO_stage2, rho_oRO_stage2) triplets with their indices
        unique_triplet_dict = {
            triplet: np.where((V_dot_oRO_stage2_array == triplet[0]) & (h_oRO_stage2_array == triplet[1]) & (rho_oRO_stage2_array == triplet[2]))[0]
            for triplet in set(zip(V_dot_oRO_stage2_array, h_oRO_stage2_array, rho_oRO_stage2_array))
        }
    
        for i in range(len(unique_triplet_dict)):
            
            # Get a unique pelton turbine inflow configuration, and all the indices that it is associated with
            V_dot_oRO = list(unique_triplet_dict.keys())[i][0]
            h_oRO_stage2 = list(unique_triplet_dict.keys())[i][1]
            rho_oRO_stage2 = list(unique_triplet_dict.keys())[i][2]
            idxs = unique_triplet_dict[(V_dot_oRO,h_oRO_stage2,rho_oRO_stage2)]
            
            # Calculate what the maximum flowrate is on the pump side
            V_dot_oRO_max = ((3.6e6)*B_pelton)/(eta_pelton_max_flow*rho_oRO_stage2*g*h_oRO_stage2) # m^3/hr
            
            # Get the efficiency associated with that flowrate
            if V_dot_oRO >= V_dot_oRO_max:
                eta_pelton = eta_pelton_max_flow
            elif V_dot_oRO < 0.11034*V_dot_oRO_max:
                eta_pelton = 0
            else:
                x = V_dot_oRO/V_dot_oRO_max
                eta_pelton_expression = ((a_pelton*x)+b_pelton)/((x**3)+(c_pelton*(x**2))+(d_pelton*x)+e_pelton)
                eta_pelton = max(0,eta_pelton_expression)
    
            # Store the efficiency
            eta_pelton_array[idxs] = np.ones(len(idxs))*eta_pelton
            
    elif ERD_choice == 1: # This means that the turbocharger is in the design
        eta_pelton_array = np.zeros(len(V_dot_oRO_stage2_array))
        
    if print_statements == 1:
        print('Start of Efficiencies Module Output')
        print('eta_hp_turbine1_array: ' + str(eta_hp_matrix[:,0]))
        print('eta_hp_turbine2_array: ' + str(eta_hp_matrix[:,1]))
        print('eta_hp_turbine3_array: ' + str(eta_hp_matrix[:,2]))
        print('eta_hp_turbine4_array: ' + str(eta_hp_matrix[:,3]))
        print('eta_hp_turbine5_array: ' + str(eta_hp_matrix[:,4]))
        print('eta_hp_turbine6_array: ' + str(eta_hp_matrix[:,5]))
        print('eta_hp_turbine7_array: ' + str(eta_hp_matrix[:,6]))
        print('eta_ht_turbine1_array: ' + str(eta_ht_matrix[:,0]))
        print('eta_ht_turbine2_array: ' + str(eta_ht_matrix[:,1]))
        print('eta_ht_turbine3_array: ' + str(eta_ht_matrix[:,2]))
        print('eta_ht_turbine4_array: ' + str(eta_ht_matrix[:,3]))
        print('eta_ht_turbine5_array: ' + str(eta_ht_matrix[:,4]))
        print('eta_ht_turbine6_array: ' + str(eta_ht_matrix[:,5]))
        print('eta_ht_turbine7_array: ' + str(eta_ht_matrix[:,6]))
        print('eta_pelton_array: ' + str(eta_pelton_array))
        print('End of Efficiencies Module Output')
        
    
    return [eta_hp_matrix,eta_ht_matrix,eta_pelton_array]




#----------------------------------------------------------
# TAKING THIS OUT FOR A TEST DRIVE! (requires running post_processing_nominal_results.py first)
#----------------------------------------------------------
'''
B_PSH = 1000e3 # kW
N_pv1 = 750
ERD_choice = 1 # turbocharger

import pickle
with open("nominal_turbo_RO_convergence.pkl", "rb") as f:
    turbo_RO_convergence = pickle.load(f)
with open("nominal_LP_efficiencies_convergence.pkl", "rb") as f:
    LP_efficiencies_convergence = pickle.load(f)


V_dot_wp1_array = LP_efficiencies_convergence['season2']['big_iteration0']['LP_efficiencies_iteration0']['V_dot_wp1_array']
V_dot_wp2_array = LP_efficiencies_convergence['season2']['big_iteration0']['LP_efficiencies_iteration0']['V_dot_wp2_array']
V_dot_wp3_array = LP_efficiencies_convergence['season2']['big_iteration0']['LP_efficiencies_iteration0']['V_dot_wp3_array']
V_dot_wp4_array = LP_efficiencies_convergence['season2']['big_iteration0']['LP_efficiencies_iteration0']['V_dot_wp4_array']
V_dot_wp5_array = LP_efficiencies_convergence['season2']['big_iteration0']['LP_efficiencies_iteration0']['V_dot_wp5_array']
V_dot_wp6_array = LP_efficiencies_convergence['season2']['big_iteration0']['LP_efficiencies_iteration0']['V_dot_wp6_array']
V_dot_wp7_array = LP_efficiencies_convergence['season2']['big_iteration0']['LP_efficiencies_iteration0']['V_dot_wp7_array']

V_dot_wp_matrix = np.zeros((168,7))
V_dot_wp_matrix[:,0] = V_dot_wp1_array
V_dot_wp_matrix[:,1] = V_dot_wp2_array
V_dot_wp_matrix[:,2] = V_dot_wp3_array
V_dot_wp_matrix[:,3] = V_dot_wp4_array
V_dot_wp_matrix[:,4] = V_dot_wp5_array
V_dot_wp_matrix[:,5] = V_dot_wp6_array
V_dot_wp_matrix[:,6] = V_dot_wp7_array

V_dot_swht1_array = LP_efficiencies_convergence['season2']['big_iteration0']['LP_efficiencies_iteration0']['V_dot_swht1_array']
V_dot_swht2_array = LP_efficiencies_convergence['season2']['big_iteration0']['LP_efficiencies_iteration0']['V_dot_swht2_array']
V_dot_swht3_array = LP_efficiencies_convergence['season2']['big_iteration0']['LP_efficiencies_iteration0']['V_dot_swht3_array']
V_dot_swht4_array = LP_efficiencies_convergence['season2']['big_iteration0']['LP_efficiencies_iteration0']['V_dot_swht4_array']
V_dot_swht5_array = LP_efficiencies_convergence['season2']['big_iteration0']['LP_efficiencies_iteration0']['V_dot_swht5_array']
V_dot_swht6_array = LP_efficiencies_convergence['season2']['big_iteration0']['LP_efficiencies_iteration0']['V_dot_swht6_array']
V_dot_swht7_array = LP_efficiencies_convergence['season2']['big_iteration0']['LP_efficiencies_iteration0']['V_dot_swht7_array']

V_dot_swht_matrix = np.zeros((168,7))
V_dot_swht_matrix[:,0] = V_dot_swht1_array
V_dot_swht_matrix[:,1] = V_dot_swht2_array
V_dot_swht_matrix[:,2] = V_dot_swht3_array
V_dot_swht_matrix[:,3] = V_dot_swht4_array
V_dot_swht_matrix[:,4] = V_dot_swht5_array
V_dot_swht_matrix[:,5] = V_dot_swht6_array
V_dot_swht_matrix[:,6] = V_dot_swht7_array

Q_oRO_stage2_array = np.zeros(168)

h_oRO_stage2_array = turbo_RO_convergence['season2']['big_iteration0']['turbo_RO_iteration2']['h_oRO_stage2_array']
rho_oRO_stage2_array = turbo_RO_convergence['season2']['big_iteration0']['turbo_RO_iteration2']['rho_oRO_stage2_array']

# State Parameters
p_state = {'rho_sw':1026.311,   # spain seawater density [kg/m^3]
           'rho_fw':998.049,    # density of freshwater at atmospheric pressure and 19.833 deg. celsius (using eqn from Nayar et al. 2016)
           'T':19.833,          # temperature [degrees celsius]
           'S_sw_g_kg':37.157,  # generic seawater salinity [g salt/kg seawater]
           'S_sw_mg_kg':37157,  # generic seawater salinity [mg salt/kg seawater]
           'M_salt':58.44,      # molar mass of NaCl [g/mol]
           'P_p':14.696,        # permeate pressure, atmospheric for now [psi]
           'MW_salt_avg':30.548 # average molar mass of NaCl [g/mol]
           }

# Reservoir Parameters
p_reservoir = {'h_res':332,             # reservoir height [m]
               'res_cap':28635496,      # reservoir capacity [m^3]
               'res_init':0.65*28635496 # initial reservoir capacity [m^3]
               }

# RO Parameters
p_RO_sys = {'N_e1':5,                               # # of stage 1 elements per pressure vessel [-]
            'N_e2':5,                               # # of stage 2 elements per pressure vessel [-]
            'N_pv_rat':1.5,                         # ratio of stage 1 pressure vessels to stage 2 pressure vessels [-]
            'A_membrane':41,                        # area of 1 seamaxx-440 membrane [m^2]
            'feed_spacer_porosity':0.89,            # spacer porosity in membrane stage [-]
            'feed_channel_height':7.112e-4,         # channel height in membrane stage [m]
            'membrane_length':1.016,                # membrane length [m]
            'B_RO_max':250000,                      # maximum allowable nameplate capacity of the RO system [m^3/day]
            'h_loss_pretreatment':34.4,             # head loss due to DAF and ceramic membrane pretreatment [m]
            'lambda_P_c':2.057510225486819,         # box-cox transform parameter for P_c
            'lambda_TDS_c':1.8630170124671173,      # box-cox transform parameter for TDS_c
            'lambda_TDS_p':-0.06430666580338582,    # box-cox transform parameter for TDS_p
            'lambda_Q_p':0.09483695370086277,       # box-cox transform parameter for Q_p
            'convex_polygon_eqns_coeffs': np.array([[1,-0,-692.6], # generated by two_stage_feasiblity_convex_hull.py
                                                    [-0,-1,4.1],
                                                    [-0.0138983,0.999903,-4.91265],
                                                    [-0.0324154,0.999474,4.44037],
                                                    [0.00608097,-0.999982,1.23822],
                                                    [0.0216166,-0.999766,-7.39341],
                                                    [0.0324154,-0.999474,-13.6755],
                                                    [0,1,-14.18],
                                                    [0.0647288,0.997903,-57.5445],
                                                    [0.149647,0.988739,-114.818],
                                                    [0.128652,0.99169,-100.548],
                                                    [-0.0808159,0.996729,24.6758],
                                                    [-0.0647288,0.997903,18.5863],
                                                    [-0.0754599,0.997149,22.728],
                                                    [-0.0462824,0.998928,11.1262],
                                                    [-0.0485912,0.998819,12.1796],
                                                    [-0.0539753,0.998542,14.3968],
                                                    [-0.0518222,0.998656,13.5341],
                                                    [-0.0162141,0.999869,-3.54818],
                                                    [0.0121613,-0.999926,-1.99837],
                                                    [0.0108102,-0.999942,-1.26749],
                                                    [0.0647288,-0.997903,-34.8721],
                                                    [0.0539753,-0.998542,-27.4578],
                                                    [0.0432029,-0.999066,-20.1914],
                                                    [0.0445503,-0.999007,-21.0653],
                                                    [-0.0153609,0.999882,-3.98141],
                                                    [-0.0152605,0.999884,-4.03943]]),
            'reduced_convex_polygon_eqns_coeffs': np.array([[ 1.55204182e-17, -1.00000000e+00,  4.11000000e+00],
                                                           [-1.38982714e-02,  9.99903414e-01, -4.90265288e+00],
                                                           [ 1.49647056e-01,  9.88739480e-01, -1.14808307e+02],
                                                           [ 1.28651656e-01,  9.91689846e-01, -1.00538238e+02],
                                                           [ 0.00000000e+00,  1.00000000e+00, -1.41700000e+01],
                                                           [ 6.47288361e-02,  9.97902890e-01, -5.75344747e+01],
                                                           [ 6.08096865e-03, -9.99981511e-01,  1.24822035e+00],
                                                           [ 1.00000000e+00,  0.00000000e+00, -6.92590000e+02],
                                                           [-6.47288361e-02,  9.97902890e-01,  1.85963459e+01],
                                                           [-1.62140845e-02,  9.99868543e-01, -3.53818215e+00],
                                                           [ 1.21612628e-02, -9.99926049e-01, -1.98836572e+00],
                                                           [ 1.08101791e-02, -9.99941568e-01, -1.25749350e+00],
                                                           [ 5.39752580e-02, -9.98542273e-01, -2.74477535e+01],
                                                           [ 6.47288361e-02, -9.97902890e-01, -3.48621210e+01],
                                                           [ 2.16165694e-02, -9.99766335e-01, -7.38340715e+00],
                                                           [ 3.24153886e-02, -9.99474483e-01, -1.36655122e+01],
                                                           [ 4.45503185e-02, -9.99007142e-01, -2.10552806e+01],
                                                           [ 4.32028679e-02, -9.99066320e-01, -2.01814003e+01],
                                                           [-8.08158688e-02,  9.96729048e-01,  2.46857786e+01],
                                                           [-7.54599119e-02,  9.97148836e-01,  2.27379865e+01],
                                                           [-3.24153886e-02,  9.99474483e-01,  4.45036799e+00],
                                                           [-4.62823967e-02,  9.98928396e-01,  1.11362110e+01],
                                                           [-4.85911825e-02,  9.98818751e-01,  1.21896498e+01],
                                                           [-1.52605439e-02,  9.99883551e-01, -4.02943417e+00],
                                                           [-1.53609186e-02,  9.99882014e-01, -3.97140786e+00],
                                                           [-5.39752580e-02,  9.98542273e-01,  1.44068206e+01],
                                                           [-5.18221663e-02,  9.98656329e-01,  1.35441145e+01]])#,
            }

# Assumed Parameters
p_assumed = {'n_pump_turbines':7, # -
             'plant_sec':3.4,       # kWh/m^3
             'pipe_length':1000,     # m
             'harmonics':[1,7] # harmonics for the S_ht fourier series [-]
             }

# Universal Constants
p_universal = {'g':9.81,    # gravitational acceleration [m/s^2]
               'R':8.3145   # universal gas constant [J/(mol*K)]
               }    

# Francis turbine coefficients
p_francis_turbo_pelton_coeffs = {'a_francis':-2.7937,
                                 'b_francis':5.7484,
                                 'c_francis':-4.0409,
                                 'd_francis':2.2726,
                                 'e_francis':-0.3489,
                                 'f_francis':0.0414,
                                 'a_turbo':-1.1518,
                                 'b_turbo':1.0550,
                                 'c_turbo':0.9845,
                                 'd_turbo':-9.416e-06,
                                 'a_pelton':2.7372,
                                 'b_pelton':-0.3020,
                                 'c_pelton':-0.9715,
                                 'd_pelton':3.0932,
                                 'e_pelton':-0.2583
                                 }

print_statements = 0

eta_hp_matrix,eta_ht_matrix,eta_pelton_array = francis_pelton_efficiencies(B_PSH,N_pv1,ERD_choice,V_dot_wp_matrix,V_dot_swht_matrix,Q_oRO_stage2_array,
                                                                           h_oRO_stage2_array,rho_oRO_stage2_array,
                                                                           p_state,p_reservoir,p_RO_sys,p_assumed,p_universal,p_francis_turbo_pelton_coeffs,
                                                                           print_statements)

'''



