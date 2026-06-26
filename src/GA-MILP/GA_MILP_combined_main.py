"""
Matt Haefner (mwh85)
SEA Lab
Desalination Paper Optimization Price Taker - Main File
Created: 2/17/25		Last Edited: 10/28/25
"""



#--------------------------------------
# NOTE: i added a couple extra print statements to GA_LP_simulation.py...remove these before running on AWS
#--------------------------------------




# Import General Packages
import numpy as np
import pandas as pd
import time
import json

import joblib
import pickle

# Import Pymoo Packages (to get pymoo, ran the command pip install -U pymoo)
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

# Import Parallelization Packages
import multiprocessing
from pymoo.core.problem import StarmapParallelization

# Import my parameters
from params_spain import *

# Import representative week function
from helper_functions.helper_function_representative_time_averaging import rep_weeks_electricity

# Import simulation function
from GA_LP_simulation import GA_LP_simulation


# Representative Weeks
electricity_prices_rep_week = rep_weeks_electricity(month_list,hours_idx,days_in_season)

# Water Price
water_price = 1.97 # Euros/m^3

# Bring in WAVE RO model
WAVE_model_bundle = joblib.load("helper_functions/RO_model_bundle.joblib")

# Bring in Epsilon predictor model
with open("helper_functions/eps_logit_bundle.pkl", "rb") as f:
    eps_logit_bundle = pickle.load(f)
'''
with open("helper_functions/logit_eps_choice.pkl", "rb") as f:
    epsilon_logit_model = pickle.load(f)
'''

# Do I want print statements?
print_statements= 0

#####################################################################################################################################################
 
# GA Multi Stuff - Implement the Problem
class IPHROS_MOGA_LP(ElementwiseProblem):
    def __init__(self, **kwargs):
        
        my_dict = {'B_PSH':Real(bounds=(0,1500e3)), # kW
                   'N_pv1_sub':Real(bounds=(int(0/3),int(3000/3))),
                   'ERD_choice':Integer(bounds=(0,1)), # 0 for pelton, 1 for turbocharger
                   'a0_spring':Real(bounds=(p_state['S_sw_g_kg'],38.26)), # psu
                   'a1_spring':Real(bounds=(-1.55,1.95)), # psu
                   'a7_spring':Real(bounds=(-1.63,1.78)), # psu
                   'b1_spring':Real(bounds=(-1.64,2.03)), # psu
                   'b7_spring':Real(bounds=(-1.77,2.06)), # psu
                   'a0_summer':Real(bounds=(p_state['S_sw_g_kg'],38.26)), # psu
                   'a1_summer':Real(bounds=(-1.55,1.95)), # psu
                   'a7_summer':Real(bounds=(-1.63,1.78)), # psu
                   'b1_summer':Real(bounds=(-1.64,2.03)), # psu
                   'b7_summer':Real(bounds=(-1.77,2.06)), # psu
                   'a0_fall':Real(bounds=(p_state['S_sw_g_kg'],38.26)), # psu
                   'a1_fall':Real(bounds=(-1.55,1.95)), # psu
                   'a7_fall':Real(bounds=(-1.63,1.78)), # psu
                   'b1_fall':Real(bounds=(-1.64,2.03)), # psu
                   'b7_fall':Real(bounds=(-1.77,2.06)), # psu
                   'a0_winter':Real(bounds=(p_state['S_sw_g_kg'],38.26)), # psu
                   'a1_winter':Real(bounds=(-1.55,1.95)), # psu
                   'a7_winter':Real(bounds=(-1.63,1.78)), # psu
                   'b1_winter':Real(bounds=(-1.64,2.03)), # psu
                   'b7_winter':Real(bounds=(-1.77,2.06)) # psu
                   }

        super().__init__(vars=my_dict,n_obj=1,n_ieq_constr=11,**kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        
        # Unpack x
        B_PSH = x['B_PSH']
        N_pv1 = int(x['N_pv1_sub']*3)
        ERD_choice = x['ERD_choice'] # -
        
        a0_spring = x['a0_spring']
        a1_spring = x['a1_spring']
        a7_spring = x['a7_spring']
        b1_spring = x['b1_spring']
        b7_spring = x['b7_spring']
        
        a_k_array_spring = [a1_spring,a7_spring]
        b_k_array_spring = [b1_spring,b7_spring]
        
        a0_summer = x['a0_summer']
        a1_summer = x['a1_summer']
        a7_summer = x['a7_summer']
        b1_summer = x['b1_summer']
        b7_summer = x['b7_summer']
        
        a_k_array_summer = [a1_summer,a7_summer]
        b_k_array_summer = [b1_summer,b7_summer]
        
        a0_fall = x['a0_fall']
        a1_fall = x['a1_fall']
        a7_fall = x['a7_fall']
        b1_fall = x['b1_fall']
        b7_fall = x['b7_fall']
        
        a_k_array_fall = [a1_fall,a7_fall]
        b_k_array_fall = [b1_fall,b7_fall]
        
        a0_winter = x['a0_winter']
        a1_winter = x['a1_winter']
        a7_winter = x['a7_winter']
        b1_winter = x['b1_winter']
        b7_winter = x['b7_winter']
        
        a_k_array_winter = [a1_winter,a7_winter]
        b_k_array_winter = [b1_winter,b7_winter]
        
        a0_array = np.array([a0_spring,a0_summer,a0_fall,a0_winter])
        a_k_arrays = np.vstack((a_k_array_spring,a_k_array_summer,a_k_array_fall,a_k_array_winter)).transpose()
        b_k_arrays = np.vstack((b_k_array_spring,b_k_array_summer,b_k_array_fall,b_k_array_winter)).transpose()
        
        # Run the simulation
        [NPV_max,B_RO_max_con,S_ht_spring_max_con1,S_ht_summer_max_con1,
         S_ht_fall_max_con1,S_ht_winter_max_con1,S_ht_spring_max_con2,
         S_ht_summer_max_con2,S_ht_fall_max_con2,
         S_ht_winter_max_con2,S_fwRO_max_cons,
         RO_feasibility_max_con,LP_feasibility_min_con] = GA_LP_simulation(B_PSH,N_pv1,ERD_choice,
                                                                           a0_array,a_k_arrays,b_k_arrays,
                                                                           eps_logit_bundle,WAVE_model_bundle,
                                                                           electricity_prices_rep_week,water_price,
                                                                           print_statements) 


        ######################## Compiling ##################################
        #out["F"] = [NPV_max,neg_buy_sell_profit_year_min,eta_RO_avg_year_max]
        out["F"] = [NPV_max]
        
        # This is with the S_fw_max constraints
        '''
        GGGGGGGG = np.hstack((B_RO_max_con,S_ht_spring_max_con1,S_ht_summer_max_con1,
                              S_ht_fall_max_con1,S_ht_winter_max_con1,S_ht_spring_max_con2,
                              S_ht_summer_max_con2,S_ht_fall_max_con2,S_ht_winter_max_con2,
                              S_fwRO_max_cons,RO_feasibility_max_con,LP_feasibility_min_con))
        '''
        # And this is without
        GGGGGGGG = np.hstack((B_RO_max_con,S_ht_spring_max_con1,S_ht_summer_max_con1,
                              S_ht_fall_max_con1,S_ht_winter_max_con1,S_ht_spring_max_con2,
                              S_ht_summer_max_con2,S_ht_fall_max_con2,S_ht_winter_max_con2,
                              RO_feasibility_max_con,LP_feasibility_min_con))
        
        out["G"] = GGGGGGGG.tolist()


# initialize the thread pool and create the runner
#'''
n_processes = 100 # 12, 95
pool = multiprocessing.Pool(n_processes)
runner = StarmapParallelization(pool.starmap)
#'''

# define the problem by passing the starmap interface of the thread pool
problem = IPHROS_MOGA_LP(elementwise_runner=runner)
#problem = IPHROS_MOGA_LP()

# Function for recreating the population
def RecreatingPopulation(X_load_in_matrix):
    samples = []

    for j in range(X_load_in_matrix.shape[0]):
        sample = {} 

        sample['B_PSH'] = X_load_in_matrix[j,-39]
        sample['N_pv1_sub'] = X_load_in_matrix[j,-38]
        sample['ERD_choice'] = X_load_in_matrix[j,-37]
        sample['a0_spring'] = X_load_in_matrix[j,-36]
        sample['a1_spring'] = X_load_in_matrix[j,-35]
        sample['a2_spring'] = X_load_in_matrix[j,-34]
        sample['a3_spring'] = X_load_in_matrix[j,-33]
        sample['a4_spring'] = X_load_in_matrix[j,-32]
        sample['b1_spring'] = X_load_in_matrix[j,-31]
        sample['b2_spring'] = X_load_in_matrix[j,-30]
        sample['b3_spring'] = X_load_in_matrix[j,-29]
        sample['b4_spring'] = X_load_in_matrix[j,-28]
        sample['a0_summer'] = X_load_in_matrix[j,-27]
        sample['a1_summer'] = X_load_in_matrix[j,-26]
        sample['a2_summer'] = X_load_in_matrix[j,-25]
        sample['a3_summer'] = X_load_in_matrix[j,-24]
        sample['a4_summer'] = X_load_in_matrix[j,-23]
        sample['b1_summer'] = X_load_in_matrix[j,-22]
        sample['b2_summer'] = X_load_in_matrix[j,-21]
        sample['b3_summer'] = X_load_in_matrix[j,-20]
        sample['b4_summer'] = X_load_in_matrix[j,-19]
        sample['a0_fall'] = X_load_in_matrix[j,-18]
        sample['a1_fall'] = X_load_in_matrix[j,-17]
        sample['a2_fall'] = X_load_in_matrix[j,-16]
        sample['a3_fall'] = X_load_in_matrix[j,-15]
        sample['a4_fall'] = X_load_in_matrix[j,-14]
        sample['b1_fall'] = X_load_in_matrix[j,-13]
        sample['b2_fall'] = X_load_in_matrix[j,-12]
        sample['b3_fall'] = X_load_in_matrix[j,-11]
        sample['b4_fall'] = X_load_in_matrix[j,-10]
        sample['a0_winter'] = X_load_in_matrix[j,-9]
        sample['a1_winter'] = X_load_in_matrix[j,-8]
        sample['a2_winter'] = X_load_in_matrix[j,-7]
        sample['a3_winter'] = X_load_in_matrix[j,-6]
        sample['a4_winter'] = X_load_in_matrix[j,-5]
        sample['b1_winter'] = X_load_in_matrix[j,-4]
        sample['b2_winter'] = X_load_in_matrix[j,-3]
        sample['b3_winter'] = X_load_in_matrix[j,-2]
        sample['b4_winter'] = X_load_in_matrix[j,-1]
        
        
        samples.append(sample)

    return samples

# If not first run, need to load in a previous population, and then run another 10 generations
first_run = True
if first_run != True:
    X_load_in_prep_df = pd.read_csv('Desal_price_taker_dvs_10_gens.csv')
    X_load_in_prep = X_load_in_prep_df.to_numpy()[:,1]

    my_mega_list = []
    for i in range(len(X_load_in_prep)):
        pop_mem_string = X_load_in_prep[i].replace("'",'"')
        pop_mem_dict = json.loads(pop_mem_string)
        my_list = list(pop_mem_dict.values())
        my_mega_list.append(my_list)
    X_load_in_matrix = np.array(my_mega_list)

    samples = RecreatingPopulation(X_load_in_matrix)

    pop_load_in = Population.new("X", samples)
    Evaluator().eval(problem, pop_load_in)

    algorithm = MixedVariableGA(pop_size=100,sampling=pop_load_in,survival=RankAndCrowdingSurvival()) # 12, 95
else:
    algorithm = MixedVariableGA(pop_size=100,survival=RankAndCrowdingSurvival()) # 12, 95

tic = time.time()
res = minimize(problem,
               algorithm,
               termination=('n_gen',100), # 2
               seed = 1,
               save_history=True,
               verbose=True,
               return_least_infeasible=True)

toc = time.time()
tic_toc = toc-tic
print("GA run time was " + str(tic_toc) + " seconds.")

# Store results if we get anything "juicy"
if str(type(res.X)) != "<class 'NoneType'>":
    print('gaba goul')
    
#'''
# Get the ENTIRE final population
final_population= res.pop

# Get the design variables, objectives, and constraints
design_vars = final_population.get("X")
objective_vars = final_population.get("F")
constraint_vars = final_population.get("G")

design_vars_df = pd.DataFrame(design_vars)
objective_vars_df = pd.DataFrame(objective_vars)
constraint_vars_df = pd.DataFrame(constraint_vars)

design_vars_df.to_csv("Desal_price_taker_dvs.csv")
objective_vars_df.to_csv("Desal_price_taker_objs.csv")
constraint_vars_df.to_csv("Desal_price_taker_cons.csv")
#'''

#'''
# Get the history
print('Trying this out!')
history = res.history

for gen, state in enumerate(history):
    pop = state.pop
    
    X = pop.get("X")
    G = pop.get("G")
    F = pop.get("F")
    
    df_X = pd.DataFrame(X)
    df_G = pd.DataFrame(G)
    df_F = pd.DataFrame(F)
    
    df_X.to_csv("Desal_design_variables_gen_" + str(gen) + ".csv")
    df_G.to_csv("Desal_constraints_gen_" + str(gen) + ".csv")
    df_F.to_csv("Desal_objectives_gen_" + str(gen) + ".csv")
#'''

