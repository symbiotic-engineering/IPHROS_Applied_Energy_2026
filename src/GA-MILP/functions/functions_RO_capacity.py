"""
Matt Haefner (mwh85)
SEA Lab
Desalination Paper Optimization Price Taker - RO capacity module
LAST UPDATED: 10/21/25     CREATED: 9/15/24
"""

import numpy as np

def RO_capacity(V_dot_fwRO_arrays,print_statements):
    
    RO_production_day1s = np.sum(V_dot_fwRO_arrays[0:24,:], axis=0) # m^3/day
    RO_production_day2s = np.sum(V_dot_fwRO_arrays[24:48,:], axis=0) # m^3/day
    RO_production_day3s = np.sum(V_dot_fwRO_arrays[48:72,:], axis=0) # m^3/day
    RO_production_day4s = np.sum(V_dot_fwRO_arrays[72:96,:], axis=0) # m^3/day
    RO_production_day5s = np.sum(V_dot_fwRO_arrays[96:120,:], axis=0) # m^3/day
    RO_production_day6s = np.sum(V_dot_fwRO_arrays[120:144,:], axis=0) # m^3/day
    RO_production_day7s = np.sum(V_dot_fwRO_arrays[144:168,:], axis=0) # m^3/day
        
    RO_productions = np.hstack((RO_production_day1s,RO_production_day2s,RO_production_day3s,RO_production_day4s,
                                RO_production_day5s,RO_production_day6s,RO_production_day7s))
    
    B_RO = max(RO_productions) # m^3/day
    
    if print_statements == 1:
        print('Start of RO Capacity Module Output')
        print('B_RO: ' + str(B_RO))
        print('End of RO Capacity Module Output')
    
    return B_RO


'''
V_dot_fwRO_arrays = np.random.randint(1, 101, size=(168, 4))
print_statements = 1

B_RO = RO_capacity(V_dot_fwRO_arrays,print_statements)
'''