# -*- coding: utf-8 -*-
"""
Matt Haefner (mwh85)
SEA Lab
Desalination Paper Optimization Price Taker - Output Salinity Creation Module
LAST UPDATED: 10/28/25     CREATED: 3/10/25
"""

import numpy as np
import math

def S_ht_array_generator(a0,a_k_array,b_k_array,p_assumed,print_statements):
    # Unpack parameters
    harmonics = p_assumed['harmonics']
    
    S_ht_array = np.zeros(168)
    t_array = np.linspace(0,167,168)
    T = 168
    for i in range(168):
        S_ht = a0
        idx = 0
        for k in harmonics: 
            S_ht += a_k_array[idx]*math.cos(2*math.pi*k*t_array[i]/T)
            S_ht += b_k_array[idx]*math.sin(2*math.pi*k*t_array[i]/T)
            idx += 1
        
        if S_ht < 38.5:
            S_ht = 38.5
        elif S_ht > 42:
            S_ht = 42
        
        S_ht_array[i] = S_ht
        
    freq_38_5 = (np.sum(S_ht_array > 38.5))/168
    freq_40 = (np.sum(S_ht_array > 40))/168
        
    if print_statements == 1:
        print('Start of Seagrass Salinity Constraint Module Output')
        print('S_ht_array: ' + str(S_ht_array))
        print('freq_38_5: ' + str(freq_38_5))
        print('freq_40: ' + str(freq_40))
        print('End of Seagrass Salinity Constraint Module Output')
        
    return S_ht_array,freq_38_5,freq_40
            






# Taking this for a spin!
'''
import random
import matplotlib.pyplot as plt
p_state = {'S_sw_g_kg':37.157  # generic seawater salinity [g salt/kg seawater]
           }
p_assumed = {'harmonics':[1,7] # harmonics for the S_ht fourier series [-]
             }
print_statements = 1


random.seed(42)

a0_spring = random.uniform(p_state['S_sw_g_kg'],38.26)
a1_spring = random.uniform(-1.55,1.95)
a7_spring = random.uniform(-1.63,1.78)
b1_spring = random.uniform(-1.64,2.03)
b7_spring = random.uniform(-1.77,2.06)

a_k_array_spring = [a1_spring,a7_spring]
b_k_array_spring = [b1_spring,b7_spring]

a0_summer = random.uniform(p_state['S_sw_g_kg'],38.26)
a1_summer = random.uniform(-1.55,1.95)
a7_summer = random.uniform(-1.63,1.78)
b1_summer = random.uniform(-1.64,2.03)
b7_summer = random.uniform(-1.77,2.06)

a_k_array_summer = [a1_summer,a7_summer]
b_k_array_summer = [b1_summer,b7_summer]

a0_fall = random.uniform(p_state['S_sw_g_kg'],38.26)
a1_fall = random.uniform(-1.55,1.95)
a7_fall = random.uniform(-1.63,1.78)
b1_fall = random.uniform(-1.64,2.03)
b7_fall = random.uniform(-1.77,2.06)

a_k_array_fall = [a1_fall,a7_fall]
b_k_array_fall = [b1_fall,b7_fall]

a0_winter = random.uniform(p_state['S_sw_g_kg'],38.26)
a1_winter = random.uniform(-1.55,1.95)
a7_winter = random.uniform(-1.63,1.78)
b1_winter = random.uniform(-1.64,2.03)
b7_winter = random.uniform(-1.77,2.06)

a_k_array_winter = [a1_winter,a7_winter]
b_k_array_winter = [b1_winter,b7_winter]

for i in range(4):
    if i == 0:
        S_ht_array_spring,freq_38_5_spring,freq_40_spring = S_ht_array_generator(a0_spring,a_k_array_spring,b_k_array_spring,p_assumed,print_statements)
    elif i == 1:
        S_ht_array_summer,freq_38_5_summer,freq_40_summer = S_ht_array_generator(a0_summer,a_k_array_summer,b_k_array_summer,p_assumed,print_statements)
    elif i == 2:
        S_ht_array_fall,freq_38_5_fall,freq_40_fall = S_ht_array_generator(a0_fall,a_k_array_fall,b_k_array_fall,p_assumed,print_statements)
    elif i == 3:
        S_ht_array_winter,freq_38_5_winter,freq_40_winter = S_ht_array_generator(a0_winter,a_k_array_winter,b_k_array_winter,p_assumed,print_statements)
        
x_array = np.linspace(0,167,168)
plt.figure()
plt.plot(x_array,S_ht_array_spring,label='spring') 
plt.plot(x_array,S_ht_array_summer,label='summer') 
plt.plot(x_array,S_ht_array_fall,label='fall') 
plt.plot(x_array,S_ht_array_winter,label='winter') 
plt.legend()
'''
        
        
        
        
        
        
        