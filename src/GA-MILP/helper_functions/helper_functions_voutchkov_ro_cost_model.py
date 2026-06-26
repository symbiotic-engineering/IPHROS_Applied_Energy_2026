# -*- coding: utf-8 -*-
"""
Matt Haefner (mwh85)
SEA Lab
Voutchkov RO Model for Simulation
"""

import numpy as np


# Functions
def opex(feed,perm,p_RO_opex_coeffs,S_mg_L,pipe_length,avg_e_price,plant_sec):
    # NOTE: capacity is no longer an input variable, based on the example in the Voutchkov 2019 textbook
    '''
    This is where you will put what the original units of everything is
    '''
    
    # Start off by converting average electricity price, in 2023 euro, to 2018 usd
    avg_e_price_2018_usd = ((avg_e_price/0.9241)/1.2142)
    
    # Isolate opex coefficients
    a_coeffs_opex = np.array(p_RO_opex_coeffs['a_coeff'])
    b_coeffs_opex = np.array(p_RO_opex_coeffs['b_coeff'])
    c_coeffs_opex = np.array(p_RO_opex_coeffs['c_coeff'])
    
    ############## Dealing with curve-based components first ###############
    # ID components with different inputs
    idxs_feed = [0,1,2,3,4,5,6,7,8,9] # NOTE: Doing membrane pretreatment, so cartridge filtration [5] and gravity filters [7] will be skipped later
    idxs_perm = [10,11,14,15,16,17]
    idxs_cap = [12,13]
    
    # Isolate coefficients based on the ID's
    a_coeffs_feed = a_coeffs_opex[idxs_feed]
    b_coeffs_feed = b_coeffs_opex[idxs_feed]
    c_coeffs_feed = c_coeffs_opex[idxs_feed]
    
    a_coeffs_perm = a_coeffs_opex[idxs_perm]
    b_coeffs_perm = b_coeffs_opex[idxs_perm]
    c_coeffs_perm = c_coeffs_opex[idxs_perm]
    
    a_coeffs_cap = a_coeffs_opex[idxs_cap]
    b_coeffs_cap = b_coeffs_opex[idxs_cap]
    c_coeffs_cap = c_coeffs_opex[idxs_cap]
    
    # Perform the fits for each curve
    feed_terms = (a_coeffs_feed*feed**b_coeffs_feed)+c_coeffs_feed
    perm_terms = (a_coeffs_perm*perm**b_coeffs_perm)+c_coeffs_perm
    cap_terms = (a_coeffs_cap*perm**b_coeffs_cap)+c_coeffs_cap
    
    # Write out each individual term to account for necessary averaging between 2 curves (also convert units to get everything to $/year)
    hdpe_pipe_opex = feed_terms[0]*pipe_length 
    pump_opex = feed_terms[1]*1000
    band_screen_opex = feed_terms[2]*1000
    wedgewire_screen_opex = feed_terms[3]*1000
    microscreen_opex = feed_terms[4]*1000
    daf_opex = feed_terms[6]*1000
    membrane_pretreatment_opex = ((feed_terms[8]+feed_terms[9])/2)*1000
    y0 = perm_terms[1]
    y1 = perm_terms[0]
    x0 = 35000
    x1 = 46000
    swro_opex = (y0+((S_mg_L-x0)*((y1-y0)/(x1-x0))))*1000
    calcite_opex = cap_terms[0]*1000
    sodium_opex = cap_terms[1]*1000
    other_direct_opex = ((perm_terms[2]+perm_terms[3])/2)*1000
    indirect_opex = ((perm_terms[4]+perm_terms[5])/2)*1000
    
    # Then for the non-curve-based ones
    energy_costs_opex = avg_e_price_2018_usd*plant_sec*perm*365 # 2018 usd 
    
    # Add everything together, convert to 2023 euro
    opex_2018_usd = (hdpe_pipe_opex + pump_opex + band_screen_opex + wedgewire_screen_opex + microscreen_opex + daf_opex + 
                     membrane_pretreatment_opex + swro_opex + calcite_opex + sodium_opex + other_direct_opex + energy_costs_opex + indirect_opex) 
    
    opex_2023_euro = opex_2018_usd*1.2142*0.9241 # 1.2142 inflation rate according to US Bureau of Labor Statistics for average of Jan. 2018 -> Jan. 2023, Feb. 2018 -> Feb. 2023, etc. (https://www.bls.gov/data/inflation_calculator.htm), 0.9241 exchange rate is from (https://www.exchangerates.org.uk/USD-EUR-spot-exchange-rates-history-2023.html)
    
    return opex_2023_euro


def capex(feed,perm,p_RO_capex_coeffs,S_mg_L,pipe_length,avg_e_price,plant_sec,p_RO_capex_project,p_RO_capex_total):
    '''
    This is where you will put what the original units of everything is
    '''
    
    # Start off by converting average electricity price, in 2023 euro, to 2018 usd
    avg_e_price_2018_usd = ((avg_e_price/0.9241)/1.2142)
    
    # Isolate capex coefficients
    a_coeffs_capex = np.array(p_RO_capex_coeffs['a_coeff'])
    b_coeffs_capex = np.array(p_RO_capex_coeffs['b_coeff'])
    c_coeffs_capex = np.array(p_RO_capex_coeffs['c_coeff'])
    
    # Unpack project cost parameters
    site_prep = p_RO_capex_project['site_prep']
    concrete_disposal = p_RO_capex_project['concrete_disposal']
    waste_solids_handling = p_RO_capex_project['waste_solids_handling']
    instrumentation = p_RO_capex_project['instrumentation']
    aux = p_RO_capex_project['aux']
    building = p_RO_capex_project['building']
    startup_a = p_RO_capex_project['startup_a']
    startup_b = p_RO_capex_project['startup_b']
    
    # Unpack total cost parameters
    preliminary = p_RO_capex_total['preliminary']
    pilot_testing_a = p_RO_capex_total['pilot_testing_a']
    pilot_testing_b = p_RO_capex_total['pilot_testing_b']
    pilot_testing_c = p_RO_capex_total['pilot_testing_c']
    detailed_design = p_RO_capex_total['detailed_design']
    management = p_RO_capex_total['management']
    admin = p_RO_capex_total['admin']
    environmental_permitting = p_RO_capex_total['environmental_permitting']
    legal_services = p_RO_capex_total['legal_services']
    interest = p_RO_capex_total['interest']
    debt_service_relief = p_RO_capex_total['debt_service_relief']
    other_financing = p_RO_capex_total['other_financing']
    contingency = p_RO_capex_total['contingency']
    
    
    ############## Dealing with curve-based components first ###############
    # ID components with different inputs
    idxs_feed = [0,1,2,3,4,5,6,7,8,9] # NOTE: Doing membrane pretreatment, so gravity filters [6] and cartridge filtration [9] will be skipped later
    idxs_perm = [10,11]
    idxs_cap = [12,13]
    
    # Isolate coefficients based on the ID's
    a_coeffs_feed = a_coeffs_capex[idxs_feed]
    b_coeffs_feed = b_coeffs_capex[idxs_feed]
    c_coeffs_feed = c_coeffs_capex[idxs_feed]
    
    a_coeffs_perm = a_coeffs_capex[idxs_perm]
    b_coeffs_perm = b_coeffs_capex[idxs_perm]
    c_coeffs_perm = c_coeffs_capex[idxs_perm]
    
    a_coeffs_cap = a_coeffs_capex[idxs_cap]
    b_coeffs_cap = b_coeffs_capex[idxs_cap]
    c_coeffs_cap = c_coeffs_capex[idxs_cap]
    
    # Perform the fits for each curve
    feed_terms = (a_coeffs_feed*feed**b_coeffs_feed)+c_coeffs_feed
    perm_terms = (a_coeffs_perm*perm**b_coeffs_perm)+c_coeffs_perm
    cap_terms = (a_coeffs_cap*perm**b_coeffs_cap)+c_coeffs_cap
    
    # Write out each individual term to account for necessary averaging between 2 curves (also convert units to get everything to $/year)
    hdpe_pipe_capex = feed_terms[0]*pipe_length*1000
    pump_capex = feed_terms[1]*1000
    band_screen_capex = feed_terms[2]*1000
    wedgewire_screen_capex = feed_terms[3]*1000
    microscreen_capex = feed_terms[4]*1000
    daf_capex = feed_terms[5]*1000
    membrane_pretreatment_capex = ((feed_terms[7]+feed_terms[8])/2)*1000
    y0 = perm_terms[1]
    y1 = perm_terms[0]
    x0 = 35000
    x1 = 46000
    swro_capex = (y0+((S_mg_L-x0)*((y1-y0)/(x1-x0))))*1000
    calcite_capex = cap_terms[0]*1000
    sodium_capex = cap_terms[1]*1000
    
    # Add all curves together
    capex_project_curves_2018_usd = (hdpe_pipe_capex + pump_capex + band_screen_capex + wedgewire_screen_capex + microscreen_capex + 
                                     daf_capex + membrane_pretreatment_capex + swro_capex + calcite_capex + sodium_capex) 
    
    # Then for the non-curve-based ones (project)
    site_prep_capex = site_prep*perm
    concrete_disposal_capex = concrete_disposal*perm
    waste_solids_handling_capex = waste_solids_handling*perm
    instrumentation_capex = instrumentation*perm
    aux_capex = aux*perm
    building_capex = building*perm
    startup_capex = (startup_a+(plant_sec*startup_b*avg_e_price_2018_usd))*perm
    
    # Add up all the non-curve project capex terms 
    capex_project_non_curve_2018_usd = (site_prep_capex + concrete_disposal_capex + waste_solids_handling_capex + 
                                        instrumentation_capex + aux_capex + building_capex + startup_capex)
    
    # Get a net project cost
    capex_project = capex_project_curves_2018_usd + capex_project_non_curve_2018_usd
    
    # Now for the non-cumulative non-curve total capex terms (basically any total capex terms dependent on flowrate)
    preliminary_capex = preliminary*perm
    pilot_testing_capex = (pilot_testing_a*perm)+(pilot_testing_b*pilot_testing_c)
    detailed_design_capex = detailed_design*perm
    management_capex = management*perm
    admin_capex = admin*perm
    environmental_permitting_capex = environmental_permitting*perm
    legal_services_capex = legal_services*perm
    
    # Add together total capex terms dependent on flowrate
    capex_total_f_flowrate = (preliminary_capex + pilot_testing_capex + detailed_design_capex + management_capex + 
                              admin_capex + environmental_permitting_capex + legal_services_capex)
    
    # A "sub-total" cost
    capex_sub_total = capex_project + capex_total_f_flowrate
    
    # Calculate the last few contributions into the total capex
    interest_capex = capex_sub_total*interest
    debt_service_relief_capex = capex_project*debt_service_relief
    other_financing_capex = capex_sub_total*other_financing
    contingency_capex = capex_project*contingency
    
    # Get a final capex total
    capex_2018_usd = capex_sub_total + interest_capex + debt_service_relief_capex + other_financing_capex + contingency_capex
    
    # Convert to 2023 euros
    capex_2023_euro = capex_2018_usd*1.2142*0.9241 # 1.2142 inflation rate according to US Bureau of Labor Statistics for average of Jan. 2018 -> Jan. 2023, Feb. 2018 -> Feb. 2023, etc. (https://www.bls.gov/data/inflation_calculator.htm), 0.9241 exchange rate is from (https://www.exchangerates.org.uk/USD-EUR-spot-exchange-rates-history-2023.html)
    
    return capex_2023_euro


#----------------------------------------
# Taking this for a test drive
#----------------------------------------
"""
# RO Capex and Opex Curve Parameters
p_RO_capex_coeffs = {'a_coeff':[0.001798687,0.024180211,0.014618901,0.148243435,0.179827215,0.352118426,0.702498317,
                                1.313602049,0.941765967,0.244790278,5.660228826,5.493331828,3.523988027,0.516522185],
                     'b_coeff':[0.783381744,0.920076499,0.974873757,0.75745577,0.766685347,0.804946451,0.807996346,
                                0.794461209,0.774983119,0.755355474,0.781771337,0.771753938,0.595483032,0.597343481],
                     'c_coeff':[-0.007682172,10.12906612,-88.36877824,-133.0162594,-170.7173626,-153.8027207,-613.7512251,
                                -406.9382034,-199.0983905,-39.10515036,-914.6272773,-439.9284306,-31.26508266,-1.686961466]
                     }

p_RO_opex_coeffs = {'a_coeff':[0.014972322,0.003050831,5.40E-04,0.007025278,0.008046379,0.026727907,0.01263469,
                               0.063893085,0.063725656,0.062280818,0.233952613,0.223691614,0.389320073,0.112536193,
                               0.114285156,0.057690226,0.422040509,0.215144932],
                    'b_coeff':[0.772995272,0.866495164,0.970970249,0.747809746,0.764072774,0.729500595,0.810396063,
                               0.782210038,0.793841094,0.77147198,0.784125094,0.771879237,0.589393163,0.590833297,
                               0.769393494,0.740279467,0.740841516,0.719216864],
                    'c_coeff':[-1.452645677,0.061354676,-3.439851698,-6.350401658,-7.503941715,-9.148871489,-5.452733029,
                               -43.0689899,-21.10184108,-13.24778772,-29.6440509,-29.00143345,-4.641796039,-1.191634105,
                               -52.53182949,-22.39551206,-33.67348319,-29.40424002]
                    }

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

# Assumed Parameters
p_assumed = {'n_pump_turbines':7, # -
             'plant_sec':3.4,       # kWh/m^3
             'pipe_length':1000,     # m
             'harmonics':[1,7], # harmonics for the S_ht fourier series [-]
             'eta_francis_max_flow':0.87367, # This comes from the efficiency curve
             'eta_pelton_max_flow':0.850457 # This also comes from the efficiency curve
             }

# Financial Parameters
p_financial = {'r':0.05,                # -
               'n':30,                  # years
               'eta_capex_RO':0.79,     # this is from my applied energy paper
               'eta_opex_RO':0.56,      # this is from my applied energy paper
               'avg_e_price':0.1468     # euros/kWh --> confirmed 2/17/26
               }

# RO capex parameters
p_RO_capex_project = {'site_prep':20,               # 2018 usd/m^3 permeate/day
                      'concrete_disposal':30,       # 2018 usd/m^3 permeate/day
                      'waste_solids_handling':30,   # 2018 usd/m^3 permeate/day
                      'instrumentation':75,         # 2018 usd/m^3 permeate/day
                      'aux':25,                     # 2018 usd/m^3 permeate/day
                      'building':60,                # 2018 usd/m^3 permeate/day
                      'startup_a':20,               # 2018 usd/m^3 permeate/day
                      'startup_b':365/6             # 2 months of days
                      }

p_RO_capex_total = {'preliminary':20,                   # 2018 usd/m^3 permeate/day
                    'pilot_testing_a':10,               # 2018 usd/m^3 permeate/day
                    'pilot_testing_b':15000,            # 2018 usd/month
                    'pilot_testing_c':9,                # months
                    'detailed_design':90,               # 2018 usd/m^3 permeate/day
                    'management':45,                    # 2018 usd/m^3 permeate/day
                    'admin':40,                         # 2018 usd/m^3 permeate/day
                    'environmental_permitting':47.5,    # 2018 usd/m^3 permeate/day
                    'legal_services':25,                # 2018 usd/m^3 permeate/day
                    'interest':0.03,                    # [-]
                    'debt_service_relief': 0.0525,      # [-]
                    'other_financing': 0.0225,          # [-]
                    'contingency': 0.075                # [-]
                    }

# Feed and permeate flows
feed = 200000
perm = 80000

# State parameters
S_sw_g_kg = p_state['S_sw_g_kg']
rho_sw = p_state['rho_sw']
S_mg_L = S_sw_g_kg*rho_sw

# Assumed parameters
pipe_length = p_assumed['pipe_length']
plant_sec = p_assumed['plant_sec']

# Financial parameters
r = p_financial['r']
n = p_financial['n']
eta_capex_RO = p_financial['eta_capex_RO']
eta_opex_RO = p_financial['eta_opex_RO']
avg_e_price = p_financial['avg_e_price']



capex_2023_euro = capex(feed,perm,p_RO_capex_coeffs,S_mg_L,pipe_length,avg_e_price,plant_sec,p_RO_capex_project,p_RO_capex_total)
opex_2023_euro = opex(feed,perm,p_RO_opex_coeffs,S_mg_L,pipe_length,avg_e_price,plant_sec)
"""
