# -*- coding: utf-8 -*-
"""
Matthew Haefner (mwh85)
SEA Lab
Power Model Pro Sims Post Processing
2/21/25
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
#import statsmodels.api as sm
import statsmodels.formula.api as smf

# Read in dataframe, only keeping the good sims
data_df = pd.read_csv("../../data/RO and Miscellaneous/power_pro_plus_turbocharger_simulations.csv")
data_df = data_df.iloc[:,1:]
data_df = data_df[data_df['warning [-]'] == 0]  

# Make a dataframe for each turbocharger model, store in a dictionary
turbocharger_models = data_df['turbo_model [-]'].unique()

all_models_dict = {
    "AT 7800":data_df[data_df['turbo_model [-]'] == 'AT 7800'],
    "AT 1500":data_df[data_df['turbo_model [-]'] == 'AT 1500'],
    "LPT 2000":data_df[data_df['turbo_model [-]'] == 'LPT 2000'],
    "AT 6000":data_df[data_df['turbo_model [-]'] == 'AT 6000'],
    "AT 3250":data_df[data_df['turbo_model [-]'] == 'AT 3250'],
    "AT 875":data_df[data_df['turbo_model [-]'] == 'AT 875'],
    "AT 4150":data_df[data_df['turbo_model [-]'] == 'AT 4150'],
    "LPT 1000":data_df[data_df['turbo_model [-]'] == 'LPT 1000'],
    "AT 2150":data_df[data_df['turbo_model [-]'] == 'AT 2150'],
    "AT 250":data_df[data_df['turbo_model [-]'] == 'AT 250'],
    "AT 95":data_df[data_df['turbo_model [-]'] == 'AT 95'],
    "AT 1100":data_df[data_df['turbo_model [-]'] == 'AT 1100'],
    "AT 350":data_df[data_df['turbo_model [-]'] == 'AT 350'],
    "AT 550":data_df[data_df['turbo_model [-]'] == 'AT 550'],
    "AT 425":data_df[data_df['turbo_model [-]'] == 'AT 425']
        }

# Determine which turbochargers are compatable with my head height
my_head = 332-34.4 # m
P_no_turbo = 0.0981*my_head*(1026.311/998.049)

truth_array = np.zeros(15)
idx = 0
for turbo in turbocharger_models:
    turbo_dict = all_models_dict[turbo]
    min_P_in = min(turbo_dict['P_hpp_outlet [bar]'])
    max_P_in = max(turbo_dict['P_hpp_outlet [bar]'])
    if min_P_in <= P_no_turbo and max_P_in >= P_no_turbo:
        truth_array[idx] = 1
    idx += 1

# Remove the turbochargers that don't work with the pressure supplied by the reservoir
good_models_dict = copy.deepcopy(all_models_dict)
del good_models_dict['LPT 2000']
del good_models_dict['LPT 1000']
del good_models_dict['AT 95'] 
good_turbocharger_models = turbocharger_models[truth_array==1]
    

#%% Key Variables
# HPP Variables
P_res = good_models_dict['AT 7800']['P_hpp_outlet [bar]']
Q_f = good_models_dict['AT 7800']['Q_f_HPP [m^3/hr]']

# Brine Variables
Q_c = Q_f*(1-(good_models_dict['AT 7800']['RO_rr [%]']/100))
P_c = good_models_dict['AT 7800']['P_c [bar]']
S_c = good_models_dict['AT 7800']['S_c [mg/L]']

# Outlet Variable of Interest
P_f = good_models_dict['AT 7800']['P_f_RO [bar]']
dP_turbo = P_f-P_res

# Misc. Variables
eta_RO = good_models_dict['AT 7800']['RO_rr [%]']
eta_turbo = good_models_dict['AT 7800']['turbo_efficiency [%]']
exhaust_pressure = good_models_dict['AT 7800']['exhaust_pressure [bar]']

#%% Single Variate Plotting
'''
# Plot some stuff
fig1, axes1 = plt.subplots(1, 2)  # Create 2 subplots side by side

# Left subplot: Outlet Pressure vs. Inlet Pressure
sc1 = axes1[0].scatter(P_res, P_f, c=efficiency, cmap='viridis', edgecolors='k')
axes1[0].set_xlabel("Inlet Pressure [bar]")
axes1[0].set_ylabel("Outlet Pressure [bar]")
cbar1 = plt.colorbar(sc1, ax=axes1[0])
cbar1.set_label("Turbocharger Efficiency [%]")

# Right subplot: Turbocharger Efficiency vs. Feed Flowrate
sc2 = axes1[1].scatter(Q_f, efficiency, c=exhaust_pressure, cmap='viridis', edgecolors='k')
axes1[1].set_xlabel("Feed Flowrate [m^3/hr]")
axes1[1].set_ylabel("Turbocharger Efficiency [%]")
cbar2 = plt.colorbar(sc2, ax=axes1[1])
cbar2.set_label("Exhaust Pressure [bar]")

# Adjust layout and show plot
plt.tight_layout()
plt.show()  # Show each plot separately

############################## Start off by plotting turbo efficiency as a function of other variables ####################
# Turbocharger Efficiency vs. HPP Pressure
fig2, ax2 = plt.subplots(1,1)
sc3 = ax2.scatter(P_res, efficiency, edgecolors='k')
ax2.set_xlabel("HPP Pressure [bar]")
ax2.set_ylabel("Turbocharger Efficiency [%]")
plt.tight_layout()
plt.show()  # Show each plot separately

# Turbocharger Efficiency vs. HPP Flowrate
fig3, ax3 = plt.subplots(1,1)
sc4 = ax3.scatter(Q_f, Q_c, edgecolors='k')
ax3.set_xlabel("HPP Flowrate [m^3/hr]")
ax3.set_ylabel("Turbocharger Efficiency [%]")
plt.tight_layout()
plt.show()  # Show each plot separately

# Turbocharger Efficiency vs. Brine Pressure
fig4, ax4 = plt.subplots(1,1)
sc5 = ax4.scatter(P_c, efficiency, edgecolors='k')
ax4.set_xlabel("Brine Pressure [bar]")
ax4.set_ylabel("Turbocharger Efficiency [%]")
plt.tight_layout()
plt.show()  # Show each plot separately

# Turbocharger Efficiency vs. Brine Flowrate
fig5, ax5 = plt.subplots(1,1)
sc6 = ax5.scatter(Q_c, efficiency, edgecolors='k')
ax5.set_xlabel("Brine Flowrate [m^3/hr]")
ax5.set_ylabel("Turbocharger Efficiency [%]")
plt.tight_layout()
plt.show()  # Show each plot separately

# Turbocharger Efficiency vs. Brine Salinity
fig6, ax6 = plt.subplots(1,1)
sc7 = ax6.scatter(S_c, efficiency, edgecolors='k')
ax6.set_xlabel("Brine Salinity [mg/L]")
ax6.set_ylabel("Turbocharger Efficiency [%]")
plt.tight_layout()
plt.show()  # Show each plot separately

# Turbocharger Efficiency vs. RO_rr
fig7, ax7 = plt.subplots(1,1)
sc8 = ax7.scatter(RO_rr, efficiency, edgecolors='k')
ax7.set_xlabel("Recovery Ratio [%]")
ax7.set_ylabel("Turbocharger Efficiency [%]")
plt.tight_layout()
plt.show()  # Show each plot separately

# Turbocharger Efficiency vs. RO Feed Pressure
fig8, ax8 = plt.subplots(1,1)
sc9 = ax8.scatter(P_f, efficiency, edgecolors='k')
ax8.set_xlabel("RO Feed Pressure [bar]")
ax8.set_ylabel("Turbocharger Efficiency [%]")
plt.tight_layout()
plt.show()  # Show each plot separately

# Turbo Pressure Increase vs. HPP Pressure
fig9, ax9 = plt.subplots(1,1)
sc10 = ax9.scatter(P_res, dP_turbo, edgecolors='k')
ax9.set_xlabel("HPP Pressure [bar]")
ax9.set_ylabel("Turbo Pressure Increase [bar]")
plt.tight_layout()
plt.show()  # Show each plot separately

# Turbo Pressure Increase vs. HPP Flowrate
fig10, ax10 = plt.subplots(1,1)
sc11 = ax10.scatter(Q_f, dP_turbo, edgecolors='k')
ax10.set_xlabel("HPP Flowrate [m^3/hr]")
ax10.set_ylabel("Turbo Pressure Increase [bar]")
plt.tight_layout()
plt.show()  # Show each plot separately

# Turbo Pressure Increase vs. Brine Pressure
fig11, ax11 = plt.subplots(1,1)
sc12 = ax11.scatter(P_c, dP_turbo, edgecolors='k')
ax11.set_xlabel("Brine Pressure [bar]")
ax11.set_ylabel("Turbo Pressure Increase [bar]")
plt.tight_layout()
plt.show()  # Show each plot separately

# Turbo Pressure Increase vs. Brine Flowrate
fig12, ax12 = plt.subplots(1,1)
sc13 = ax12.scatter(Q_c, dP_turbo, edgecolors='k')
ax12.set_xlabel("Brine Flowrate [m^3/hr]")
ax12.set_ylabel("Turbo Pressure Increase [bar]")
plt.tight_layout()
plt.show()  # Show each plot separately

# Turbo Pressure Increase vs. Brine Salinity
fig13, ax13 = plt.subplots(1,1)
sc14 = ax13.scatter(S_c, dP_turbo, edgecolors='k')
ax13.set_xlabel("Brine Salinity [mg/L]")
ax13.set_ylabel("Turbo Pressure Increase [bar]")
plt.tight_layout()
plt.show()  # Show each plot separately

# Turbo Pressure Increase vs. turbo efficiency
fig14, ax14 = plt.subplots(1,1)
sc15 = ax14.scatter(efficiency, dP_turbo, edgecolors='k')
ax14.set_xlabel("Turbocharger Efficiency [%]")
ax14.set_ylabel("Turbo Pressure Increase [bar]")
plt.tight_layout()
plt.show()  # Show each plot separately
'''

#%% Multivariate Analysis
# Prepare data
AT7800_df_full = all_models_dict["AT 7800"]
AT7800_df_reduced = AT7800_df_full.drop(columns = ['q_f [m^3/hr]','num_turbos [-]','turbo_model [-]','suction_pressure [bar]','exhaust_pressure [bar]',
                                          'bypass_flowrate [m^3/hr]','valve_position [-]','hpp_efficiency [%','motor_efficiency [%]',
                                          'vfd_efficiency [%]','sec_system [kWh/m^3]','warning [-]'])
AT7800_df = AT7800_df_reduced.rename(columns={'Q_f_HPP [m^3/hr]': 'Q_f', 'P_f_RO [bar]': 'P_f', 'RO_rr [%]': 'eta_RO', 'P_c [bar]': 'P_c', 
                                               'S_c [mg/L]': 'S_c', 'turbo_efficiency [%]': 'eta_turbo', 'P_hpp_outlet [bar]': 'P_res'})

# Use Kendall's tau coefficent to get a sense for which variables give the best relations
corr_matrix = AT7800_df.corr(method='kendall')

# Fit regression model
reg_mdl_P_f_option1 = smf.ols('P_f ~ P_c + P_res', data=AT7800_df).fit()
reg_mdl_P_f_option2 = smf.ols('P_f ~ P_c + P_res + P_c:S_c', data=AT7800_df).fit()

# Get parameters, generate predicted values
parameters_P_f_option1 = reg_mdl_P_f_option1.params
parameters_P_f_option2 = reg_mdl_P_f_option2.params
yhat_P_f_option1 = parameters_P_f_option1.iloc[0] + ((parameters_P_f_option1.iloc[1]*P_c) + (parameters_P_f_option1.iloc[2]*P_res))
yhat_P_f_option2 = parameters_P_f_option2.iloc[0] + ((parameters_P_f_option2.iloc[1]*P_c) + (parameters_P_f_option2.iloc[2]*P_res) + 
                                                     (parameters_P_f_option2.iloc[3]*P_c*S_c)) # This came from having all variables and all pairwise interactions, removing terms based on p-value
    
# Inspect the results, get R^2
print(reg_mdl_P_f_option1.summary())
print(reg_mdl_P_f_option2.summary())
R2_P_f_option1 = round(reg_mdl_P_f_option1.rsquared,5)
R2_P_f_option2 = round(reg_mdl_P_f_option2.rsquared,5)

# Get the residuals
residuals_P_f_option1 = P_f - yhat_P_f_option1
residuals_P_f_option2 = P_f - yhat_P_f_option2

# Plot the fits
fig, ax = plt.subplots(1,1)
x = [30,70]
y = [30,70]
ax.scatter(yhat_P_f_option1,P_f)
ax.plot(x,y,c='r')
ax.set_xlabel("Predicted P_f")
ax.set_ylabel("Actual P_f")
plt.tight_layout()
plt.title('P_c + P_res')
plt.show()  # Show each plot separately

fig, ax = plt.subplots(1,1)
sc = ax.scatter(yhat_P_f_option1, residuals_P_f_option1)
ax.set_xlabel("Predicted P_f")
ax.set_ylabel("Residuals P_f")
plt.tight_layout()
plt.title('P_c + P_res')
plt.show()  # Show each plot separately

fig, ax = plt.subplots(1,1)
x = [30,70]
y = [30,70]
ax.scatter(yhat_P_f_option2,P_f)
ax.plot(x,y,c='r')
ax.set_xlabel("Predicted P_f")
ax.set_ylabel("Actual P_f")
plt.tight_layout()
plt.title('P_c + P_res + P_c:S_c')
plt.show()  # Show each plot separately

fig, ax = plt.subplots(1,1)
sc = ax.scatter(yhat_P_f_option2, residuals_P_f_option2)
ax.set_xlabel("Predicted P_f")
ax.set_ylabel("Residuals P_f")
plt.tight_layout()
plt.title('P_c + P_res + P_c:S_c')
plt.show()  # Show each plot separately



#----------------------------------------------------------------------------
# FINAL PLOTS FOR JOURNAL PAPER
#----------------------------------------------------------------------------

fig, ax = plt.subplots(1,2, figsize=(10, 5))
x = [min([min(yhat_P_f_option2),min(yhat_P_f_option2)]),max([max(yhat_P_f_option2),max(yhat_P_f_option2)])]
y = x
ax[0].scatter(yhat_P_f_option2,P_f)
ax[0].plot(x,y,c='r')
ax[0].set_xlabel(r'$Predicted \; P_{[f,bar],pv\text{x}} \; [bar]$', fontsize=14)
ax[0].set_ylabel(r'$Actual \; P_{[f,bar],pv\text{x}} \; [bar]$', fontsize=14)
ax[0].set_title(r'$(a)$')
ax[1].scatter(yhat_P_f_option2, residuals_P_f_option2)
ax[1].set_xlabel(r'$Predicted \; P_{[f,bar],pv\text{x}} \; [bar]$', fontsize=14)
ax[1].set_ylabel(r'$Residuals \; [bar]$', fontsize=14)
ax[1].set_title(r'$(b)$')
plt.tight_layout()
plt.savefig("Figure_A9.pdf")
plt.show()  


#ax[0].set_aspect('equal')
#ax[1].set_aspect('equal')






