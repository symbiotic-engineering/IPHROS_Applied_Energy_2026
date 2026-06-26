# -*- coding: utf-8 -*-
"""
Visualizing Raw Data Prior to transform
"""

import pandas as pd
from scipy.stats import boxcox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


#%% Load in the data
# With the way I am doing things now, I need to re-generate the training-validation-test sets
df_wave_stage1_reg = pd.read_csv('../../data/RO and Miscellaneous/df_wave_stage1_reg.csv').iloc[:,1:]
df_wave_stage2_reg = pd.read_csv('../../data/RO and Miscellaneous/df_wave_stage2_reg.csv').iloc[:,1:]

df_wave_stage1_reg["stage_id"] = 0
df_wave_stage2_reg["stage_id"] = 1

df_wave_reg = pd.concat([df_wave_stage1_reg, df_wave_stage2_reg], ignore_index=True)

# Add columns to this dataframe that are transformed output variables based on WAVE_autoML_pre_analysis.py
transformed_P_c, lambda_P_c = boxcox(np.array(df_wave_reg['P_c_i [psi]']))
df_wave_reg['P_c_i [psi] trans'] = transformed_P_c

transformed_TDS_c, lambda_TDS_c = boxcox(np.array(df_wave_reg['TDS_c_i [mg/L]']))
df_wave_reg['TDS_c_i [mg/L] trans'] = transformed_TDS_c

transformed_Q_c = np.sqrt(np.array(df_wave_reg['Q_c_i [m^3/hr]']))
df_wave_reg['Q_c_i [m^3/hr] trans'] = transformed_Q_c

transformed_TDS_p, lambda_TDS_p = boxcox(np.array(df_wave_reg['TDS_p_i [mg/L]']))
df_wave_reg['TDS_p_i [mg/L] trans'] = transformed_TDS_p

transformed_Q_p, lambda_Q_p = boxcox(np.array(df_wave_reg['Q_p_i [m^3/hr]']))
df_wave_reg['Q_p_i [m^3/hr] trans'] = transformed_Q_p



# --- columns in your df (pre + post) ---
vars_info = [
    ("P_c",   "P_c_i [psi]",           "P_c_i [psi] trans"),
    ("TDS_c", "TDS_c_i [mg/L]",         "TDS_c_i [mg/L] trans"),
    ("Q_c",   "Q_c_i [m^3/hr]",         "Q_c_i [m^3/hr] trans"),
    ("TDS_p", "TDS_p_i [mg/L]",         "TDS_p_i [mg/L] trans"),
    ("Q_p",   "Q_p_i [m^3/hr]",         "Q_p_i [m^3/hr] trans"),
]

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4.3), constrained_layout=True)

for j, (short_name, pre_col, post_col) in enumerate(vars_info):
    # Pre-transform
    x_pre = df_wave_reg[pre_col].to_numpy()
    x_pre = x_pre[np.isfinite(x_pre)]
    axes[0, j].hist(x_pre, bins=40, rwidth=1)
    if j == 0:
        axes[0, j].set_xlabel(r'$P_{[c,psi],pv\text{x}} \; [psi]$', fontsize=12)
        axes[0, j].xaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
    elif j == 1:
        axes[0, j].set_xlabel(r'$S_{[c,mg/L],pv\text{x}} \; [mg/L]$', fontsize=12)
        axes[0, j].xaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
    elif j == 2:
        axes[0, j].set_xlabel(r'$Q_{c,pv\text{x}} \; [m^3/hr]$', fontsize=12)
        axes[0, j].xaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
    elif j == 3:
        axes[0, j].set_xlabel(r'$S_{[p,mg/L],pv\text{x}} \; [mg/L]$', fontsize=12)
        axes[0, j].xaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
    elif j == 4:
        axes[0, j].set_xlabel(r'$Q_{p,pv\text{x}} \; [m^3/hr]$', fontsize=12)
        axes[0, j].xaxis.set_major_locator(mticker.MaxNLocator(nbins=2))

    # Post-transform
    x_post = df_wave_reg[post_col].to_numpy()
    x_post = x_post[np.isfinite(x_post)]
    if j == 1: # Manually scaling the axis
        x_post = x_post / 1e8
    axes[1, j].hist(x_post, bins=40, rwidth=1)
    if j == 0:
        axes[1, j].set_xlabel(r'$P_{[c,psi],pv\text{x}} \; Trans.$', fontsize=12)
        axes[1, j].xaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
    elif j == 1:
        axes[1, j].set_xlabel(r'$S_{[c,mg/L],pv\text{x}} \; Trans. \; [\times 10^8]$', fontsize=12)
        axes[1, j].xaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
        #axes[1, j].xaxis.get_offset_text().set_fontsize(9)
        #axes[1, j].ticklabel_format(axis='x', style='sci', scilimits=(-2, 3))
        #axes[1, j].xaxis.get_offset_text().set_y(1)
    elif j == 2:
        axes[1, j].set_xlabel(r'$Q_{c,pv\text{x}} \; Trans.$', fontsize=12)
        axes[1, j].xaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
    elif j == 3:
        axes[1, j].set_xlabel(r'$S_{[p,mg/L],pv\text{x}} \; Trans.$', fontsize=12)
        axes[1, j].xaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
    elif j == 4:
        axes[1, j].set_xlabel(r'$Q_{p,pv\text{x}} \; Trans.$', fontsize=12)
        axes[1, j].xaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
    
fig.supylabel(r'$Count$', fontsize=14)

plt.savefig("Figure_A2.pdf", bbox_inches="tight")
plt.show()