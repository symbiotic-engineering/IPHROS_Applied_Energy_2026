# -*- coding: utf-8 -*-
"""
Matt Haefner (mwh85)
SEA Lab
Epsilon Prediction Function
1/20/26
"""

import numpy as np

def epsilon_predictor(B_PSH, N_pv1, eps_logit_bundle,print_statements):
    params = eps_logit_bundle["params"]

    # Statsmodels formula names these typically:
    # 'Intercept', 'B_PSH', 'N_pv1', 'B_PSH:N_pv1'
    z = (
        params.get("Intercept", 0.0)
        + params["B_PSH"] * B_PSH
        + params["N_pv1"] * N_pv1
        + params["B_PSH:N_pv1"] * (B_PSH * N_pv1)
    )

    p_eps_on = 1.0 / (1.0 + np.exp(-z))
    
    if p_eps_on >= eps_logit_bundle["p_star"]:
        eps_MILP = eps_logit_bundle["eps_on"]
    else:
        eps_MILP = eps_logit_bundle["eps_off"]
        
    if print_statements == 1:
        print('Start of Epsilon Predictor Module Output')
        print('eps_MILP: ' + str(eps_MILP))
        print('End of Epsilon Predictor Module Output')

    return eps_MILP










'''
import pandas as pd
import numpy as np

def epsilon_predictor(B_PSH,N_pv1,epsilon_logit_model):
    
    # In order to recreate the dataframe, I need B_PSH,N_pv1,B_PSH:N_PV1
    dataframe_for_epsilon = pd.DataFrame({
        "B_PSH": [B_PSH],
        "N_pv1": [N_pv1]
    })
    
    
    p_eps_on_series = epsilon_logit_model.predict(dataframe_for_epsilon)
    p_eps_on = np.array(p_eps_on_series)[0]
    
    if p_eps_on >= 0.12:
        epsilon = 3.17e-5
    elif p_eps_on < 0.12:
        epsilon = 0
    else:
        print('gotta fix this')
        
    return epsilon
'''
