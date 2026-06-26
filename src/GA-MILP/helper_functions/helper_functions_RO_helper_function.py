# -*- coding: utf-8 -*-
"""
Matt Haefner (mwh85)
SEA Lab
RO Helper Function
1/27/26
"""

import torch
import torch.nn as nn
import joblib
import numpy as np


def undo_boxcox_trans(y, lam):
    if lam == 0:
        return np.exp(y)
    else:
        return np.power(lam * y + 1, 1 / lam)
    
def undo_sqrt_trans(y):
    return np.square(y)

def inverse_transform_outputs_to_physical(Y_transformed,WAVE_model_bundle):
    """
    Y_transformed: numpy array (N,5) in transformed space (after inverse scaling),
    columns = [P_c_trans, TDS_c_trans, Q_c_trans, TDS_p_trans, Q_p_trans]
    returns: dict of physical numpy arrays
    """
    lambda_P_c = WAVE_model_bundle['boxcox_lambdas']['lambda_P_c']
    lambda_TDS_c = WAVE_model_bundle['boxcox_lambdas']['lambda_TDS_c']
    lambda_TDS_p = WAVE_model_bundle['boxcox_lambdas']['lambda_TDS_p']
    lambda_Q_p = WAVE_model_bundle['boxcox_lambdas']['lambda_Q_p']
    
    P_c   = undo_boxcox_trans(Y_transformed[:, 0], lambda_P_c)
    TDS_c = undo_boxcox_trans(Y_transformed[:, 1], lambda_TDS_c)
    Q_c   = undo_sqrt_trans(Y_transformed[:, 2])
    TDS_p = undo_boxcox_trans(Y_transformed[:, 3], lambda_TDS_p)
    Q_p   = undo_boxcox_trans(Y_transformed[:, 4], lambda_Q_p)
    return {"P_c": P_c, "TDS_c": TDS_c, "Q_c": Q_c, "TDS_p": TDS_p, "Q_p": Q_p}

def apply_linear_calibration_helper(y_pred, a, b):
    return a + b * y_pred

def apply_calibration_to_phys_dict(pred_phys_dict, calibration):
    pred_dict = pred_phys_dict.copy()
    for k, ab in calibration.items():
        pred_dict[k] = apply_linear_calibration_helper(pred_dict[k], ab[0], ab[1]) # 0 is a, 1 is b
    return pred_dict

# =========================
# Define the MLP model
# =========================
class RO_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation='relu'):
        super().__init__()
        layers = []
        prev_dim = input_dim
        act_fn = nn.ReLU if activation=='relu' else nn.Tanh
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_fn())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


def RO_helper_function(X_phys_np,WAVE_model_bundle):
    """
    X_phys_np: (N,4) in physical units including stage_id
    Returns calibrated physical outputs as dict.
    """
    
    # Unpack relevant parts of RO model bundle
    X_scaler = WAVE_model_bundle['scalers']['X_scaler_2000_final']
    Y_scaler = WAVE_model_bundle['scalers']['Y_scaler_2000_final']
    model_stage_dict = WAVE_model_bundle['model_state_dict']
    calibration = WAVE_model_bundle['calibration']
    
    # Specifically model architecture stuff
    my_input_dim = WAVE_model_bundle['model_arch']['input_dim'] # 4
    my_output_dim = WAVE_model_bundle['model_arch']['output_dim'] # 5
    my_hidden_layers = WAVE_model_bundle['model_arch']['hidden_layers'] # [56,74]
    my_activation = WAVE_model_bundle['model_arch']['activation'] # tanh
    
    # Rebuild my model
    final_model = RO_MLP(
        input_dim=my_input_dim,
        output_dim=my_output_dim,
        hidden_layers=my_hidden_layers,
        activation=my_activation
    )
    
    final_model.load_state_dict(model_stage_dict)
    
    # Ju-lee, do the thing
    X_scaled = X_scaler.transform(X_phys_np)
    x_t = torch.tensor(X_scaled, dtype=torch.float32)

    final_model.eval()
    with torch.no_grad():
        y_scaled = final_model(x_t).numpy()

    y_trans = Y_scaler.inverse_transform(y_scaled)
    pred_phys_dict = inverse_transform_outputs_to_physical(y_trans,WAVE_model_bundle)
    Y_phys_dict = apply_calibration_to_phys_dict(pred_phys_dict, calibration)
    
    return Y_phys_dict