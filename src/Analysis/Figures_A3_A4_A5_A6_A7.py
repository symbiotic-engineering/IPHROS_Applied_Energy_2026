# -*- coding: utf-8 -*-
"""
Reloading optuna pytorch model
"""

import torch
import torch.nn as nn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


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


# Generate training and test sets
X_df = df_wave_reg[['P_f_i [psi]','TDS_f_i [mg/L]','Q_f_i [m^3/hr]', 'stage_id']]
Y_df = df_wave_reg[['P_c_i [psi] trans','TDS_c_i [mg/L] trans','Q_c_i [m^3/hr] trans','TDS_p_i [mg/L] trans','Q_p_i [m^3/hr] trans']]

X_train_full_df, X_test_df, Y_train_full_df, Y_test_df = train_test_split(
    X_df, Y_df, test_size=0.2, random_state=42
)

X_train_df, X_val_df, Y_train_df, Y_val_df = train_test_split(
    X_train_full_df, Y_train_full_df, test_size=0.25, random_state=42
)


# Convert everything to a numpy array
X_train_actual_units = X_train_df.to_numpy() 
Y_train_transformed = Y_train_df.to_numpy()

X_val_actual_units = X_val_df.to_numpy()
Y_val_transformed = Y_val_df.to_numpy()

X_train_full_actual_units = X_train_full_df.to_numpy()
Y_train_full_transformed = Y_train_full_df.to_numpy()

X_test_actual_units = X_test_df.to_numpy()
Y_test_transformed = Y_test_df.to_numpy()



#%% Functions needed for neural net stuff
# Couple helper functions for undoing transforms
def undo_boxcox_trans(y, lam):
    if lam == 0:
        return np.exp(y)
    else:
        return np.power(lam * y + 1, 1 / lam)

def undo_sqrt_trans(y):
    return np.square(y)

def inv_boxcox_torch(y, lam, eps=1e-12):
    # y is torch tensor
    if lam == 0:
        return torch.exp(y)
    inside = lam * y + 1.0
    inside = torch.clamp(inside, min=eps)
    return inside ** (1.0 / lam)

def unscale_y_torch(y_scaled, Y_scaler, device):
    # y_scaled: torch tensor (N,5)
    mean = torch.tensor(Y_scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(Y_scaler.scale_, dtype=torch.float32, device=device)
    return y_scaled * scale + mean

def unscale_x_torch(x_scaled, X_scaler, device):
    # x_scaled: torch tensor (N,4)
    mean = torch.tensor(X_scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(X_scaler.scale_, dtype=torch.float32, device=device)
    return x_scaled * scale + mean

def inverse_transform_outputs_to_physical(Y_transformed):
    """
    Y_transformed: numpy array (N,5) in transformed space (after inverse scaling),
    columns = [P_c_trans, TDS_c_trans, Q_c_trans, TDS_p_trans, Q_p_trans]
    returns: dict of physical numpy arrays
    """
    P_c   = undo_boxcox_trans(Y_transformed[:, 0], lambda_P_c)
    TDS_c = undo_boxcox_trans(Y_transformed[:, 1], lambda_TDS_c)
    Q_c   = undo_sqrt_trans(Y_transformed[:, 2])
    TDS_p = undo_boxcox_trans(Y_transformed[:, 3], lambda_TDS_p)
    Q_p   = undo_boxcox_trans(Y_transformed[:, 4], lambda_Q_p)
    return {"P_c": P_c, "TDS_c": TDS_c, "Q_c": Q_c, "TDS_p": TDS_p, "Q_p": Q_p}


def fit_linear_calibration(y_pred, y_true):
    """
    Fit y_true ≈ a + b*y_pred, returns (a,b)
    """
    b, a = np.polyfit(y_pred, y_true, deg=1)
    return float(a), float(b)


def apply_linear_calibration(y_pred, a, b):
    return a + b * y_pred

def apply_calibration_to_phys_dict(pred_phys_dict, calibration):
    pred = pred_phys_dict.copy()
    for k, ab in calibration.items():
        pred[k] = apply_linear_calibration(pred[k], ab["a"], ab["b"])
    return pred




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


# =========================
# Load in model
# =========================
X_scaler_2000_final = joblib.load("../../data/RO and Miscellaneous/X_scaler_2000_final_v2_no_lambdas_4_inputs.joblib")
Y_scaler_2000_final = joblib.load("../../data/RO and Miscellaneous/Y_scaler_2000_final_v2_no_lambdas_4_inputs.joblib")

hidden_layers = [56, 74]   # <-- example: replace with your actual values
activation = "tanh"        # <-- maybe need to change this too

final_model = RO_MLP(
    input_dim=4,
    output_dim=5,
    hidden_layers=hidden_layers,
    activation=activation
)

final_model.load_state_dict(torch.load("../../data/RO and Miscellaneous/RO_model_final_v2_no_lambdas_4_inputs.pt"))


# =========================
# Post-process data
# =========================
X_train_full_actual_units_scaled = X_scaler_2000_final.transform(X_train_full_actual_units)
Y_train_full_transformed_scaled = Y_scaler_2000_final.transform(Y_train_full_transformed)

X_val_actual_units_scaled = X_scaler_2000_final.transform(X_val_actual_units)
Y_val_transformed_scaled = Y_scaler_2000_final.transform(Y_val_transformed)

X_test_actual_units_scaled = X_scaler_2000_final.transform(X_test_actual_units)
Y_test_transformed_scaled = Y_scaler_2000_final.transform(Y_test_transformed)

X_train_full_actual_units_scaled_t = torch.tensor(X_train_full_actual_units_scaled, dtype=torch.float32)
Y_train_full_transformed_scaled_t = torch.tensor(Y_train_full_transformed_scaled, dtype=torch.float32)

X_val_actual_units_scaled_t = torch.tensor(X_val_actual_units_scaled, dtype=torch.float32)
Y_val_transformed_scaled_t = torch.tensor(Y_val_transformed_scaled, dtype=torch.float32)

X_test_actual_units_scaled_t = torch.tensor(X_test_actual_units_scaled, dtype=torch.float32)
Y_test_transformed_scaled_t = torch.tensor(Y_test_transformed_scaled, dtype=torch.float32)


# =========================
# Evaluate on TEST set
# =========================
final_model.eval()
with torch.no_grad():
    Y_test_pred_transformed_scaled = final_model(X_test_actual_units_scaled_t)

Y_test_pred_transformed = Y_scaler_2000_final.inverse_transform(Y_test_pred_transformed_scaled)


# =========================
# Evaluate on FULL TRAIN set
# =========================
final_model.eval()
with torch.no_grad():
    Y_train_full_pred_transformed_scaled = final_model(X_train_full_actual_units_scaled_t)

Y_train_full_pred_transformed = Y_scaler_2000_final.inverse_transform(Y_train_full_pred_transformed_scaled)


# =========================
# Post-hoc calibration (fit on VAL set)
# =========================
final_model.eval()
with torch.no_grad():
    Y_val_pred_transformed_scaled = final_model(X_val_actual_units_scaled_t)

# Convert VAL predictions to transformed space (numpy)
Y_val_pred_transformed = Y_scaler_2000_final.inverse_transform(Y_val_pred_transformed_scaled.numpy())

# True VAL outputs are already in transformed space: Y_val_transformed
# Convert both pred and true VAL to PHYSICAL units for the variables we calibrate
P_c_val_true   = undo_boxcox_trans(Y_val_transformed[:, 0], lambda_P_c)
TDS_c_val_true = undo_boxcox_trans(Y_val_transformed[:, 1], lambda_TDS_c)
Q_p_val_true   = undo_boxcox_trans(Y_val_transformed[:, 4], lambda_Q_p)

P_c_val_pred   = undo_boxcox_trans(Y_val_pred_transformed[:, 0], lambda_P_c)
TDS_c_val_pred = undo_boxcox_trans(Y_val_pred_transformed[:, 1], lambda_TDS_c)
Q_p_val_pred   = undo_boxcox_trans(Y_val_pred_transformed[:, 4], lambda_Q_p)

calibration = {}
calibration["P_c"]   = fit_linear_calibration(P_c_val_pred,   P_c_val_true)
calibration["TDS_c"] = fit_linear_calibration(TDS_c_val_pred, TDS_c_val_true)
calibration["Q_p"]   = fit_linear_calibration(Q_p_val_pred,   Q_p_val_true)

print("\nPOST-HOC CALIBRATION (fit on VAL)")
for k, (a, b) in calibration.items():
    print(f"{k:6s}: a={a:.6g}, b={b:.6g}")


# =========================
# Metrics on TEST set and FULL TRAIN set
# =========================
P_c_train_full = undo_boxcox_trans(Y_train_full_transformed[:,0],lambda_P_c)
TDS_c_train_full = undo_boxcox_trans(Y_train_full_transformed[:,1],lambda_TDS_c)
Q_c_train_full = undo_sqrt_trans(Y_train_full_transformed[:,2])
TDS_p_train_full = undo_boxcox_trans(Y_train_full_transformed[:,3],lambda_TDS_p)
Q_p_train_full = undo_boxcox_trans(Y_train_full_transformed[:,4],lambda_Q_p)

y_train_full = np.column_stack((P_c_train_full,TDS_c_train_full,Q_c_train_full,TDS_p_train_full,Q_p_train_full))

P_c_train_full_pred = undo_boxcox_trans(Y_train_full_pred_transformed[:,0],lambda_P_c)
TDS_c_train_full_pred = undo_boxcox_trans(Y_train_full_pred_transformed[:,1],lambda_TDS_c)
Q_c_train_full_pred = undo_sqrt_trans(Y_train_full_pred_transformed[:,2])
TDS_p_train_full_pred = undo_boxcox_trans(Y_train_full_pred_transformed[:,3],lambda_TDS_p)
Q_p_train_full_pred = undo_boxcox_trans(Y_train_full_pred_transformed[:,4],lambda_Q_p)

# Apply calibration to FULL TRAIN predictions (physical units)
a, b = calibration["P_c"]
P_c_train_full_pred = apply_linear_calibration(P_c_train_full_pred, a, b)

a, b = calibration["TDS_c"]
TDS_c_train_full_pred = apply_linear_calibration(TDS_c_train_full_pred, a, b)

a, b = calibration["Q_p"]
Q_p_train_full_pred = apply_linear_calibration(Q_p_train_full_pred, a, b)

y_train_full_pred = np.column_stack((P_c_train_full_pred,TDS_c_train_full_pred,Q_c_train_full_pred,TDS_p_train_full_pred,Q_p_train_full_pred))


P_c_test = undo_boxcox_trans(Y_test_transformed[:,0],lambda_P_c)
TDS_c_test = undo_boxcox_trans(Y_test_transformed[:,1],lambda_TDS_c)
Q_c_test = undo_sqrt_trans(Y_test_transformed[:,2])
TDS_p_test = undo_boxcox_trans(Y_test_transformed[:,3],lambda_TDS_p)
Q_p_test = undo_boxcox_trans(Y_test_transformed[:,4],lambda_Q_p)

y_test = np.column_stack((P_c_test,TDS_c_test,Q_c_test,TDS_p_test,Q_p_test))

P_c_test_pred = undo_boxcox_trans(Y_test_pred_transformed[:,0],lambda_P_c)
TDS_c_test_pred = undo_boxcox_trans(Y_test_pred_transformed[:,1],lambda_TDS_c)
Q_c_test_pred = undo_sqrt_trans(Y_test_pred_transformed[:,2])
TDS_p_test_pred = undo_boxcox_trans(Y_test_pred_transformed[:,3],lambda_TDS_p)
Q_p_test_pred = undo_boxcox_trans(Y_test_pred_transformed[:,4],lambda_Q_p)

# Apply calibration to TEST predictions (physical units)
a, b = calibration["P_c"]
P_c_test_pred = apply_linear_calibration(P_c_test_pred, a, b)

a, b = calibration["TDS_c"]
TDS_c_test_pred = apply_linear_calibration(TDS_c_test_pred, a, b)

a, b = calibration["Q_p"]
Q_p_test_pred = apply_linear_calibration(Q_p_test_pred, a, b)

y_test_pred = np.column_stack((P_c_test_pred,TDS_c_test_pred,Q_c_test_pred,TDS_p_test_pred,Q_p_test_pred))


output_names = [
    "P_c", "TDS_c", "Q_c",
    "TDS_p", "Q_p"
]

r2_train_full = r2_score(y_train_full, y_train_full_pred, multioutput="raw_values")
rmse_train_full = np.sqrt(mean_squared_error(y_train_full, y_train_full_pred, multioutput="raw_values"))

r2_test = r2_score(y_test, y_test_pred, multioutput="raw_values")
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred, multioutput="raw_values"))

P_c_nom = 554.08
TDS_c_nom = 48921
Q_c_nom = 8.5415
TDS_p_nom = 2574
Q_p_nom = 1.0565

nominal_values = np.array([P_c_nom,TDS_c_nom,Q_c_nom,TDS_p_nom,Q_p_nom])
percent_values_train_full = (rmse_train_full/nominal_values)*100
percent_values_test = (rmse_test/nominal_values)*100


print("\nFULL TRAIN SET METRICS")
for i, name in enumerate(output_names):
    print(f"{name:8s} | R2 = {r2_train_full[i]:6.5f} | RMSE = {rmse_train_full[i]:10.4e} | % Diff. = {percent_values_train_full[i]:6.3f}%")
    
    
print("\nTEST SET METRICS")
for i, name in enumerate(output_names):
    print(f"{name:8s} | R2 = {r2_test[i]:6.5f} | RMSE = {rmse_test[i]:10.4e} | % Diff. = {percent_values_test[i]:6.3f}%")


#%% Post-processing (part 2)
# =========================
# Physics diagnostics --> this will be used to determine if I actually need to turn these on or not
# =========================
P_conc = y_test_pred[:, 0]
S_conc = y_test_pred[:, 1]
Q_conc = y_test_pred[:, 2]
S_perm = y_test_pred[:, 3]
Q_perm = y_test_pred[:, 4]

P_feed = X_test_actual_units[:, 0]
S_feed = X_test_actual_units[:, 1]
Q_feed = X_test_actual_units[:, 2]

mass_error = Q_feed - Q_perm - Q_conc
mass_error_norm = mass_error/(Q_feed + 1e-6)
salt_error = Q_feed*S_feed - Q_perm*S_perm - Q_conc*S_conc
salt_error_norm = salt_error/((Q_feed*S_feed) + 1e-6)

flow_balance_95th_percentile = np.percentile(mass_error, 95)
salt_balance_95th_percentile = np.percentile(salt_error, 95)

flow_balance_95th_percentile_norm = np.percentile(mass_error_norm, 95)
salt_balance_95th_percentile_norm = np.percentile(salt_error_norm, 95)

pressure_violation1 = np.maximum(0, P_conc - P_feed)
pressure_violation2 = np.maximum(0, 0 - P_conc)
salinity_violation1 = np.maximum(0, S_feed - S_conc)
salinity_violation2 = np.maximum(0, S_perm - S_feed)
salinity_violation3 = np.maximum(0, 0 - S_perm)
flow_violation1 = np.maximum(0, 0 - Q_perm)
flow_violation2 = np.maximum(0, Q_perm - Q_conc)
flow_violation3 = np.maximum(0, Q_conc - Q_feed)

'''
print("\nPHYSICS DIAGNOSTICS (TEST SET)")
print(f"Mass balance RMSE      : {np.sqrt(np.mean(mass_error**2)):.4e}")
print(f"Salt balance RMSE      : {np.sqrt(np.mean(salt_error**2)):.4e}")
print(f"Pressure violations   : {(pressure_violation > 0).mean()*100:.2f}% of samples")
'''

print("\nBETTER PHYSICS DIAGNOSTICS (TEST SET)")
print(f"Mass balance residual, 95th percentile: {flow_balance_95th_percentile:.4e}")
print(f"Salt balance residual, 95th percentile: {salt_balance_95th_percentile:.4e}")
print(f"Normalized mass balance residual, 95th percentile: {flow_balance_95th_percentile_norm:.4e}")
print(f"Normalized sal balance residual, 95th percentile: {salt_balance_95th_percentile_norm:.4e}")
print(f"Mass balance residual, Max absolute value: {max(abs(mass_error)):.4e}")
print(f"Salt balance residual, Max absolute value: {max(abs(salt_error)):.4e}")
print(f"Normalized mass balance residual, Max absolute value: {max(abs(mass_error_norm)):.4e}")
print(f"Normalized salt balance residual, Max absolute value: {max(abs(salt_error_norm)):.4e}")
print(f"Max pressure violations, P_c: {(pressure_violation1 > 0).mean()*100:.2f}% of samples")
print(f"Min pressure violations, P_c: {(pressure_violation2 > 0).mean()*100:.2f}% of samples")
print(f"Min salinity violations, S_c: {(salinity_violation1 > 0).mean()*100:.2f}% of samples")
print(f"Max salinity violations, S_p: {(salinity_violation2 > 0).mean()*100:.2f}% of samples")
print(f"Min salinity violations, S_p: {(salinity_violation3 > 0).mean()*100:.2f}% of samples")
print(f"Min flow violations, Q_p: {(flow_violation1 > 0).mean()*100:.2f}% of samples")
print(f"Max/min flow violations, Q_p & Q_c: {(flow_violation2 > 0).mean()*100:.2f}% of samples")
print(f"Max flow violations, Q_c: {(flow_violation3 > 0).mean()*100:.2f}% of samples")

RR = Q_perm/Q_feed


# =========================
# Plotting
# =========================
import matplotlib.pyplot as plt

residuals_dict = {}
residuals_dict['0'] = P_c_test - P_c_test_pred
residuals_dict['1'] = TDS_c_test - TDS_c_test_pred
residuals_dict['2'] = Q_c_test - Q_c_test_pred
residuals_dict['3'] = TDS_p_test - TDS_p_test_pred
residuals_dict['4'] = Q_p_test - Q_p_test_pred


output_strings = ['P_c [psi]', 'TDS_c [mg/L]', 'Q_c [m^3/hr]', 'TDS_p [mg/L]', 'Q_p [m^3/hr]']
for i in range(5):
    # Plots of predicted vs. actual and residuals vs. predicted
    actual = y_test[:,i]
    predicted = y_test_pred[:,i]
    residuals = residuals_dict[str(i)]
    min_val = min([min(actual), min(predicted)])
    max_val = max([max(actual), max(predicted)])
    #'''
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].scatter(actual,predicted)
    ax[0].plot([min_val, max_val], [min_val, max_val], color='red')
    ax[1].scatter(predicted,residuals)
    save_figs = 0
    if i == 0:
        ax[0].set_xlabel(r'$Actual \; P_{[c,psi],pv\text{x}} \; [psi]$', fontsize=14)
        ax[1].set_xlabel(r'$Predicted \; P_{[c,psi],pv\text{x}} \; [psi]$', fontsize=14)
        ax[0].set_ylabel(r'$Predicted \; P_{[c,psi],pv\text{x}} \; [psi]$', fontsize=14)
        ax[1].set_ylabel(r'$Residual \; [psi]$', fontsize=14)
        ax[0].set_title(r'$(a)$')
        ax[1].set_title(r'$(b)$')
        plt.tight_layout()
        if save_figs == 1:
            plt.savefig("Figure_A3.pdf")
    elif i == 1:
        ax[0].set_xlabel(r'$Actual \; S_{[c,mg/L],pv\text{x}} \; [mg/L]$', fontsize=14)
        ax[1].set_xlabel(r'$Predicted \; S_{[c,mg/L],pv\text{x}} \; [mg/L]$', fontsize=14)
        ax[0].set_ylabel(r'$Predicted \; S_{[c,mg/L],pv\text{x}} \; [mg/L]$', fontsize=14)
        ax[1].set_ylabel(r'$Residual \; [mg/L]$', fontsize=14)
        ax[0].set_title(r'$(a)$')
        ax[1].set_title(r'$(b)$')
        plt.tight_layout()
        if save_figs == 1:
            plt.savefig("Figure_A4.pdf")
    elif i == 2:
        ax[0].set_xlabel(r'$Actual \; Q_{c,pv\text{x}} \; [m^3/hr]$', fontsize=14)
        ax[1].set_xlabel(r'$Predicted \; Q_{c,pv\text{x}} \; [m^3/hr]$', fontsize=14)
        ax[0].set_ylabel(r'$Predicted \; Q_{c,pv\text{x}} \; [m^3/hr]$', fontsize=14)
        ax[1].set_ylabel(r'$Residual \; [m^3/hr]$', fontsize=14)
        ax[0].set_title(r'$(a)$')
        ax[1].set_title(r'$(b)$')
        plt.tight_layout()
        if save_figs == 1:
            plt.savefig("Figure_A5.pdf")
    elif i == 3:
        ax[0].set_xlabel(r'$Actual \; S_{[p,mg/L],pv\text{x}} \; [mg/L]$', fontsize=14)
        ax[1].set_xlabel(r'$Predicted \; S_{[p,mg/L],pv\text{x}} \; [mg/L]$', fontsize=14)
        ax[0].set_ylabel(r'$Predicted \; S_{[p,mg/L],pv\text{x}} \; [mg/L]$', fontsize=14)
        ax[1].set_ylabel(r'$Residual \; [mg/L]$', fontsize=14)
        ax[0].set_title(r'$(a)$')
        ax[1].set_title(r'$(b)$')
        plt.tight_layout()
        if save_figs == 1:
            plt.savefig("Figure_A6.pdf")
    elif i == 4:
        ax[0].set_xlabel(r'$Actual \; Q_{p,pv\text{x}} \; [m^3/hr]$', fontsize=14)
        ax[1].set_xlabel(r'$Predicted \; Q_{p,pv\text{x}} \; [m^3/hr]$', fontsize=14)
        ax[0].set_ylabel(r'$Predicted \; Q_{p,pv\text{x}} \; [m^3/hr]$', fontsize=14)
        ax[1].set_ylabel(r'$Residual \; [m^3/hr]$', fontsize=14)
        ax[0].set_title(r'$(a)$')
        ax[1].set_title(r'$(b)$')
        plt.tight_layout()
        if save_figs == 1:
            plt.savefig("Figure_A7.pdf")
    
    '''
    ax[0].set_xlabel('Actual ' + output_strings[i])
    ax[1].set_xlabel('Predicted ' + output_strings[i])
    ax[0].set_ylabel('Predicted ' + output_strings[i])
    ax[1].set_ylabel('Residual')
    plt.suptitle(output_strings[i])
    '''
    #'''
    
    #'''
    plt.figure(figsize=(4,4))
    plt.scatter(predicted,residuals)
    plt.xlabel('Predicted ' + output_strings[i])
    plt.ylabel('Residual')
    plt.title('Residuals vs. Predicted, ' + output_strings[i])
    #'''
    
    # Print the R2 value for each variable while I'm here
    #print("R2_test, " + output_strings[i] + ": " + str(r2_score(actual, predicted)))
    #print("RMSE_test, " + output_strings[i] + ": " + str(root_mean_squared_error(actual, predicted)))
    
    # Plots of residuals vs. each input
    """
    fig,ax = plt.subplots(nrows=1, ncols=3, figsize=(12.6, 4))
    ax[0].scatter(X_test_actual_units[:,0],residuals)
    ax[0].set_xlabel('P_f [psi]')
    ax[0].set_ylabel('Residual')
    ax[1].scatter(X_test_actual_units[:,1],residuals)
    ax[1].set_xlabel('TDS_f [mg/L]')
    ax[2].scatter(X_test_actual_units[:,2],residuals)
    ax[2].set_xlabel('Q_f [m^3/hr]')
    plt.suptitle('Residuals vs Each Input, ' + output_strings[i])
    """
    
    # Plots of residuals vs. recovery ratio
    """
    plt.figure(figsize=(4,4))
    plt.scatter(RR,residuals)
    plt.xlabel('Predicted Recovery Ratio')
    plt.ylabel('Residual')
    plt.title('Residuals vs. Recovery Ratio, ' + output_strings[i])
    """
    
    # Plot relative residuals
    #"""
    relative_error = (predicted-actual)/actual
    plt.figure(figsize=(4,4))
    plt.scatter(predicted,relative_error)
    plt.xlabel('Predicted ' + output_strings[i])
    plt.ylabel('Relative Residual')
    plt.title('Relative Residuals vs. Predicted, ' + output_strings[i])
    #"""
    
    # Let's fit lines to the residuals and see if the trends are meaningful
    m, b = np.polyfit(predicted, residuals, deg=1) # x is the predicted value, y is the residual
    print(f"Slope (m): {m}, Intercept (b): {b}")
    y_fit = m*predicted+b
    r2_fit = r2_score(residuals, y_fit)
    print(f"R2: {r2_fit}")
    bias = m*(max(predicted) - min(predicted))
    print(f"Bias: {bias}")


# ====================================
# Saving everything as one exportable "thing"
# ====================================
bundle = {
    "model_state_dict": final_model.state_dict(),
    "model_arch": {
        "input_dim": 4,
        "output_dim": 5,
        "hidden_layers": hidden_layers,
        "activation": activation,
    },
    "scalers": {
        "X_scaler_2000_final": X_scaler_2000_final,
        "Y_scaler_2000_final": Y_scaler_2000_final,
    },
    "boxcox_lambdas": {
        "lambda_P_c": lambda_P_c,
        "lambda_TDS_c": lambda_TDS_c,
        "lambda_TDS_p": lambda_TDS_p,
        "lambda_Q_p": lambda_Q_p,
    },
    "calibration": calibration,  # <-- the new piece
    "meta": {
        "calibration_fit_split": "val",
        "stage_id_encoding": "stage1=0, stage2=1",
        "outputs_order": ["P_c", "TDS_c", "Q_c", "TDS_p", "Q_p"],
        "inputs_order": ["P_f", "TDS_f", "Q_f", "stage_id"],
        "notes": "Apply calibration in physical units to P_c, TDS_c, Q_p",
    }
}

save_this = 0
if save_this == 1:
    joblib.dump(bundle, "RO_model_bundle.joblib")
    print("Saved RO_model_bundle_v2.joblib")
else:
    print("No joblib files saved")


# Maybe need this for later
'''
def predict_physical_calibrated(model, X_phys_np, X_scaler, Y_scaler, calibration):
    """
    X_phys_np: (N,4) in physical units including stage_id
    Returns calibrated physical outputs as dict.
    """
    X_scaled = X_scaler.transform(X_phys_np)
    x_t = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_scaled = model(x_t).numpy()

    y_trans = Y_scaler.inverse_transform(y_scaled)
    pred_phys = inverse_transform_outputs_to_physical(y_trans)
    pred_phys_cal = apply_calibration_to_phys_dict(pred_phys, calibration)
    return pred_phys_cal
'''












