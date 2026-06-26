# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 03:33:42 2026

@author: mwh85
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#%% Official State Analysis (why things happen when)

master_df_spring = pd.read_pickle("../../data/Optimal IPHROS/master_df_spring.pkl")
master_df_summer = pd.read_pickle("../../data/Optimal IPHROS/master_df_summer.pkl")
master_df_fall = pd.read_pickle("../../data/Optimal IPHROS/master_df_fall.pkl")
master_df_winter = pd.read_pickle("../../data/Optimal IPHROS/master_df_winter.pkl")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, accuracy_score, r2_score
from sklearn.model_selection import train_test_split

# ============================================================
# 0) BUILD DATAFRAME
# ============================================================

df_spring = pd.DataFrame(master_df_spring).copy()
df_summer = pd.DataFrame(master_df_summer).copy()
df_fall = pd.DataFrame(master_df_fall).copy()
df_winter = pd.DataFrame(master_df_winter).copy()

def add_dataframe_columns(df):
    #print('new season')
    # Combined state label
    df["state_label"] = (
        "con_" + df["state_num_turbines_con"].astype(int).astype(str)
        + "_gen_" + df["state_num_turbines_gen"].astype(int).astype(str)
        + "_RO_" + df["state_RO_on"].astype(int).astype(str)
    )
    
    # A compact integer state ID if useful (will be over-riding in a sec)
    df["state_id"] = pd.factorize(df["state_label"])[0]
    
    # Make some other dataframes that have some different conglomerations
    grouped_state_id = np.array(df["state_id"])
    grouped_state_label = list(df["state_label"])
    
    # List of different combos and listings (note: these are exhaustive lists)
    combo1 = np.array([0,1,1])
    combo2 = np.array([6,1,1])
    combo3 = np.array([0,7,1])
    combo4 = np.array([7,0,0])
    group1_combos = np.array([[1,1,1],[2,1,1],[3,1,1],[4,1,1],[5,1,1]]) # Non-maximal HSC 'con_>0&<6_gen_1_RO_1'
    group2_combos = np.array([[0,0,0],[0,1,0],[0,2,0],[0,3,0],[1,0,0],[2,0,0],[3,0,0]]) # RO off, minimal PSH activity 'con_0_gen_<4_RO_0'
    group3_combos = np.array([[0,2,1],[0,3,1],[0,4,1],[0,5,1],[0,6,1]]) # Non-maximal generation, RO on 'con_0_gen_>1&<7_RO_1'
    
    for i in range(168):
        hour_state = np.array(list(np.array(df.iloc[i,2:5])))
        if np.array_equal(hour_state, combo1):
            grouped_state_id[i] = 0
            #grouped_state_label[i] = 'con_0_gen_1_RO_1'
            grouped_state_label[i] = '# con=0, # gen=1, RO=1'
        elif np.array_equal(hour_state, combo2):
            grouped_state_id[i] = 1
            #grouped_state_label[i] = 'con_6_gen_1_RO_1'
            grouped_state_label[i] = '# con=6, # gen=1, RO=1'
        elif np.array_equal(hour_state, combo3):
            grouped_state_id[i] = 2
            #grouped_state_label[i] = 'con_0_gen_7_RO_1'
            grouped_state_label[i] = '# con=0, # gen=7, RO=1'
        elif np.array_equal(hour_state, combo4):
            grouped_state_id[i] = 3
            #grouped_state_label[i] = 'con_7_gen_0_RO_0'
            grouped_state_label[i] = '# con=7, # gen=0, RO=0'
        elif np.any(np.all(group1_combos == hour_state, axis=1)):
            grouped_state_id[i] = 4
            #grouped_state_label[i] = 'con_>0&<6_gen_1_RO_1'
            grouped_state_label[i] = '0<# con<6, # gen=1, RO=1'
        elif np.any(np.all(group2_combos == hour_state, axis=1)):
            grouped_state_id[i] = 5
            #grouped_state_label[i] = 'con_<4_gen_<4_RO_0'
            grouped_state_label[i] = '# con<4, # gen<4, RO=0'
        elif np.any(np.all(group3_combos == hour_state, axis=1)):
            grouped_state_id[i] = 6
            #grouped_state_label[i] = 'con_0_gen_>1&<7_RO_1'
            grouped_state_label[i] = '# con=0, 1<# gen<7, RO=1'
        else:
            print('shit I messed up')
            print(hour_state)
        
        
    df["grouped_state_label"] = grouped_state_label
    df["grouped_state_id"] = grouped_state_id
    
    return df

df_spring = add_dataframe_columns(df_spring)
df_summer = add_dataframe_columns(df_summer)
df_fall = add_dataframe_columns(df_fall)
df_winter = add_dataframe_columns(df_winter)


# ============================================================
# 1) ADD HELPER COLUMNS (for all seasons)
# ============================================================

# Aggregate cycle slack columns
cycle_slack_cols = [
    "slack_cycle_counting_PSH1",
    "slack_cycle_counting_PSH2",
    "slack_cycle_counting_PSH3",
    "slack_cycle_counting_PSH4",
    "slack_cycle_counting_PSH5",
    "slack_cycle_counting_PSH6",
    "slack_cycle_counting_PSH7",
]

df_spring["slack_cycle_min"] = df_spring[cycle_slack_cols].min(axis=1)
df_spring["slack_cycle_mean"] = df_spring[cycle_slack_cols].mean(axis=1)

df_summer["slack_cycle_min"] = df_summer[cycle_slack_cols].min(axis=1)
df_summer["slack_cycle_mean"] = df_summer[cycle_slack_cols].mean(axis=1)

df_fall["slack_cycle_min"] = df_fall[cycle_slack_cols].min(axis=1)
df_fall["slack_cycle_mean"] = df_fall[cycle_slack_cols].mean(axis=1)

df_winter["slack_cycle_min"] = df_winter[cycle_slack_cols].min(axis=1)
df_winter["slack_cycle_mean"] = df_winter[cycle_slack_cols].mean(axis=1)



#%%
# ============================================================
# 2) CONDITION DISTRIBUTIONS BY STATE
# ============================================================

def condition_distributions_by_state(df, signal_cols, label_column_key, top_n_states=None, figsize=(14, 4)):
    ###
    # Prints grouped stats and makes boxplots of signals by state.
    ###
    state_counts = df[label_column_key].value_counts()
    if top_n_states is not None:
        states_to_plot = state_counts.index[:top_n_states]
        dff = df[df[label_column_key].isin(states_to_plot)].copy()
    else:
        dff = df.copy()

    print("\n=== Condition distributions by state ===")
    print(dff.groupby(label_column_key)[signal_cols].describe())

    for col in signal_cols:
        if col == "signal_elec_price":
            plt.figure(figsize=figsize)
            ordered_states = dff[label_column_key].value_counts().index.tolist()
            data = [dff.loc[dff[label_column_key] == s, col].dropna().values for s in ordered_states]
            plt.boxplot(data, labels=ordered_states, vert=True)
            plt.xticks(rotation=45, ha="right",fontsize=12)
        
            plt.ylabel(r"$\lambda_e \; [€/kW]$",fontsize=14)
            #plt.title(f"{col} distribution by state")
            plt.ylim((0,0.35))
            plt.tight_layout()
            plt.savefig('Figure_12.pdf')
            plt.show()
        

# Example signals/slacks to compare by state
distribution_cols = [
    "signal_elec_price",
    "signal_max_salinity",
    "signal_res_level",
    "slack_lower_res_bound",
    "slack_upper_res_bound",
    "slack_cycle_min",
    "slack_res_end_of_week",
]

#condition_distributions_by_state(df=df, signal_cols=distribution_cols, label_column_key = 'state_label') #, top_n_states=10)
condition_distributions_by_state(df=df_spring, signal_cols=distribution_cols, label_column_key = 'grouped_state_label') #, top_n_states=10)


#%%
# ============================================================
# 6) FEATURE IMPORTANCE VIA CLASSIFICATION / REGRESSION
# ============================================================

# Combined dataframe
df_all_seasons = pd.concat([df_spring, df_summer, df_fall, df_winter], ignore_index=True)

feature_cols = [
    "signal_elec_price_norm",
    "signal_max_salinity",
    #"signal_res_level",
    "slack_lower_res_bound",
    "slack_upper_res_bound",
    "slack_cycle_mean",
    "slack_res_end_of_week",
    #"hour",
]

target_cols = [
    "grouped_state_id",
    "state_num_turbines_con",
    "state_num_turbines_gen",
    "state_RO_on"]

# ---------- Classification: predict operating state ----------
def remove_rare_classes(dff, target_col, min_count=3):
    counts = dff[target_col].value_counts()
    valid_classes = counts[counts >= min_count].index
    return dff[dff[target_col].isin(valid_classes)]

def state_classification_analysis(df, feature_cols, target_col="grouped_state_id", random_state=0):
    
    dff = df.dropna(subset=feature_cols + [target_col]).copy()
    dff = remove_rare_classes(dff, target_col)

    X = dff[feature_cols].values
    y = dff[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Random forest classifier
    clf_rf = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=500,
        random_state=random_state
    )
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)

    print("\n=== Random forest classifier ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))

    rf_imp_series = pd.Series(clf_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nRandom forest feature importance:")
    print(rf_imp_series)

    plt.figure(figsize=(8, 4))
    rf_imp_series.plot(kind="bar")
    plt.ylabel("Importance")
    plt.title("Random forest feature importance: " + target_col)
    plt.tight_layout()
    plt.show()

    # Permutation importance
    perm = permutation_importance(clf_rf, X_test, y_test, n_repeats=20, random_state=random_state)
    perm_imp_series = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)

    plt.figure(figsize=(8, 4))
    perm_imp_series.plot(kind="bar")
    plt.ylabel("Permutation importance")
    plt.title("Permutation importance for state classification")
    plt.tight_layout()
    plt.show()

    return {
        "rf_model": clf_rf,
        "scaler": scaler,
        "rf_importance": rf_imp_series,
        "perm_importance": perm_imp_series,
    }


def conditional_state_analysis(df, feature_cols, target_col, condition, label):
    #############
    # condition: boolean mask applied to df
    # label: string for printing/plots
    #############
    
    dff = df.loc[condition].copy()
    dff = remove_rare_classes(dff, target_col)
    
    print(f"\n==============================")
    print(f"Condition: {label}")
    print(f"Samples: {len(dff)}")
    print(f"==============================")

    if len(dff) < 50:
        print("Too few samples, results may be unstable")
        return None

    return state_classification_analysis(
        dff,
        feature_cols=feature_cols,
        target_col=target_col
    )

state_models_group = state_classification_analysis(df_all_seasons, feature_cols, target_col=target_cols[0])
state_models_con = state_classification_analysis(df_all_seasons, feature_cols, target_col=target_cols[1])
state_models_gen = state_classification_analysis(df_all_seasons, feature_cols, target_col=target_cols[2])
state_models_RO = state_classification_analysis(df_all_seasons, feature_cols, target_col=target_cols[3])

# Make a combined plot ######################################################
# Pull the rf importance series from your model dictionaries
rf_group = state_models_group["rf_importance"].copy()
rf_con   = state_models_con["rf_importance"].copy()
rf_gen   = state_models_gen["rf_importance"].copy()
rf_ro    = state_models_RO["rf_importance"].copy()

# Combine into one dataframe
rf_df = pd.concat(
    [rf_group, rf_con, rf_gen, rf_ro],
    axis=1
)

rf_df.columns = [
    "grouped_state_id",
    "state_num_turbines_con",
    "state_num_turbines_gen",
    "state_RO_on"
]

# Optional: enforce a specific feature order
feature_order = [
    "signal_elec_price_norm",
    "slack_cycle_mean",
    "slack_res_end_of_week",
    "slack_upper_res_bound",
    "slack_lower_res_bound",
    "signal_max_salinity"
]
rf_df = rf_df.reindex(feature_order)

# Plot settings
x = np.arange(len(rf_df.index))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(x - 1.5*width, rf_df["grouped_state_id"], width, label=r"$Aggregated \; IPHROS \; States$")
ax.bar(x - 0.5*width, rf_df["state_num_turbines_con"], width, label=r"$\# \; Turbines \;Consumption \;Mode$")
ax.bar(x + 0.5*width, rf_df["state_num_turbines_gen"], width, label=r"$\# \; Turbines \;Generation \;Mode$")
ax.bar(x + 1.5*width, rf_df["state_RO_on"], width, label=r"$RO \; On/Off$")

ax.set_xticks(x)
#ax.set_xticklabels(rf_df.index, rotation=45, ha="right", fontsize=12)
ax.set_xticklabels(['Normalized Electricity Price','Mean Cycle Count','End-of-Week Reservoir Level','Reservoir Capacity UB','Reservoir Capacity LB','Discharge Salinity Limit'], rotation=45, ha="right", fontsize=12)
ax.set_ylabel(r"$Importance$",fontsize=14)
#ax.set_title("Random forest feature importance")
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('Figure_14.pdf')
plt.show()

'''
res_high_sal = conditional_state_analysis(
    df_all_seasons,
    feature_cols,
    target_col="grouped_state_id",
    condition=df_all_seasons["signal_max_salinity"] > 38.5,
    label="High salinity (> 38.5)"
)

res_ro_off = conditional_state_analysis(
    df_all_seasons,
    feature_cols,
    "grouped_state_id",
    df_all_seasons["state_RO_on"] == 0,
    "RO OFF"
)

res_ro_on = conditional_state_analysis(
    df_all_seasons,
    feature_cols,
    "grouped_state_id",
    df_all_seasons["state_RO_on"] == 1,
    "RO ON"
)

res_ro_off_high_salinity = conditional_state_analysis(
    df_all_seasons,
    feature_cols,
    "grouped_state_id",
    ((df_all_seasons["signal_max_salinity"] > 38.5) & (df_all_seasons["state_RO_on"] == 0)),
    "RO OFF"
)
'''




#%% Modified HSC Analysis - Setting some things up

#---------------------------------------------------------
# Modified HSC Analysis
#---------------------------------------------------------

# First thing's first, add a column for determining if a given hour is a modified HSC event
df_all_seasons['mod_HSC'] = np.zeros(168*4)
for i in range(168*4):
    if df_all_seasons['grouped_state_id'][i] == 1 or df_all_seasons['grouped_state_id'][i] == 4:
        df_all_seasons.loc[i,'mod_HSC'] = 1

# We're then going to need max values
eta_francis_max_flow = 0.87367
nameplates = np.array([213413.2720506368, 213413.2720506368, 213413.2720506368, 213413.2720506368, 213413.2720506368, 213413.2720506368, 213413.2720506368])
rho_sw = 1026.311
g = 9.81
h_res = 332

V_dot_wp_max_array = (eta_francis_max_flow*nameplates*(3.6e6))/(rho_sw*g*h_res)
V_dot_swht_max_array = (nameplates*(3.6e6))/(eta_francis_max_flow*rho_sw*g*h_res)
P_con_PSH_max_array = np.array([df_all_seasons['P_con_PSH1'].max(),
                                df_all_seasons['P_con_PSH2'].max(),
                                df_all_seasons['P_con_PSH3'].max(),
                                df_all_seasons['P_con_PSH4'].max(),
                                df_all_seasons['P_con_PSH5'].max(),
                                df_all_seasons['P_con_PSH6'].max(),
                                df_all_seasons['P_con_PSH7'].max(),
                                ])
P_gen_PSH_max_array = np.array([df_all_seasons['P_gen_PSH1'].max(),
                                df_all_seasons['P_gen_PSH2'].max(),
                                df_all_seasons['P_gen_PSH3'].max(),
                                df_all_seasons['P_gen_PSH4'].max(),
                                df_all_seasons['P_gen_PSH5'].max(),
                                df_all_seasons['P_gen_PSH6'].max(),
                                df_all_seasons['P_gen_PSH7'].max(),
                                ])



#%% Modified HSC Analysis - Part 1
# ============================================================
# 1) ANALYSIS 1:
#    For the one generating turbine used for dilution,
#    what fraction of max allowable flow is it operating at?
#    How often is salinity at its maximum?
# ============================================================

# Indicator for "salinity limit above 38.5"
df_all_seasons["salinity_limit_above_38_5"] = (df_all_seasons["signal_max_salinity"] > 38.5).astype(int)

# Since modified HSC requires exactly 1 generating unit,
# the total generating flow equals the single-unit generating flow.
df_mod_HSC = df_all_seasons.loc[df_all_seasons["mod_HSC"] == 1].copy()


# Fraction of max allowable flow for the single generating turbine
flow_cols = [f"V_dot_swht{i}" for i in range(1, 8)]
efficiency_cols = [f"eta_gen_PSH{i}" for i in range(1, 8)]

df_mod_HSC["gen_turbine_flow_frac_of_max"] = (
    df_mod_HSC[flow_cols]
    .div(V_dot_swht_max_array, axis=1)  # column-wise division
    .max(axis=1)                        # row-wise max
)

df_mod_HSC["gen_turbine_flow_efficiency"] = (
    df_mod_HSC[efficiency_cols]
    .max(axis=1)                        # row-wise max
)


print("="*70)
print("Analysis 1A: Single generating turbine loading during modified HSC")
print("="*70)
print(df_mod_HSC["gen_turbine_flow_frac_of_max"].describe(percentiles=[0.1,0.25,0.5,0.75,0.9]))
print()

'''
# Optional: if you want to compare actual dilution flow to RO brine flow
# (i.e., a "dilution ratio"), this can also be informative.
df_mod_HSC["dilution_to_brine_flow_ratio"] = df_mod_HSC["Q_gen_total"] / df_mod_HSC["Q_brine_RO"].replace(0, np.nan)

print("="*70)
print("Analysis 1A (optional): Dilution flow / RO brine flow during modified HSC")
print("="*70)
print(df_mod_HSC["dilution_to_brine_flow_ratio"].describe(percentiles=[0.1,0.25,0.5,0.75,0.9]))
print()
'''

# How often is brine/discharge salinity at the maximum?
# Use tolerance because real data may have tiny numerical differences.
SAL_TOL = 1e-6
df_mod_HSC["at_salinity_max"] = np.isclose(df_mod_HSC["S_ht"], df_mod_HSC["signal_max_salinity"], atol=SAL_TOL)

print("="*70)
print("Analysis 1B: How often salinity is at the maximum during modified HSC")
print("="*70)
print(f"Count at max: {df_mod_HSC['at_salinity_max'].sum()} / {len(df_mod_HSC)}")
print(f"Fraction at max: {df_mod_HSC['at_salinity_max'].mean():.3f}")
print()

# Also useful: how often each max-salinity tier is active during modified HSC
salinity_tier_counts = (
    df_mod_HSC.groupby("signal_max_salinity")
    .agg(
        #modified_hsc_hours=("hour_global", "count"),
        frac_at_max=("at_salinity_max", "mean"),
        mean_gen_flow_frac=("gen_turbine_flow_frac_of_max", "mean"),
        #mean_dilution_ratio=("dilution_to_brine_flow_ratio", "mean")
    )
)

print("="*70)
print("Modified HSC metrics by allowable salinity level")
print("="*70)
print(salinity_tier_counts)
print()

# Plotting Time!!!
fig, ax1 = plt.subplots()

# Left axis → Objective
ax1.scatter(df_mod_HSC["gen_turbine_flow_frac_of_max"].to_numpy(), df_mod_HSC["S_ht"].to_numpy(), linewidth=2, label=r'$S_{ht}$')
ax1.set_xlabel(r"$\dot{V}_{swht}/\dot{V}_{swht,max}$",fontsize=14)
ax1.set_ylabel(r"$Discharge \; Salinity \; [g/kg]$",fontsize=14)
ax1.tick_params(axis='both', labelsize=11)
ax1.set_xlim([df_mod_HSC["gen_turbine_flow_frac_of_max"].min()-0.03,1])

# Right axis → Constraint Violation
ax2 = ax1.twinx()
ax2.scatter(df_mod_HSC["gen_turbine_flow_frac_of_max"].to_numpy(), df_mod_HSC["gen_turbine_flow_efficiency"].to_numpy(), color='tab:orange', linewidth=2, label=r'$\eta_{ht}$')
ax2.set_ylabel(r"$Turbine \; Efficiency \; [-]$",fontsize=14)
ax2.tick_params(axis='both', labelsize=11)

handles_part1, labels_part1 = ax1.get_legend_handles_labels()
handles_part2, labels_part2 = ax2.get_legend_handles_labels()
handles = handles_part1+handles_part2
labels = labels_part1+labels_part2
fig.legend(handles, labels,fontsize=12,bbox_to_anchor=(0.845, 0.6),frameon=True)

plt.tight_layout()   
plt.savefig('Figure_15.pdf')        
plt.show()


#---------------------------------
# Statistical analysis to see if the 46% overlap between modified HSC and elevated salinity is a coincidence or not
#---------------------------------

# Basic stats
n_total = len(df_all_seasons)
n_sal_high = df_all_seasons["salinity_limit_above_38_5"].sum()
n_sal_low = n_total - n_sal_high

n_hsc_total = df_all_seasons["mod_HSC"].sum()
n_hsc_sal_high = df_all_seasons.loc[df_all_seasons["salinity_limit_above_38_5"] == 1, "mod_HSC"].sum()
n_hsc_sal_low = df_all_seasons.loc[df_all_seasons["salinity_limit_above_38_5"] == 0, "mod_HSC"].sum()

frac_hsc_given_high = n_hsc_sal_high / n_sal_high if n_sal_high > 0 else np.nan
frac_hsc_given_low = n_hsc_sal_low / n_sal_low if n_sal_low > 0 else np.nan
frac_sal_high_given_hsc = (
    n_hsc_sal_high / n_hsc_total if n_hsc_total > 0 else np.nan
)

print("=" * 70)
print("Basic counts")
print("=" * 70)
print(f"Total hours: {n_total}")
print(f"Hours with salinity limit > 38.5: {n_sal_high}")
print(f"Hours with salinity limit <= 38.5: {n_sal_low}")
print(f"Total modified HSC hours: {n_hsc_total}")
print(f"Modified HSC hours when salinity limit > 38.5: {n_hsc_sal_high}")
print(f"Modified HSC hours when salinity limit <= 38.5: {n_hsc_sal_low}")
print()
print(f"P(modified HSC | salinity limit > 38.5)  = {frac_hsc_given_high:.4f}")
print(f"P(modified HSC | salinity limit <= 38.5) = {frac_hsc_given_low:.4f}")
print(f"P(salinity limit > 38.5 | modified HSC)  = {frac_sal_high_given_hsc:.4f}")
print()

# ============================================================
# 3) CONTINGENCY TABLE
# ============================================================
# rows   = salinity condition (high / not high)
# cols   = modified HSC (yes / no)
#
#                HSC=1   HSC=0
# sal_high=1       a       b
# sal_high=0       c       d

a = int(((df_all_seasons["salinity_limit_above_38_5"] == 1) & (df_all_seasons["mod_HSC"] == 1)).sum())
b = int(((df_all_seasons["salinity_limit_above_38_5"] == 1) & (df_all_seasons["mod_HSC"] == 0)).sum())
c = int(((df_all_seasons["salinity_limit_above_38_5"] == 0) & (df_all_seasons["mod_HSC"] == 1)).sum())
d = int(((df_all_seasons["salinity_limit_above_38_5"] == 0) & (df_all_seasons["mod_HSC"] == 0)).sum())

table = np.array([[a, b],
                  [c, d]])

print("=" * 70)
print("2x2 contingency table")
print("=" * 70)
print(pd.DataFrame(
    table,
    index=["salinity > 38.5", "salinity <= 38.5"],
    columns=["modified HSC = 1", "modified HSC = 0"]
))
print()

# ============================================================
# 4) STATISTICAL TESTS
# ============================================================
from scipy.stats import chi2_contingency, fisher_exact

# Chi-square test of independence
chi2, p_chi2, dof, expected = chi2_contingency(table, correction=False)

print("=" * 70)
print("Chi-square test of independence")
print("=" * 70)
print(f"chi2 statistic = {chi2:.6f}")
print(f"p-value        = {p_chi2:.6g}")
print("Expected counts under independence:")
print(pd.DataFrame(
    expected,
    index=["salinity > 38.5", "salinity <= 38.5"],
    columns=["modified HSC = 1", "modified HSC = 0"]
))
print()

# Fisher exact test (especially nice for 2x2 tables)
oddsratio_fisher, p_fisher = fisher_exact(table, alternative="two-sided")

print("=" * 70)
print("Fisher exact test")
print("=" * 70)
print(f"odds ratio = {oddsratio_fisher:.6f}")
print(f"p-value    = {p_fisher:.6g}")
print()

# ============================================================
# 5) EFFECT SIZES
# ============================================================
# Risk ratio:
#   P(HSC | sal high) / P(HSC | sal low)
risk_high = a / (a + b) if (a + b) > 0 else np.nan
risk_low  = c / (c + d) if (c + d) > 0 else np.nan
risk_ratio = risk_high / risk_low if risk_low > 0 else np.nan

# Odds ratio computed directly from table
# (same concept as Fisher output, but shown explicitly)
odds_ratio = (a * d) / (b * c) if (b * c) > 0 else np.nan

# Phi coefficient / Cramer's V for 2x2
phi = np.sqrt(chi2 / n_total) if n_total > 0 else np.nan

print("=" * 70)
print("Effect sizes")
print("=" * 70)
print(f"Risk high  = P(HSC | salinity > 38.5)  = {risk_high:.6f}")
print(f"Risk low   = P(HSC | salinity <= 38.5) = {risk_low:.6f}")
print(f"Risk ratio = {risk_ratio:.6f}")
print(f"Odds ratio = {odds_ratio:.6f}")
print(f"Phi coeff  = {phi:.6f}")
print()

# ============================================================
# 6) "HOW SURPRISING IS 46%?" UNDER INDEPENDENCE
# ============================================================
# If HSC and salinity-high were independent, expected overlap would be:
expected_overlap = n_sal_high * (n_hsc_total / n_total) if n_total > 0 else np.nan
expected_frac_hsc_given_high = n_hsc_total / n_total if n_total > 0 else np.nan

print("=" * 70)
print("Observed vs expected under independence")
print("=" * 70)
print(f"Observed overlap count                    = {a}")
print(f"Expected overlap count under independence = {expected_overlap:.4f}")
print(f"Observed P(HSC | salinity > 38.5) = {frac_hsc_given_high:.4f}")
print(f"Expected under independence                 = {expected_frac_hsc_given_high:.4f}")
print()

# ============================================================
# 7) PERMUTATION TEST
# ============================================================
# Null hypothesis:
#   Modified HSC timing is unrelated to salinity-high timing.
# We keep the number of HSC hours fixed, but randomly shuffle when they occur.
# Then compare the observed overlap count to this null distribution.

rng = np.random.default_rng(1)

hsc_array = df_all_seasons["mod_HSC"].to_numpy()
sal_high_array = df_all_seasons["salinity_limit_above_38_5"].to_numpy()

observed_overlap = int(np.sum((hsc_array == 1) & (sal_high_array == 1)))

perm_overlaps = np.zeros(10000, dtype=int)

for i in range(10000):
    perm_hsc = rng.permutation(hsc_array)
    perm_overlaps[i] = np.sum((perm_hsc == 1) & (sal_high_array == 1))

# One-sided p-value:
# how often do we get overlap >= observed just by chance?
p_perm_one_sided = (np.sum(perm_overlaps >= observed_overlap) + 1) / (10000 + 1)

# Two-sided-ish version based on distance from null mean
null_mean = perm_overlaps.mean()
obs_dist = abs(observed_overlap - null_mean)
perm_dist = abs(perm_overlaps - null_mean)
p_perm_two_sided = (np.sum(perm_dist >= obs_dist) + 1) / (10000 + 1)

print("=" * 70)
print("Permutation test")
print("=" * 70)
print(f"Observed overlap count = {observed_overlap}")
print(f"Null mean overlap      = {null_mean:.4f}")
print(f"One-sided p-value      = {p_perm_one_sided:.6g}")
print(f"Two-sided p-value      = {p_perm_two_sided:.6g}")
print()



#%% Modified HSC Analysis - Part 2
# ============================================================
# 2) ANALYSIS 2:
#    Opportunity cost of modified HSC
#
#    Instead of discharging now to dilute RO now, what if you charged now to 
#    discharge later with no RO? How much profit would you be missing out on?
# ============================================================

# How much of the available pumping/generating flowrate/power is being used, given the number of turbines that are active in each mode?
flow_cols = [f"V_dot_wp{i}" for i in range(1, 8)]

df_mod_HSC["con_unit_average_loading_frac"] = (
    df_mod_HSC[flow_cols]
    .div(V_dot_wp_max_array, axis=1)   # normalize
    .replace(0, np.nan)                # ignore zeros
    .mean(axis=1)                      # row-wise mean
)

# Based on the loading of the consuming turbines, how much consuming power is added?
df_mod_HSC["P_con_added_h"] = nameplates[0] * df_mod_HSC["con_unit_average_loading_frac"]

# The power discharged is the current hour is assumed to be discharged at a later hour
df_mod_HSC["P_gen_later_h"] = df_mod_HSC["P_gen_PSH_total"]

# Determine what the maximum electricity price is each season
lambda_e_max_array = np.array([df_spring['signal_elec_price'].max(),
                               df_summer['signal_elec_price'].max(),
                               df_fall['signal_elec_price'].max(),
                               df_winter['signal_elec_price'].max()
                               ])

# Calculate the different terms that go into the opportunity cost
water_price = 1.97 # Euros/m^3

df_mod_HSC["profit_e_h"] = df_mod_HSC["P_gen_PSH_total"]*df_mod_HSC['signal_elec_price']
df_mod_HSC["profit_w_h"] = df_mod_HSC["V_dot_fwRO"]*water_price

df_mod_HSC["cost_e_h_mod"] = df_mod_HSC["P_con_added_h"]*df_mod_HSC['signal_elec_price']
df_mod_HSC["profit_e_h_later"] = df_mod_HSC["P_gen_later_h"]*lambda_e_max_array[df_mod_HSC["rep_week"].to_numpy().astype(int)]

# Here is the opportunity cost calculation
df_mod_HSC["current_profit"] = df_mod_HSC["profit_e_h"] + df_mod_HSC["profit_w_h"] 
df_mod_HSC["alternate_profit"] = df_mod_HSC["profit_e_h_later"] - df_mod_HSC["cost_e_h_mod"]
df_mod_HSC["opportunity_cost"] = df_mod_HSC["profit_e_h"] + df_mod_HSC["profit_w_h"] - (df_mod_HSC["profit_e_h_later"] - df_mod_HSC["cost_e_h_mod"])
df_mod_HSC["percent_profit_lost"] = (df_mod_HSC["profit_e_h_later"] - df_mod_HSC["cost_e_h_mod"])/(df_mod_HSC["profit_e_h"] + df_mod_HSC["profit_w_h"])


# Calculate the difference between the "now vs. later" electricity prices
df_mod_HSC["lambda_e_diff"] = lambda_e_max_array[df_mod_HSC["rep_week"].to_numpy().astype(int)] - df_mod_HSC['signal_elec_price']


# Plotting Time!!! ####################################################
# Group by number of consuming turbines and compute averages
avg_df = (
    df_mod_HSC
    .groupby("state_num_turbines_con", as_index=False)[["current_profit", "alternate_profit", "opportunity_cost", "percent_profit_lost", "lambda_e_diff"]]
    .mean()
    .sort_values("state_num_turbines_con")
)

fig, ax1 = plt.subplots()

# Left axis → Objective
ax1.plot(avg_df["state_num_turbines_con"].to_numpy(), avg_df["current_profit"].to_numpy(), linewidth=2, label=r'$Current \; Hour \; Profit$')
ax1.plot(avg_df["state_num_turbines_con"].to_numpy(), avg_df["alternate_profit"].to_numpy(), linewidth=2, label=r'$Alternate \; Hour \; Profit$')
#ax1.plot(avg_df["state_num_turbines_con"].to_numpy(), avg_df["opportunity_cost"].to_numpy(), linewidth=2, label=r'$Penalty$')
#ax1.plot(avg_df["state_num_turbines_con"].to_numpy(), avg_df["percent_profit_lost"].to_numpy(), linewidth=2, label=r'$\% \; Loss$')
ax1.set_xlabel(r"$Modified \; HSC \; State \; [\# \; Charging \; Units]$",fontsize=14)
ax1.set_ylabel(r"$Average \; Modified \; HSC$" "\n" r"$Hour \; Profit \; [2023\;€]$",fontsize=14)
ax1.tick_params(axis='both', labelsize=11)

# Right axis → Constraint Violation
ax2 = ax1.twinx()
ax2.plot(avg_df["state_num_turbines_con"].to_numpy(), avg_df["lambda_e_diff"].to_numpy(), color='tab:green', linewidth=2, label=r'$\lambda_e \; Difference$')
ax2.set_ylabel(r"$Average \; \lambda_{e,s,max} - \lambda_{e,h} \; [2023\;€]$",fontsize=14)
ax2.tick_params(axis='both', labelsize=11)

handles_part1, labels_part1 = ax1.get_legend_handles_labels()
handles_part2, labels_part2 = ax2.get_legend_handles_labels()
handles = handles_part1+handles_part2
labels = labels_part1+labels_part2
fig.legend(handles, labels,fontsize=12,bbox_to_anchor=(0.65, 0.75),frameon=True)

plt.tight_layout()   
plt.savefig('Figure_16.pdf')        
plt.show()

