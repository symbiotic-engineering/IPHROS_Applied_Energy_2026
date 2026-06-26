# -*- coding: utf-8 -*-
"""
Comparing HiGHS 1.2 and HiGHS 1.8 AND SETTING TERMINATION CRITERIA for Maha meeting on 1/14 (not official!!!)
"""

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict


# Read in data, isolate fully converged designs
df_epsilon_0 = pd.read_pickle("../../data/RO and Miscellaneous/key_vals_df_epsilon_0.pkl")
df_epsilon_3_17 = pd.read_pickle("../../data/RO and Miscellaneous/key_vals_df_epsilon_3_17.pkl")

# Need to fix a couple reduced duality gap values for this analysis
df_epsilon_0.loc[0,"Duality_Gap_Reduced"] = max(np.array(df_epsilon_0["Duality_Gap_Reduced"]))
df_epsilon_0.loc[26,"Duality_Gap_Reduced"] = max(np.array(df_epsilon_0["Duality_Gap_Reduced"]))
df_epsilon_3_17.loc[0,"Duality_Gap_Reduced"] = max(np.array(df_epsilon_3_17["Duality_Gap_Reduced"]))

with open("../../data/RO and Miscellaneous/summary_values_epsilon_0.pkl", "rb") as f:
    summary_values_epsilon_0 = pickle.load(f)   
with open("../../data/RO and Miscellaneous/summary_values_epsilon_3_17.pkl", "rb") as f:
    summary_values_epsilon_3_17 = pickle.load(f)

df_epsilon_0_cat2 = df_epsilon_0[df_epsilon_0["Category"] == 2]
df_epsilon_3_17_cat2 = df_epsilon_3_17[df_epsilon_3_17["Category"] == 2]

df_epsilon_0_cat_other = df_epsilon_0[df_epsilon_0["Category"] != 2]
df_epsilon_3_17_cat_other = df_epsilon_3_17[df_epsilon_3_17["Category"] != 2]

# First need to get a list of these values
converged_idxs = df_epsilon_0_cat2.index.tolist()
long_idxs = df_epsilon_0_cat_other.index.tolist()



#%% Analysis

print('STEP 1!')
results = defaultdict(dict)
epsilons = [0,3.17e-5]

LHS_matrix = pd.read_csv("../../data/RO and Miscellaneous/MILP_options_LHS_new_fixed_epsilon.csv",header=None).to_numpy()

# Format the data
for design in range(40):
    
    if design in converged_idxs:
        my_converged = True
    elif design in long_idxs:
        my_converged = False
    else:
        print('oops1')
        
    for epsilon in epsilons:
        
        if epsilon == 0:
            my_runtime = df_epsilon_0['Run_Time'][design]
            my_nodes = df_epsilon_0['Nodes'][design]
            my_runtime_reduced = df_epsilon_0['Run_Time_Reduced'][design]
            my_nodes_reduced = df_epsilon_0['Nodes_Reduced'][design]
            my_gap_reduced = df_epsilon_0['Duality_Gap_Reduced'][design]
            
            my_pd_integral = summary_values_epsilon_0[design]['P-D integral']
            my_iterations_total = summary_values_epsilon_0[design]['LP iterations']
            my_iterations_strong_branch = summary_values_epsilon_0[design]['(strong br.)']
            my_iterations_separation = summary_values_epsilon_0[design]['(separation)']
            my_iterations_heuristic = summary_values_epsilon_0[design]['(heuristics)']
            
            my_nodes_2_hrs = summary_values_epsilon_0[design]['Nodes']
            my_gap_2_hrs = summary_values_epsilon_0[design]['Gap']
        elif epsilon == 3.17e-5:
            my_runtime = df_epsilon_3_17['Run_Time'][design]
            my_nodes = df_epsilon_3_17['Nodes'][design]
            my_runtime_reduced = df_epsilon_3_17['Run_Time_Reduced'][design]
            my_nodes_reduced = df_epsilon_3_17['Nodes_Reduced'][design]
            my_gap_reduced = df_epsilon_3_17['Duality_Gap_Reduced'][design]
            
            my_pd_integral = summary_values_epsilon_3_17[design]['P-D integral']
            my_iterations_total = summary_values_epsilon_3_17[design]['LP iterations']
            my_iterations_strong_branch = summary_values_epsilon_3_17[design]['(strong br.)']
            my_iterations_separation = summary_values_epsilon_3_17[design]['(separation)']
            my_iterations_heuristic = summary_values_epsilon_3_17[design]['(heuristics)']
            
            my_nodes_2_hrs = summary_values_epsilon_3_17[design]['Nodes']
            my_gap_2_hrs = summary_values_epsilon_3_17[design]['Gap']
        else:
            print('oops2')

        results[design][epsilon] = {
            "B_PSH": LHS_matrix[design,0],
            "N_pv1": LHS_matrix[design,1],
            "converged": my_converged,
            "runtime": my_runtime,
            "nodes": my_nodes,
            "pd_integral": my_pd_integral, # by the 2-hour mark
            "iterations": {
                "total": my_iterations_total, # by the 2-hour mark
                "strong": my_iterations_strong_branch, # by the 2-hour mark
                "separation": my_iterations_separation, # by the 2-hour mark
                "heuristics": my_iterations_heuristic # by the 2-hour mark
            },
            "primal_2100": {
                "time": my_runtime_reduced,
                "nodes": my_nodes_reduced,
                "gap": my_gap_reduced
            },
            "2_hr_vals": {
                "nodes": my_nodes_2_hrs,
                "gap": my_gap_2_hrs
            },
        }


rows = []

for design_id, design_data in results.items():
    for epsilon, run in design_data.items():
        rows.append({
            "design_id": design_id,
            "epsilon": epsilon,
            "B_PSH": run["B_PSH"],
            "N_pv1": run["N_pv1"],

            # convergence info
            "converged": int(run["converged"]),
            "runtime": run["runtime"],
            "nodes": run["nodes"],
            "pd_integral": run["pd_integral"],

            # iterations
            "iter_total": run["iterations"]["total"],
            "iter_strong": run["iterations"]["strong"],
            "iter_sep": run["iterations"]["separation"],
            "iter_heur": run["iterations"]["heuristics"],

            # primal within $2100
            "t_primal_2100": run["primal_2100"]["time"],
            "nodes_primal_2100": run["primal_2100"]["nodes"],
            "gap_primal_2100": run["primal_2100"]["gap"],
            
            # other values at the 2-hr mark
            "nodes_2_hr": run["2_hr_vals"]["nodes"],
            "gap_2_hr": run["2_hr_vals"]["gap"],
        })

analysis_df = pd.DataFrame(rows)

TIME_CAP = 7200

analysis_df["censored"] = 1 - analysis_df["converged"]
analysis_df["event_time"] = analysis_df["runtime"].clip(upper=TIME_CAP)

analysis_df_converged = analysis_df[analysis_df["converged"] == 1]
analysis_df_no_converge = analysis_df[analysis_df["converged"] != 1]


metric_cols = [
    "runtime",
    "nodes",
    "pd_integral",
    "iter_total",
    "iter_strong",
    "iter_sep",
    "iter_heur",
    "t_primal_2100",
    "nodes_primal_2100",
    "gap_primal_2100",
    "nodes_2_hr",
    "gap_2_hr"
]

long_df = analysis_df.melt(
    id_vars=["design_id", "B_PSH", "N_pv1", "epsilon", "converged"],
    value_vars=metric_cols,
    var_name="metric",
    value_name="value"
)

long_df_converged = analysis_df_converged.melt(
    id_vars=["design_id", "B_PSH", "N_pv1", "epsilon", "converged"],
    value_vars=metric_cols,
    var_name="metric",
    value_name="value"
)

long_df_no_converge = analysis_df_no_converge.melt(
    id_vars=["design_id", "B_PSH", "N_pv1", "epsilon", "converged"],
    value_vars=metric_cols,
    var_name="metric",
    value_name="value"
)


print('STEP 2A')

# convergence dominance table
conv_tbl = (
    analysis_df
    .pivot_table(
        index="design_id",
        columns="epsilon",
        values="converged",
        aggfunc="first"
    )
)

conv_tbl["status"] = (
    conv_tbl.fillna(0).astype(int).astype(str).agg("".join, axis=1)
)

status_counts = conv_tbl["status"].value_counts()
print(status_counts)

print('STEP 2B')

# Paired deltas --> no convergence
paired_no_converge = (
    analysis_df_no_converge
    .pivot(
        index="design_id",
        columns="epsilon",
        values=[
            "pd_integral",
            "iter_total",
            "iter_strong",
            "iter_sep",
            "iter_heur",
            "t_primal_2100",
            "nodes_primal_2100",
            "nodes_2_hr",
            "gap_2_hr"
        ]
    )
)

# flatten columns
paired_no_converge.columns = ["_".join(map(str, c)) for c in paired_no_converge.columns]
paired_no_converge = paired_no_converge.dropna()

paired_no_converge["delta_pd"] = paired_no_converge["pd_integral_3.17e-05"] - paired_no_converge["pd_integral_0.0"]
paired_no_converge["delta_iter_total"] = paired_no_converge["iter_total_3.17e-05"] - paired_no_converge["iter_total_0.0"]
paired_no_converge["delta_iter_strong"] = paired_no_converge["iter_strong_3.17e-05"] - paired_no_converge["iter_strong_0.0"]
paired_no_converge["delta_iter_sep"] = paired_no_converge["iter_sep_3.17e-05"] - paired_no_converge["iter_sep_0.0"]
paired_no_converge["delta_iter_heur"] = paired_no_converge["iter_heur_3.17e-05"] - paired_no_converge["iter_heur_0.0"]
paired_no_converge["delta_t_primal_2100"] = paired_no_converge["t_primal_2100_3.17e-05"] - paired_no_converge["t_primal_2100_0.0"]
paired_no_converge["delta_nodes_primal_2100"] = paired_no_converge["nodes_primal_2100_3.17e-05"] - paired_no_converge["nodes_primal_2100_0.0"]
paired_no_converge["delta_nodes_2_hr"] = paired_no_converge["nodes_2_hr_3.17e-05"] - paired_no_converge["nodes_2_hr_0.0"]
paired_no_converge["delta_gap_2_hr"] = paired_no_converge["gap_2_hr_3.17e-05"] - paired_no_converge["gap_2_hr_0.0"]


paired_no_converge_summary = paired_no_converge[[
    "delta_pd",
    "delta_iter_total",
    "delta_iter_strong",
    "delta_iter_sep",
    "delta_iter_heur",
    "delta_t_primal_2100",
    "delta_nodes_primal_2100",
    "delta_nodes_2_hr",
    "delta_gap_2_hr"
]].describe()

# Note: positive deltas is an indication that epsilon = 0 is best

# Paired deltas --> converged designs
paired_converged = (
    analysis_df_converged
    .pivot(
        index="design_id",
        columns="epsilon",
        values=[
            "runtime",
            "nodes",
            "pd_integral",
            "iter_total",
            "iter_strong",
            "iter_sep",
            "iter_heur",
            "t_primal_2100",
            "nodes_primal_2100"
        ]
    )
)

# flatten columns
paired_converged.columns = ["_".join(map(str, c)) for c in paired_converged.columns]
paired_converged = paired_converged.dropna()

paired_converged["delta_runtime"] = paired_converged["runtime_3.17e-05"] - paired_converged["runtime_0.0"]
paired_converged["delta_nodes"] = paired_converged["nodes_3.17e-05"] - paired_converged["nodes_0.0"]
paired_converged["delta_pd"] = paired_converged["pd_integral_3.17e-05"] - paired_converged["pd_integral_0.0"]
paired_converged["delta_iter_total"] = paired_converged["iter_total_3.17e-05"] - paired_converged["iter_total_0.0"]
paired_converged["delta_iter_strong"] = paired_converged["iter_strong_3.17e-05"] - paired_converged["iter_strong_0.0"]
paired_converged["delta_iter_sep"] = paired_converged["iter_sep_3.17e-05"] - paired_converged["iter_sep_0.0"]
paired_converged["delta_iter_heur"] = paired_converged["iter_heur_3.17e-05"] - paired_converged["iter_heur_0.0"]
paired_converged["delta_t2100"] = paired_converged["t_primal_2100_3.17e-05"] - paired_converged["t_primal_2100_0.0"]
paired_converged["delta_n2100"] = paired_converged["nodes_primal_2100_3.17e-05"] - paired_converged["nodes_primal_2100_0.0"]

paired_converged_summary = paired_converged[[
    "delta_runtime",
    "delta_nodes",
    "delta_pd",
    "delta_iter_total",
    "delta_iter_strong",
    "delta_iter_sep",
    "delta_iter_heur",
    "delta_t2100",
    "delta_n2100",
]].describe()



print('STEP 3A')

# Survival analysis
# Kaplan-Meier curves
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

kmf = KaplanMeierFitter()

plt.figure()
for eps in analysis_df["epsilon"].unique():
    mask = analysis_df["epsilon"] == eps
    if eps == 0:
        my_label = r"$\epsilon_{PSH} = 0$"
    else:
        my_label = r"$\epsilon_{PSH} = 3.17 \times 10^{-5}$"
    kmf.fit(
        durations=analysis_df.loc[mask, "event_time"],
        event_observed=analysis_df.loc[mask, "converged"],
        label=my_label#f"epsilon={eps}"
    )
    kmf.plot_survival_function()

plt.xlabel(r"$Time \; [s]$",fontsize=14)
plt.ylabel(r"$1 - P(Converged)$",fontsize=14)
plt.legend(fontsize=12)
#plt.title("Convergence Survival Curves")
plt.tight_layout()
plt.savefig("Figure_A10.pdf")
plt.show()



print('STEP 3B')

# Cox proportional hazards model
from lifelines import CoxPHFitter

cox_df = analysis_df.copy()

cox_df["eps_on"] = (cox_df["epsilon"] > 0).astype(int)
cox_df["eps_on_B"] = cox_df["eps_on"] * cox_df["B_PSH"]
cox_df["eps_on_N"] = cox_df["eps_on"] * cox_df["N_pv1"]

cph = CoxPHFitter()
cph.fit(
    cox_df[[
        "event_time",
        "converged",
        "eps_on",
        "B_PSH",
        "N_pv1",
        "eps_on_B",
        "eps_on_N"
    ]],
    duration_col="event_time",
    event_col="converged"
)

cph.print_summary()



print('STEP 4 (note: I am adding this to step 2) --> actually not useful')
"""

# Early usefulness (time to good primal)
# Compare time to 2100 primal
t2100 = analysis_df.pivot(
    index="design_id",
    columns="epsilon",
    values="t_primal_2100"
)

t2100["delta_t2100"] = t2100[3.17e-5] - t2100[0.0]
t2100["better_eps"] = t2100["delta_t2100"].apply(
    lambda x: "eps>0" if x < 0 else "eps=0"
)

print(t2100["better_eps"].value_counts())


# Early usefulness (number of nodes to good primal)
# Compare time to 2100 primal
nodes2100 = analysis_df.pivot(
    index="design_id",
    columns="epsilon",
    values="nodes_primal_2100"
)

nodes2100["delta_nodes2100"] = nodes2100[3.17e-5] - nodes2100[0.0]
nodes2100["better_eps"] = nodes2100["delta_nodes2100"].apply(
    lambda x: "eps>0" if x < 0 else "eps=0"
)

print(nodes2100["better_eps"].value_counts())
"""


print('STEP 5A --> not using')
"""
# “Do covariate effects on time-to-event change when censoring is removed?”


cox_df_step5 = analysis_df.copy()

cox_df_step5["eps_on"] = (cox_df_step5["epsilon"] > 0).astype(int)
cox_df_step5["eps_on_B"] = cox_df_step5["eps_on"] * cox_df_step5["B_PSH"]
cox_df_step5["eps_on_N"] = cox_df_step5["eps_on"] * cox_df_step5["N_pv1"]

covariates = [
    "eps_on",
    "B_PSH",
    "N_pv1",
    "eps_on_B",
    "eps_on_N",
]

cph_obs = CoxPHFitter()

cph_obs.fit(
    cox_df_step5[
        ["event_time", "converged"] + covariates
    ],
    duration_col="event_time",
    event_col="converged"
)

cph_obs.print_summary()


cph_proj = CoxPHFitter()

cph_proj.fit(
    cox_df_step5[
        ["runtime"] + covariates
    ],
    duration_col="runtime"
    # NOTE: no event_col
)

cph_proj.print_summary()


summary_compare = (
    cph_obs.summary[["coef", "exp(coef)", "p"]]
    .rename(columns=lambda c: f"{c}_obs")
    .join(
        cph_proj.summary[["coef", "exp(coef)", "p"]]
        .rename(columns=lambda c: f"{c}_proj")
    )
)

print(summary_compare)
"""


print('STEP 5B --> not using')

"""
kmf_step5 = KaplanMeierFitter()

plt.figure()
for eps in analysis_df["epsilon"].unique():
    mask = analysis_df["epsilon"] == eps
    kmf_step5.fit(
        durations=analysis_df.loc[mask, "runtime"],
        #event_observed=analysis_df.loc[mask, "converged"],
        label=f"epsilon={eps}"
    )
    kmf_step5.plot_survival_function()

plt.xlabel("Time (s)")
plt.ylabel("P(Not yet converged)")
plt.title("Convergence Survival Curves")
plt.show()


"""

print('STEP 6A')

from sklearn.preprocessing import StandardScaler

score_cols = [
    "runtime",
    "nodes",
    "pd_integral",
    "iter_total",
    "t_primal_2100",
    "nodes_primal_2100",
    "gap_2_hr"
]

scaler = StandardScaler()
scaled = scaler.fit_transform(analysis_df[score_cols])

scaled_df = pd.DataFrame(
    scaled,
    columns=[f"z_{c}" for c in score_cols],
    index=analysis_df.index
)

analysis_df = pd.concat([analysis_df, scaled_df], axis=1)



print('STEP 6B and STEP 7')
    
w_tier1 = 0.8 # np.random.uniform(0.6,0.9) # most important
w_tier2 = 0.5 # np.random.uniform(0.35, 0.65)
w_tier3 = 0.2 #np.random.uniform(0.1, 0.4) # least important

analysis_df["score"] = (
    -w_tier2 * analysis_df["z_runtime"]
    -w_tier3 * analysis_df["z_nodes"]
    -w_tier2 * analysis_df["z_pd_integral"]
    -w_tier3 * analysis_df["z_iter_total"]
    -w_tier1 * analysis_df["z_t_primal_2100"]
    -w_tier2 * analysis_df["z_nodes_primal_2100"] # changed from tier1 to tier2
    -w_tier1 * analysis_df["z_gap_2_hr"]
)


aa = analysis_df.groupby("epsilon")["score"].describe()


score_pivot = analysis_df.pivot(
    index="design_id",
    columns="epsilon",
    values="score"
)

score_pivot["preferred_eps"] = score_pivot.idxmax(axis=1)

score_pivot["t_primal_2100_0.0"] = df_epsilon_0["Run_Time_Reduced"]
score_pivot["t_primal_2100_3.17e-05"] = df_epsilon_3_17["Run_Time_Reduced"]

# Included for later setting of MILP termination criteria
score_pivot["nodes_primal_2100_0.0"] = df_epsilon_0["Nodes_Reduced"]
score_pivot["nodes_primal_2100_3.17e-05"] = df_epsilon_3_17["Nodes_Reduced"]

# Included for later setting of MILP termination criteria
score_pivot["gap_primal_2100_0.0"] = np.array(analysis_df[analysis_df["epsilon"] == 0]["gap_primal_2100"])
score_pivot["gap_primal_2100_3.17e-05"] = np.array(analysis_df[analysis_df["epsilon"] == 3.17e-5]["gap_primal_2100"])

design_info = (
    analysis_df
    .drop_duplicates("design_id")
    .set_index("design_id")[["B_PSH", "N_pv1"]]
)

pref_df = score_pivot.join(design_info)


# Make a boolean array for which epsilon is preferred
pref_eps_bool = pref_df["preferred_eps"] == 3.17e-5

# Define marker based on convergence
markers = np.where(
    analysis_df.drop_duplicates("design_id").set_index("design_id")["converged"] == 1,
    'o',  # circle for converged
    's'   # square for not converged
)

# Create scatter plot
fig, ax = plt.subplots(1, 1, figsize=(7, 4))

for marker_type in ['o', 's']:
    mask = markers == marker_type
    sc = ax.scatter(
        pref_df.loc[mask, "B_PSH"],
        pref_df.loc[mask, "N_pv1"],
        c=pref_eps_bool[mask],
        cmap="coolwarm",
        edgecolor='k',  # optional: black edge for visibility
        marker=marker_type,
        s=100,  # optional: make markers larger
        label=r"$Converged$" if marker_type == 'o' else r"$Did \; not \; converge$"
    )

# Create custom legend for colors (preferred epsilon)
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=r'$\epsilon_{PSH} = 0$', markerfacecolor=plt.cm.coolwarm(0.), markersize=10),
    Line2D([0], [0], marker='o', color='w', label=r'$\epsilon_{PSH} = 3.17 \times 10^{-5}$', markerfacecolor=plt.cm.coolwarm(1.), markersize=10)
]

ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0], loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

ax.set_xlabel(r"$B_{PSH} \; [kW]$",fontsize=14)
ax.set_ylabel(r"$N_{pv1} \; [-]$",fontsize=14)
#ax.set_title("Winning Epsilons with Convergence Status")
plt.tight_layout()
plt.savefig('Figure_A11a.pdf')
plt.show()


print('STEP 8!')

pref_df["eps_choice"] = (pref_df["preferred_eps"] == 3.17e-5).astype(int)

import statsmodels.formula.api as smf

logit_model = smf.logit(
    "eps_choice ~ B_PSH + N_pv1 + B_PSH:N_pv1",
    data=pref_df
).fit()

print(logit_model.summary())

pref_df["p_eps_on"] = logit_model.predict(pref_df)


term1_c_fp_time = np.array(pref_df[pref_df["preferred_eps"] == 0]["t_primal_2100_3.17e-05"])
term2_c_fp_time = np.array(pref_df[pref_df["preferred_eps"] == 0]["t_primal_2100_0.0"])
term1_c_fn_time = np.array(pref_df[pref_df["preferred_eps"] == 3.17e-5]["t_primal_2100_0.0"])
term2_c_fn_time = np.array(pref_df[pref_df["preferred_eps"] == 3.17e-5]["t_primal_2100_3.17e-05"])

C_FP_time = np.mean(term1_c_fp_time-term2_c_fp_time)
C_FN_time = np.mean(term1_c_fn_time-term2_c_fn_time)

p_star = C_FP_time/(C_FP_time+C_FN_time)

pref_df["eps_policy"] = np.where(
    pref_df["p_eps_on"] > round(p_star,2),
    3.17e-5,
    0.0
)

plt.figure()
plt.scatter(
    pref_df["B_PSH"],
    pref_df["N_pv1"],
    c=pref_df["p_eps_on"],
    cmap="viridis"
)
plt.colorbar(label="P(epsilon > 0)")
plt.xlabel(r"$B_[PSH]$",fontsize=14)
plt.ylabel(r"$N_[pv1]$",fontsize=14)
#plt.title("Predicted Preference for epsilon > 0")
plt.show()



#----------------------------------------------------------------------------
# Logit heat map
#----------------------------------------------------------------------------
# Define grid
B_vals = np.linspace(pref_df["B_PSH"].min(), pref_df["B_PSH"].max(), 101)
N_vals = np.linspace(pref_df["N_pv1"].min(), pref_df["N_pv1"].max(), 101)

B_grid, N_grid = np.meshgrid(B_vals, N_vals)

grid_df = pd.DataFrame({
    "B_PSH": B_grid.ravel(),
    "N_pv1": N_grid.ravel()
})

# Add interaction term automatically handled by formula
p_grid = logit_model.predict(grid_df)

P = p_grid.values.reshape(B_grid.shape)

plt.figure()

# Smooth heatmap
contour = plt.contourf(
    B_grid, N_grid, P,
    levels=50,
    cmap="viridis"
)

cbar = plt.colorbar(contour, label=r"$P(\epsilon_{PSH} = 3.17 \times 10^{-5})$")

p_star_mine = round(p_star, 2)  # your 0.12
print('Confirming, my p star is: ' + str(p_star_mine))

cbar.ax.axhline(p_star_mine, color='red', linewidth=2)

# --- Add label next to the line ---
cbar.ax.text(
    -1.3, p_star,              # x>1 moves it slightly to the right of the bar
    r"$p^*$",
    color='red',
    va='center',
    ha='left',
    transform=cbar.ax.transData,
    fontsize=12
)

plt.contour(
    B_grid, N_grid, P,
    levels=[p_star_mine],
    colors='red',
    linewidths=2
)

plt.xlabel(r"$B_{PSH} \; [kW]$",fontsize=14)
plt.ylabel(r"$N_{pv1} \; [-]$",fontsize=14)
# plt.title(f"Logit Preference Surface with Decision Boundary (p* = {p_star_mine})")
plt.tight_layout()
plt.savefig('Figure_A11b.pdf')
plt.show()



# save only what you need for prediction
bundle = {
    "params": logit_model.params.to_dict(),  # coefficients
    "p_star": float(p_star),
    "eps_on": 3.17e-5,
    "eps_off": 0.0
}

'''
with open("eps_logit_bundle.pkl", "wb") as f:
    pickle.dump(bundle, f)
'''


# Saving the logit model for use in another function
"""
import pickle

with open("logit_eps_choice.pkl", "wb") as f:
    pickle.dump(logit_model, f)
"""


# Now I need to get the arrays that I am basing the termination criteria off of
pref_df["t_convergence"] = np.ones(40)*(-69)
pref_df["nodes_convergence"] = np.ones(40)*(-69)
pref_df["gap_convergence"] = np.ones(40)*(-69)

for i in range(40):
    if pref_df["eps_policy"][i] == 0:
        pref_df.loc[i,"t_convergence"] = pref_df["t_primal_2100_0.0"][i]
        pref_df.loc[i,"nodes_convergence"] = pref_df["nodes_primal_2100_0.0"][i]
        pref_df.loc[i,"gap_convergence"] = pref_df["gap_primal_2100_0.0"][i]
    elif pref_df["eps_policy"][i] == 3.17e-5:
        pref_df.loc[i,"t_convergence"] = pref_df["t_primal_2100_3.17e-05"][i]
        pref_df.loc[i,"nodes_convergence"] = pref_df["nodes_primal_2100_3.17e-05"][i]
        pref_df.loc[i,"gap_convergence"] = pref_df["gap_primal_2100_3.17e-05"][i]
    else:
        print('ruh roh raggy')





# This down is all just to set termination criteria
times_2100_array = np.array(pref_df['t_convergence'])
nodes_2100_array = np.array(pref_df['nodes_convergence'])
nodes_2100_array = nodes_2100_array[nodes_2100_array > 0]
duality_gaps_2100_array = np.array(pref_df['gap_convergence'])


nodes_termination_option1 = np.mean(nodes_2100_array)
nodes_termination_option2 = np.mean(nodes_2100_array) + np.std(nodes_2100_array)
nodes_termination_option3 = np.percentile(nodes_2100_array,90)

duality_gaps_termination_option1 = np.mean(duality_gaps_2100_array)
duality_gaps_termination_option2 = np.mean(duality_gaps_2100_array) - np.std(duality_gaps_2100_array)
duality_gaps_termination_option3 = np.percentile(duality_gaps_2100_array,10)

times_termination_option1 = np.mean(times_2100_array)
times_termination_option2 = np.mean(times_2100_array) + np.std(times_2100_array)
times_termination_option3 = np.percentile(times_2100_array,90)

# NOTE: THESE TERMINATION CRITERIA ARE IMPLEMENTED IN THE 
# functions_milp_reversible_unequal_turbines_simplified.py FILE IN 
# Documents --> Desalination Paper --> Code_Analysis --> functions folder

