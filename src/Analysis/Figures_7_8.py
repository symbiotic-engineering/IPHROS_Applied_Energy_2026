# -*- coding: utf-8 -*-
"""
Convergence Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


n_generations = 100
n_dv_vars = 23

# =========================
# Read in design variables, constraints, and objectives
# =========================
with open("../../data/GA-MILP/AWS_GA_data.pkl", "rb") as f:
    data = pickle.load(f)

all_design_variables_dict = data["all_design_variables_dict"]
design_variable_global_max_array = data["design_variable_global_max_array"]
design_variable_global_min_array = data["design_variable_global_min_array"]
design_variable_global_range_array = data["design_variable_global_range_array"]

all_constraints_dict = data["all_constraints_dict"]
all_objectives_dict = data["all_objectives_dict"]


# =========================
# Some helper functions for future analysis
# =========================
def compute_cv_array(G):
    """
    Compute per-design total constraint violation from a constraint matrix G.

    Parameters
    ----------
    G : numpy array, shape (n_designs, n_constraints)
        Raw inequality constraint values, where feasible means G <= 0.

    Returns
    -------
    cv : ndarray, shape (n_designs,)
        Total constraint violation for each design.
    """

    # Positive parts only contribute to violation
    cv_helper = np.maximum(G,0.0)
    cv = cv_helper.sum(axis=1)
    
    # Average cv value
    cv_avg = np.mean(cv)
    
    return cv, cv_avg



def feasible_objectives_by_generation(J,cv_gen_i):
    """
    Only consider designs that are feasible.

    Parameters
    ----------
    J : numpy array, shape (n_designs, )
        Raw objective values, where negative is good (will be making these positive at the end of the function).
                                                      
    cv_gen_i : numpy array, shape (n_designs, )
        Constraint violation values for each design in generation i

    Returns
    -------
    npv_avg: the average value of feasible IPRHOS designs
    npv_best: the best feasible objective value
    """
    feasible_designs = []

    for i in range(100):
        if cv_gen_i[i] <= 0:
            feasible_designs.append(J[i])       
    
    if len(feasible_designs) > 0:
        
        feasible_designs = np.array(feasible_designs)
        feasible_designs_positive = feasible_designs * -1 
        
        npv_avg = np.mean(feasible_designs_positive)
        npv_best = np.max(feasible_designs_positive)
    else:
        npv_avg = -69
        npv_best = -69

    return npv_avg, npv_best


# =========================
# Plot objective convergence plot
# =========================

cv_avg_array = []
npv_avg_array = []
npv_best_array = []

for gen in range(100):
    G = all_constraints_dict[str(gen)]
    J = all_objectives_dict[str(gen)]
    
    cv_gen_i, cv_avg_gen_i = compute_cv_array(G)
    
    npv_avg_gen_i, npv_best_gen_i = feasible_objectives_by_generation(J,cv_gen_i)
    
    cv_avg_array.append(cv_avg_gen_i)
    if npv_best_gen_i != -69:
        npv_avg_array.append(npv_avg_gen_i)
        npv_best_array.append(npv_best_gen_i)
    
generation = np.linspace(0,99,100)+1

fig, ax1 = plt.subplots()

# Left axis → Objective
ax1.plot(generation, np.insert(np.array(npv_best_array),0,np.nan), linewidth=2, label=r'$Best \; NPV$')
ax1.plot(generation, np.insert(np.array(npv_avg_array),0,np.nan), linewidth=2, label=r'$Average \; NPV$')
ax1.set_xlabel(r"$Generation$",fontsize=14)
ax1.set_ylabel(r"$NPV \; [2023\;€]$",fontsize=14)
ax1.tick_params(axis='both', labelsize=11)

# Right axis → Constraint Violation
ax2 = ax1.twinx()
ax2.plot(generation, cv_avg_array, color='green', linewidth=2, label=r'$Avgerage \; Constraint \; Violation$')
ax2.set_ylabel(r"$Average \; Constraint \; Violation$",fontsize=14)
ax2.set_yscale("log")
ax2.tick_params(axis='both', labelsize=11)

handles_part1, labels_part1 = ax1.get_legend_handles_labels()
handles_part2, labels_part2 = ax2.get_legend_handles_labels()
handles = handles_part1+handles_part2
labels = labels_part1+labels_part2
fig.legend(handles, labels,bbox_to_anchor=(0.845, 0.41),frameon=True,fontsize=11)

plt.tight_layout()   
plt.savefig('Figure_7.pdf')        
plt.show()


# =========================
# Compute normalized diversity per generation
# =========================
diversity_array = np.ones(100)*(-69)
all_diversities_matrix = np.ones((100,23))*(-69)
top_spread = []

for gen in range(n_generations):
    design_variables_gen_i = all_design_variables_dict[str(gen)]

    # Standard deviation of each variable within this generation
    std_array_gen_i = np.std(design_variables_gen_i, axis=0, ddof=0)

    # Normalize by global variable range
    normalized_std_array_gen_i = std_array_gen_i / design_variable_global_range_array

    # If a variable never changes at all globally, treat its contribution as 0
    #normalized_std_array_gen_i = normalized_std_array_gen_i.fillna(0.0)

    # Average across variables
    diversity_gen_i = normalized_std_array_gen_i.mean()

    # Store
    diversity_array[gen] = diversity_gen_i
    all_diversities_matrix[gen,:] = normalized_std_array_gen_i
    
    # For the last generation, get values for box plots of the best 10% of designs
    if gen == n_generations-1:
        top10_final = design_variables_gen_i[:10,:] # top 10 designs
        top10_final_normalized = (top10_final-design_variable_global_min_array)/design_variable_global_range_array
    
    # Spread of the top 10% vs generation (are the elite designs stills changing)
    top10 = design_variables_gen_i[:10,:] # top 10 designs
    top10_std = np.std(top10, axis=0)
    top10_spread = np.mean(top10_std)
    top_spread.append(top10_spread)
    

# =========================
# Plot normalized diversity
# =========================
x_array = np.linspace(0,99,100)+1
plt.figure()#figsize=(8, 5))
plt.plot(x_array, diversity_array, linewidth=2)
plt.xlabel(r"$Generation$",fontsize=14)
plt.ylabel(r"$Normalized \; Diversity$",fontsize=14)
#plt.title("Normalized Diversity per Generation")
#plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_8.pdf')
plt.show()




