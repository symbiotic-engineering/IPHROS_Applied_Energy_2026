# -*- coding: utf-8 -*-
"""
Shrinking convex hull based on hull.equations
"""

from scipy.spatial import ConvexHull, HalfspaceIntersection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#------------------------------------------------------------------
# The final plots from this are shown at the end of this file
#------------------------------------------------------------------

"""
# Load in the polygon, and my variables
raw_data_df = pd.read_csv('two_stage_feasibility_points.csv')
points = raw_data_df.to_numpy()[:,1:]
hull = ConvexHull(points)

# Plot the convex hull edges
pressures = np.array([
    689.4212524,  689.4212524,  691.90043694, 691.90043694, 691.90043694,
    691.90043694, 657.5281205,  689.4212524,  657.5281205,  657.5281205,
    657.5281205,  657.5281205,  657.5281205,  657.5281205,  689.4212524,
    657.5281205,  691.90043694, 691.90043694, 691.90043694, 691.90043694,
    691.8621476,  689.4212524,  691.90043694, 657.5281205
])
flowrates = np.array([
    25697.79963547, 25697.79963547,     0.        ,     0.        ,
        0.        , 25697.79963547,     0.        , 25697.79963547,
        0.        ,     0.        ,     0.        ,     0.        ,
        0.        ,     0.        , 25697.79963547,     0.        ,
    24221.79502944, 15090.16845571, 25697.79963547, 25697.79963547,
    25697.79963547, 25697.79963547, 25697.79963547,     0.        
])/(3*610)
plt.figure()
flowrates_original = np.array([
    1.81996903e+04, 1.81996903e+04, 2.08519373e+04, 2.08519373e+04,
    2.08519373e+04, 2.08519373e+04, 0.00000000e+00, 1.81996903e+04,
    0.00000000e+00, 0.00000000e+00, 2.45716154e-11, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00, 1.81996903e+04, 0.00000000e+00,
    2.08519373e+04, 2.08519373e+04, 2.08519373e+04, 2.08519373e+04,
    2.06087786e+04, 1.81996903e+04, 2.08519373e+04, 0.00000000e+00
])/(3*610)
legend_counter = 0
for simplex in hull.simplices:
    if legend_counter == 26:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-', label="Convex Hull")
    else:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    legend_counter += 1
plt.scatter(pressures,flowrates_original,label="RO Inlet flows post turbo-RO")
plt.scatter(pressures,flowrates,label="RO Inlet flows post LP")
#plt.scatter(P_RO_stage1_array,Q_wRO_stage1_array)
# Optionally, fill the convex hull
#plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], 'lightblue', alpha=0.5)

plt.title("2D Convex Hull")
plt.xlabel("Pressure [psi]")
plt.ylabel('Flowrate [m^3/hr]')
plt.legend()

plt.show()

#'''
# Step 2: Shrink the halfspaces
delta = 0.5  # Shrink distance 0.01

A = hull.equations[:, :2]  # Normal vectors (a, b)
c = hull.equations[:, 2]   # Constant term

norms = np.linalg.norm(A, axis=1)#.reshape(-1, 1)
c_shrunk = c + delta * norms

# Each halfspace is: a*x + b*y + c <= 0 -> [a, b, c]
halfspaces = np.hstack((A, c_shrunk[:, np.newaxis]))

# Step 3: Find a feasible point in the shrunk region
# We'll use the centroid of the original hull as a feasible point guess
centroid = np.mean(points[hull.vertices], axis=0)

# Step 4: Construct and plot the shrunk hull
shrunk_hull = HalfspaceIntersection(halfspaces, centroid)
shrunk_points = shrunk_hull.intersections
shrunk_hull = ConvexHull(shrunk_points)

# Step 5: Plot original and shrunk hulls
plt.figure(figsize=(6,6))

# Original hull
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
#plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], alpha=0.2, label='Original Hull')

# Shrunk hull
for simplex in shrunk_hull.simplices:
    plt.plot(shrunk_points[simplex, 0], shrunk_points[simplex, 1], 'r-')
#plt.fill(shrunk_points[shrunk_hull.vertices, 0], shrunk_points[shrunk_hull.vertices, 1], color='red', alpha=0.4, label='Shrunk Hull')

plt.legend()
#plt.axis('equal')
plt.title("Original and Shrunk Convex Hull")
plt.xlabel("Pressure [psi]")
plt.ylabel('Flowrate [m^3/hr]')
plt.show()
#'''
"""


#---------------------------------------------------------------------------------------------------------
# FINAL PLOTTING
#---------------------------------------------------------------------------------------------------------

# Load in the polygon, and my variables
raw_data_df = pd.read_csv('../../data/RO and Miscellaneous/two_stage_feasibility_points.csv')
points = raw_data_df.to_numpy()[:,1:]
hull = ConvexHull(points)

# Plot the convex hull edges
plt.figure()
legend_counter = 0
for simplex in hull.simplices:
    if legend_counter == 26:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-', label="Convex Hull")
    else:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    legend_counter += 1
# Optionally, fill the convex hull
plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], 'lightblue', alpha=0.5)

plt.title("2D Convex Hull")
plt.xlabel("Pressure [psi]")
plt.ylabel('Flowrate [m^3/hr]')
plt.legend()

plt.show()

#'''
# Step 2: Shrink the halfspaces
delta = 0.5  # Shrink distance 0.01

A = hull.equations[:, :2]  # Normal vectors (a, b)
c = hull.equations[:, 2]   # Constant term

norms = np.linalg.norm(A, axis=1)#.reshape(-1, 1)
c_shrunk = c + delta * norms

# Each halfspace is: a*x + b*y + c <= 0 -> [a, b, c]
halfspaces = np.hstack((A, c_shrunk[:, np.newaxis]))

# Step 3: Find a feasible point in the shrunk region
# We'll use the centroid of the original hull as a feasible point guess
centroid = np.mean(points[hull.vertices], axis=0)

# Step 4: Construct and plot the shrunk hull
shrunk_hull = HalfspaceIntersection(halfspaces, centroid)
shrunk_points = shrunk_hull.intersections
shrunk_hull = ConvexHull(shrunk_points)

# Step 5: Plot original and shrunk hulls
plt.figure(figsize=(6,6))

# Original hull
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], alpha=0.2, label='Original Hull')

# Shrunk hull
for simplex in shrunk_hull.simplices:
    plt.plot(shrunk_points[simplex, 0], shrunk_points[simplex, 1], 'r-')
#plt.fill(shrunk_points[shrunk_hull.vertices, 0], shrunk_points[shrunk_hull.vertices, 1], color='red', alpha=0.4, label='Shrunk Hull')

plt.legend()
plt.axis('equal')
plt.title("Original and Shrunk Convex Hull")
plt.xlabel("Pressure [psi]")
plt.ylabel('Flowrate [m^3/hr]')
plt.show()
#'''

# This is the final plot!!
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
legend_counter = 0
for simplex in hull.simplices:
    if legend_counter == 0:
        ax[0].plot(points[simplex, 0], points[simplex, 1], 'k-', label="Convex Polygon")
    else:
        ax[0].plot(points[simplex, 0], points[simplex, 1], 'k-')
    legend_counter += 1
'''
legend_counter = 0
for simplex in shrunk_hull.simplices:
    if legend_counter == 0:
        ax[0].plot(shrunk_points[simplex, 0], shrunk_points[simplex, 1], 'r-', label="Reduced Convex Polygon")
    else:
        ax[0].plot(shrunk_points[simplex, 0], shrunk_points[simplex, 1], 'r-')
    legend_counter += 1
'''
ax[0].fill(points[hull.vertices, 0], points[hull.vertices, 1], 'lightblue', alpha=0.5)
    
legend_counter = 0
for simplex in hull.simplices:
    if legend_counter == 0:
        ax[1].plot(points[simplex, 0], points[simplex, 1], 'k-', label="Convex Polygon")
    else:
        ax[1].plot(points[simplex, 0], points[simplex, 1], 'k-')
    legend_counter += 1
legend_counter = 0
for simplex in shrunk_hull.simplices:
    if legend_counter == 0:
        ax[1].plot(shrunk_points[simplex, 0], shrunk_points[simplex, 1], 'r-', label="Reduced Convex Polygon")
    else:
        ax[1].plot(shrunk_points[simplex, 0], shrunk_points[simplex, 1], 'r-')
    legend_counter += 1
ax[1].fill(points[hull.vertices, 0], points[hull.vertices, 1], 'lightblue', alpha=0.5)

ax[1].axis('equal')
ax[1].set_xlim([665,695])
ax[1].set_ylim([0,25])

ax[0].legend()
ax[1].legend()

ax[0].set_xlabel(r'$P_{[f,psi],pv\text{x}} \; [psi]$', fontsize=14)
ax[1].set_xlabel(r'$P_{[f,psi],pv\text{x}} \; [psi]$', fontsize=14)
ax[0].set_ylabel(r'$Q_{f,pv\text{x}} \; [m^3/hr]$', fontsize=14)
ax[1].set_ylabel(r'$Q_{f,pv\text{x}} \; [m^3/hr]$', fontsize=14)
ax[0].set_title(r'$(a)$')
ax[1].set_title(r'$(b)$')
plt.tight_layout()
plt.savefig("Figure_4.pdf")



