# -*- coding: utf-8 -*-
"""
XDSM for GA-MILP Applied Energy Paper
"""


import os
from pyxdsm.XDSM import (
    XDSM,
    OPT,
    SUBOPT,
    SOLVER,
    DOE,
    IFUNC,
    FUNC,
    GROUP,
    IGROUP,
    METAMODEL,
    LEFT,
    RIGHT,
)

# Initialize the XDSM
x = XDSM(use_sfmath=True)

# Add my systems (i.e. the diagonal elements)
x.add_system("optGA",OPT,r"NSGA-II")
x.add_system("seagrassConstraint",IFUNC,r"\shortstack{Seagrass \\ Salinity \\ Constraint}")
x.add_system("epsilonPredictor",IFUNC,r"\shortstack{MILP \\ Epsilon \\ Predictor}")
x.add_system("pretreatment",IFUNC,r"Pretreatment")
x.add_system("turbocharger",IFUNC,r"Turbocharger")
x.add_system("roModel",IFUNC,r"RO \: Model")
x.add_system("optLP",OPT,r"\shortstack{MILP \\ (HiGHS) \\ (1 per Rep. Season $s$)}")
x.add_system("postMILP",IFUNC,r"\shortstack{MILP \: Post \\ Processing}")
x.add_system("efficiencies",IFUNC,r"\shortstack{Turbine \\ Efficiencies}")
x.add_system("roCapacity",IFUNC,r"RO \: Capacity")
x.add_system("yearSummator",IFUNC,r"\shortstack{Year \\ Summator}")
x.add_system("npvCost",IFUNC,r"\shortstack{NPV \: Cost}")

# Add connectors - left to right
x.connect("optGA","seagrassConstraint",[r"a_{0,s}",r"a_{k_f,s}",r"b_{k_f,s}"],label_width=2)
x.connect("optGA","epsilonPredictor",[r"B_{PSH}",r"N_{pv1}"],label_width=2)
x.connect("optGA","turbocharger",r"ERD")
x.connect("optGA","roModel",r"N_{pv1}")
x.connect("optGA","optLP",[r"B_{PSH}",r"N_{pv1}",r"ERD"],label_width=3)
x.connect("optGA","postMILP",r"N_{pv1}")
x.connect("optGA","efficiencies",[r"B_{PSH}",r"N_{pv1}",r"ERD"],label_width=2)
x.connect("optGA","roCapacity",r"N_{pv1}")
x.connect("optGA","npvCost",r"B_{PSH}")
x.connect("seagrassConstraint","optLP",r"S_{ht,h,s,adj.}")
x.connect("pretreatment","turbocharger",r"h_{turbo}")
x.connect("epsilonPredictor","optLP",r"\epsilon_{PSH}")
x.connect("turbocharger","roModel",r"h_{f,pv1,h,s}")
x.connect("turbocharger","optLP",r"h_{f,pv1,h,s}")
x.connect("roModel","optLP",[r"S_{[c,g/kg],pv2,h,s}",r"\rho_{c,pv2,h,s}",r"h_{c,pv2,h,s}",r"\eta_{RO,pv1,h,s}",r"\eta_{RO,pv2,h,s}",r"(Q_{f,pv1,h,s,iter.})"],label_width=2)
x.connect("roModel","postMILP",[r"\eta_{RO,pv1,h,s}",r"\eta_{RO,pv2,h,s}",r"\rho_{c,pv2,h,s}",r"h_{c,pv2,h,s}"],label_width=2)
x.connect("roModel","efficiencies",[r"\rho_{c,pv2,h,s}",r"h_{c,pv2,h,s}"],label_width=1)
x.connect("roModel","yearSummator",[r"\eta_{RO,pv1,h,s}",r"\eta_{RO,pv2,h,s}"])
x.connect("optLP","postMILP",[r"\dot{V}_{wp,i,h,s}^*",r"\dot{V}_{swht,i,h,s}^*",r"Q_{f,pv1,h,s}^*",r"(\eta_{hp,i,h,s}",r"\eta_{ht,i,h,s}",r"\eta_{pelton,h,s})"],label_width=2)
x.connect("optLP","efficiencies",[r"\dot{V}_{wp,i,h,s}^*",r"\dot{V}_{swht,i,h,s}^*"],label_width=1)
x.connect("optLP","yearSummator",[r"J_{MILP,s}^*"],label_width=1)
x.connect("postMILP","efficiencies",[r"\dot{V}_{oRO,h,s}",r"P_{gen,pelton,h,s}"],label_width=1)
x.connect("postMILP","roCapacity",[r"\dot{V}_{fwRO,h,s}"],label_width=1)
x.connect("postMILP","yearSummator",[r"P_{con,PSH,i,h,s}",r"P_{gen,PSH,i,h,s}"],label_width=1)
x.connect("efficiencies","yearSummator",r"\eta_{ht,i,h,s}")
x.connect("roCapacity","npvCost",r"B_{RO}")
x.connect("yearSummator","npvCost",[r"P_{con,PSH,year}",r"P_{gen,PSH,year}",r"EBIT_{year}",r"\bar{\eta}_{RO,year}",r"\bar{\eta}_{ht,year}"],label_width=1)

# Add connectors - right to left
x.connect("npvCost","optGA",r"NPV")
x.connect("roCapacity","optGA",r"B_{RO}")
x.connect("efficiencies","optLP",[r"\eta_{hp,i,h,s}",r"\eta_{ht,i,h,s}",r"\eta_{pelton,h,s}"],label_width=2)
x.connect("optLP","roModel",r"Q_{f,pv1,h,s,iter.}^*")
x.connect("optLP","optGA",r"Feas. LP_s")
x.connect("roModel","turbocharger",[r"P_{[c,psi],pv2,h,s}",r"S_{[c,g/kg],pv2,h,s}",r"\rho_{c,pv2,h,s}"])
x.connect("roModel","optGA",[r"Feas. RO_{h,s}"],label_width=1)
x.connect("seagrassConstraint","optGA",[r"f_{38.5,s}",r"f_{40,s}"],label_width=2)

# Add in external inputs
#x.add_input("optGA", r"[B_{PSH}, N_{pv1}, ERD, a_{0,s}, a_{k,s}, b_{k,s}]^{(0)}")
x.add_input("optGA",[r"[B_{PSH}",r"N_{pv1},",r"ERD",r"a_{0,s},",r"a_{k_f,s}",r"b_{k_f,s}]^{(0)}"],label_width=2)
x.add_input("epsilonPredictor",[r"Epsilon \; Logit \; Model"],label_width=1)
x.add_input("pretreatment",[r"h_{res}"],label_width=1)
x.add_input("turbocharger",[r"P_{[c,psi],pv2,h,s}^0",r"S_{[c,g/kg],pv2,h,s}^0",r"\rho_{c,pv2,h,s}^0"],label_width=1)
x.add_input("roModel",[r"Q_{f,pv1,h}^0",r"WAVE \; Model"],label_width=1)
x.add_input("optLP",[r"\lambda_{e,h,s}",r"\lambda_{w}",r"\eta_{hp,i,h,s}^0",r"\eta_{ht,i,h,s}^0",r"\eta_{pelton,h,s}^0",r"g_{\text{convex polygon}}^{red.}"],label_width=4)


# Add in outputs
x.add_output("optGA",[r"B_{PSH}^*",r"N_{pv1}^*",r"ERD^*",r"a_{0,s}^*",r"a_{k,s}^*",r"b_{k,s}^*"],side=LEFT,label_width=2)
x.add_output("optLP",[r"\dot{V}_{wp,i,h,s}^*",r"\dot{V}_{swht,i,h,s}^*",r"Q_{f,pv1,h,s}^*",r"V_{res,h,s}^*"],side=LEFT,label_width=1)
x.add_output("npvCost",r"NPV^*")

# Add Processes
x.add_process(["turbocharger", "roModel", "turbocharger"],arrow=True)
x.add_process(["optLP","efficiencies","optLP"],arrow=True)
x.add_process(["optLP","roModel","optLP"],arrow=True)

x.write("Figure_A12")
os.startfile("Figure_A12.pdf")