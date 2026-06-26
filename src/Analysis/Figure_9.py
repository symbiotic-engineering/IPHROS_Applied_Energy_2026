# -*- coding: utf-8 -*-
"""
Note: I also put the energy prices in here too just in case I want them later
"""

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt


#----------------------------------------------
# Load in Raw Data
#----------------------------------------------
with open("../../data/Optimal IPHROS/AWS_turbo_dicts.pkl", "rb") as f:
    dict_turbo_RO_convergence, dict_LP_efficiencies_convergence, dict_MILP_post_processing, dict_big_convergence = pickle.load(f)
    
x_MILP_season0 = pd.read_csv('../../data/Optimal IPHROS/x_MILP_season0.csv',header=None).to_numpy()
x_MILP_season1 = pd.read_csv('../../data/Optimal IPHROS/x_MILP_season1.csv',header=None).to_numpy()
x_MILP_season2 = pd.read_csv('../../data/Optimal IPHROS/x_MILP_season2.csv',header=None).to_numpy()
x_MILP_season3 = pd.read_csv('../../data/Optimal IPHROS/x_MILP_season3.csv',header=None).to_numpy()


S_sw = 37.157
rho_sw = 1026.311


#%% Spring
energy_prices_spring = np.array([
    0.13181929, 0.12878071, 0.1275,     0.12684143, 0.13030571, 0.13941214,
    0.15305714, 0.17593286, 0.15322071, 0.17227786, 0.16057714, 0.156655,
    0.15346286, 0.11335857, 0.10445786, 0.10608286, 0.11597,    0.16552,
    0.19589786, 0.23200857, 0.23435857, 0.18546429, 0.17071929, 0.14342143,
    0.13377643, 0.13001714, 0.13007286, 0.126685,   0.13280786, 0.14116571,
    0.15359,    0.18088,    0.16186857, 0.19044857, 0.18564143, 0.18095,
    0.17583786, 0.12746,    0.11501929, 0.11457071, 0.12050357, 0.17337429,
    0.20219857, 0.22955071, 0.23485786, 0.18539429, 0.16947643, 0.14020357,
    0.12987308, 0.12523308, 0.12635308, 0.12469923, 0.12834615, 0.13673,
    0.14815231, 0.17254154, 0.15598615, 0.17948462, 0.17059231, 0.16624846,
    0.16274385, 0.11532308, 0.10811077, 0.1088,     0.11528769, 0.16876154,
    0.19526154, 0.21868308, 0.23010769, 0.18221154, 0.16515923, 0.13593308,
    0.12842,    0.12591154, 0.12422769, 0.12477923, 0.12705846, 0.13434538,
    0.14644615, 0.16831077, 0.15389846, 0.17320462, 0.16444308, 0.15617385,
    0.15184923, 0.10726538, 0.10316308, 0.10373,    0.11097923, 0.16182231,
    0.19360769, 0.21253692, 0.22401692, 0.17743615, 0.16527692, 0.13723538,
    0.12611077, 0.12377385, 0.12492385, 0.11955231, 0.12129077, 0.12835692,
    0.13880923, 0.16293231, 0.14465769, 0.17383923, 0.16239385, 0.15785077,
    0.15163077, 0.10142231, 0.09393462, 0.09342077, 0.10389615, 0.15986385,
    0.18883231, 0.21356538, 0.21989462, 0.17673231, 0.16365692, 0.13902615,
    0.12585769, 0.12454308, 0.12143231, 0.12257077, 0.12369308, 0.12448538,
    0.12632769, 0.12416231, 0.11414154, 0.08893385, 0.07860231, 0.07374231,
    0.06614692, 0.05871,    0.05540769, 0.05231154, 0.05907923, 0.07618846,
    0.10667923, 0.12775385, 0.13874846, 0.13570154, 0.12885,    0.1225,
    0.11711923, 0.11526308, 0.11441769, 0.11221385, 0.11113385, 0.11216846,
    0.10851308, 0.10239077, 0.08127615, 0.05928692, 0.05262385, 0.05074923,
    0.04715154, 0.04518,    0.04333769, 0.04460769, 0.05111154, 0.06855154,
    0.09172538, 0.12746385, 0.14677692, 0.14473154, 0.13034769, 0.13416769
])

S_ht_max_array_spring = np.array([38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.52112244,38.65320476,38.7074187,38.68177763,38.5797322,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.61113667,
 38.86379145,39.05720684,39.17929499,39.22277898,39.18568718,39.07148626,
 38.88884365,38.65103561,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.51247705,38.78718121,
 39.0184762,39.19031546,39.29064132,39.31220729,39.25307239,39.11673413,
 38.91189091,38.65185001,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.66603612,38.82021479,38.90339769,38.90836228,38.83319082,38.68140325,
 38.5,38.5,38.5,38.5,38.5,38.5       
 ])


#%% Summer

energy_prices_summer = np.array([
    0.13911857, 0.13428857, 0.13194286, 0.13028214, 0.13179786, 0.14429357,
    0.15581357, 0.18314,    0.16937786, 0.20280071, 0.19519786, 0.19394571,
    0.19127929, 0.14138857, 0.13780357, 0.13889,    0.15044857, 0.20821929,
    0.22390143, 0.23683429, 0.24077571, 0.19161214, 0.18095714, 0.14943714,
    0.14350214, 0.13733429, 0.137225,   0.13601143, 0.14033429, 0.14911071,
    0.15893071, 0.18263286, 0.16860071, 0.20301786, 0.19224357, 0.19122143,
    0.18779571, 0.13900143, 0.13882857, 0.14020143, 0.146945,   0.20297357,
    0.22004286, 0.23666643, 0.24013786, 0.19108571, 0.18262,    0.15131286,
    0.14368429, 0.138075,   0.13597143, 0.13315929, 0.13509071, 0.14457071,
    0.15295357, 0.17766071, 0.16475071, 0.19949071, 0.19341786, 0.19070357,
    0.18648714, 0.13483,    0.131785,   0.13402857, 0.13958357, 0.19872214,
    0.21684071, 0.23187643, 0.23341714, 0.18463,    0.17814571, 0.14930643,
    0.14311846, 0.13893769, 0.13631538, 0.13544538, 0.13537231, 0.13800385,
    0.13745538, 0.13328077, 0.12271231, 0.11078308, 0.09381308, 0.08865077,
    0.08461077, 0.07882385, 0.07623769, 0.08295077, 0.09482,    0.11385846,
    0.13674231, 0.15334769, 0.16391769, 0.16210769, 0.15591,    0.14606385,
    0.13673615, 0.13125462, 0.12918923, 0.12724846, 0.12738154, 0.12944154,
    0.12523385, 0.11134769, 0.09322308, 0.07166692, 0.05734923, 0.05343308,
    0.05125692, 0.05023231, 0.04810154, 0.05298769, 0.06572692, 0.08357923,
    0.11905615, 0.14739846, 0.15800846, 0.15752538, 0.15101385, 0.13845615,
    0.13257385, 0.12435538, 0.12316231, 0.12313,    0.12911615, 0.14112462,
    0.15324615, 0.17874231, 0.16651538, 0.19763077, 0.18802615, 0.18637615,
    0.18308231, 0.13254462, 0.13015,    0.13050923, 0.14173308, 0.20081077,
    0.21733769, 0.23611769, 0.24513077, 0.18985385, 0.18122846, 0.15062154,
    0.14215769, 0.13638385, 0.13420615, 0.13083308, 0.13420769, 0.14618615,
    0.15356615, 0.17799077, 0.16074923, 0.18999385, 0.18496615, 0.18258,
    0.18106,    0.13670615, 0.13423538, 0.13911462, 0.14655769, 0.19831692,
    0.21784231, 0.23253462, 0.23827,    0.18933385, 0.17876154, 0.14558308
])

S_ht_max_array_summer = np.array([38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.55231302,38.60333405,
 38.58615169,38.50522649,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.76484656,39.01969603,39.23162692,39.38794754,39.47964443,39.50199436,
 39.45488403,39.34281586,39.17459932,38.96274896,38.72263009,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.69662367,
 38.93487677,39.14548096,39.31299066,39.42477629,39.47188669,39.44966097,
 39.35804818,39.20161299,38.98922691,38.73346628,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.51599542,38.53488152,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5       
 ])


#%% Fall

energy_prices_fall = np.array([
    0.11037385, 0.10598077, 0.10453154, 0.10330077, 0.10252615, 0.10451692,
    0.10754923, 0.10901,    0.10800154, 0.09771846, 0.08660462, 0.07836154,
    0.07766231, 0.07520385, 0.07574769, 0.08308462, 0.09625846, 0.11659462,
    0.13164923, 0.14256846, 0.13986154, 0.13073,    0.12433692, 0.12464692,
    0.12290308, 0.11943,    0.11816,    0.11775385, 0.11760462, 0.11765385,
    0.11811385, 0.11781,    0.11239077, 0.09394538, 0.07868,    0.07428,
    0.07251077, 0.07285385, 0.07310923, 0.08447077, 0.10054308, 0.12210462,
    0.13559462, 0.14398308, 0.14563308, 0.13734538, 0.12628846, 0.12440077,
    0.12099,    0.11705308, 0.11329692, 0.11256538, 0.11279769, 0.12253077,
    0.13841615, 0.16967,    0.18275769, 0.17309615, 0.18848385, 0.18033,
    0.17675231, 0.15715154, 0.12915769, 0.13779462, 0.15039154, 0.19253231,
    0.24434308, 0.25531846, 0.24068231, 0.20614538, 0.16799462, 0.14787154,
    0.12866077, 0.12530077, 0.12278692, 0.12260769, 0.12383462, 0.12977077,
    0.14289462, 0.16459308, 0.17967154, 0.18032308, 0.19350154, 0.18754,
    0.18433077, 0.16394846, 0.13607923, 0.14047923, 0.15809077, 0.19529538,
    0.23555308, 0.23950154, 0.23160769, 0.20337385, 0.16075154, 0.13720231,
    0.11115077, 0.10725538, 0.10488231, 0.10118231, 0.10101692, 0.10957231,
    0.11777923, 0.13747692, 0.14968077, 0.15407923, 0.16458846, 0.15604692,
    0.15208,    0.13110154, 0.1086,     0.11422462, 0.12664846, 0.16382,
    0.19809692, 0.20248154, 0.19992,    0.17103231, 0.14294462, 0.11772769,
    0.10476083, 0.10292833, 0.102525,   0.10094917, 0.10150583, 0.108045,
    0.116375,   0.13870083, 0.15402917, 0.15380083, 0.16873917, 0.1654425,
    0.16455,    0.14880333, 0.1204025,  0.12186417, 0.1316025,  0.16338333,
    0.2017975,  0.2129175,  0.20732167, 0.18256583, 0.14633417, 0.12496417,
    0.1064525,  0.10232917, 0.10004167, 0.0973525,  0.09594,    0.10337667,
    0.11045917, 0.13022583, 0.14060333, 0.14226583, 0.1521275,  0.14728167,
    0.1463425,  0.125295,   0.1012825,  0.1052575,  0.1190075,  0.15456167,
    0.19140667, 0.20277917, 0.19727083, 0.17009333, 0.14096083, 0.1238075
])

S_ht_max_array_fall = np.array([39.24443616,39.44444616,39.5988591,39.69497613,39.7239432,39.68135763,
 39.56757161,39.38767169,39.15113592,38.87119181,38.56391893,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.55293773,38.79778898,38.98546469,39.10698275,39.15781106,
 39.13817114,39.05301787,38.91169679,38.72730225,38.5157802,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.74551592,39.04096434,
 39.32403428,39.57707101,39.78432519,39.93302406,40.01424004,40.02349739,
 39.96107576,39.83198989,39.64564674,39.41520364,39.15667092,38.88782047,
 38.62697438,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.53590099,38.77223091,39.01450426
 ])


#%% Winter

energy_prices_winter = np.array([
    0.14483615, 0.13315231, 0.12796462, 0.12801769, 0.12756,    0.13049538,
    0.14547846, 0.15816231, 0.19891692, 0.18881692, 0.21962077, 0.20515923,
    0.20165538, 0.19790385, 0.14940154, 0.14899846, 0.15970154, 0.18128615,
    0.24209692, 0.27062615, 0.26939615, 0.25205692, 0.18884154, 0.17553154,
    0.14526154, 0.13822846, 0.13642615, 0.13423692, 0.13136385, 0.13547308,
    0.14759846, 0.16229692, 0.19802385, 0.18322077, 0.20847,    0.20152385,
    0.19607769, 0.19058769, 0.13943308, 0.13915308, 0.14739538, 0.16565846,
    0.23267385, 0.24818077, 0.24939769, 0.23628769, 0.17807769, 0.16428385,
    0.13190923, 0.12739692, 0.12430538, 0.11971308, 0.11869077, 0.12406615,
    0.13590231, 0.14856308, 0.17813615, 0.16742923, 0.18887769, 0.17916769,
    0.17243462, 0.16564538, 0.12718154, 0.12565769, 0.13467769, 0.16040538,
    0.21312692, 0.23506308, 0.23992692, 0.22862385, 0.17593923, 0.16459846,
    0.13830538, 0.13055308, 0.12706923, 0.12284769, 0.12163923, 0.11997769,
    0.11956077, 0.12471692, 0.12596077, 0.12186,    0.11112538, 0.10330077,
    0.09915154, 0.09519538, 0.09257538, 0.09152769, 0.10246538, 0.11989538,
    0.13906308, 0.15438231, 0.16064308, 0.15177846, 0.14112538, 0.13341385,
    0.12654077, 0.12096154, 0.12093923, 0.12018,    0.11831462, 0.11882,
    0.11862462, 0.11813846, 0.11930308, 0.10907385, 0.09118923, 0.08649769,
    0.08518615, 0.08442308, 0.08346154, 0.08270769, 0.08998077, 0.11099615,
    0.13718923, 0.15585692, 0.16383077, 0.15811385, 0.14252231, 0.13032923,
    0.12882667, 0.12553583, 0.12508417, 0.12295917, 0.12044583, 0.12758417,
    0.1340875,  0.15503833, 0.19415167, 0.18315333, 0.21500417, 0.210085,
    0.20398667, 0.20086333, 0.15071833, 0.15137,    0.16005083, 0.17162167,
    0.24207917, 0.26423167, 0.2668875,  0.248365,   0.18309667, 0.17140667,
    0.14496333, 0.13916583, 0.13348,    0.13193667, 0.13111583, 0.13489583,
    0.1451325,  0.1630225,  0.1972925,  0.18556917, 0.21343083, 0.20482,
    0.20028417, 0.19733833, 0.15075,    0.14954,    0.15827917, 0.175755,
    0.24269333, 0.2709725,  0.27112167, 0.25499917, 0.19295083, 0.17740583
])

S_ht_max_array_winter = np.array([38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.51893209,38.70741515,
 38.82577583,38.86751257,38.83129788,38.72106665,38.54574598,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.63060922,38.85545322,39.02093892,
 39.11586376,39.13375894,39.07333032,38.93854666,38.73836939,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.58932793,38.73097011,
 38.80245698,38.79735252,38.71439475,38.55758394,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5,
 38.5,38.5,38.5,38.5,38.5,38.5
 ])


#%% Reservoir levels

V_res_spring = x_MILP_season0[:,15]
V_res_summer = x_MILP_season1[:,15]
V_res_fall = x_MILP_season2[:,15]
V_res_winter = x_MILP_season3[:,15]

V_res_max = 28635496
V_res_min = 0.3*V_res_max
V_res_init = 0.65*V_res_max

V_res_spring = np.insert(V_res_spring, 0, V_res_init)
V_res_summer = np.insert(V_res_summer, 0, V_res_init)
V_res_fall = np.insert(V_res_fall, 0, V_res_init)
V_res_winter = np.insert(V_res_winter, 0, V_res_init)



#%% Bundle reservoir values and save
reservoir_level_dict = {'spring':V_res_spring,
                        'summer':V_res_summer,
                        'fall':V_res_fall,
                        'winter':V_res_winter
                        }

# Save the dictionaries for later usage
#with open("reservoir_levels_dict.pkl", "wb") as f:
#    pickle.dump(reservoir_level_dict, f)


#%% Other normalization parameters
elec_max = np.max([np.max(energy_prices_spring),np.max(energy_prices_summer),
                   np.max(energy_prices_fall),np.max(energy_prices_winter)])

elec_min = np.min([np.min(energy_prices_spring),np.min(energy_prices_summer),
                   np.min(energy_prices_fall),np.min(energy_prices_winter)])

S_ht_max_limit = 42
S_ht_min_limit = 38.5


#%% Normalize all the different signals
V_res_spring_norm = (V_res_spring-V_res_min)/(V_res_max-V_res_min)
V_res_summer_norm = (V_res_summer-V_res_min)/(V_res_max-V_res_min)
V_res_fall_norm = (V_res_fall-V_res_min)/(V_res_max-V_res_min)
V_res_winter_norm = (V_res_winter-V_res_min)/(V_res_max-V_res_min)

energy_prices_spring_norm = (energy_prices_spring-elec_min)/(elec_max-elec_min)
energy_prices_summer_norm = (energy_prices_summer-elec_min)/(elec_max-elec_min)
energy_prices_fall_norm = (energy_prices_fall-elec_min)/(elec_max-elec_min)
energy_prices_winter_norm = (energy_prices_winter-elec_min)/(elec_max-elec_min)

S_ht_max_array_spring_norm = (S_ht_max_array_spring-S_ht_min_limit)/(S_ht_max_limit-S_ht_min_limit)
S_ht_max_array_summer_norm = (S_ht_max_array_summer-S_ht_min_limit)/(S_ht_max_limit-S_ht_min_limit)
S_ht_max_array_fall_norm = (S_ht_max_array_fall-S_ht_min_limit)/(S_ht_max_limit-S_ht_min_limit)
S_ht_max_array_winter_norm = (S_ht_max_array_winter-S_ht_min_limit)/(S_ht_max_limit-S_ht_min_limit)


#%% Plot salinity policy and electricity price and reservoir level on the same plot (a 2x2 plot)

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

axes[0,0].plot(S_ht_max_array_spring_norm,label='Discharge Salinity')
axes[0,0].plot(V_res_spring_norm,label='Reservoir Volume')
axes[0,0].plot(energy_prices_spring_norm,label='Electricity Price')
axes[0,0].set_title('Spring')

axes[0,1].plot(S_ht_max_array_summer_norm,label='Discharge Salinity')
axes[0,1].plot(V_res_summer_norm,label='Reservoir Volume')
axes[0,1].plot(energy_prices_summer_norm,label='Electricity Price')
axes[0,1].set_title('Summer')

axes[1,0].plot(S_ht_max_array_fall_norm,label='Discharge Salinity')
axes[1,0].plot(V_res_fall_norm,label='Reservoir Volume')
axes[1,0].plot(energy_prices_fall_norm,label='Electricity Price')
axes[1,0].set_title('Fall')

axes[1,1].plot(S_ht_max_array_winter_norm,label='Discharge Salinity')
axes[1,1].plot(V_res_winter_norm,label='Reservoir Volume')
axes[1,1].plot(energy_prices_winter_norm,label='Electricity Price')
axes[1,1].set_title('Winter')

# Leave space at bottom manually
fig.subplots_adjust(bottom=0.18)

# Shared axis labels
fig.supxlabel(r"$Time \; [hr]$",fontsize=14, y=0.09)
fig.supylabel(r"$Normalized \; Value$",fontsize=14)

# One legend for entire figure (cleaner than 4 legends)
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3,bbox_to_anchor=(0.5, 0.01),frameon=True)

plt.tight_layout(rect=[0, 0.08, 1, 1])  # leave space for legend
plt.show()


#%% Calculate S_ht_arrays for each season
S_oRO_array_spring = dict_turbo_RO_convergence['season0']['big_iteration2']['turbo_RO_iteration0']['RO_eval0']['S_oRO_stage2_array']
rho_oRO_array_spring = dict_turbo_RO_convergence['season0']['big_iteration2']['turbo_RO_iteration0']['RO_eval0']['rho_oRO_stage2_array']

S_oRO_array_summer = dict_turbo_RO_convergence['season1']['big_iteration2']['turbo_RO_iteration0']['RO_eval0']['S_oRO_stage2_array']
rho_oRO_array_summer = dict_turbo_RO_convergence['season1']['big_iteration2']['turbo_RO_iteration0']['RO_eval0']['rho_oRO_stage2_array']

S_oRO_array_fall = dict_turbo_RO_convergence['season2']['big_iteration2']['turbo_RO_iteration0']['RO_eval0']['S_oRO_stage2_array']
rho_oRO_array_fall = dict_turbo_RO_convergence['season2']['big_iteration2']['turbo_RO_iteration0']['RO_eval0']['rho_oRO_stage2_array']

S_oRO_array_winter = dict_turbo_RO_convergence['season3']['big_iteration2']['turbo_RO_iteration1']['RO_eval1']['S_oRO_stage2_array']
rho_oRO_array_winter = dict_turbo_RO_convergence['season3']['big_iteration2']['turbo_RO_iteration1']['RO_eval1']['rho_oRO_stage2_array']

V_dot_oRO_array_spring = dict_MILP_post_processing['season0']['big_iteration2']['V_dot_oRO_stage2_array']
V_dot_oRO_array_summer = dict_MILP_post_processing['season1']['big_iteration2']['V_dot_oRO_stage2_array']
V_dot_oRO_array_fall = dict_MILP_post_processing['season2']['big_iteration2']['V_dot_oRO_stage2_array']
V_dot_oRO_array_winter = dict_MILP_post_processing['season3']['big_iteration2']['V_dot_oRO_stage2_array']

V_dot_swht_matrix_spring = x_MILP_season0[:,7:14]
V_dot_swht_matrix_summer = x_MILP_season1[:,7:14]
V_dot_swht_matrix_fall = x_MILP_season2[:,7:14]
V_dot_swht_matrix_winter = x_MILP_season3[:,7:14]

V_dot_swht_array_spring = np.sum(V_dot_swht_matrix_spring,axis=1)
V_dot_swht_array_summer = np.sum(V_dot_swht_matrix_summer,axis=1)
V_dot_swht_array_fall = np.sum(V_dot_swht_matrix_fall,axis=1)
V_dot_swht_array_winter = np.sum(V_dot_swht_matrix_winter,axis=1)

# The calculation
S_ht_spring = (((S_oRO_array_spring*V_dot_oRO_array_spring*rho_oRO_array_spring)+(S_sw*V_dot_swht_array_spring*rho_sw))/
               ((V_dot_oRO_array_spring*rho_oRO_array_spring)+(V_dot_swht_array_spring*rho_sw)))
S_ht_summer = (((S_oRO_array_summer*V_dot_oRO_array_summer*rho_oRO_array_summer)+(S_sw*V_dot_swht_array_summer*rho_sw))/
               ((V_dot_oRO_array_summer*rho_oRO_array_summer)+(V_dot_swht_array_summer*rho_sw)))
S_ht_fall = (((S_oRO_array_fall*V_dot_oRO_array_fall*rho_oRO_array_fall)+(S_sw*V_dot_swht_array_fall*rho_sw))/
               ((V_dot_oRO_array_fall*rho_oRO_array_fall)+(V_dot_swht_array_fall*rho_sw)))
S_ht_winter = (((S_oRO_array_winter*V_dot_oRO_array_winter*rho_oRO_array_winter)+(S_sw*V_dot_swht_array_winter*rho_sw))/
               ((V_dot_oRO_array_winter*rho_oRO_array_winter)+(V_dot_swht_array_winter*rho_sw)))


#%% Make a plot of the output salinity as an active constraint
x_array = np.linspace(0,167,168)+1

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)

nan_mask_spring = np.isnan(S_ht_spring)
valid_mask_spring = ~nan_mask_spring
axes[0,0].step(x_array, S_ht_max_array_spring, where='mid', color='red', label=r'$S_{ht,max}$')
axes[0,0].axhline(y=S_sw,color='orange', linestyle='--',label=r'$S_{sw}$')
axes[0,0].bar(x_array[valid_mask_spring], S_ht_spring[valid_mask_spring], width=1.0, label=r'$S_{ht}$')
axes[0,0].bar(
    x_array[nan_mask_spring],
    [40.5]*np.sum(nan_mask_spring),  # full height
    width=1.0,
    color='none',
    edgecolor='black',
    hatch='//',
    alpha=0.5,
    label=r'$Inactive \; PSH \; Generation \; and \; RO$'
)
axes[0,0].set_title(r'$Spring$')
axes[0,0].set_ylim([37,40.5])
axes[0,0].set_xticks(np.arange(0, max(x_array)+2, 24))  # ticks every 24

nan_mask_summer = np.isnan(S_ht_summer)
valid_mask_summer = ~nan_mask_summer
axes[0,1].step(x_array, S_ht_max_array_summer, where='mid', color='red', label=r'$S_{ht,max}$')
axes[0,1].axhline(y=S_sw,color='orange', linestyle='--',label=r'$S_{sw}$')
axes[0,1].bar(x_array[valid_mask_summer], S_ht_summer[valid_mask_summer], width=1.0, label=r'$S_{ht}$')
axes[0,1].bar(
    x_array[nan_mask_summer],
    [40.5]*np.sum(nan_mask_summer),  # full height
    width=1.0,
    color='none',
    edgecolor='black',
    hatch='//',
    alpha=0.5,
    label=r'$Inactive \; PSH \; Generation \; and \; RO$'
)
axes[0,1].set_title(r'$Summer$')
axes[0,1].set_ylim([37,40.5])
axes[0,1].set_xticks(np.arange(0, max(x_array)+2, 24))  # ticks every 24

nan_mask_fall = np.isnan(S_ht_fall)
valid_mask_fall = ~nan_mask_fall
axes[1,0].step(x_array, S_ht_max_array_fall, where='mid', color='red', label=r'$S_{ht,max}$')
axes[1,0].axhline(y=S_sw,color='orange', linestyle='--',label=r'$S_{sw}$')
axes[1,0].bar(x_array[valid_mask_fall], S_ht_fall[valid_mask_fall], width=1.0, label=r'$S_{ht}$')
axes[1,0].bar(
    x_array[nan_mask_fall],
    [40.5]*np.sum(nan_mask_fall),  # full height
    width=1.0,
    color='none',
    edgecolor='black',
    hatch='//',
    alpha=0.5,
    label=r'$Inactive \; PSH \; Generation \; and \; RO$'
)
axes[1,0].set_title(r'$Fall$')
axes[1,0].set_ylim([37,40.5])
axes[1,0].set_xticks(np.arange(0, max(x_array)+2, 24))  # ticks every 24

nan_mask_winter = np.isnan(S_ht_winter)
valid_mask_winter = ~nan_mask_winter
axes[1,1].step(x_array, S_ht_max_array_winter, where='mid', color='red', label=r'$S_{ht,max}$')
axes[1,1].axhline(y=S_sw,color='orange', linestyle='--',label=r'$S_{sw}$')
axes[1,1].bar(x_array[valid_mask_winter], S_ht_winter[valid_mask_winter], width=1.0, label=r'$S_{ht}$')
axes[1,1].bar(
    x_array[nan_mask_winter],
    [40.5]*np.sum(nan_mask_winter),  # full height
    width=1.0,
    color='none',
    edgecolor='black',
    hatch='//',
    alpha=0.5,
    label=r'$Inactive \; RO \; and \; PSH \; Generation$'
)
axes[1,1].set_title(r'$Winter$')
axes[1,1].set_ylim([37,40.5])
axes[1,1].set_xticks(np.arange(0, max(x_array)+2, 24))  # ticks every 24

# Leave space at bottom manually
fig.subplots_adjust(bottom=0.18)

# Shared axis labels
fig.supxlabel(r"$Time \; [hr]$",fontsize=14, y=0.08)
fig.supylabel(r"$Salinity \; [g/kg]$",fontsize=14)

# One legend for entire figure (cleaner than 4 legends)
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4,bbox_to_anchor=(0.5, 0.01),frameon=True)

plt.tight_layout(rect=[0, 0.06, 1, 1])  # leave space for legend
plt.savefig('Figure_9.pdf')
plt.show()

