import scipy.io as spio
import matplotlib.pyplot as plt
import csv
import ast
from pysit_extensions.elastic_solver.boundary_integral_helper import *

cda_mat = spio.loadmat('greens_out/cda.mat') #pysit
el_mat  = spio.loadmat('greens_out/el.mat')  #elastic

if cda_mat['perturb'] != el_mat['perturb']:
    raise Exception("perturbation is not consistent") 

c_shot                  = el_mat['c_shot'][0][0]
c_rec                   = el_mat['c_rec'][0][0]
int_sc_bound_p_el       = el_mat['int_sc_bound_p']
greens_to_bound_el      = el_mat['greens_to_bound']
t_arr_greens_el         = el_mat['t_arr'][0]
full_domain_scatter_el  = el_mat['full_domain_scatter'][0]
el_dt                   = t_arr_greens_el[1] - t_arr_greens_el[0]

greens_to_bound_cda     = cda_mat['greens_to_bound']
int_sc_bound_p_cda      = cda_mat['int_sc_bound_p']
cda_dt                  = cda_mat['dt'][0]
full_domain_scatter_cda = cda_mat['full_domain_scatter'][0]

trunc_geom_dict_el  = {}
for key, val in csv.reader(open("greens_out/trunc_geom_dict_el.csv")):
    trunc_geom_dict_el[key]  = ast.literal_eval(val) #Turn back into int or float respectively

trunc_geom_dict_cda = {}
for key, val in csv.reader(open("greens_out/trunc_geom_dict_cda.csv")):
    trunc_geom_dict_cda[key] = ast.literal_eval(val) #Turn back into int or float respectively

#DO CORRECTIONS TO MAKE EL CLOSE TO CDA
full_domain_scatter_el *= c_shot**2
greens_to_bound_el     *= c_rec**2
int_sc_bound_p_el      *= c_shot**2
dt_el_greens            = t_arr_greens_el[1] - t_arr_greens_el[0]
center_integral_cda  = compute_acoustic_boundary_integral(trunc_geom_dict_cda, greens_to_bound_cda, int_sc_bound_p_cda, deriv = 'center')
center_integral_el   = compute_acoustic_boundary_integral(trunc_geom_dict_el , greens_to_bound_el , int_sc_bound_p_el , deriv = 'center', greens_el = True, dt_green = dt_el_greens)

#EXTRA CORRECTIONS
#Shift the cda result to the left by cda_dt.
#In the EL code the source acts directly at t = 0, while in CDA it needs cda_dt to propagate
full_domain_scatter_cda = full_domain_scatter_cda[1:]
center_integral_cda     = center_integral_cda[1:]

plt.plot(cda_dt*np.arange(full_domain_scatter_cda.size) , full_domain_scatter_cda, 'k', label='$p_{sc}$ full domain CDA')
plt.plot( el_dt*np.arange(full_domain_scatter_el.size)  , full_domain_scatter_el , 'r', label='$p_{sc}$ full domain EL')
plt.plot(cda_dt*np.arange(center_integral_cda.size)     , center_integral_cda    , 'b', label='$p_{sc}$ center deriv CDA')
plt.plot( el_dt*np.arange(center_integral_el.size)      , center_integral_el     , 'g', label='$p_{sc}$ center deriv EL')
plt.legend()
plt.show()


