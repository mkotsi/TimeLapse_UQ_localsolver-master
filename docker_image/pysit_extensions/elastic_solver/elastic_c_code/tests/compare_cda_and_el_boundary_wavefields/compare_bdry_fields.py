# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

from pysit import *
from pysit_extensions.elastic_solver.precompute_greens import precompute_using_cda_solver
from pysit_extensions.elastic_solver.precompute_greens import get_trunc_geom_dict, get_nr_physical
from pysit_extensions.elastic_solver.wrapping_functions import elastic_solve, get_source_wavelet
from pysit_extensions.elastic_solver.boundary_integral_helper import *
from shared_routines import *

if __name__ == '__main__':

    m, shots = compute_shot_and_mesh()

    vp_bg_2d, rho_2d, vs_2d, dt = give_background_models(m)
    
    t_min             = 0.0
    t_max_desired     = 1.0
    rec_boundary_geom = { 'rec_x_l': 200.0,  'rec_x_r': 650.0,  'rec_z_t': 200.0, 'rec_z_b': 500.0}
    trunc_geom_dict   = get_trunc_geom_dict(m, rec_boundary_geom)

    #For simplicity, look at single shot (shots[1])
    shot_nr  = 1
    shot     = shots[shot_nr]
    x_1, z_1 = shot.sources.position
    xps1 = int(np.round(x_1/m.x.delta))
    zps1 = int(np.round(z_1/m.z.delta))
    
    c_shot                     = vp_bg_2d[zps1,xps1]
    load_prefix                = 'greens_out/step1_'
    #############FIRST EL RESULTS################
    indict_el                  = spio.loadmat(load_prefix + "s_pos_x_%.2f_z_%.2f"%(x_1, z_1))
    green_rec_times_el         = indict_el['shotgathers_times'][0] 
    dt_el                      = green_rec_times_el[1]-green_rec_times_el[0]      
    
    match_pysit_el             = True #Effective wavelet passed to C code is integral of source wavelet. Doesn't matter from a test perspective
    itimestep_el               = int(np.floor((t_max_desired-t_min)/dt_el) + 1); #number of timesteps
    t_max_el                   = (itimestep_el-1) * dt_el

    green_gather_comb_el       = indict_el['shotgathers'][0] #Contains both real receivers and boundary integral receivers
    nr_physical_el             = get_nr_physical(green_gather_comb_el, trunc_geom_dict)
    wavelet_el                 = get_source_wavelet(shot, green_rec_times_el, match_pysit_el)
    
    #Convolve recorded green's functions with the wavelet to get background field.
    
    p0_shotgather_combined_el  = convolve_shotgather_with_wavelet(green_gather_comb_el, wavelet_el)    
    p0_shotgather_el           = p0_shotgather_combined_el[:itimestep_el,:nr_physical_el]    
     
    green_boundary_fields_el   = indict_el['boundary_fields'][0]
    boundary_fields_el         = convolve_boundary_fields(green_boundary_fields_el, wavelet_el, itimestep_el)    
    
    #scale results
    p0_shotgather_el          *= c_shot**2
    boundary_fields_el        *= c_shot**2

    #############NOW CDA RESULTS################
    indict_cda                 = spio.loadmat(load_prefix + "s_pos_x_%.2f_z_%.2f_CDA"%(x_1, z_1))
    green_rec_times_cda        = indict_cda['shotgathers_times'][0]
    dt_cda                     = green_rec_times_cda[1]-green_rec_times_cda[0]      
    
    match_pysit_cda            = False #no need to integrate. We generated everything with PySIT
    itimestep_cda              = int(np.floor((t_max_desired-t_min)/dt_cda) + 1); #number of timesteps
    t_max_cda                  = (itimestep_cda-1) * dt_cda

    green_gather_comb_cda      = indict_cda['shotgathers'][0] #Contains both real receivers and boundary integral receivers
    nr_physical_cda            = get_nr_physical(green_gather_comb_cda, trunc_geom_dict)
    wavelet_cda                = get_source_wavelet(shot, green_rec_times_cda, match_pysit_cda)   
    
    #Convolve recorded green's functions with the wavelet to get background field.
    
    p0_shotgather_combined_cda = convolve_shotgather_with_wavelet(green_gather_comb_cda, wavelet_cda)    
    p0_shotgather_cda          = p0_shotgather_combined_cda[:itimestep_cda,:nr_physical_cda]    
     
    green_boundary_fields_cda  = indict_cda['boundary_fields'][0]
    boundary_fields_cda        = convolve_boundary_fields(green_boundary_fields_cda, wavelet_cda, itimestep_cda)
    
    ############# MAKE COMPARISON PLOTS #############

    #CORRECTIONS
    #Shift the cda result to the left by cda_dt.
    #In the EL code the source acts directly at t = 0, while in CDA it needs cda_dt to propagate
    green_rec_times_cda = green_rec_times_cda[:-1]
    p0_shotgather_cda   = p0_shotgather_cda[1:,:]
    
    #First a trace from background shotgather
    rec_nr             = 4
    rec_of_interest    = shot.receivers.receiver_list[rec_nr]
    plt.figure(1)
    plt.plot(green_rec_times_el , p0_shotgather_el[ :, rec_nr], 'r', label =  'EL')
    plt.plot(green_rec_times_cda, p0_shotgather_cda[:, rec_nr], 'b', label = 'CDA')
    plt.title("Trace of background gather")
    plt.legend()
    
    #Then a trace of the boundary wavefields
    nvals_bdry_each_el  = boundary_fields_el.size /itimestep_el
    nvals_bdry_each_cda = boundary_fields_cda.size/itimestep_cda
    
    boundary_fields_el_2d  = np.reshape(boundary_fields_el , (itimestep_el , nvals_bdry_each_el ), 'C')
    boundary_fields_cda_2d = np.reshape(boundary_fields_cda, (itimestep_cda, nvals_bdry_each_cda), 'C')
    
    #Time correction for boundary field as well
    boundary_fields_cda_2d = boundary_fields_cda_2d[1:,:]
    
    boundary_pix      = 300 #random pixel number
    txx_off           = 0
    tzz_off           = 1
    txz_off           = 2
    vx_off            = 3
    vz_off            = 4    
    boundary_ind_base = 5*boundary_pix
    
    ind_txx           = boundary_ind_base + txx_off
    ind_vx            = boundary_ind_base + vx_off
    ind_vz            = boundary_ind_base + vz_off
    
    plt.figure(2)
    plt.plot(green_rec_times_el  , boundary_fields_el_2d[ :, ind_txx], 'r', label =   'P EL')
    plt.plot(green_rec_times_cda , boundary_fields_cda_2d[:, ind_txx], 'b', label =  'P CDA')
    plt.title("Trace of boundary field")
    plt.legend()
    
    plt.figure(3)
    plt.plot(green_rec_times_el  , boundary_fields_el_2d[ :, ind_vx], 'r', label =   'Vx EL')
    plt.plot(green_rec_times_el  , boundary_fields_el_2d[ :, ind_vz], 'k', label =   'Vz EL')
    plt.plot(green_rec_times_cda , boundary_fields_cda_2d[:, ind_vx], 'b', label =  'Vx CDA')
    plt.plot(green_rec_times_cda , boundary_fields_cda_2d[:, ind_vz], 'g', label =  'Vz CDA')
    plt.title("Trace of boundary field")
    plt.legend()        
    
    print "Here I am using different timesteps for CDA and EL. For a real application it is probably best to decide beforehand on a uniform timestep which statisfies stability conditions of both CDA and EL."
    
    #Show all
    plt.show()
     