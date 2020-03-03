#IF DT CHANGES IN THE PERTURBED MODEL COMPARED TO THE MODEL WHERE THE GREEN'S FUNCTIONS ARE COMPUTED, THEN DO SOME INTERPOLATION?'

# Std import block
import time

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt

from pysit import *
from pysit_extensions.elastic_solver.wrapping_functions import elastic_solve, get_source_wavelet
from pysit_extensions.elastic_solver.precompute_greens import get_trunc_geom_dict, get_nr_physical, get_rec_pos, get_nr_physical
from pysit_extensions.elastic_solver.boundary_integral_helper import *
from pysit_extensions.ximage.ximage import *
from pysit_extensions.impulse_time.impulse_time import ImpulseTimeWavelet
from shared_routines import *

import csv

def make_shot(m, x_pos_src, z_pos_src, x_pos_rec, z_pos_rec, wavelet):
    if wavelet == 'Ricker':
        w = RickerWavelet(6.0)
    elif wavelet == 'Delta':
        w = ImpulseTimeWavelet()
    else:
        raise Exception("Wrong option provided")
    
    source = PointSource(m, (x_pos_src, z_pos_src), w, approximation='delta')
    receivers = ReceiverSet(m, [PointReceiver(m, (x, z), approximation='delta') for x,z in zip(x_pos_rec, z_pos_rec)])
    return Shot(source,receivers)

if __name__ == '__main__':
    print "Kind of hacky implementation. Quickly want to see if PySIT gives same boundary integral"

    m, shots = compute_shot_and_mesh('PySIT')
    del shots #Will get my own shots,  
    
    print "First look at a negative perturbation so that the timestep does not change"
    perturb = 500.0
    vp_bg_2d, rho_2d, vs_2d, dt_el = give_background_models(m)
    vp_ptb_2d, rho_2d, vs_2d, dt_el = give_perturbed_models(m, perturb)
    
    t_min             = 0.0
    t_max_desired     = 1.0
    rec_boundary_geom = { 'rec_x_l': 200.0,  'rec_x_r': 600.0,  'rec_z_t': 200.0, 'rec_z_b': 600.0}
    trunc_geom_dict   = get_trunc_geom_dict(m, rec_boundary_geom)    
    
    x_pos_int, z_pos_int = get_boundary_integral_positions(m, trunc_geom_dict)
    
    #ADD REAL RECEIVER LOCATION SO I CAN GET REAL SCATTERED FIELD IN PYSIT
    rec_nr = -2
    x_pos_recs = np.linspace(0.0, 800.0, 9)
    xr_1       = x_pos_recs[rec_nr]
    zr_1       = 30.0

    x_pos = np.concatenate([[xr_1], x_pos_int])
    z_pos = np.concatenate([[zr_1], z_pos_int])
    
    #Get Ricker wavelet for incident field
    shot_ricker_bg  = make_shot(m, 400.0, 10.0, x_pos, z_pos, 'Ricker') #source location 
    shot_ricker_ptb = make_shot(m, 400.0, 10.0, x_pos, z_pos, 'Ricker') #source location
    shot_green      = make_shot(m, xr_1 , zr_1, x_pos_int, z_pos_int, 'Delta') #receiver location
    
    trange = (0.0,t_max_desired)

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=4,
                                         trange=trange,
                                         kernel_implementation='cpp')
    

    vp_bg_1d  = np.reshape(vp_bg_2d,  (np.prod(vp_bg_2d.shape) ,1), 'F')
    vp_ptb_1d = np.reshape(vp_ptb_2d, (np.prod(vp_ptb_2d.shape),1), 'F')
    
    bg_model  = solver.ModelParameters(m,{'C': vp_bg_1d})
    ptb_model = solver.ModelParameters(m,{'C': vp_ptb_1d})
    solver.model_parameters = bg_model #This will set dt
    if dt_el < solver.dt:
        solver.dt = dt_el
        solver.nsteps = int(np.floor((t_max_desired-t_min)/dt_el+1))
    else:
        raise Exception("probably should in this case do exactly half of dt_el if i would later want to use these green's for the EL local solver to avoid interpolation?")
    print "bg green" 
    generate_seismic_data([shot_green]     , solver,  bg_model)    
    print "bg ricker"
    generate_seismic_data([shot_ricker_bg] , solver,  bg_model)
    print "ptb ricker"
    solver.model_parameters = ptb_model #This will set dt
    if dt_el < solver.dt:
        solver.dt = dt_el
        solver.nsteps = int(np.floor((t_max_desired-t_min)/dt_el+1))
    else:
        raise Exception("probably should in this case do exactly half of dt_el if i would later want to use these green's for the EL local solver to avoid interpolation?")        
    generate_seismic_data([shot_ricker_ptb], solver, ptb_model)

    
    nr_physical = get_nr_physical(shot_ricker_ptb.receivers.data, trunc_geom_dict)
    
    int_sc_bound_p  = shot_ricker_ptb.receivers.data[:, nr_physical:] - shot_ricker_bg.receivers.data[:, nr_physical:]
    
    greens_to_bound = shot_green.receivers.data
    outdict = dict()
    outdict['greens_to_bound']     = np.copy(greens_to_bound) #save before manipulate
    greens_to_bound /= solver.dt                              #Did not normalize the green's function yet
    
    inward_integral = compute_acoustic_boundary_integral(trunc_geom_dict, greens_to_bound, int_sc_bound_p, deriv = 'inward')
    center_integral = compute_acoustic_boundary_integral(trunc_geom_dict, greens_to_bound, int_sc_bound_p, deriv = 'center')

    full_domain_scatter = shot_ricker_ptb.receivers.data[:, 0] - shot_ricker_bg.receivers.data[:, 0]
    
    outdict['rec_nr']              = rec_nr
    outdict['full_domain_scatter'] = full_domain_scatter
    outdict['int_sc_bound_p']      = np.copy(int_sc_bound_p)
    outdict['dt'] = solver.dt
    outdict['perturb'] = perturb
    spio.savemat('greens_out/cda.mat', outdict)

    #save trunc_geom_dict
    myfile = open('greens_out/trunc_geom_dict_cda.csv','wb')
    wrtr = csv.writer(myfile)
    for key, val in trunc_geom_dict.items():
        wrtr.writerow([key, val])
    myfile.close() # when you're done.    
    
    #Show scattered field at integral point 0. Want to compare with step_3_compute_perturbed_wavefield.py
    plt.plot(solver.ts(), int_sc_bound_p[:,0], 'k', label = 'PySIT scatter')
    plt.show()
    
    multiplier  = solver.dt #This seems to be what is required. Maybe because I do not use dt in convolution in boundary integral?
    
    print "In the end (for PySIT) I would be fine not normalizing the green's function by 1./dt and then not multiplying the integral by dt below. (I think that dt comes from the convolution, where in continuous form I needed dt)"
    
    plt.plot(solver.dt*np.arange(full_domain_scatter.size) , full_domain_scatter, 'k', label='$p_{sc}$ full domain')
    plt.plot(solver.dt*np.arange(inward_integral.size)     , multiplier*inward_integral, 'r', label='$p_{sc}$ inward deriv')
    plt.plot(solver.dt*np.arange(center_integral.size)     , multiplier*center_integral, 'b', label='$p_{sc}$ center deriv')
    plt.legend()
    plt.show()    