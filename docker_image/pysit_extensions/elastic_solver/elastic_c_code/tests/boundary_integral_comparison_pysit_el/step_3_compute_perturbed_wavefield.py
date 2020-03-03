#IF DT CHANGES IN THE PERTURBED MODEL COMPARED TO THE MODEL WHERE THE GREEN'S FUNCTIONS ARE COMPUTED, THEN DO SOME INTERPOLATION?'

# Std import block
import time

import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt

from pysit import *
from pysit_extensions.elastic_solver.wrapping_functions import elastic_solve, get_source_wavelet
from pysit_extensions.elastic_solver.precompute_greens import get_trunc_geom_dict, get_nr_physical
from pysit_extensions.elastic_solver.boundary_integral_helper import *
from pysit_extensions.ximage.ximage import *
from shared_routines import *
import csv

def make_shot(m, x_pos_src, z_pos_src, x_pos_rec, z_pos_rec, peakfreq):
    w = RickerWavelet(peakfreq)
    source = PointSource(m, (x_pos_src, z_pos_src), w, approximation='delta')
    receivers = ReceiverSet(m, [PointReceiver(m, (x, z), approximation='delta') for x,z in zip(x_pos_rec, z_pos_rec)])
    return Shot(source,receivers)

if __name__ == '__main__':
    m, shots = compute_shot_and_mesh()
    
    print "First look at a negative perturbation so that the timestep does not change"
    perturb = 500.0
    vp_bg_2d, rho_bg_2d, vs_bg_2d, dt  = give_background_models(m)
    vp_ptb_2d, rho_ptb_2d, vs_ptb_2d, dt = give_perturbed_models(m, perturb)
    
    
    t_min             = 0.0
    t_max_desired     = 1.0
    match_pysit       = True #Effective wavelet passed to C code is integral of source wavelet. Doesn't matter from a test perspective
    itimestep         = int(np.floor((t_max_desired-t_min)/dt) + 1); #number of timesteps
    t_max             = (itimestep-1) * dt
    rec_boundary_geom = { 'rec_x_l': 200.0,  'rec_x_r': 600.0,  'rec_z_t': 200.0, 'rec_z_b': 600.0}
    trunc_geom_dict   = get_trunc_geom_dict(m, rec_boundary_geom)

    ############### TEMPORARY TEST FOR LARGER PERTURB ############################

    
    pert_l = int(rec_boundary_geom['rec_x_l']/m.x.delta + 1)
    pert_r = int(rec_boundary_geom['rec_x_r']/m.x.delta - 1)
    pert_t = int(rec_boundary_geom['rec_z_t']/m.z.delta + 1)
    pert_b = int(rec_boundary_geom['rec_z_b']/m.z.delta - 1)
    
    #MOSTLY THE VP CHANGE SEEMS TO CAUSE SOME PROBLEMS
    #np.random.seed(42)
    #vp_ptb_2d[pert_t:pert_b+1,pert_l:pert_r+1]  += 3000*np.random.rand(pert_b-pert_t +1, pert_r - pert_l + 1)
    #rho_ptb_2d[pert_t:pert_b+1,pert_l:pert_r+1] += 1000*np.random.rand(pert_b-pert_t +1, pert_r - pert_l + 1)
    #vs_ptb_2d[pert_t:pert_b+1,pert_l:pert_r+1]  += 1000*np.random.rand(pert_b-pert_t +1, pert_r - pert_l + 1)

    
    ############### END TEMPORARY #################################################
    
    #For simplicity, look at single shot (shots[1])
    shot_nr  = 1
    shot     = shots[shot_nr]
    x_1, z_1 = shot.sources.position
    xps1 = int(np.round(x_1/m.x.delta))
    zps1 = int(np.round(z_1/m.z.delta))
    
    #1: Convolve recorded green's functions with the wavelet to get background field. 
    #In step_2_compute_background_wavefield.py I already verified this gives exactly the same as a full domain solve 
    load_prefix = 'greens_out/step1_'
    indict      = spio.loadmat(load_prefix + "s_pos_x_%.2f_z_%.2f"%(x_1, z_1))

    green_shotgather_combined  = indict['shotgathers'][0] #Contains both real receivers and boundary integral receivers
    nr_physical                = get_nr_physical(green_shotgather_combined, trunc_geom_dict)
    #ximage(shotgather_combined, perc=80)
    shotgather_recording_times = indict['shotgathers_times'][0]
    #This is the wavelet we use, the integral of the Ricker. Should give similar results as using a normal ricker in the normal acoustic simulation code.    
    wavelet                    = get_source_wavelet(shot, shotgather_recording_times, match_pysit)
    p0_shotgather_combined     = convolve_shotgather_with_wavelet(green_shotgather_combined, wavelet)    
    p0_shotgather              = p0_shotgather_combined[:itimestep,:nr_physical]
    
    #2: Do full domain simulation on perturbed background model 
    recording_period = 1
    elastic_options_full = {'dt': dt,
                            'amp0': 1.0,
                            'itimestep':itimestep,
                            'iwavelet': 3, #Use from shot, which will give same 'wavelet' array as used above
                            'traces_output': 1,
                            'traces_step_length': recording_period, 
                            'traces_mem': True,
                            }
    
    retval_full       = elastic_solve([shots[shot_nr]], m, vp_ptb_2d, rho_ptb_2d, vs_ptb_2d, elastic_options_full)        
    #Only contains physical receivers here.
    p_shotgather_full = retval_full['shotgathers'][0]
    
    p_sc_full          = p_shotgather_full - p0_shotgather
    c_shot             = vp_bg_2d[zps1,xps1]
    rec_nr             = -2
    rec_of_interest    = shot.receivers.receiver_list[rec_nr]
    xr_1, zr_1         = rec_of_interest.position
    xpr1               = int(np.round(xr_1/m.x.delta))
    zpr1               = int(np.round(zr_1/m.z.delta))    
    c_rec              = vp_bg_2d[zpr1,xpr1]
    
    p_sc_full_rec_copy = np.copy(p_sc_full[:,rec_nr]) #For the savedict    
    p_sc_full         *= c_shot**2 #c**2 is amplitude difference with what would have been generated in pysit
        
    #3 Now get scattered field from boundary integral
    green_boundary_fields = indict['boundary_fields'][0]
    boundary_fields       = convolve_boundary_fields(green_boundary_fields, wavelet, itimestep)
    
    elastic_options_full['rec_boundary_geom'] = rec_boundary_geom
    boundary_fields            = [boundary_fields] #need input to be a list with same length as number of shots in shots array we pass

    #Compute the pressures at the integral points
    
    #n_pad_nodes = 15 #WITH THIS WE INCLUDE THE FULL MODEL AND HAE A PERFECT 'LOCAL' SOLVER.  
    n_pad_nodes = 5
    int_sc_bound_p, int_p_times = compute_boundary_recordings_local_solver([shot], m, vp_ptb_2d, rho_ptb_2d, vs_ptb_2d, elastic_options_full, boundary_fields, n_pad_nodes = n_pad_nodes, PMLwidth = 1000.0)
    int_sc_bound_copy = np.copy(int_sc_bound_p[0]) 
    int_sc_bound_p[0]    *= c_shot**2 #COULD SIMPLY HAVE MULTIPLIED boundary_fields with c**2. That would have worked (tested)
    
    print "MUCH THINNER LOCAL PML WILL CAUSE SOME DIFFERENCE BETWEEN LOCAL AND FULL DOMAIN SCATTERED FIELD. BUT I DONT THINK IT INFLUENCED THE BOUNDARY INTEGRAL PROPAGATED RESULT WITHIN THE 1.0 SECOND RANGE THAT MATTERED"
    
    ################################################################################
    #Compare this scattered field with what a local solver would generate!
    #DOING DOUBLE WORK. COULD HAVE OBTAINED EARLIER. JUST DOING FOR EXTRA CERTAINTY. 
    #RIGHT NOW RECORD AT INTEGRAL NODES DIRECTLY WITHOUT USING LOCAL SOLVER
    x_pos_int, z_pos_int = get_boundary_integral_positions(m, trunc_geom_dict)
    peakfreq = shot.sources.w.peak_frequency
     
    shot_bg  = make_shot(m, shot.sources.position[0], shot.sources.position[1], x_pos_int, z_pos_int, peakfreq)
    shot_ptb = make_shot(m, shot.sources.position[0], shot.sources.position[1], x_pos_int, z_pos_int, peakfreq)
     
    retval_bg       = elastic_solve([shot_bg] , m,  vp_bg_2d, rho_bg_2d , vs_bg_2d , elastic_options_full)
    retval_ptb      = elastic_solve([shot_ptb], m, vp_ptb_2d, rho_ptb_2d, vs_ptb_2d, elastic_options_full)
     
    int_sc_bound_p_full_domain = retval_ptb['shotgathers'][0] - retval_bg['shotgathers'][0] 
    int_sc_bound_p_full_domain *= c_shot**2
    
    #compare from local solver with direct computation
    int_point = 0 #random integral point to look at
    plt.plot(retval_bg['shotgathers_times'][0] , int_sc_bound_p[0][:,int_point], 'k', label='from local solver')
    plt.plot(retval_ptb['shotgathers_times'][0], int_sc_bound_p_full_domain[:,int_point], 'r', label='from full solver')
    plt.legend()
    plt.show()
    ################################################################################
    

    
    indict_rec      = spio.loadmat(load_prefix + "r_pos_x_%.2f_z_%.2f"%(xr_1, zr_1))
    greens_to_bound = indict_rec['shotgathers'][0]
    t_arr_greens    = indict_rec['shotgathers_times'][0]
    #We want to have Green's function corresponding to 'delta(x - x_0)*delta(t-t_0), which integrates to one after space and time integral
    #In the elastic code the '1' force is divided by dx**2 and multiplied by dt
    #You want to enter unit energy. The division by dx**2 is fine.
    #The multiplication by dt is not fine. Undo here to get result close to analytic Green's function (unit energy input)
    
    greens_to_bound_copy = np.copy(greens_to_bound)
      
    greens_to_bound = greens_to_bound / dt
    greens_to_bound = greens_to_bound * c_rec**2 
    dt_greens       = t_arr_greens[1] - t_arr_greens[0]
    
    inward_integral = compute_acoustic_boundary_integral(trunc_geom_dict, greens_to_bound, int_sc_bound_p[0], deriv = 'inward', greens_el = True, dt_green=dt_greens)
    center_integral = compute_acoustic_boundary_integral(trunc_geom_dict, greens_to_bound, int_sc_bound_p[0], deriv = 'center', greens_el = True, dt_green=dt_greens)
    
    outdict                        = dict()
    outdict['rec_nr']              = rec_nr
    outdict['int_sc_bound_p']      = int_sc_bound_copy
    outdict['c_shot']              = c_shot
    outdict['c_rec']               = c_rec
    outdict['t_arr']               = t_arr_greens
    outdict['greens_to_bound']     = greens_to_bound_copy
    outdict['full_domain_scatter'] = p_sc_full_rec_copy
    outdict['perturb']             = perturb
    spio.savemat('greens_out/el.mat', outdict)

    #save trunc_geom_dict
    myfile = open('greens_out/trunc_geom_dict_el.csv','wb')
    wrtr = csv.writer(myfile, delimiter=',', quotechar='"')
    for key, val in trunc_geom_dict.items():
        wrtr.writerow([key, val])
    myfile.close() # when you're done.  
    
    mult = dt       #the dt is the same as in the pysit version of the boundary integral. Probably because I did not use dt when evaluating convolution integral.
                    #Effectively this dt cancels the 1/dt I used to scale the green's function
                    #the c**2 may be related to the c**2 I needed to scale the EL simulation to the pysit in 'compare_pysit_and_wrapped_elastic.py in the pysit_extension source   
    plt.plot(dt*np.arange(p_sc_full[:, rec_nr].size), p_sc_full[:, rec_nr], 'k', label='$p_{sc}$ full domain')
    plt.plot(dt*np.arange(inward_integral.size)     , mult*inward_integral, 'r', label='$p_{sc}$ inward deriv')
    plt.plot(dt*np.arange(center_integral.size)     , mult*center_integral, 'b', label='$p_{sc}$ center deriv')
    plt.legend()
    plt.show()