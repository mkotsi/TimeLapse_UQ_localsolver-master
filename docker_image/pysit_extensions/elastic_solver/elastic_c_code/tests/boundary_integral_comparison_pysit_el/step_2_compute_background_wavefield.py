#Use the recorded green's functions and convolve them with the source wavelet. Compare with direct full domain solve

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

if __name__ == '__main__':

    m, shots = compute_shot_and_mesh()

    vp_2d, rho_2d, vs_2d, dt = give_background_models(m)
    
    t_min             = 0.0
    t_max_desired     = 1.0
    match_pysit       = True #Effective wavelet passed to C code is integral of source wavelet. Doesn't matter from a test perspective
    itimestep         = int(np.floor((t_max_desired-t_min)/dt) + 1); #number of timesteps
    t_max             = (itimestep-1) * dt
    rec_boundary_geom = { 'rec_x_l': 200.0,  'rec_x_r': 600.0,  'rec_z_t': 200.0, 'rec_z_b': 600.0}
    trunc_geom_dict   = get_trunc_geom_dict(m, rec_boundary_geom)
    
    #For simplicity, look at single shot (shots[1])
    shot_nr  = 1
    shot     = shots[shot_nr]
    x_1, z_1 = shot.sources.position
    xps1 = int(np.round(x_1/m.x.delta))
    zps1 = int(np.round(z_1/m.z.delta))
    
    #Convolve recorded green's functions with the wavelet and compare with a full domain simulation using the same wavelet as input. 
    #Result should be the same up to machine precision
    
    print "Only looking at the pressure recordings to see if convolution is consistent." 
    print "Assuming that the convolution with the other physical quantities around the recording boundary is similar."
    
    #1:Convolve recorded green's functions    
    load_prefix = 'greens_out/step1_'
    indict      = spio.loadmat(load_prefix + "s_pos_x_%.2f_z_%.2f"%(x_1, z_1))

    green_shotgather_combined  = indict['shotgathers'][0] #Contains both real receivers and boundary integral receivers
    nr_physical                = get_nr_physical(green_shotgather_combined, trunc_geom_dict)
    #ximage(shotgather_combined, perc=80)
    shotgather_recording_times = indict['shotgathers_times'][0]
    #This is the wavelet we use, the integral of the Ricker. Should give similar results as using a normal ricker in the normal acoustic simulation code.    
    wavelet                    = get_source_wavelet(shot, shotgather_recording_times, match_pysit)
    shotgather_combined        = convolve_shotgather_with_wavelet(green_shotgather_combined, wavelet)
    
    trace_nr = 5 #400m
    get_intuition_convolve_implementation = True
    if get_intuition_convolve_implementation:
        trace_1        = green_shotgather_combined[:,trace_nr]
        trace_2        = wavelet
        conv_naive     = temp_naive_convolve(trace_1, trace_2); plt.plot(np.arange(conv_naive.size)*dt, conv_naive, 'r')
        
    conv_implement = shotgather_combined[:,trace_nr]; plt.plot(np.arange(conv_implement.size)*dt, conv_implement, 'b')
    
    
    #2: Do simulation with ricker on the background model
    recording_period = 1
    elastic_options_full = {'dt': dt,
                            'amp0': 1.0,
                            'itimestep':conv_implement.size, #See if convolution past 'itimestep' steps still exact
                            'iwavelet': 3, #Use from shot, which will give same 'wavelet' array as used above
                            'traces_output': 1,
                            'traces_step_length': recording_period, 
                            'traces_mem': True,
                            }
    
    retval_full     = elastic_solve(shots, m, vp_2d, rho_2d, vs_2d, elastic_options_full)        
    #Only contains physical receivers here.
    shotgather_full = retval_full['shotgathers'][shot_nr]
    trace_full      = shotgather_full[:,trace_nr]
    plt.plot(np.arange(trace_full.size)*dt, trace_full, 'k')
    plt.title("EXACT UP TO 1.0 SEC: LEN GREENS FUN")
    plt.show()

    print "Exact match at physical receivers. Since the boundary integral receivers are exactly the same I assume my convolution is working there as well"
    print "Move on to boundary fields..."
    
    green_boundary_fields = indict['boundary_fields'][0]
    boundary_fields       = convolve_boundary_fields(green_boundary_fields, wavelet, itimestep)

    #temporarily reshape boundary_fields to a 2D array again. 
    #Extract a pressure trace that coincides with one of the boundary integral receivers to verify they are equal
    #Looking at the top-left outer pixel of the boundary field. Index 0. This is directly one pixel above the left boundary of S_i
    
    number_traces      = boundary_fields.size / itimestep 
    boundary_fields_2d = np.reshape(boundary_fields, (itimestep, number_traces), 'C')
    
    txx_offset = 0 #both txx and tzz are equal in a pressure field
    tzz_offset = 1
    trace_from_boundary_field = 0.5*(boundary_fields_2d[:, 0 + txx_offset] + boundary_fields_2d[:, 0 + tzz_offset])  
    plt.plot(trace_from_boundary_field, 'r', label='from boundary field')
    boundary_int_rec_nr = 4 * trunc_geom_dict['int_h_n'] + 5 
    plt.plot(shotgather_combined[:, nr_physical + boundary_int_rec_nr], 'b', label='from receiver')
    plt.legend()
    plt.show()
    
    #See if the boundary integral correctly evaluates to 0.
    #Do local solve on background model, should give same field as verified before, get pressure from here.
    #Then run boundary integral on this pressure field to some random receiver associated with the shot we look at.
    
    elastic_options_full['rec_boundary_geom'] = rec_boundary_geom
    boundary_fields       = [boundary_fields] #need input to be a list with same length as number of shots in shots array we pass
    
    #Compute the pressures at the integral points
    int_sc_bound_p, int_p_times = compute_boundary_recordings_local_solver([shot], m, vp_2d, rho_2d, vs_2d, elastic_options_full, boundary_fields, n_pad_nodes = 5, PMLwidth = 100.0)
    c_shot = vp_2d[zps1,xps1]
    int_sc_bound_p[0]    *= c_shot**2 #COULD SIMPLY HAVE MULTIPLIED boundary_fields with c**2. That would have worked (tested)
    
    #Since we are doing the local solve on the background model, int_sc_bound_p is basically zero (scattered field). 
    #Not very useful to determine if boundary integral is implemented correctly. 
    #Boundary integral should be zero if we add the background pressure to it. 
    int_bg_bound_p  = shotgather_combined[:,nr_physical:]
    int_ptb_bound_p = int_bg_bound_p[:itimestep,:] + int_sc_bound_p[0][:itimestep,:] 
    
    rec_nr = (nr_physical-1)/2
    rec_of_interest = shot.receivers.receiver_list[rec_nr]
    xr_1, zr_1        = rec_of_interest.position
    xpr1              = int(np.round(xr_1/m.x.delta))
    zpr1              = int(np.round(zr_1/m.z.delta))   
    c_rec             = vp_2d[zpr1,xpr1]
    indict_rec        = spio.loadmat(load_prefix + "r_pos_x_%.2f_z_%.2f"%(xr_1, zr_1))
    greens_to_bound   = indict_rec['shotgathers'][0]
    t_arr_greens      = indict_rec['shotgathers_times'][0]
    
    #We want to have Green's function corresponding to 'delta(x - x_0)*delta(t-t_0), which integrates to one after space and time integral
    #In the elastic code the '1' force is multiplied dx**2 and divided by dt
    #You want to enter unit energy. The division by dx**2 is fine.
    #The multiplication by dt is not fine. Undo here to get result close to analytic Green's function  
    greens_to_bound      = greens_to_bound / dt     
    greens_to_bound      = greens_to_bound * c_rec**2 
    shotgather_combined *= c_rec**2
    dt_greens            = t_arr_greens[1] - t_arr_greens[0]
    inward_integral = compute_acoustic_boundary_integral(trunc_geom_dict, greens_to_bound, int_ptb_bound_p, deriv = 'inward', greens_el = True, dt_green = dt_greens)
    center_integral = compute_acoustic_boundary_integral(trunc_geom_dict, greens_to_bound, int_ptb_bound_p, deriv = 'center', greens_el = True, dt_green = dt_greens)
    
    #Both integrals should approximately be zero within the 1.0s range
    mult = dt       #the dt is the same as in the pysit version of the boundary integral. Probably because I did not use dt when evaluating convolution integral.
                    #Effectively this dt cancels the 1/dt I used to scale the green's function
                    #the c**2 may be related to the c**2 I needed to scale the EL simulation to the pysit in 'compare_pysit_and_wrapped_elastic.py in the pysit_extension source
    plt.plot(dt*np.arange(inward_integral.size), inward_integral, 'r', label='$p_{sc}$ inward deriv')
    plt.plot(dt*np.arange(center_integral.size), center_integral, 'b', label='$p_{sc}$ center deriv')
    #For reference, also true background field 
    plt.plot(dt*np.arange(shotgather_combined[:,rec_nr].size), shotgather_combined[:,rec_nr], 'k', label='$p_0$')
    plt.title("$p_0$ ONLY TRUSTWORTHY UP TO 1.0 SEC, LEN GREEN FUNCTION")
    plt.legend()
    plt.show()