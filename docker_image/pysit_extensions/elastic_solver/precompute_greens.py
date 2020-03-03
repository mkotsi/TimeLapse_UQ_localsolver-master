import copy
import numpy as np
import scipy.io as spio
import warnings
from mpi4py import MPI #FOR THE ACOUSTIC SOLVER
from pysit.core import *
from pysit.solvers import *
from pysit.modeling import *
from pysit_extensions.elastic_solver.wrapping_functions import elastic_solve
from pysit_extensions.elastic_solver.boundary_integral_helper import get_boundary_integral_positions, get_additional_required_boundary_fields_positions, get_additional_required_boundary_fields_positions_new, global_geometry, get_trunc_geom_dict, validate_shots, float_eq, interpolate_gather_in_time_2d, convolve_shotgather_with_wavelet
from pysit_extensions.impulse_time.impulse_time import ImpulseTimeWavelet

def get_nr_physical(green_shotgather_combined, trunc_geom_dict, additional_boundary_field = False, new_implementation = False):
    #Backward compatability is making this quite messy...
    #New implementation refers to the new precomputation routine where I only save pressure fields and compute the boundary field from that later
    #That is the routine I'm using for injecting on a medium with mu != 0 
    
    if not additional_boundary_field:
        return green_shotgather_combined.shape[1] - trunc_geom_dict['int_n']
    elif additional_boundary_field and not new_implementation:
        return green_shotgather_combined.shape[1] - (trunc_geom_dict['int_n'] + trunc_geom_dict['bdry_n']) 
    elif new_implementation:
        return green_shotgather_combined.shape[1] - (trunc_geom_dict['int_n'] + trunc_geom_dict['bdry_extra_n'])
    else:
        raise Exception("??")

#Derived from the 2D truncated domain helmholtz solver version
def validate_truncation_params(m, rec_boundary_geom):
    #Verify global domain
    [x_min, x_max, z_min, z_max, dx, dz, domain_global] = global_geometry(m)

    if domain_global.dim != 2:
        raise Exception("Right now I only work with 2D")

    if dz != dx:
        raise Exception('C code currently assumes uniform spacing')
    
    #now check the truncated domain against this
    #The recording boundary corresponds with S_i
    #Right now I implement the boundary integral by using up to five layers right outside.
    #Because of this I will pad with five uniform spacings the quantities below,
    #This gives the outmost positions of the truncated domain whose nodal positions must exist on the full domain grid.
    xpos_top_left_corner  = rec_boundary_geom['rec_x_l'] - 5*dx
    xpos_top_right_corner = rec_boundary_geom['rec_x_r'] + 5*dx
    zpos_top_left_corner  = rec_boundary_geom['rec_z_t'] - 5*dz
    zpos_bot_left_corner  = rec_boundary_geom['rec_z_b'] + 5*dz
    
    width  = xpos_top_right_corner - xpos_top_left_corner
    height = zpos_bot_left_corner  - zpos_top_left_corner 
    
    if xpos_top_left_corner + width > x_max:
        raise Exception('Truncated domain exceeds right boundary full domain!')
    
    if zpos_top_left_corner + height > z_max:
        raise Exception('Truncated domain exceeds bottom boundary of full domain!')
    
    #allow some slack up to a millimeter or so. Sometimes there are some rounding errors
    eps = 1.0e-3
    
    #Modulus is buggy sometimes for large numbers (even double precision). So I am first getting the numbers to be between 0 and spacing.
    left_remainder = xpos_top_left_corner - int(np.floor((xpos_top_left_corner-x_min)/dx))*dx - x_min
    top_remainder  = zpos_top_left_corner - int(np.floor((zpos_top_left_corner-z_min)/dz))*dz - z_min
    width_remainder = width - np.floor(width/dx)*dx
    height_remainder = height - np.floor(height/dz)*dz
    
    if np.mod(left_remainder + eps, dx) > eps and np.mod(left_remainder, dx) > eps:
        raise Exception('The left part of one of the truncated grids would not align with nodes on the full grid')
    if np.mod(top_remainder + eps, dz) > eps and np.mod(top_remainder, dz) > eps:
        raise Exception('The top part of one of the truncated grids would not align with nodes on the full grid') 
    if np.mod(width_remainder + eps, dx) > eps and np.mod(width_remainder, dx) > eps:
        raise Exception('The width of one of the truncated grids is not an integral number of nodes')    
    if np.mod(height_remainder + eps, dz) > eps and np.mod(height_remainder, dz) > eps:
        raise Exception('The height of one of the truncated grids is not an integral number of nodes')    

def get_rec_pos(shot):
    nrec = shot.receivers.receiver_count
    rec_pos_x = np.zeros(nrec)
    rec_pos_z = np.zeros(nrec)
    for i in xrange(nrec):
        receiver = shot.receivers.receiver_list[i]
        (rec_pos_x[i],rec_pos_z[i]) = receiver.position
        
    return rec_pos_x, rec_pos_z  

def get_source_shots(shots_in, m, x_pos_int, z_pos_int, distr_shots=True):
    #Get new shots which also record wavefield along recording boundary
    
    nshots = len(shots_in)
    if distr_shots: #Distribute shots over MPI processes 
        indices_for_process  = shot_indices_for_process(nshots)
        shots_in_for_process = [shots_in[i] for i in indices_for_process]
    else: #each process has acccess to all the shots. Useful if you want to distribute shots later.
        shots_in_for_process = shots_in
        
    
    shots_out_for_process = []
    for shot_in in shots_in_for_process:
        (s_pos_x, s_pos_z) = shot_in.sources.position

        rec_pos_x, rec_pos_z = get_rec_pos(shot_in)
        
        #make new shot
        source = PointSource(m, (s_pos_x, s_pos_z), ImpulseTimeWavelet(), approximation='delta') #When using EL solver, the wavelet is just a dummy variable. Will use flag in elastic solver to use delta source
        receivers = ReceiverSet(m, [PointReceiver(m, (x, z), approximation='delta') 
                                    for x,z in zip(np.concatenate((rec_pos_x, x_pos_int)),np.concatenate((rec_pos_z, z_pos_int)))])

        shots_out_for_process.append(Shot(source, receivers))
        
    return shots_out_for_process

def get_unique_receiver_positions(shots):
    #Getting unique positions can conveniently be done with dict()
    #I believe the lookup for whether the key exist is very cheap (hash table?) -> O(1)
    pos_dict = dict()
    for shot in shots:
        for receiver in shot.receivers.receiver_list:
            pos = receiver.position
            if pos not in pos_dict:
                pos_dict[pos] = True

    u_rec_pos   = pos_dict.keys()
    n_u         = len(u_rec_pos)
    u_rec_pos_x = np.zeros(n_u)
    u_rec_pos_z = np.zeros(n_u)  
    for i in xrange(n_u):
        u_rec_pos_x[i], u_rec_pos_z[i] = u_rec_pos[i]

    return u_rec_pos_x, u_rec_pos_z 

def get_receiver_shots(m, u_rec_pos_x, u_rec_pos_z, x_pos_int, z_pos_int, distr_shots=True):
    nshots                  = len(u_rec_pos_x)
    
    if distr_shots: #Distribute shots over MPI processes 
        indices_for_process  = shot_indices_for_process(nshots)
    else: #each process has acccess to all the shots. Useful if you want to distribute shots later.
        indices_for_process  = np.arange(nshots)    
    
    if len(indices_for_process) > 0:
        u_rec_pos_x_for_process = u_rec_pos_x[indices_for_process]
        u_rec_pos_z_for_process = u_rec_pos_z[indices_for_process]     
        
        shots_out_for_process = []
        for s_pos_x, s_pos_z in zip(u_rec_pos_x_for_process, u_rec_pos_z_for_process):
            source = PointSource(m, (s_pos_x, s_pos_z), ImpulseTimeWavelet(), approximation='delta') #When using EL solver, the wavelet is just a dummy variable. Will use flag in elastic solver to use delta source 
            receivers = ReceiverSet(m, [PointReceiver(m, (x, z), approximation='delta') for x,z in zip(x_pos_int, z_pos_int)])
    
            shots_out_for_process.append(Shot(source, receivers))
            
    else:
        shots_out_for_process = []
        
    return shots_out_for_process

def initiate_solve(shot, m, trunc_geom_dict):
    print "call elastic solver here"

def compute_greens_functions(shots_s, shots_r, m, vp_2d, rho_2d, vs_2d, elastic_options_dict, save_prefix):
    rec_boundary_geom = elastic_options_dict['rec_boundary_geom']
    
    vmax = np.max([vp_2d,rho_2d,vs_2d])
    dt_max = m.x.delta/(np.sqrt(2)*vmax*(9./8 +1./24)) #assuming equal spacing
    dt = 1./3*dt_max #    
    
    delta_flag = 9
    iwavelet = delta_flag
    
    #Since we only introduce energy at the first timestep, the values could maybe become very small in a large model?
    #When evaluating the laplacian we add/subtract small numbers. If the field would get close to machine precision, 
    #addition of these small numbers could result in inaccuracies? 
    #Not sure if it matters, but will introduce a bit more energy, and then divide it out before saving the green's functions
    #IN HINDSIGHT, NOT USEFUL. ROUNDING ERRORS ONLY OCCUR WHEN TRYING TO ADD 1e-16 to 1.0 FOR INSTANCE. NO PROBLEM TO ADD 1e-20 to 1e-18. STILL VERY HIGH ACCURACY.
    #MULTIPLYING BY SCALAR IS KIND OF POINTLESS HERE
    default_amp0 = 1.0e6
    
    #Define default elastic options
    el_opts = {'dt': dt,
               'amp0': default_amp0,
               'snap_output': 0,
               'rec_boundary': True,
               'rec_boundary_geom': rec_boundary_geom,
               'local_solve': False 
               }

   
    #allow for user to overwrite defaults
    el_opts.update(elastic_options_dict)
    
    #These two should not be overwritten for Green's function computation. Stop execution and warn user
    if el_opts['iwavelet'] != delta_flag:
        raise Exception("iwavelet should correspond to green's function: " + str(delta_flag))
    if el_opts['amp0']     != default_amp0:
        raise Exception("Do not change amp0")

    #Only record shotgathers for source shots (real receivers and boundary integral receivers)
    el_opts_s = el_opts.copy()
    el_opts_s['traces_output']      = 1
    el_opts_s['traces_step_length'] = 1  #Not my focus in this test. But want to keep on so I can verify it still doesnt segfault
    el_opts_s['traces_mem']         = True
    
    #Don't record shotgathers for receiver shots. 
    el_opts_r = el_opts.copy()
    el_opts_r['traces_output']      = 1
    el_opts_r['traces_step_length'] = 1  #Not my focus in this test. But want to keep on so I can verify it still doesnt segfault
    el_opts_r['traces_mem']         = True

    recording_period = el_opts_s['traces_step_length']
    if recording_period != 1:
        raise Exception("Right now hardcode recording_period = 1. If I want to subsample I should make sure that rec_boundary_geom also allows subsampling and that the same rate is used!")

    #Start computing green's functions
    for shot in shots_s:
        retdict = elastic_solve([shot], m, vp_2d, rho_2d, vs_2d, el_opts_s)
       
        retdict['rec_boundary_geom']                = rec_boundary_geom #From this we can deduce how many boundary integral receivers we used
        retdict['itimestep']                        = el_opts['itimestep']
        retdict['compute_dt']                       = dt #can be different from record_times 
        retdict['recording_period']                 = recording_period
        retdict['src_x'], retdict['src_z']          = shot.sources.position
        retdict['rec_pos_x'], retdict['rec_pos_z']  = get_rec_pos(shot)
        retdict['dx']                               = m.x.delta
        retdict['dz']                               = m.z.delta

        #Undo the amplitude scaling
        retdict['shotgathers'][0]     /= el_opts['amp0']
        retdict['boundary_fields'][0] /= el_opts['amp0']
        
        #Save shotgathers (real receivers and boundary integral values
        spio.savemat(save_prefix + "s_pos_x_%.2f_z_%.2f"%(retdict['src_x'], retdict['src_z']), retdict) 
        
    for shot in shots_r:
        retdict = elastic_solve([shot], m, vp_2d, rho_2d, vs_2d, el_opts_r)
        retdict['rec_boundary_geom']                = rec_boundary_geom #From this we can deduce how many boundary integral receivers we used
        retdict['itimestep']                        = el_opts['itimestep']
        retdict['compute_dt']                       = dt #can be different from record_times 
        retdict['recording_period']                 = recording_period
        retdict['src_x'], retdict['src_z']          = shot.sources.position
        retdict['rec_pos_x'], retdict['rec_pos_z']  = get_rec_pos(shot)
        retdict['dx']                               = m.x.delta
        retdict['dz']                               = m.z.delta
        
        #Undo the amplitude scaling
        retdict['shotgathers'][0]     /= el_opts['amp0']
        retdict['boundary_fields'][0] /= el_opts['amp0']
                        
        spio.savemat(save_prefix + "r_pos_x_%.2f_z_%.2f"%(retdict['src_x'], retdict['src_z']), retdict)
            
    warnings.warn("Right now using 1.0e6 as amp because we inject only at one timestep. I suspect energy may be so small that otherwise we encounter 1e-16 values too early ? ")
    
    print "Right now save every timestep. Later I can do some subsampling in time and maybe also space."

def compute_greens_functions_cda_solver(shots_s, shots_r, m, vp_2d, trunc_geom_dict, cda_options_dict, save_prefix):
    rec_boundary_geom = trunc_geom_dict['rec_boundary_geom']
    
    try:
        recording_period = int(cda_options_dict['recording_period'])
    except:
        recording_period = 1
    
    nx = m.x.n
    nz = m.z.n
    
    t_min  = 0.0                       #Green's functions start at t=0. That is when ImpulseTimeWavelet fires
    t_max  = cda_options_dict['t_max']
    rho    = cda_options_dict['rho']   #Even though we do CDA solve, computing velocity from pressure requires a value of rho
    trange = (t_min, t_max)
    
    vp_1d  = np.reshape( vp_2d, (nz*nx,1), 'F')
    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=4, #Also use 4th order accuracy like the elastic solver
                                         trange=trange,
                                         kernel_implementation='cpp')
        
    model  = solver.ModelParameters(m,{'C': vp_1d})
    solver.model_parameters = model #This will set dt
    
    try: #If an alternative dt is provided, change solver
        desired_dt = cda_options_dict['dt']
        solver.dt = desired_dt
        solver.nsteps = int(np.floor((t_max-t_min)/desired_dt+1))
    except: #If no dt provided we will end up here. Just move on with the default dt
        pass
    
    dt = solver.dt
    
    #First shots at shot location
    for shot in shots_s:
        print "Starting new source shot: %.2f"%shot.sources.position[0]
        generate_seismic_data([shot] ,solver, model)
        gather_full = shot.receivers.data
        
        #SUBSAMPLE
        ts_full = solver.ts()
        nt_full = len(ts_full)
        subsample_index = np.arange(0,nt_full,recording_period)
        
        ts_subsampled     = ts_full[subsample_index]
        nt_subsampled     = ts_subsampled.size
        gather_subsampled = gather_full[subsample_index,:] 
        
        #Get boundary fields from receivers in same was as elastic code would have gotten
        boundary_fields = compute_boundary_fields_cda_solver(gather_full, m, dt, rho, trunc_geom_dict)
        
        #reshape boundary fields and subsample
        nvals_bdry_full    = boundary_fields.size
        nvals_bdry_each    = nvals_bdry_full / nt_full
        boundary_fields_2d_full = np.reshape(boundary_fields, (nt_full, nvals_bdry_each), 'C')
        boundary_fields_2d_subsampled = boundary_fields_2d_full[subsample_index,:]   
        boundary_fields_subsampled = np.reshape(boundary_fields_2d_subsampled, (nvals_bdry_each*nt_subsampled,) ,'C')
        
        #remove the extra receivers i used for the boundary fields
        #these receivers are no longer needed. Not used for the integral
        shotgather = strip_extra_receivers(gather_subsampled, trunc_geom_dict)
        savedict = dict()
        savedict['rho']                               = rho
        savedict['recording_period']                  = recording_period
        savedict['shotgathers']                       = [shotgather]      #From EL solver it is also in its own array. Consistency
        savedict['boundary_fields']                   = [boundary_fields_subsampled] #From EL solver it is also in its own array. Consistency
        savedict['shotgathers_times']                 = [ts_subsampled]     #From EL solver it is also in its own array. Consistency
        savedict['rec_boundary_geom']                 = rec_boundary_geom
        savedict['itimestep']                         = solver.nsteps
        savedict['src_x'], savedict['src_z']          = shot.sources.position
        savedict['rec_pos_x'], savedict['rec_pos_z']  = get_rec_pos(shot)
        savedict['dx']                                = m.x.delta
        savedict['dz']                                = m.z.delta
                
        #Save shotgathers (real receivers and boundary integral values
        spio.savemat(save_prefix + "s_pos_x_%.2f_z_%.2f_CDA"%(savedict['src_x'], savedict['src_z']), savedict)
        del savedict, shot, gather_full, gather_subsampled, boundary_fields, boundary_fields_2d_full
         
    for shot in shots_r:
        print "Starting new receiver shot: %.2f"%shot.sources.position[0]
        generate_seismic_data([shot] ,solver, model)
        gather_full = shot.receivers.data
        
        #SUBSAMPLE
        ts_full = solver.ts()
        nt_full = len(ts_full)
        subsample_index = np.arange(0,nt_full,recording_period)
        
        ts_subsampled     = ts_full[subsample_index]
        nt_subsampled     = ts_subsampled.size
        gather_subsampled = gather_full[subsample_index,:]         
        
        #Get boundary fields from receivers in same was as elastic code would have gotten
        boundary_fields = compute_boundary_fields_cda_solver(gather_full, m, dt, rho, trunc_geom_dict)

        #reshape boundary fields and subsample
        nvals_bdry_full    = boundary_fields.size
        nvals_bdry_each    = nvals_bdry_full / nt_full
        boundary_fields_2d_full = np.reshape(boundary_fields, (nt_full, nvals_bdry_each), 'C')
        boundary_fields_2d_subsampled = boundary_fields_2d_full[subsample_index,:]   
        boundary_fields_subsampled = np.reshape(boundary_fields_2d_subsampled, (nvals_bdry_each*nt_subsampled,) ,'C')
        
        #remove the extra receivers i used for the boundary fields
        #these receivers are no longer needed. Not used for the integral
        shotgather = strip_extra_receivers(gather_subsampled, trunc_geom_dict)
        savedict = dict()
        savedict['rho']                               = rho
        savedict['recording_period']                  = recording_period
        savedict['shotgathers']                       = [shotgather]      #From EL solver it is also in its own array. Consistency
        savedict['boundary_fields']                   = [boundary_fields_subsampled] #From EL solver it is also in its own array. Consistency
        savedict['shotgathers_times']                 = [ts_subsampled]     #From EL solver it is also in its own array. Consistency
        savedict['rec_boundary_geom']                 = rec_boundary_geom
        savedict['itimestep']                         = solver.nsteps
        savedict['src_x'], savedict['src_z']          = shot.sources.position
        savedict['rec_pos_x'], savedict['rec_pos_z']  = get_rec_pos(shot)
        savedict['dx']                                = m.x.delta
        savedict['dz']                                = m.z.delta      
        
        #Save shotgathers (real receivers and boundary integral values
        spio.savemat(save_prefix + "r_pos_x_%.2f_z_%.2f_CDA"%(savedict['src_x'], savedict['src_z']), savedict)
        del savedict, shot, gather_full, gather_subsampled, boundary_fields, boundary_fields_2d_full
        
    print "Right now save every timestep. Later I can do some subsampling in time and maybe also space."

def compute_p_greens_functions_only_cda_solver(shots, m, vp_2d, trunc_geom_dict, cda_options_dict, save_prefix):
    """Similar to function compute_greens_functions_cda_solver.
       A difference is that instead of computing the boundary fields vector completely,
       I will will just compute and store pressure at sufficiently many layers so for the 
       local solve I can get the velocity values and their spatial derivatives at several layers around the injection boundary.
       With these velocity derivates and therefore evaluate txx, txz, tzz at the boundary positions.
    """

    rec_boundary_geom = trunc_geom_dict['rec_boundary_geom']
    
    try:
        recording_period = int(cda_options_dict['recording_period'])
    except:
        recording_period = 1
    
    nx = m.x.n
    nz = m.z.n
    
    t_min  = 0.0                       #Green's functions start at t=0. That is when ImpulseTimeWavelet fires
    t_max  = cda_options_dict['t_max']
    trange = (t_min, t_max)
    
    vp_1d  = np.reshape( vp_2d, (nz*nx,1), 'F')
    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=4, #Also use 4th order accuracy like the elastic solver
                                         trange=trange,
                                         kernel_implementation='cpp')
        
    model  = solver.ModelParameters(m,{'C': vp_1d})
    solver.model_parameters = model #This will set dt
    
    try: #If an alternative dt is provided, change solver
        desired_dt = cda_options_dict['dt']
        solver.dt = desired_dt
        solver.nsteps = int(np.floor((t_max-t_min)/desired_dt+1))
    except: #If no dt provided we will end up here. Just move on with the default dt
        pass
    
    dt = solver.dt    
    
    #First shots at shot location
    for shot in shots:
        ###################### DEBUG FOR MEMORY LEAK WHICH IS PREVENTED BY SETTING THE RECEIVERS FIELD OF THE SHOT TO 0 ##############
        ###################### SHOULD MAYBE INVESTIGATE WHY THIS HAPPENS? USE guppy and objgraph 
        ###################### using link http://chase-seibert.github.io/blog/2013/08/03/diagnosing-memory-leaks-python.html 
        ###################### and references therein (for guppy) to trace it down?
        #PARALLEL VARIABLES
        #comm = MPI.COMM_WORLD
        #size = comm.Get_size()
        #rank = comm.Get_rank()
        
        #if rank == 0:
        #    try: 
        #        after = hp.heap()
        #        leftover = after - before
        #        #import pdb; pdb.set_trace()   #TRIGGERS FIRST EVERYTIME          
        #        print leftover
                
        #        #finished this iter
        #        before = after
        #    except: #First time
        #        print "FIRST TIME"
        #        from guppy import hpy
        #        hp = hpy()
        #        before = hp.heap()

            
        #    # critical section here
            
        #    import gc
        #    gc.collect()  # don't care about stuff that would be garbage collected properly
        #    import objgraph
             
        #    print "MOST COMMON TYPES\n"
        #    objgraph.show_most_common_types()
             
        #    print "\nSHOW GROWTH\n"
        #    objgraph.show_growth()
             
        #    import random
        #    objgraph.show_chain(
        #                        objgraph.find_backref_chain(
        #                                                    random.choice(objgraph.by_type('interp1d')),
        #                                                    objgraph.is_proper_module),
        #                        filename='chain.png')            
        
        print "Starting new shot: %.2f"%shot.sources.position[0]
        generate_seismic_data([shot] ,solver, model)
        gather_full = shot.receivers.data
        
        #SUBSAMPLE. 
        #Somewhat inefficient, should implement such that the subsampled field is directly recorded.
        #Now the full 'dense' wavefield is unnecessarily put in memory first, which may result in trouble if too much memory is claimed.
        ts_full = solver.ts()
        nt_full = len(ts_full)
        subsample_index = np.arange(0,nt_full,recording_period)
        
        ts_subsampled     = ts_full[subsample_index]
        nt_subsampled     = ts_subsampled.size
        gather_subsampled = gather_full[subsample_index,:] 
        
        savedict = dict()
        savedict['recording_period']                  = recording_period
        savedict['shotgathers']                       = [gather_subsampled]      #From EL solver it is also in its own array. Consistency
        savedict['shotgathers_times']                 = [ts_subsampled]     #From EL solver it is also in its own array. Consistency
        savedict['rec_boundary_geom']                 = rec_boundary_geom
        savedict['itimestep']                         = solver.nsteps
        savedict['src_x'], savedict['src_z']          = shot.sources.position
        savedict['rec_pos_x'], savedict['rec_pos_z']  = get_rec_pos(shot)
        savedict['dx']                                = m.x.delta
        savedict['dz']                                = m.z.delta
                
        #Save shotgathers [WHAT I CURRENTLY DO IS NOT ROBUST! SHOULD ADD TO NAME IF SOURCE OR RECEIVER BASED GREENS FUNCTION. NOW IF SOURCE AND RECEIVER ARE OVERLAPPING IN SPACE, A RECEIVER FILE COULD OVERWRITE A SHOT FILE, WHICH WOULD RESULT IN US LOSING BACKGROUND GREENS FUNCTIONS BETWEEN SOURCE AND ITS PHYSICAL RECEIVERS 
        spio.savemat(save_prefix + "pos_x_%.2f_z_%.2f_CDA"%(savedict['src_x'], savedict['src_z']), savedict)
        
        #WITHOUT SETTING REFERENCE TO RECEIVERS TO NONE, THE MEMORY IS NOT CORRECTLY RELEASED FOR SOME REASON.
        #SIMPLY DELETING REFERENCE TO THE SHOT DOES NOT WORK!!!
        shot.receivers=None    
   
def strip_extra_receivers(gather, trunc_geom_dict):
    rec_physical = get_nr_physical(gather, trunc_geom_dict, additional_boundary_field = True)
    rec_int      = trunc_geom_dict['int_n']
    return gather[:,:rec_physical+rec_int]
    
 
def compute_boundary_fields_cda_solver_new(gather_bdry, ts_in, m, vp_2d, rho_2d, vs_2d, trunc_geom_dict, pos_x_src, pos_z_src, recording_period=1, wavelet=None, rho_src = None):
    """ I will use the CDA boundary pressure arrays to compute vx, vz, tau_xx, tau_zz and tau_xz.
        The amplitude will be scaled in function 'empirical_amplitude_fix_density_contrast based on the difference between the density at the source and the boundary pixel.
        This mimics the natural amplitude variation experienced in the EL solver when smoothly going from density 1 to density 2
        When injecting on a local domain with Vs != 0, tau_xz will also be != 0 and the pressure in general will be lower than in the CDA input field
        This is in agreement with the full domain EL solver, where the pressure also scales proportionally with (Vp**2-Vs**2)
        This scaling will happen automatically in the 'compute_stresses' subroutine.
        At the end the amplitude of the injected field in the EL local solver should be similar as the amplitude obtained in the full domain elastic simulation.
        The primary difference should come from different transmission losses (which we cannot compensate for with the scaling coefficients based on smooth transition)
    """

    #rho_src: This value is used to specify what the density at the source pixel is in the real 2d elastic model you want to simulate the wavefield on
    #since rho_2d could be a tapered density model, the density at the source pixel in this model may not be the density
    #you really want to work with when scaling the input field. rho_src is a scalar value which can be provided instead

    #precompute indices for the gather_bdry array. Very similar to local function within compute_boundary_fields_cda_solver.  
    def precompute_indices(m, trunc_geom_dict):
        #COMPUTES INDICES FOR GATHER WITHOUT PHYSICAL RECEIVERS (ibg below). 
         
        
        #WE ALSO NEED TO CALCULATE VX AND VZ UPDATES FROM THE PRESSURE TRACES USING 
        #THE FOURTH ORDER DERIVATIVE STENCIL. DO STAGGERED GRID EVALUATIONS.
        #INSTEAD OF USING THREE LAYERS AROUND EACH BOUNDARY, NOW PAD WITH 2 LAYERS ON EACH SIDE.
        #EACH LAYER IS ALSO 2 WIDER
        #JUST AS IN fd2d_update_SSG.c    
    
        dx = m.x.delta
        dz = m.z.delta
        ind_dict = dict()
        
        #LOTS OF REPETITION ON CODE BELOW. SHOULD BE ABLE TO MAKE MORE COMPACT
        
        ###################### TOP #############################################
        
        x_min = trunc_geom_dict['rec_x_l'] - 4*dx #pad by 4 for horizontal derivs
        x_max = trunc_geom_dict['rec_x_r'] + 4*dx #pad by 4 for horizontal derivs
        nx    = trunc_geom_dict['rec_h_n'] + 8 
        
        z_min = trunc_geom_dict['rec_z_t'] - 4*dz
        z_max = trunc_geom_dict['rec_z_t'] + 4*dz
        nz    = 9
    
        top_x = np.linspace(x_min, x_max, nx)
        top_z = np.linspace(z_min, z_max, nz)
        
        #Loop is not very efficient in python. But we cache the results, so only need to do once. 
        #Probably not necessary to optimize because of that
        name_arr = ['ind_top_m4',
                    'ind_top_m3',
                    'ind_top_m2',
                    'ind_top_m1',
                    'ind_top_0' ,
                    'ind_top_p1',
                    'ind_top_p2',
                    'ind_top_p3',
                    'ind_top_p4'
                    ]
        iz = 0
        for z in top_z:
            name = name_arr[iz]
            ind_arr = np.zeros(nx, dtype='int64')
            ix = 0
            for x in top_x:
                ind_arr[ix] = pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation=True)
                ix         +=1
                
            ind_dict[name] = ind_arr
            iz            += 1
            
        ###################### END TOP #########################################
    
        ###################### RIGHT############################################
        x_min = trunc_geom_dict['rec_x_r'] - 4*dx #pad by 4 for horizontal derivs
        x_max = trunc_geom_dict['rec_x_r'] + 4*dx #pad by 4 for horizontal derivs
        nx    = 9 
        
        z_min = trunc_geom_dict['rec_z_t'] - 4*dz
        z_max = trunc_geom_dict['rec_z_b'] + 4*dz
        nz    = trunc_geom_dict['rec_v_n'] + 8 
    
        right_x = np.linspace(x_min, x_max, nx)
        right_z = np.linspace(z_min, z_max, nz)        
        
        name_arr = ['ind_right_m4',
                    'ind_right_m3',
                    'ind_right_m2',
                    'ind_right_m1',
                    'ind_right_0' ,
                    'ind_right_p1',
                    'ind_right_p2',
                    'ind_right_p3',
                    'ind_right_p4']        
        
        ix = 0
        for x in right_x:
            name = name_arr[ix]
            ind_arr = np.zeros(nz, dtype='int64')
            iz = 0
            for z in right_z:
                ind_arr[iz] = pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation=True)
                iz         +=1
                
            ind_dict[name] = ind_arr        
            ix            += 1
        ###################### END RIGHT########################################
        
        ###################### BOT #############################################
        
        x_min = trunc_geom_dict['rec_x_l'] - 4*dx #pad by 4 for horizontal derivs
        x_max = trunc_geom_dict['rec_x_r'] + 4*dx #pad by 4 for horizontal derivs
        nx    = trunc_geom_dict['rec_h_n'] + 8 
        
        z_min = trunc_geom_dict['rec_z_b'] - 4*dz
        z_max = trunc_geom_dict['rec_z_b'] + 4*dz
        nz    = 9
    
        bot_x = np.linspace(x_min, x_max, nx)
        bot_z = np.linspace(z_min, z_max, nz)
        
        #Loop is not very efficient in python. But we cache the results, so only need to do once. 
        #Probably not necessary to optimize because of that
        name_arr = ['ind_bot_m4',
                    'ind_bot_m3',
                    'ind_bot_m2',
                    'ind_bot_m1',
                    'ind_bot_0' ,
                    'ind_bot_p1',
                    'ind_bot_p2',
                    'ind_bot_p3',
                    'ind_bot_p4']
        iz = 0
        for z in bot_z:
            name = name_arr[iz]
            ind_arr = np.zeros(nx, dtype='int64')
            ix = 0
            for x in bot_x:
                ind_arr[ix] = pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation=True)
                ix         +=1
                
            ind_dict[name] = ind_arr
            iz            += 1      
            
        ###################### END BOT #########################################
        
        ###################### LEFT ############################################
        
        x_min = trunc_geom_dict['rec_x_l'] - 4*dx #pad by 3 for horizontal derivs
        x_max = trunc_geom_dict['rec_x_l'] + 4*dx #pad by 3 for horizontal derivs
        nx    = 9 
        
        z_min = trunc_geom_dict['rec_z_t'] - 4*dz
        z_max = trunc_geom_dict['rec_z_b'] + 4*dz
        nz    = trunc_geom_dict['rec_v_n'] + 8 
    
        left_x = np.linspace(x_min, x_max, nx)
        left_z = np.linspace(z_min, z_max, nz)        
        
        name_arr = ['ind_left_m4',
                    'ind_left_m3',
                    'ind_left_m2',
                    'ind_left_m1',
                    'ind_left_0' ,
                    'ind_left_p1',
                    'ind_left_p2',
                    'ind_left_p3',
                    'ind_left_p4']       
        
        ix = 0
        for x in left_x:
            name = name_arr[ix]
            ind_arr = np.zeros(nz, dtype='int64')
            iz = 0
            for z in left_z:
                ind_arr[iz] = pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation=True)
                iz         +=1
                
            ind_dict[name] = ind_arr        
            ix            += 1
        
        ###################### END LEFT ########################################
        return ind_dict    
    
    #For now check if vs_2d is uniform on the 3 layers of the injections boundaries
    #I suspect it may be possible to inject on variable vs_2d as well if uniform vs_2d works, but should try later
    #Variable density would probably require VDA green's functions instead of CDA? Unless I could compensate easily for density
    def empirical_amplitude_fix_density_contrast(m, trunc_geom_dict, vp_2d, rho_2d, vs_2d, src_x, src_z, rho_src):
        uniform_spacing = trunc_geom_dict['dx'] 
        [x_min, _, z_min, _, _, _, _] = global_geometry(m)
        src_x_ind                     = np.round((src_x   - x_min)/uniform_spacing).astype('int32')
        src_z_ind                     = np.round((src_z   - z_min)/uniform_spacing).astype('int32')

        #If not providing manually, grab from 2d density model
        #when using a tapered density model this may not be the true density at the source!        
        if rho_src == None: 
            rho_src                   = rho_2d[src_z_ind, src_x_ind]
            
            
        _, _, _, rho_bdry             = get_lambda_mu_rho(m, trunc_geom_dict, vp_2d, rho_2d, vs_2d)
        multiplier                    = np.sqrt(rho_bdry/rho_src)
        return multiplier
        
        
    
    def get_lambda_mu_rho(m, trunc_geom_dict, vp_2d, rho_2d, vs_2d):
        #Get elastic parameters from velocity models.
        #Make sure to implement mu and lambda exactly in the same way as is done in the elastic C code 
        
        #Go over all possible boundary positions, and compute lam2mu, lam and mu exactly the same way as the elastic C code.
        #Return lam2mu, lam, c19 and rho. These are the arrays used in fd2d_update_SSG in function 'update_T_SSG'
        #Put the results in vectors the size of the vx, vz, tau_xx, tau_zz and tau_xz arrays
        
        x_pos_int, z_pos_int     = get_boundary_integral_positions(m, trunc_geom_dict)
        x_pos_extra, z_pos_extra = get_additional_required_boundary_fields_positions_new(m, trunc_geom_dict)
        
        x_pos = np.concatenate([x_pos_int, x_pos_extra])
        z_pos = np.concatenate([z_pos_int, z_pos_extra])
        
        #we now have all the required positions, loop over them
        npos   = len(x_pos)
        lam2mu = np.zeros(npos)
        lam    = np.zeros(npos)
        c19    = np.zeros(npos)
        rho    = np.zeros(npos)
        
        dx = m.x.delta
        dz = m.z.delta
        
        x_min = m.domain.x.lbound
        z_min = m.domain.z.lbound
        
        #A for loop is not very efficient, but probably not bottleneck here
        #We also compute at some positions for which we actually do not need lamda and mu
        #The tau_xx. tau_zz and tau_xz update arrays we later multiply with have zero columns.  
        for x, z in zip(x_pos,z_pos):
            #get associated index in 2d arrays of elastic parameters
            full_dom_ind_x = int(np.round((x-x_min)/dx))
            full_dom_ind_z = int(np.round((z-z_min)/dz))
            
            #get column index in the gathers tau_xx, tau_zz, tau_xz, vx, vz
            gather_ind         = pos_to_ind_int_bdry_gather(x, z, m, trunc_geom_dict, new_implementation=True)
        
            rho[gather_ind]    = rho_2d[full_dom_ind_z, full_dom_ind_x]
            lam2mu[gather_ind] = rho[gather_ind]*vp_2d[full_dom_ind_z, full_dom_ind_x]**2
            
            #mu is calculated this way in fd2d_model.c
            mu = rho[gather_ind]*vs_2d[full_dom_ind_z, full_dom_ind_x]**2
            
            lam[gather_ind] =  lam2mu[gather_ind] - 2*mu
            
            #when computing the shear stress updates, mu is averages between 4 cells in function 'update_T_SSG' in fd2d_update_SSG. See variable c19
            mu_left = rho[gather_ind]*vs_2d[full_dom_ind_z  , full_dom_ind_x-1]**2
            mu_bot  = rho[gather_ind]*vs_2d[full_dom_ind_z+1, full_dom_ind_x  ]**2
            mu_diag = rho[gather_ind]*vs_2d[full_dom_ind_z+1, full_dom_ind_x-1]**2
            c19[gather_ind] =  0.25*(mu + mu_left + mu_bot + mu_diag)
        
        return lam2mu, lam, c19, rho
    
    def compute_velocities(gb, m, rho_2d, dtdx, ind_dict, trunc_geom_dict, recording_period, ts_in, ts_dense):
        #Compute velocities from the CDA pressure values. Since pressure, tau_xz = 0, tau_xx = tau_zz = gb. 
        #Need stress derivatives for velocities. Since pressure, only need to compute pressure derivatives
        
        #We will have fewer velocity values than pressure values 
        #(since we only get velocities at positions we can evaluate the full derivative stencil)
        #but to make indexing easier I will just use the same indexes and sizes.
        #Will first compute derivatives of pressure at cells where the velocity stagger locations need it
        dp_dx = np.zeros_like(gb)
        dp_dz = np.zeros_like(gb)
        
        
        
        #It's possible to write the things down more compact for sure
        
        coe1    =  9./8
        coe2    = -1./24
        ###################### TOP #######################
        
        #Will compute Vz: We need Vz at more places than the injection boundary since its vertical derivatives give tau_zz
        #We also need its horizontal derivatives to compute tau_xz. These two requirements give slc_vz positions where vz needs to be known 
        #Recall that ind_dict['ind_top_0' ] and the other arrays like _p1, _m1 etc range from 4 cells left of middle left injection boundary to 4 right of middle right injection boundary
        slc_vz = 1+np.arange(trunc_geom_dict['rec_h_n']+5)  #slice (3 points left of middle left injection boundary and 2 right of middle right boundary
        
        #vertical deriv of tau_zz (pressure) evaluated at vz stagger points
        dp_dz[:,ind_dict['ind_top_m3'][slc_vz]] = coe1*(gb[:, ind_dict['ind_top_m2'][slc_vz]] - gb[:, ind_dict['ind_top_m3'][slc_vz]]) + \
                                                  coe2*(gb[:, ind_dict['ind_top_m1'][slc_vz]] - gb[:, ind_dict['ind_top_m4'][slc_vz]])
        dp_dz[:,ind_dict['ind_top_m2'][slc_vz]] = coe1*(gb[:, ind_dict['ind_top_m1'][slc_vz]] - gb[:, ind_dict['ind_top_m2'][slc_vz]]) + \
                                                  coe2*(gb[:, ind_dict['ind_top_0' ][slc_vz]] - gb[:, ind_dict['ind_top_m3'][slc_vz]])        
        dp_dz[:,ind_dict['ind_top_m1'][slc_vz]] = coe1*(gb[:, ind_dict['ind_top_0' ][slc_vz]] - gb[:, ind_dict['ind_top_m1'][slc_vz]]) + \
                                                  coe2*(gb[:, ind_dict['ind_top_p1'][slc_vz]] - gb[:, ind_dict['ind_top_m2'][slc_vz]])        
        dp_dz[:,ind_dict['ind_top_0' ][slc_vz]] = coe1*(gb[:, ind_dict['ind_top_p1'][slc_vz]] - gb[:, ind_dict['ind_top_0' ][slc_vz]]) + \
                                                  coe2*(gb[:, ind_dict['ind_top_p2'][slc_vz]] - gb[:, ind_dict['ind_top_m1'][slc_vz]])        
        dp_dz[:,ind_dict['ind_top_p1'][slc_vz]] = coe1*(gb[:, ind_dict['ind_top_p2'][slc_vz]] - gb[:, ind_dict['ind_top_p1'][slc_vz]]) + \
                                                  coe2*(gb[:, ind_dict['ind_top_p3'][slc_vz]] - gb[:, ind_dict['ind_top_0' ][slc_vz]])
        dp_dz[:,ind_dict['ind_top_p2'][slc_vz]] = coe1*(gb[:, ind_dict['ind_top_p3'][slc_vz]] - gb[:, ind_dict['ind_top_p2'][slc_vz]]) + \
                                                  coe2*(gb[:, ind_dict['ind_top_p4'][slc_vz]] - gb[:, ind_dict['ind_top_p1'][slc_vz]])
        
        #Will compute Vx now. Vertical derivatives of Vx are needed for tau_xz later. Horizontal derivatives are needed for tau_xx later.
        #So we need to compute Vx at a sufficient amount of layers.
        slc_vx = 2+np.arange(trunc_geom_dict['rec_h_n']+5) #slice (2 points on left of left middle boundary and 3 on right of right middle injection boundary)
        dp_dx[:,ind_dict['ind_top_m2'][slc_vx]] = coe1*(gb[:, ind_dict['ind_top_m2'][slc_vx  ]] - gb[:, ind_dict['ind_top_m2'][slc_vx-1]]) + \
                                                  coe2*(gb[:, ind_dict['ind_top_m2'][slc_vx+1]] - gb[:, ind_dict['ind_top_m2'][slc_vx-2]])
        dp_dx[:,ind_dict['ind_top_m1'][slc_vx]] = coe1*(gb[:, ind_dict['ind_top_m1'][slc_vx  ]] - gb[:, ind_dict['ind_top_m1'][slc_vx-1]]) + \
                                                  coe2*(gb[:, ind_dict['ind_top_m1'][slc_vx+1]] - gb[:, ind_dict['ind_top_m1'][slc_vx-2]])        
        dp_dx[:,ind_dict['ind_top_0' ][slc_vx]] = coe1*(gb[:, ind_dict['ind_top_0' ][slc_vx  ]] - gb[:, ind_dict['ind_top_0' ][slc_vx-1]]) + \
                                                  coe2*(gb[:, ind_dict['ind_top_0' ][slc_vx+1]] - gb[:, ind_dict['ind_top_0' ][slc_vx-2]])
        dp_dx[:,ind_dict['ind_top_p1'][slc_vx]] = coe1*(gb[:, ind_dict['ind_top_p1'][slc_vx  ]] - gb[:, ind_dict['ind_top_p1'][slc_vx-1]]) + \
                                                  coe2*(gb[:, ind_dict['ind_top_p1'][slc_vx+1]] - gb[:, ind_dict['ind_top_p1'][slc_vx-2]])                
        dp_dx[:,ind_dict['ind_top_p2'][slc_vx]] = coe1*(gb[:, ind_dict['ind_top_p2'][slc_vx  ]] - gb[:, ind_dict['ind_top_p2'][slc_vx-1]]) + \
                                                  coe2*(gb[:, ind_dict['ind_top_p2'][slc_vx+1]] - gb[:, ind_dict['ind_top_p2'][slc_vx-2]])
        dp_dx[:,ind_dict['ind_top_p3'][slc_vx]] = coe1*(gb[:, ind_dict['ind_top_p3'][slc_vx  ]] - gb[:, ind_dict['ind_top_p3'][slc_vx-1]]) + \
                                                  coe2*(gb[:, ind_dict['ind_top_p3'][slc_vx+1]] - gb[:, ind_dict['ind_top_p3'][slc_vx-2]])
                                                  
        ###################### RIGHT #######################
        slc_vz = 1 + np.arange(trunc_geom_dict['rec_v_n']+5) #3 points above top middle injection boundary and 2 points below bot middle injection boundary. Needed for tau_zz. Will need multiple vertical slices as well since tau_xz needs horizontal derivatives of vz
        
        #need from m3 (3 on left of middle right inj boundary) to p2 (2 on the right for middle right inj boundary). Need these vertical slices to be able to compute tau_xz from these vz later on the 3 injection boundaries
        dp_dz[:,ind_dict['ind_right_m3'][slc_vz]] = coe1*(gb[:, ind_dict['ind_right_m3'][slc_vz+1]] - gb[:, ind_dict['ind_right_m3'][slc_vz  ]]) + \
                                                    coe2*(gb[:, ind_dict['ind_right_m3'][slc_vz+2]] - gb[:, ind_dict['ind_right_m3'][slc_vz-1]])
        dp_dz[:,ind_dict['ind_right_m2'][slc_vz]] = coe1*(gb[:, ind_dict['ind_right_m2'][slc_vz+1]] - gb[:, ind_dict['ind_right_m2'][slc_vz  ]]) + \
                                                    coe2*(gb[:, ind_dict['ind_right_m2'][slc_vz+2]] - gb[:, ind_dict['ind_right_m2'][slc_vz-1]])
        dp_dz[:,ind_dict['ind_right_m1'][slc_vz]] = coe1*(gb[:, ind_dict['ind_right_m1'][slc_vz+1]] - gb[:, ind_dict['ind_right_m1'][slc_vz  ]]) + \
                                                    coe2*(gb[:, ind_dict['ind_right_m1'][slc_vz+2]] - gb[:, ind_dict['ind_right_m1'][slc_vz-1]])                                                    
        dp_dz[:,ind_dict['ind_right_0' ][slc_vz]] = coe1*(gb[:, ind_dict['ind_right_0' ][slc_vz+1]] - gb[:, ind_dict['ind_right_0' ][slc_vz  ]]) + \
                                                    coe2*(gb[:, ind_dict['ind_right_0' ][slc_vz+2]] - gb[:, ind_dict['ind_right_0' ][slc_vz-1]])
        dp_dz[:,ind_dict['ind_right_p1'][slc_vz]] = coe1*(gb[:, ind_dict['ind_right_p1'][slc_vz+1]] - gb[:, ind_dict['ind_right_p1'][slc_vz  ]]) + \
                                                    coe2*(gb[:, ind_dict['ind_right_p1'][slc_vz+2]] - gb[:, ind_dict['ind_right_p1'][slc_vz-1]])
        dp_dz[:,ind_dict['ind_right_p2'][slc_vz]] = coe1*(gb[:, ind_dict['ind_right_p2'][slc_vz+1]] - gb[:, ind_dict['ind_right_p2'][slc_vz  ]]) + \
                                                    coe2*(gb[:, ind_dict['ind_right_p2'][slc_vz+2]] - gb[:, ind_dict['ind_right_p2'][slc_vz-1]])

        slc_vx = 2 + np.arange(trunc_geom_dict['rec_v_n']+5) #2 points above top middle injection boundary and 3 points below bot middle injection boundary. Vertical derivative of vx is needed for tau_xz. Will need multiple vertical slices as well since tau_xx needs horizontal derivatives of vx
        
        #need from m2 (2 on left of middle right inj boundary) to p3 (3 on the right of middle right inj boundary). These bounds are determined by tau_xx which will require horizontal derivatives of vx later
        dp_dx[:,ind_dict['ind_right_m2'][slc_vx]] = coe1*(gb[:, ind_dict['ind_right_m2'][slc_vx]] - gb[:, ind_dict['ind_right_m3'][slc_vx]]) + \
                                                    coe2*(gb[:, ind_dict['ind_right_m1'][slc_vx]] - gb[:, ind_dict['ind_right_m4'][slc_vx]])
        dp_dx[:,ind_dict['ind_right_m1'][slc_vx]] = coe1*(gb[:, ind_dict['ind_right_m1'][slc_vx]] - gb[:, ind_dict['ind_right_m2'][slc_vx]]) + \
                                                    coe2*(gb[:, ind_dict['ind_right_0' ][slc_vx]] - gb[:, ind_dict['ind_right_m3'][slc_vx]])
        dp_dx[:,ind_dict['ind_right_0' ][slc_vx]] = coe1*(gb[:, ind_dict['ind_right_0' ][slc_vx]] - gb[:, ind_dict['ind_right_m1'][slc_vx]]) + \
                                                    coe2*(gb[:, ind_dict['ind_right_p1'][slc_vx]] - gb[:, ind_dict['ind_right_m2'][slc_vx]])
        dp_dx[:,ind_dict['ind_right_p1'][slc_vx]] = coe1*(gb[:, ind_dict['ind_right_p1'][slc_vx]] - gb[:, ind_dict['ind_right_0' ][slc_vx]]) + \
                                                    coe2*(gb[:, ind_dict['ind_right_p2'][slc_vx]] - gb[:, ind_dict['ind_right_m1'][slc_vx]])
        dp_dx[:,ind_dict['ind_right_p2'][slc_vx]] = coe1*(gb[:, ind_dict['ind_right_p2'][slc_vx]] - gb[:, ind_dict['ind_right_p1'][slc_vx]]) + \
                                                    coe2*(gb[:, ind_dict['ind_right_p3'][slc_vx]] - gb[:, ind_dict['ind_right_0' ][slc_vx]])
        dp_dx[:,ind_dict['ind_right_p3'][slc_vx]] = coe1*(gb[:, ind_dict['ind_right_p3'][slc_vx]] - gb[:, ind_dict['ind_right_p2'][slc_vx]]) + \
                                                    coe2*(gb[:, ind_dict['ind_right_p4'][slc_vx]] - gb[:, ind_dict['ind_right_p1'][slc_vx]])
    
        ###################### BOT #########################
        slc_vz = 1+np.arange(trunc_geom_dict['rec_h_n']+5)  #slice (3 points left of middle left injection boundary and 2 right of middle right boundary
    
        #need from m3 (3 on top of middle bot inj boundary) to p2 (2 on the bot for middle bot inj boundary). Need these horizontal slices to be able to compute tau_zz from these vz later on the 3 injection boundaries
        dp_dz[:,ind_dict['ind_bot_m3'][slc_vz]] = coe1*(gb[:, ind_dict['ind_bot_m2'][slc_vz]] - gb[:, ind_dict['ind_bot_m3'][slc_vz]]) + \
                                                  coe2*(gb[:, ind_dict['ind_bot_m1'][slc_vz]] - gb[:, ind_dict['ind_bot_m4'][slc_vz]])
        dp_dz[:,ind_dict['ind_bot_m2'][slc_vz]] = coe1*(gb[:, ind_dict['ind_bot_m1'][slc_vz]] - gb[:, ind_dict['ind_bot_m2'][slc_vz]]) + \
                                                  coe2*(gb[:, ind_dict['ind_bot_0' ][slc_vz]] - gb[:, ind_dict['ind_bot_m3'][slc_vz]])
        dp_dz[:,ind_dict['ind_bot_m1'][slc_vz]] = coe1*(gb[:, ind_dict['ind_bot_0' ][slc_vz]] - gb[:, ind_dict['ind_bot_m1'][slc_vz]]) + \
                                                  coe2*(gb[:, ind_dict['ind_bot_p1'][slc_vz]] - gb[:, ind_dict['ind_bot_m2'][slc_vz]])
        dp_dz[:,ind_dict['ind_bot_0' ][slc_vz]] = coe1*(gb[:, ind_dict['ind_bot_p1'][slc_vz]] - gb[:, ind_dict['ind_bot_0' ][slc_vz]]) + \
                                                  coe2*(gb[:, ind_dict['ind_bot_p2'][slc_vz]] - gb[:, ind_dict['ind_bot_m1'][slc_vz]])
        dp_dz[:,ind_dict['ind_bot_p1'][slc_vz]] = coe1*(gb[:, ind_dict['ind_bot_p2'][slc_vz]] - gb[:, ind_dict['ind_bot_p1'][slc_vz]]) + \
                                                  coe2*(gb[:, ind_dict['ind_bot_p3'][slc_vz]] - gb[:, ind_dict['ind_bot_0' ][slc_vz]])
        dp_dz[:,ind_dict['ind_bot_p2'][slc_vz]] = coe1*(gb[:, ind_dict['ind_bot_p3'][slc_vz]] - gb[:, ind_dict['ind_bot_p2'][slc_vz]]) + \
                                                  coe2*(gb[:, ind_dict['ind_bot_p4'][slc_vz]] - gb[:, ind_dict['ind_bot_p1'][slc_vz]])                                                  
        
        slc_vx = 2+np.arange(trunc_geom_dict['rec_h_n']+5)  #slice (2 points left of middle left injection boundary and 3 right of middle right boundary
        
        #need from m2 (2 op top of middle bot inj boundary) to p3 (3 on the bot for middle bot inj boundary). Need those horizontal slices to be able to compute tau_xz which will require vertical derivatives of vx. 
        dp_dx[:,ind_dict['ind_bot_m2'][slc_vx]] = coe1*(gb[:, ind_dict['ind_bot_m2'][slc_vx  ]] - gb[:, ind_dict['ind_bot_m2'][slc_vx-1]]) + \
                                                  coe2*(gb[:, ind_dict['ind_bot_m2'][slc_vx+1]] - gb[:, ind_dict['ind_bot_m2'][slc_vx-2]])
        dp_dx[:,ind_dict['ind_bot_m1'][slc_vx]] = coe1*(gb[:, ind_dict['ind_bot_m1'][slc_vx  ]] - gb[:, ind_dict['ind_bot_m1'][slc_vx-1]]) + \
                                                  coe2*(gb[:, ind_dict['ind_bot_m1'][slc_vx+1]] - gb[:, ind_dict['ind_bot_m1'][slc_vx-2]])
        dp_dx[:,ind_dict['ind_bot_0' ][slc_vx]] = coe1*(gb[:, ind_dict['ind_bot_0' ][slc_vx  ]] - gb[:, ind_dict['ind_bot_0' ][slc_vx-1]]) + \
                                                  coe2*(gb[:, ind_dict['ind_bot_0' ][slc_vx+1]] - gb[:, ind_dict['ind_bot_0' ][slc_vx-2]])
        dp_dx[:,ind_dict['ind_bot_p1'][slc_vx]] = coe1*(gb[:, ind_dict['ind_bot_p1'][slc_vx  ]] - gb[:, ind_dict['ind_bot_p1'][slc_vx-1]]) + \
                                                  coe2*(gb[:, ind_dict['ind_bot_p1'][slc_vx+1]] - gb[:, ind_dict['ind_bot_p1'][slc_vx-2]])
        dp_dx[:,ind_dict['ind_bot_p2'][slc_vx]] = coe1*(gb[:, ind_dict['ind_bot_p2'][slc_vx  ]] - gb[:, ind_dict['ind_bot_p2'][slc_vx-1]]) + \
                                                  coe2*(gb[:, ind_dict['ind_bot_p2'][slc_vx+1]] - gb[:, ind_dict['ind_bot_p2'][slc_vx-2]])
        dp_dx[:,ind_dict['ind_bot_p3'][slc_vx]] = coe1*(gb[:, ind_dict['ind_bot_p3'][slc_vx  ]] - gb[:, ind_dict['ind_bot_p3'][slc_vx-1]]) + \
                                                  coe2*(gb[:, ind_dict['ind_bot_p3'][slc_vx+1]] - gb[:, ind_dict['ind_bot_p3'][slc_vx-2]])                                                  
        
        ###################### LEFT ########################
        slc_vz = 1 + np.arange(trunc_geom_dict['rec_v_n']+5) #3 points above top middle injection boundary and 2 points below bot middle injection boundary. Needed for tau_zz. Will need multiple vertical slices as well since tau_xz needs horizontal derivatives of vz

        #need from m3 (3 on left of middle left inj boundary) to p2 (2 on right of middle left inj boundary). Need these vertical slices to be able to take the horizontal derivatives of vz later (required for tau_xz)
        dp_dz[:,ind_dict['ind_left_m3'][slc_vz]] = coe1*(gb[:, ind_dict['ind_left_m3'][slc_vz+1]] - gb[:, ind_dict['ind_left_m3'][slc_vz  ]]) + \
                                                   coe2*(gb[:, ind_dict['ind_left_m3'][slc_vz+2]] - gb[:, ind_dict['ind_left_m3'][slc_vz-1]])
        dp_dz[:,ind_dict['ind_left_m2'][slc_vz]] = coe1*(gb[:, ind_dict['ind_left_m2'][slc_vz+1]] - gb[:, ind_dict['ind_left_m2'][slc_vz  ]]) + \
                                                   coe2*(gb[:, ind_dict['ind_left_m2'][slc_vz+2]] - gb[:, ind_dict['ind_left_m2'][slc_vz-1]])
        dp_dz[:,ind_dict['ind_left_m1'][slc_vz]] = coe1*(gb[:, ind_dict['ind_left_m1'][slc_vz+1]] - gb[:, ind_dict['ind_left_m1'][slc_vz  ]]) + \
                                                   coe2*(gb[:, ind_dict['ind_left_m1'][slc_vz+2]] - gb[:, ind_dict['ind_left_m1'][slc_vz-1]])
        dp_dz[:,ind_dict['ind_left_0' ][slc_vz]] = coe1*(gb[:, ind_dict['ind_left_0' ][slc_vz+1]] - gb[:, ind_dict['ind_left_0' ][slc_vz  ]]) + \
                                                   coe2*(gb[:, ind_dict['ind_left_0' ][slc_vz+2]] - gb[:, ind_dict['ind_left_0' ][slc_vz-1]])
        dp_dz[:,ind_dict['ind_left_p1'][slc_vz]] = coe1*(gb[:, ind_dict['ind_left_p1'][slc_vz+1]] - gb[:, ind_dict['ind_left_p1'][slc_vz  ]]) + \
                                                   coe2*(gb[:, ind_dict['ind_left_p1'][slc_vz+2]] - gb[:, ind_dict['ind_left_p1'][slc_vz-1]])
        dp_dz[:,ind_dict['ind_left_p2'][slc_vz]] = coe1*(gb[:, ind_dict['ind_left_p2'][slc_vz+1]] - gb[:, ind_dict['ind_left_p2'][slc_vz  ]]) + \
                                                   coe2*(gb[:, ind_dict['ind_left_p2'][slc_vz+2]] - gb[:, ind_dict['ind_left_p2'][slc_vz-1]])                                                                                                      
                                                   
        slc_vx = 2 + np.arange(trunc_geom_dict['rec_v_n']+5) #2 points above top middle injection boundary and 3 points below bot middle injection boundary. Vertical derivative of vx is needed for tau_xz. Will need multiple vertical slices as well since tau_xx needs horizontal derivatives of vx
        
        #need from m2 (2 on left of middle left inj boundary) to p3 (3 on right of middle left inj boundary). Need these vertical slices to be able to take the horizontal derivatives of vx later (required for tau_xx)
        dp_dx[:,ind_dict['ind_left_m2'][slc_vx]] = coe1*(gb[:, ind_dict['ind_left_m2'][slc_vx]] - gb[:, ind_dict['ind_left_m3'][slc_vx]]) + \
                                                   coe2*(gb[:, ind_dict['ind_left_m1'][slc_vx]] - gb[:, ind_dict['ind_left_m4'][slc_vx]])
        dp_dx[:,ind_dict['ind_left_m1'][slc_vx]] = coe1*(gb[:, ind_dict['ind_left_m1'][slc_vx]] - gb[:, ind_dict['ind_left_m2'][slc_vx]]) + \
                                                   coe2*(gb[:, ind_dict['ind_left_0' ][slc_vx]] - gb[:, ind_dict['ind_left_m3'][slc_vx]])
        dp_dx[:,ind_dict['ind_left_0' ][slc_vx]] = coe1*(gb[:, ind_dict['ind_left_0' ][slc_vx]] - gb[:, ind_dict['ind_left_m1'][slc_vx]]) + \
                                                   coe2*(gb[:, ind_dict['ind_left_p1'][slc_vx]] - gb[:, ind_dict['ind_left_m2'][slc_vx]])
        dp_dx[:,ind_dict['ind_left_p1'][slc_vx]] = coe1*(gb[:, ind_dict['ind_left_p1'][slc_vx]] - gb[:, ind_dict['ind_left_0' ][slc_vx]]) + \
                                                   coe2*(gb[:, ind_dict['ind_left_p2'][slc_vx]] - gb[:, ind_dict['ind_left_m1'][slc_vx]])
        dp_dx[:,ind_dict['ind_left_p2'][slc_vx]] = coe1*(gb[:, ind_dict['ind_left_p2'][slc_vx]] - gb[:, ind_dict['ind_left_p1'][slc_vx]]) + \
                                                   coe2*(gb[:, ind_dict['ind_left_p3'][slc_vx]] - gb[:, ind_dict['ind_left_0' ][slc_vx]])
        dp_dx[:,ind_dict['ind_left_p3'][slc_vx]] = coe1*(gb[:, ind_dict['ind_left_p3'][slc_vx]] - gb[:, ind_dict['ind_left_p2'][slc_vx]]) + \
                                                   coe2*(gb[:, ind_dict['ind_left_p4'][slc_vx]] - gb[:, ind_dict['ind_left_p1'][slc_vx]])
                                                                                                                                                                                                                                                                                                                  
        #WE COMPUTED THE PRESSURE DERIVATIVES. NOW TURN THIS INTO VELOCITIES USING THE DENSITY

        _, _, _, rho = get_lambda_mu_rho(m, trunc_geom_dict, vp_2d, rho_2d, vs_2d)
        
        #get scaled velocity time derivative in similar way as in C code. Shear stress contribution to velocity is zero since CDA solver assumes tau_xz = 0
        #Use appropriate average densities at stagger points just like in elastic C code
        
        #initialize. Will later have to divide, so initialize as 1's (so locations where dp_dx and dp_dz = 0 (locations where we do not compute velocities) are not getting wrong value
        midpoint_rho_vx = np.ones(gb.shape[1])
        midpoint_rho_vz = np.ones(gb.shape[1])
        
        #top
        slc_vx = 2+np.arange(trunc_geom_dict['rec_h_n']+5)
        slc_vz = 1+np.arange(trunc_geom_dict['rec_h_n']+5)
        
        midpoint_rho_vx[ind_dict['ind_top_m2'][slc_vx]] = 0.5*(rho[ind_dict['ind_top_m2'][slc_vx]] + rho[ind_dict['ind_top_m2'][slc_vx-1]])
        midpoint_rho_vx[ind_dict['ind_top_m1'][slc_vx]] = 0.5*(rho[ind_dict['ind_top_m1'][slc_vx]] + rho[ind_dict['ind_top_m1'][slc_vx-1]])
        midpoint_rho_vx[ind_dict['ind_top_0' ][slc_vx]] = 0.5*(rho[ind_dict['ind_top_0' ][slc_vx]] + rho[ind_dict['ind_top_0' ][slc_vx-1]])
        midpoint_rho_vx[ind_dict['ind_top_p1'][slc_vx]] = 0.5*(rho[ind_dict['ind_top_p1'][slc_vx]] + rho[ind_dict['ind_top_p1'][slc_vx-1]])
        midpoint_rho_vx[ind_dict['ind_top_p2'][slc_vx]] = 0.5*(rho[ind_dict['ind_top_p2'][slc_vx]] + rho[ind_dict['ind_top_p2'][slc_vx-1]])
        midpoint_rho_vx[ind_dict['ind_top_p3'][slc_vx]] = 0.5*(rho[ind_dict['ind_top_p3'][slc_vx]] + rho[ind_dict['ind_top_p3'][slc_vx-1]])
        
        midpoint_rho_vz[ind_dict['ind_top_m3'][slc_vz]] = 0.5*(rho[ind_dict['ind_top_m3'][slc_vz]] + rho[ind_dict['ind_top_m2'][slc_vz]])
        midpoint_rho_vz[ind_dict['ind_top_m2'][slc_vz]] = 0.5*(rho[ind_dict['ind_top_m2'][slc_vz]] + rho[ind_dict['ind_top_m1'][slc_vz]])
        midpoint_rho_vz[ind_dict['ind_top_m1'][slc_vz]] = 0.5*(rho[ind_dict['ind_top_m1'][slc_vz]] + rho[ind_dict['ind_top_0' ][slc_vz]])
        midpoint_rho_vz[ind_dict['ind_top_0' ][slc_vz]] = 0.5*(rho[ind_dict['ind_top_0' ][slc_vz]] + rho[ind_dict['ind_top_p1'][slc_vz]])
        midpoint_rho_vz[ind_dict['ind_top_p1'][slc_vz]] = 0.5*(rho[ind_dict['ind_top_p1'][slc_vz]] + rho[ind_dict['ind_top_p2'][slc_vz]])
        midpoint_rho_vz[ind_dict['ind_top_p2'][slc_vz]] = 0.5*(rho[ind_dict['ind_top_p2'][slc_vz]] + rho[ind_dict['ind_top_p3'][slc_vz]])
        
        #right
        slc_vx = 2 + np.arange(trunc_geom_dict['rec_v_n']+5)
        slc_vz = 1 + np.arange(trunc_geom_dict['rec_v_n']+5)
        
        midpoint_rho_vx[ind_dict['ind_right_m2'][slc_vx]] = 0.5*(rho[ind_dict['ind_right_m2'][slc_vx]] + rho[ind_dict['ind_right_m3'][slc_vx]])
        midpoint_rho_vx[ind_dict['ind_right_m1'][slc_vx]] = 0.5*(rho[ind_dict['ind_right_m1'][slc_vx]] + rho[ind_dict['ind_right_m2'][slc_vx]])
        midpoint_rho_vx[ind_dict['ind_right_0' ][slc_vx]] = 0.5*(rho[ind_dict['ind_right_0' ][slc_vx]] + rho[ind_dict['ind_right_m1'][slc_vx]])
        midpoint_rho_vx[ind_dict['ind_right_p1'][slc_vx]] = 0.5*(rho[ind_dict['ind_right_p1'][slc_vx]] + rho[ind_dict['ind_right_0' ][slc_vx]])
        midpoint_rho_vx[ind_dict['ind_right_p2'][slc_vx]] = 0.5*(rho[ind_dict['ind_right_p2'][slc_vx]] + rho[ind_dict['ind_right_p1'][slc_vx]])
        midpoint_rho_vx[ind_dict['ind_right_p3'][slc_vx]] = 0.5*(rho[ind_dict['ind_right_p3'][slc_vx]] + rho[ind_dict['ind_right_p2'][slc_vx]])
        
        midpoint_rho_vz[ind_dict['ind_right_m3'][slc_vz]] = 0.5*(rho[ind_dict['ind_right_m3'][slc_vz]] + rho[ind_dict['ind_right_m3'][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_right_m2'][slc_vz]] = 0.5*(rho[ind_dict['ind_right_m2'][slc_vz]] + rho[ind_dict['ind_right_m2'][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_right_m1'][slc_vz]] = 0.5*(rho[ind_dict['ind_right_m1'][slc_vz]] + rho[ind_dict['ind_right_m1'][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_right_0' ][slc_vz]] = 0.5*(rho[ind_dict['ind_right_0' ][slc_vz]] + rho[ind_dict['ind_right_0' ][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_right_p1'][slc_vz]] = 0.5*(rho[ind_dict['ind_right_p1'][slc_vz]] + rho[ind_dict['ind_right_p1'][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_right_p2'][slc_vz]] = 0.5*(rho[ind_dict['ind_right_p2'][slc_vz]] + rho[ind_dict['ind_right_p2'][slc_vz+1]])
        
        #bot
        slc_vx = 2+np.arange(trunc_geom_dict['rec_h_n']+5)
        slc_vz = 1+np.arange(trunc_geom_dict['rec_h_n']+5)
        
        midpoint_rho_vx[ind_dict['ind_bot_m2'][slc_vx]] = 0.5*(rho[ind_dict['ind_bot_m2'][slc_vx]] + rho[ind_dict['ind_bot_m2'][slc_vx-1]])
        midpoint_rho_vx[ind_dict['ind_bot_m1'][slc_vx]] = 0.5*(rho[ind_dict['ind_bot_m1'][slc_vx]] + rho[ind_dict['ind_bot_m1'][slc_vx-1]])
        midpoint_rho_vx[ind_dict['ind_bot_0' ][slc_vx]] = 0.5*(rho[ind_dict['ind_bot_0' ][slc_vx]] + rho[ind_dict['ind_bot_0' ][slc_vx-1]])
        midpoint_rho_vx[ind_dict['ind_bot_p1'][slc_vx]] = 0.5*(rho[ind_dict['ind_bot_p1'][slc_vx]] + rho[ind_dict['ind_bot_p1'][slc_vx-1]])
        midpoint_rho_vx[ind_dict['ind_bot_p2'][slc_vx]] = 0.5*(rho[ind_dict['ind_bot_p2'][slc_vx]] + rho[ind_dict['ind_bot_p2'][slc_vx-1]])
        midpoint_rho_vx[ind_dict['ind_bot_p3'][slc_vx]] = 0.5*(rho[ind_dict['ind_bot_p3'][slc_vx]] + rho[ind_dict['ind_bot_p3'][slc_vx-1]])
        
        midpoint_rho_vz[ind_dict['ind_bot_m3'][slc_vz]] = 0.5*(rho[ind_dict['ind_bot_m3'][slc_vz]] + rho[ind_dict['ind_bot_m3'][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_bot_m2'][slc_vz]] = 0.5*(rho[ind_dict['ind_bot_m2'][slc_vz]] + rho[ind_dict['ind_bot_m2'][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_bot_m1'][slc_vz]] = 0.5*(rho[ind_dict['ind_bot_m1'][slc_vz]] + rho[ind_dict['ind_bot_m1'][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_bot_0' ][slc_vz]] = 0.5*(rho[ind_dict['ind_bot_0' ][slc_vz]] + rho[ind_dict['ind_bot_0' ][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_bot_p1'][slc_vz]] = 0.5*(rho[ind_dict['ind_bot_p1'][slc_vz]] + rho[ind_dict['ind_bot_p1'][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_bot_p2'][slc_vz]] = 0.5*(rho[ind_dict['ind_bot_p2'][slc_vz]] + rho[ind_dict['ind_bot_p2'][slc_vz+1]])
        
        #left
        slc_vx = 2 + np.arange(trunc_geom_dict['rec_v_n']+5)
        slc_vz = 1 + np.arange(trunc_geom_dict['rec_v_n']+5)
        
        midpoint_rho_vx[ind_dict['ind_left_m2'][slc_vx]] = 0.5*(rho[ind_dict['ind_left_m2'][slc_vx]] + rho[ind_dict['ind_left_m3'][slc_vx]])
        midpoint_rho_vx[ind_dict['ind_left_m1'][slc_vx]] = 0.5*(rho[ind_dict['ind_left_m1'][slc_vx]] + rho[ind_dict['ind_left_m2'][slc_vx]])
        midpoint_rho_vx[ind_dict['ind_left_0' ][slc_vx]] = 0.5*(rho[ind_dict['ind_left_0' ][slc_vx]] + rho[ind_dict['ind_left_m1'][slc_vx]])
        midpoint_rho_vx[ind_dict['ind_left_p1'][slc_vx]] = 0.5*(rho[ind_dict['ind_left_p1'][slc_vx]] + rho[ind_dict['ind_left_0' ][slc_vx]])
        midpoint_rho_vx[ind_dict['ind_left_p2'][slc_vx]] = 0.5*(rho[ind_dict['ind_left_p2'][slc_vx]] + rho[ind_dict['ind_left_p1'][slc_vx]])
        midpoint_rho_vx[ind_dict['ind_left_p3'][slc_vx]] = 0.5*(rho[ind_dict['ind_left_p3'][slc_vx]] + rho[ind_dict['ind_left_p2'][slc_vx]])
        
        midpoint_rho_vz[ind_dict['ind_left_m3'][slc_vz]] = 0.5*(rho[ind_dict['ind_left_m3'][slc_vz]] + rho[ind_dict['ind_left_m3'][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_left_m2'][slc_vz]] = 0.5*(rho[ind_dict['ind_left_m2'][slc_vz]] + rho[ind_dict['ind_left_m2'][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_left_m1'][slc_vz]] = 0.5*(rho[ind_dict['ind_left_m1'][slc_vz]] + rho[ind_dict['ind_left_m1'][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_left_0' ][slc_vz]] = 0.5*(rho[ind_dict['ind_left_0' ][slc_vz]] + rho[ind_dict['ind_left_0' ][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_left_p1'][slc_vz]] = 0.5*(rho[ind_dict['ind_left_p1'][slc_vz]] + rho[ind_dict['ind_left_p1'][slc_vz+1]])
        midpoint_rho_vz[ind_dict['ind_left_p2'][slc_vz]] = 0.5*(rho[ind_dict['ind_left_p2'][slc_vz]] + rho[ind_dict['ind_left_p2'][slc_vz+1]]) 
        
        vx_updates = dp_dx*dtdx/midpoint_rho_vx
        vz_updates = dp_dz*dtdx/midpoint_rho_vz
        
        #IF THE INPUT GATHER WAS SUBSAMPLED, THE vx_updates and vz_updates arrays will be exactly the same as the not subsampled version at the timesteps where ts_in matches ts_sub 
        #If recording_period != 1 (subsampled), interpolate vx_updates and vz_updates. Will still agree at times where ts_in = ts_sub
        #If subsampled, interpolate these velocities
        if recording_period != 1:
            vx_updates = interpolate_gather_in_time_2d(vx_updates, ts_in, ts_dense)
            vz_updates = interpolate_gather_in_time_2d(vz_updates, ts_in, ts_dense)
        
        
        #integrate these updates
        vx = np.cumsum(vx_updates, axis=0)
        vz = np.cumsum(vz_updates, axis=0)
        
        return vx, vz
        
    def compute_stresses(vx, vz, m, vp_2d, rho_2d, vs_2d, dtdx, ind_dict, trunc_geom_dict):
        #Compute stresses from the velocities we computed in 'compute_velocities'.  
        #These stresses will depend on lambda and mu of the injection domain.
        #These lambda and mu will be extracted from the velocity models using the same logic as in the C code
        #Will need to take both horizontal derivatives of both vx and vz.
        
        dvx_dx = np.zeros_like(vx)
        dvx_dz = np.zeros_like(vx)
        
        dvz_dx = np.zeros_like(vz)
        dvz_dz = np.zeros_like(vz)        
        
        #It's possible to write the things down more compact for sure
        
        coe1    =  9./8
        coe2    = -1./24

        ###################### TOP #######################
        
        slc = 3+np.arange(trunc_geom_dict['rec_h_n']+2) #slice (From outer injection boundary to outer injection boundary)

        #dv_x/dx        
        dvx_dx[:,ind_dict['ind_top_m1'][slc]] = coe1*(vx[:, ind_dict['ind_top_m1'][slc+1]] - vx[:, ind_dict['ind_top_m1'][slc  ]]) + \
                                                coe2*(vx[:, ind_dict['ind_top_m1'][slc+2]] - vx[:, ind_dict['ind_top_m1'][slc-1]])        
        dvx_dx[:,ind_dict['ind_top_0' ][slc]] = coe1*(vx[:, ind_dict['ind_top_0' ][slc+1]] - vx[:, ind_dict['ind_top_0' ][slc  ]]) + \
                                                coe2*(vx[:, ind_dict['ind_top_0' ][slc+2]] - vx[:, ind_dict['ind_top_0' ][slc-1]])
        dvx_dx[:,ind_dict['ind_top_p1'][slc]] = coe1*(vx[:, ind_dict['ind_top_p1'][slc+1]] - vx[:, ind_dict['ind_top_p1'][slc  ]]) + \
                                                coe2*(vx[:, ind_dict['ind_top_p1'][slc+2]] - vx[:, ind_dict['ind_top_p1'][slc-1]])
                                                   
        #dv_x/dz
        dvx_dz[:,ind_dict['ind_top_m1'][slc]] = coe1*(vx[:, ind_dict['ind_top_0' ][slc]] - vx[:, ind_dict['ind_top_m1'][slc]]) + \
                                                coe2*(vx[:, ind_dict['ind_top_p1'][slc]] - vx[:, ind_dict['ind_top_m2'][slc]])
        dvx_dz[:,ind_dict['ind_top_0' ][slc]] = coe1*(vx[:, ind_dict['ind_top_p1'][slc]] - vx[:, ind_dict['ind_top_0' ][slc]]) + \
                                                coe2*(vx[:, ind_dict['ind_top_p2'][slc]] - vx[:, ind_dict['ind_top_m1'][slc]])                                                
        dvx_dz[:,ind_dict['ind_top_p1'][slc]] = coe1*(vx[:, ind_dict['ind_top_p2'][slc]] - vx[:, ind_dict['ind_top_p1'][slc]]) + \
                                                coe2*(vx[:, ind_dict['ind_top_p3'][slc]] - vx[:, ind_dict['ind_top_0' ][slc]])
                                                
        #dv_z/dx
        dvz_dx[:,ind_dict['ind_top_m1'][slc]] = coe1*(vz[:, ind_dict['ind_top_m1'][slc  ]] - vz[:, ind_dict['ind_top_m1'][slc-1]]) + \
                                                coe2*(vz[:, ind_dict['ind_top_m1'][slc+1]] - vz[:, ind_dict['ind_top_m1'][slc-2]])
        dvz_dx[:,ind_dict['ind_top_0' ][slc]] = coe1*(vz[:, ind_dict['ind_top_0' ][slc  ]] - vz[:, ind_dict['ind_top_0' ][slc-1]]) + \
                                                coe2*(vz[:, ind_dict['ind_top_0' ][slc+1]] - vz[:, ind_dict['ind_top_0' ][slc-2]])
        dvz_dx[:,ind_dict['ind_top_p1'][slc]] = coe1*(vz[:, ind_dict['ind_top_p1'][slc  ]] - vz[:, ind_dict['ind_top_p1'][slc-1]]) + \
                                                coe2*(vz[:, ind_dict['ind_top_p1'][slc+1]] - vz[:, ind_dict['ind_top_p1'][slc-2]])        
                                                
        #dv_z/dz
        dvz_dz[:,ind_dict['ind_top_m1'][slc]] = coe1*(vz[:, ind_dict['ind_top_m1'][slc]] - vz[:, ind_dict['ind_top_m2'][slc]]) + \
                                                coe2*(vz[:, ind_dict['ind_top_0' ][slc]] - vz[:, ind_dict['ind_top_m3'][slc]])
        dvz_dz[:,ind_dict['ind_top_0' ][slc]] = coe1*(vz[:, ind_dict['ind_top_0' ][slc]] - vz[:, ind_dict['ind_top_m1'][slc]]) + \
                                                coe2*(vz[:, ind_dict['ind_top_p1'][slc]] - vz[:, ind_dict['ind_top_m2'][slc]])
        dvz_dz[:,ind_dict['ind_top_p1'][slc]] = coe1*(vz[:, ind_dict['ind_top_p1'][slc]] - vz[:, ind_dict['ind_top_0' ][slc]]) + \
                                                coe2*(vz[:, ind_dict['ind_top_p2'][slc]] - vz[:, ind_dict['ind_top_m1'][slc]])
                                                
        ###################### RIGHT #######################
        
        slc = 3 + np.arange(trunc_geom_dict['rec_v_n']+2) #slice (From outer injection boundary to outer injection boundary)
        
        #dv_x/dx
        dvx_dx[:,ind_dict['ind_right_m1'][slc]] = coe1*(vx[:, ind_dict['ind_right_0' ][slc]] - vx[:, ind_dict['ind_right_m1'][slc]]) + \
                                                  coe2*(vx[:, ind_dict['ind_right_p1'][slc]] - vx[:, ind_dict['ind_right_m2'][slc]])
        dvx_dx[:,ind_dict['ind_right_0' ][slc]] = coe1*(vx[:, ind_dict['ind_right_p1'][slc]] - vx[:, ind_dict['ind_right_0' ][slc]]) + \
                                                  coe2*(vx[:, ind_dict['ind_right_p2'][slc]] - vx[:, ind_dict['ind_right_m1'][slc]])        
        dvx_dx[:,ind_dict['ind_right_p1'][slc]] = coe1*(vx[:, ind_dict['ind_right_p2'][slc]] - vx[:, ind_dict['ind_right_p1'][slc]]) + \
                                                  coe2*(vx[:, ind_dict['ind_right_p3'][slc]] - vx[:, ind_dict['ind_right_0' ][slc]])
        
        #dv_x/dz
        dvx_dz[:,ind_dict['ind_right_m1'][slc]] = coe1*(vx[:, ind_dict['ind_right_m1'][slc+1]] - vx[:, ind_dict['ind_right_m1'][slc  ]]) + \
                                                  coe2*(vx[:, ind_dict['ind_right_m1'][slc+2]] - vx[:, ind_dict['ind_right_m1'][slc-1]])
        dvx_dz[:,ind_dict['ind_right_0' ][slc]] = coe1*(vx[:, ind_dict['ind_right_0' ][slc+1]] - vx[:, ind_dict['ind_right_0' ][slc  ]]) + \
                                                  coe2*(vx[:, ind_dict['ind_right_0' ][slc+2]] - vx[:, ind_dict['ind_right_0' ][slc-1]])
        dvx_dz[:,ind_dict['ind_right_p1'][slc]] = coe1*(vx[:, ind_dict['ind_right_p1'][slc+1]] - vx[:, ind_dict['ind_right_p1'][slc  ]]) + \
                                                  coe2*(vx[:, ind_dict['ind_right_p1'][slc+2]] - vx[:, ind_dict['ind_right_p1'][slc-1]])                                                                                                    
                                                  
        #dv_z/dx
        dvz_dx[:,ind_dict['ind_right_m1'][slc]] = coe1*(vz[:, ind_dict['ind_right_m1'][slc]] - vz[:, ind_dict['ind_right_m2'][slc]]) + \
                                                  coe2*(vz[:, ind_dict['ind_right_0' ][slc]] - vz[:, ind_dict['ind_right_m3'][slc]])
        dvz_dx[:,ind_dict['ind_right_0' ][slc]] = coe1*(vz[:, ind_dict['ind_right_0' ][slc]] - vz[:, ind_dict['ind_right_m1'][slc]]) + \
                                                  coe2*(vz[:, ind_dict['ind_right_p1'][slc]] - vz[:, ind_dict['ind_right_m2'][slc]])
        dvz_dx[:,ind_dict['ind_right_p1'][slc]] = coe1*(vz[:, ind_dict['ind_right_p1'][slc]] - vz[:, ind_dict['ind_right_0' ][slc]]) + \
                                                  coe2*(vz[:, ind_dict['ind_right_p2'][slc]] - vz[:, ind_dict['ind_right_m1'][slc]])
                                                  
        #dv_z/dz
        dvz_dz[:,ind_dict['ind_right_m1'][slc]] = coe1*(vz[:, ind_dict['ind_right_m1'][slc  ]] - vz[:, ind_dict['ind_right_m1'][slc-1]]) + \
                                                  coe2*(vz[:, ind_dict['ind_right_m1'][slc+1]] - vz[:, ind_dict['ind_right_m1'][slc-2]])
        dvz_dz[:,ind_dict['ind_right_0' ][slc]] = coe1*(vz[:, ind_dict['ind_right_0' ][slc  ]] - vz[:, ind_dict['ind_right_0' ][slc-1]]) + \
                                                  coe2*(vz[:, ind_dict['ind_right_0' ][slc+1]] - vz[:, ind_dict['ind_right_0' ][slc-2]])
        dvz_dz[:,ind_dict['ind_right_p1'][slc]] = coe1*(vz[:, ind_dict['ind_right_p1'][slc  ]] - vz[:, ind_dict['ind_right_p1'][slc-1]]) + \
                                                  coe2*(vz[:, ind_dict['ind_right_p1'][slc+1]] - vz[:, ind_dict['ind_right_p1'][slc-2]])
                                                  
        ###################### BOT #######################
        
        slc = 3+np.arange(trunc_geom_dict['rec_h_n']+2) #slice (From outer injection boundary to outer injection boundary)
        
        #dv_x/dx        
        dvx_dx[:,ind_dict['ind_bot_m1'][slc]] = coe1*(vx[:, ind_dict['ind_bot_m1'][slc+1]] - vx[:, ind_dict['ind_bot_m1'][slc  ]]) + \
                                                coe2*(vx[:, ind_dict['ind_bot_m1'][slc+2]] - vx[:, ind_dict['ind_bot_m1'][slc-1]])        
        dvx_dx[:,ind_dict['ind_bot_0' ][slc]] = coe1*(vx[:, ind_dict['ind_bot_0' ][slc+1]] - vx[:, ind_dict['ind_bot_0' ][slc  ]]) + \
                                                coe2*(vx[:, ind_dict['ind_bot_0' ][slc+2]] - vx[:, ind_dict['ind_bot_0' ][slc-1]])
        dvx_dx[:,ind_dict['ind_bot_p1'][slc]] = coe1*(vx[:, ind_dict['ind_bot_p1'][slc+1]] - vx[:, ind_dict['ind_bot_p1'][slc  ]]) + \
                                                coe2*(vx[:, ind_dict['ind_bot_p1'][slc+2]] - vx[:, ind_dict['ind_bot_p1'][slc-1]])
                                                   
        #dv_x/dz
        dvx_dz[:,ind_dict['ind_bot_m1'][slc]] = coe1*(vx[:, ind_dict['ind_bot_0' ][slc]] - vx[:, ind_dict['ind_bot_m1'][slc]]) + \
                                                coe2*(vx[:, ind_dict['ind_bot_p1'][slc]] - vx[:, ind_dict['ind_bot_m2'][slc]])
        dvx_dz[:,ind_dict['ind_bot_0' ][slc]] = coe1*(vx[:, ind_dict['ind_bot_p1'][slc]] - vx[:, ind_dict['ind_bot_0' ][slc]]) + \
                                                coe2*(vx[:, ind_dict['ind_bot_p2'][slc]] - vx[:, ind_dict['ind_bot_m1'][slc]])                                                
        dvx_dz[:,ind_dict['ind_bot_p1'][slc]] = coe1*(vx[:, ind_dict['ind_bot_p2'][slc]] - vx[:, ind_dict['ind_bot_p1'][slc]]) + \
                                                coe2*(vx[:, ind_dict['ind_bot_p3'][slc]] - vx[:, ind_dict['ind_bot_0' ][slc]])
                                                
        #dv_z/dx
        dvz_dx[:,ind_dict['ind_bot_m1'][slc]] = coe1*(vz[:, ind_dict['ind_bot_m1'][slc  ]] - vz[:, ind_dict['ind_bot_m1'][slc-1]]) + \
                                                coe2*(vz[:, ind_dict['ind_bot_m1'][slc+1]] - vz[:, ind_dict['ind_bot_m1'][slc-2]])
        dvz_dx[:,ind_dict['ind_bot_0' ][slc]] = coe1*(vz[:, ind_dict['ind_bot_0' ][slc  ]] - vz[:, ind_dict['ind_bot_0' ][slc-1]]) + \
                                                coe2*(vz[:, ind_dict['ind_bot_0' ][slc+1]] - vz[:, ind_dict['ind_bot_0' ][slc-2]])
        dvz_dx[:,ind_dict['ind_bot_p1'][slc]] = coe1*(vz[:, ind_dict['ind_bot_p1'][slc  ]] - vz[:, ind_dict['ind_bot_p1'][slc-1]]) + \
                                                coe2*(vz[:, ind_dict['ind_bot_p1'][slc+1]] - vz[:, ind_dict['ind_bot_p1'][slc-2]])        
                                                
        #dv_z/dz
        dvz_dz[:,ind_dict['ind_bot_m1'][slc]] = coe1*(vz[:, ind_dict['ind_bot_m1'][slc]] - vz[:, ind_dict['ind_bot_m2'][slc]]) + \
                                                coe2*(vz[:, ind_dict['ind_bot_0' ][slc]] - vz[:, ind_dict['ind_bot_m3'][slc]])
        dvz_dz[:,ind_dict['ind_bot_0' ][slc]] = coe1*(vz[:, ind_dict['ind_bot_0' ][slc]] - vz[:, ind_dict['ind_bot_m1'][slc]]) + \
                                                coe2*(vz[:, ind_dict['ind_bot_p1'][slc]] - vz[:, ind_dict['ind_bot_m2'][slc]])
        dvz_dz[:,ind_dict['ind_bot_p1'][slc]] = coe1*(vz[:, ind_dict['ind_bot_p1'][slc]] - vz[:, ind_dict['ind_bot_0' ][slc]]) + \
                                                coe2*(vz[:, ind_dict['ind_bot_p2'][slc]] - vz[:, ind_dict['ind_bot_m1'][slc]])        
        
        ###################### LEFT ######################
        
        slc = 3 + np.arange(trunc_geom_dict['rec_v_n']+2) #slice (From outer injection boundary to outer injection boundary)
        
        #dv_x/dx
        dvx_dx[:,ind_dict['ind_left_m1'][slc]] = coe1*(vx[:, ind_dict['ind_left_0' ][slc]] - vx[:, ind_dict['ind_left_m1'][slc]]) + \
                                                 coe2*(vx[:, ind_dict['ind_left_p1'][slc]] - vx[:, ind_dict['ind_left_m2'][slc]])
        dvx_dx[:,ind_dict['ind_left_0' ][slc]] = coe1*(vx[:, ind_dict['ind_left_p1'][slc]] - vx[:, ind_dict['ind_left_0' ][slc]]) + \
                                                 coe2*(vx[:, ind_dict['ind_left_p2'][slc]] - vx[:, ind_dict['ind_left_m1'][slc]])        
        dvx_dx[:,ind_dict['ind_left_p1'][slc]] = coe1*(vx[:, ind_dict['ind_left_p2'][slc]] - vx[:, ind_dict['ind_left_p1'][slc]]) + \
                                                 coe2*(vx[:, ind_dict['ind_left_p3'][slc]] - vx[:, ind_dict['ind_left_0' ][slc]])
        
        #dv_x/dz
        dvx_dz[:,ind_dict['ind_left_m1'][slc]] = coe1*(vx[:, ind_dict['ind_left_m1'][slc+1]] - vx[:, ind_dict['ind_left_m1'][slc  ]]) + \
                                                 coe2*(vx[:, ind_dict['ind_left_m1'][slc+2]] - vx[:, ind_dict['ind_left_m1'][slc-1]])
        dvx_dz[:,ind_dict['ind_left_0' ][slc]] = coe1*(vx[:, ind_dict['ind_left_0' ][slc+1]] - vx[:, ind_dict['ind_left_0' ][slc  ]]) + \
                                                 coe2*(vx[:, ind_dict['ind_left_0' ][slc+2]] - vx[:, ind_dict['ind_left_0' ][slc-1]])
        dvx_dz[:,ind_dict['ind_left_p1'][slc]] = coe1*(vx[:, ind_dict['ind_left_p1'][slc+1]] - vx[:, ind_dict['ind_left_p1'][slc  ]]) + \
                                                 coe2*(vx[:, ind_dict['ind_left_p1'][slc+2]] - vx[:, ind_dict['ind_left_p1'][slc-1]])                                                                                                    
                                                  
        #dv_z/dx
        dvz_dx[:,ind_dict['ind_left_m1'][slc]] = coe1*(vz[:, ind_dict['ind_left_m1'][slc]] - vz[:, ind_dict['ind_left_m2'][slc]]) + \
                                                 coe2*(vz[:, ind_dict['ind_left_0' ][slc]] - vz[:, ind_dict['ind_left_m3'][slc]])
        dvz_dx[:,ind_dict['ind_left_0' ][slc]] = coe1*(vz[:, ind_dict['ind_left_0' ][slc]] - vz[:, ind_dict['ind_left_m1'][slc]]) + \
                                                 coe2*(vz[:, ind_dict['ind_left_p1'][slc]] - vz[:, ind_dict['ind_left_m2'][slc]])
        dvz_dx[:,ind_dict['ind_left_p1'][slc]] = coe1*(vz[:, ind_dict['ind_left_p1'][slc]] - vz[:, ind_dict['ind_left_0' ][slc]]) + \
                                                 coe2*(vz[:, ind_dict['ind_left_p2'][slc]] - vz[:, ind_dict['ind_left_m1'][slc]])
                                                  
        #dv_z/dz
        dvz_dz[:,ind_dict['ind_left_m1'][slc]] = coe1*(vz[:, ind_dict['ind_left_m1'][slc  ]] - vz[:, ind_dict['ind_left_m1'][slc-1]]) + \
                                                 coe2*(vz[:, ind_dict['ind_left_m1'][slc+1]] - vz[:, ind_dict['ind_left_m1'][slc-2]])
        dvz_dz[:,ind_dict['ind_left_0' ][slc]] = coe1*(vz[:, ind_dict['ind_left_0' ][slc  ]] - vz[:, ind_dict['ind_left_0' ][slc-1]]) + \
                                                 coe2*(vz[:, ind_dict['ind_left_0' ][slc+1]] - vz[:, ind_dict['ind_left_0' ][slc-2]])
        dvz_dz[:,ind_dict['ind_left_p1'][slc]] = coe1*(vz[:, ind_dict['ind_left_p1'][slc  ]] - vz[:, ind_dict['ind_left_p1'][slc-1]]) + \
                                                 coe2*(vz[:, ind_dict['ind_left_p1'][slc+1]] - vz[:, ind_dict['ind_left_p1'][slc-2]])
        
        ############ WE HAVE ALL THE VELOCITY DERIVATIVES NOW AT THE STRESS POINTS. NOW COMPUTE STRESSES ##########
        
        #First get lam2mu, lam and c19 in same way from vp, rho and vs as elastic C code
        #Place these values in vectors the same size as the number of columns of the velocity derivative arrays.
        #These velocity derivative arrays have many columns with zero (places around boundary where the derivatives were not calculated), because they have the same size as vx and vz.
        #Later we extract the correct columns when we calculate the boundary fields (not in this function)
        lam2mu, lam, c19, _ = get_lambda_mu_rho(m, trunc_geom_dict, vp_2d, rho_2d, vs_2d)
        
        #get stress rates 
        tau_xx_updates = dtdx*(dvx_dx*lam2mu + dvz_dz*lam   )
        tau_zz_updates = dtdx*(dvx_dx*lam    + dvz_dz*lam2mu)
        tau_xz_updates = dtdx*(dvx_dz        + dvz_dx       )*c19
        
        #Compute stresses
        tau_xx = np.cumsum(tau_xx_updates, axis=0)
        tau_zz = np.cumsum(tau_zz_updates, axis=0)
        tau_xz = np.cumsum(tau_xz_updates, axis=0)
        
        return tau_xx, tau_zz, tau_xz
    
#     def do_empirical_amplitude_correction(gb, vx, vz, tau_xx, tau_zz, tau_xz, m, trunc_geom_dict, ind_dict, vp_2d, rho, vs_2d, amp_correction_mode):
#         #OLD IMPLEMENTATION FOR INJECTING ON MATERIAL WITH NONZERO VS. I WOULD SCALE AFTER COMPUTING vx, vz, txx, txz and tzz. 
#         
#         def movingaverage(interval, window_size): #http://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
#             #modified to pad last values 
#             
#             #pad with ceil(window_size/2.) on left and right
#             npad = int(np.ceil(window_size/2.))
#             padded_interval = np.zeros(interval.size + 2*npad)
#             padded_interval[npad:npad + interval.size] = interval 
#             
#             leftval = interval[0]
#             rightval = interval[-1]
#             
#             padded_interval[0:npad] = leftval
#             padded_interval[npad + interval.size:] = rightval
#             
#             window = np.ones(int(window_size))/float(window_size)
#             padded_average = np.convolve(padded_interval, window, 'same')
#             average = padded_average[npad:npad + interval.size]
#             return average    
#         
#         def fill_and_avg(arr1, arr2):
#             #when filling arr2 into arr1, first check if certain columns in arr1 we want to fill are filled already
#             #if so, average the old and the new
#             
#             filled_arr1 = arr1 > 0 #Filled entries 
#             filled_arr2 = arr2 > 0 #Filled entries
#             both_filled = np.logical_and(filled_arr1, filled_arr2)
# 
#             avg_both_filled  = 0.5*(arr1[both_filled] + arr2[both_filled])  
# 
#             #first do normal fill
#             arr1[:] = arr2[:]
#             if np.any(both_filled):
#                 arr1[both_filled] = avg_both_filled
# 
#             return arr1
#         #when mu != 0, the abs value of stress rate will be lower than when mu = 0
#         #Adding the equation for dtau_xx/dt and dtau_zz/dt and reorganizing will give
#         #dp/dt = rho(vp**2 - vs**2)(dv_x/dx + dv_z/dz). So if mu != 0 the amplitude of the pressure will be lower in the local solver
#         #Injecting on mu != 0 is just a trick to reduce the Vs taper amplitude. 
#         #We would still like to inject the same amplitude pressure field. 
#         #-> Go from the tau_xx and tau_zz updates which are currently multiplied by a (vp**2-vs**2) factor to a vp**2 factor only.
#         #Will also have to use the same multiplier on the velocities.
#         lam2mu, lam, _ = get_lambda_mu(m, trunc_geom_dict, vp_2d, rho, vs_2d)
#         mu = (lam2mu - lam)/2.
#         vp_bdry = np.sqrt(lam2mu/rho)
#         vs_bdry = np.sqrt(mu/rho)
#         factor_bdry = vp_bdry**2/(vp_bdry**2-vs_bdry**2) #vector of values. Each value corresponds to a pixel.
#         
#         
#         if amp_correction_mode == 'local':         
#             #At each pixel scale everything with the same value for that pixel to get the same pressure amplitude as if Vs = 0.
#             #Not sure if this will result in some (hopefully minor?) inconsistencies between the stresses and velocities ? 
#             #They are tied together through gradients. By multiplying with values that vary on a pixel basis some of these gradients may be inconsistent ? 
#             vx *= factor_bdry
#             vz *= factor_bdry
#              
#             tau_xx *= factor_bdry
#             tau_xz *= factor_bdry
#             tau_zz *= factor_bdry
#         elif amp_correction_mode == 'avg':
#             #Cannot do the pointwise corrections below outside this 'if' statement, they may no longer be small ? Would make stresses and velocities too inconsistent with EL equations?
#             #So when Vs = 0, there will still be some difference compared to the old implementation which gave slightly better results
#             
#             #NOT CORRECT, WHEN COMPUTING THE AVERAGE THIS WAY, WE ALSO TAKE INTO ACCOUNT POINTS THAT ARE NOT ON THE BOUNDARY m1, 0 and p1
#             #avg_factor = np.average(factor_bdry)
#             #print "AVG FACTOR = " + str(avg_factor)
#             #tau_xx*=avg_factor
#             #tau_xz*=avg_factor
#             #tau_zz*=avg_factor
#             #vx    *=avg_factor
#             #vz    *=avg_factor
#             
#             #By looking at the average difference in max amplitude we should be able to correct both for the effect of 
#             #Vs no longer be 0, but also for the small difference with the old implementation which just used the input pressure wavefield
#             p = 0.5*(tau_xx + tau_zz)
#             bdry_indices  = np.max(p, axis=0) > 0 #Columns which are filled
#             max_p_traces  = np.max( p[:,bdry_indices], axis=0)
#             max_in_gather = np.max(gb[:,bdry_indices], axis=0)
#             ratios = max_in_gather / max_p_traces
#             ratios = np.abs(max_in_gather / max_p_traces - 1) + 1
#             avg_ratio = np.average(ratios)
#             print "AVG_RATIO " + str(avg_ratio)
#             tau_xx*=avg_ratio
#             tau_xz*=avg_ratio
#             tau_zz*=avg_ratio
#             vx    *=avg_ratio
#             vz    *=avg_ratio             
#             
#             return              
#             
#         elif amp_correction_mode == 'sliding':
#             #Intermediate between 'local' and 'avg'
#             
#             #First average the factor over the three layers at each boundary. Will not allow variation in that direction
#             
#             
#             N = 10 #Sliding window length
#             
#             #TOP
#             slc_h = 3+np.arange(trunc_geom_dict['rec_h_n']+2) #slice (From outer injection boundary to outer injection boundary)
#             
#             avg_top = 1./3*(factor_bdry[ind_dict['ind_top_m1'][slc_h]] + factor_bdry[ind_dict['ind_top_0'][slc_h]] + factor_bdry[ind_dict['ind_top_p1'][slc_h]])
#             
#             #sliding window over this
#             avg_top_sliding = movingaverage(avg_top, N)
#             
#             #RIGHT
#             slc_v = 3+np.arange(trunc_geom_dict['rec_v_n']+2) #slice (From outer injection boundary to outer injection boundary)
#             
#             avg_right = 1./3*(factor_bdry[ind_dict['ind_right_m1'][slc_v]] + factor_bdry[ind_dict['ind_right_0'][slc_v]] + factor_bdry[ind_dict['ind_right_p1'][slc_v]])
#             
#             #sliding window over this
#             avg_right_sliding = movingaverage(avg_right, N)
#             
#             #BOT
#             avg_bot = 1./3*(factor_bdry[ind_dict['ind_bot_m1'][slc_h]] + factor_bdry[ind_dict['ind_bot_0'][slc_h]] + factor_bdry[ind_dict['ind_bot_p1'][slc_h]])
#             
#             #sliding window over this
#             avg_bot_sliding = movingaverage(avg_bot, N)        
# 
#             #RIGHT
#             avg_left = 1./3*(factor_bdry[ind_dict['ind_left_m1'][slc_v]] + factor_bdry[ind_dict['ind_left_0'][slc_v]] + factor_bdry[ind_dict['ind_left_p1'][slc_v]])
#             
#             #sliding window over this
#             avg_left_sliding = movingaverage(avg_left, N)
# 
#             
#             #Sliding window may have changed the total average a bit. Want it to be the same to inject the same amount of energy
#             avg_top_sliding   *= np.mean(avg_top)/np.mean(avg_top_sliding)
#             avg_right_sliding *= np.mean(avg_right)/np.mean(avg_right_sliding)
#             avg_bot_sliding   *= np.mean(avg_bot)/np.mean(avg_bot_sliding)
#             avg_left_sliding  *= np.mean(avg_left)/np.mean(avg_left_sliding)
#         
#             #Now scale build scaling array
#             sliding_scale = np.zeros_like(factor_bdry)
#             
#             sliding_scale[ind_dict['ind_top_m1'][slc_h]] = fill_and_avg(sliding_scale[ind_dict['ind_top_m1'][slc_h]], avg_top_sliding)
#             sliding_scale[ind_dict['ind_top_0' ][slc_h]] = fill_and_avg(sliding_scale[ind_dict['ind_top_0' ][slc_h]], avg_top_sliding)
#             sliding_scale[ind_dict['ind_top_p1'][slc_h]] = fill_and_avg(sliding_scale[ind_dict['ind_top_p1'][slc_h]], avg_top_sliding)
#             
#             sliding_scale[ind_dict['ind_right_m1'][slc_v]] = fill_and_avg(sliding_scale[ind_dict['ind_right_m1'][slc_v]], avg_right_sliding)
#             sliding_scale[ind_dict['ind_right_0' ][slc_v]] = fill_and_avg(sliding_scale[ind_dict['ind_right_0' ][slc_v]], avg_right_sliding)
#             sliding_scale[ind_dict['ind_right_p1'][slc_v]] = fill_and_avg(sliding_scale[ind_dict['ind_right_p1'][slc_v]], avg_right_sliding)
# 
#             sliding_scale[ind_dict['ind_bot_m1'][slc_h]] = fill_and_avg(sliding_scale[ind_dict['ind_bot_m1'][slc_h]], avg_bot_sliding)
#             sliding_scale[ind_dict['ind_bot_0' ][slc_h]] = fill_and_avg(sliding_scale[ind_dict['ind_bot_0' ][slc_h]], avg_bot_sliding)
#             sliding_scale[ind_dict['ind_bot_p1'][slc_h]] = fill_and_avg(sliding_scale[ind_dict['ind_bot_p1'][slc_h]], avg_bot_sliding)
# 
#             sliding_scale[ind_dict['ind_left_m1'][slc_v]] = fill_and_avg(sliding_scale[ind_dict['ind_left_m1'][slc_v]], avg_left_sliding)
#             sliding_scale[ind_dict['ind_left_0' ][slc_v]] = fill_and_avg(sliding_scale[ind_dict['ind_left_0' ][slc_v]], avg_left_sliding)
#             sliding_scale[ind_dict['ind_left_p1'][slc_v]] = fill_and_avg(sliding_scale[ind_dict['ind_left_p1'][slc_v]], avg_left_sliding)
#            
#             #Apply scale
#             tau_xx*=sliding_scale
#             tau_xz*=sliding_scale
#             tau_zz*=sliding_scale
#             vx    *=sliding_scale
#             vz    *=sliding_scale
#             return
#         #When Vs = 0, factor_bdry = 1.0 so the part above does nothing.
#         #I observe that when Vp contrasts intersect the boundary of the truncated domain, the boundary txx and tzz are different than in the old implementation         
#         #The velocities on the other hand are exactly the same as in the old implementation
#         #The old implementation gave a scattered field ~100 times lower than the new implementation when there was no perturbation in the local domain (it remained the same background Vp and mu = 0 and rho is constant). 
#         #So I'd like txx and tzz to have similar amplitude as in the old implementation, where we just used the input boundary pressure gathers.
#         #Inspection on some simple examples shows that the shape is exactly the same in txx, tzz and the input gather gb
#         #So I will correct amplitude of txx and tzz by comparing the max with the max of gb in each trace. Should result in minor corrections. Very empirical.
# 
#         #TOP
#         slc = 3+np.arange(trunc_geom_dict['rec_h_n']+2) #slice (From outer injection boundary to outer injection boundary)
#         
#         #TOP ABOVE
#         p                = 0.5*(tau_xx[:,ind_dict['ind_top_m1'][slc]] + tau_zz[:,ind_dict['ind_top_m1'][slc]])
#         max_in_gather    = np.max(gb[:,ind_dict['ind_top_m1'][slc]], axis=0)
#         max_p_now        = np.max(p, axis=0)
#         ratio            = max_in_gather/max_p_now
#         
#         if np.any(np.abs(ratio-1) > 0.05): 
#             raise Exception("Much larger empirical fix than expected. Probably good to halt here and investigate what happens")
#         
#         tau_xx[:,ind_dict['ind_top_m1'][slc]] *= ratio
#         tau_zz[:,ind_dict['ind_top_m1'][slc]] *= ratio
#         tau_xz[:,ind_dict['ind_top_m1'][slc]] *= ratio
#         
#         #TOP ON
#         p                = 0.5*(tau_xx[:,ind_dict['ind_top_0'][slc]] + tau_zz[:,ind_dict['ind_top_0'][slc]])
#         max_in_gather    = np.max(gb[:,ind_dict['ind_top_0'][slc]], axis=0)
#         max_p_now        = np.max(p, axis=0)
#         ratio            = max_in_gather/max_p_now
# 
#         if np.any(np.abs(ratio-1) > 0.05): 
#             raise Exception("Much larger empirical fix than expected. Probably good to halt here and investigate what happens")
# 
#         
#         tau_xx[:,ind_dict['ind_top_0'][slc]] *= ratio
#         tau_zz[:,ind_dict['ind_top_0'][slc]] *= ratio
#         tau_xz[:,ind_dict['ind_top_0'][slc]] *= ratio
# 
#         #TOP BELOW
#         p                = 0.5*(tau_xx[:,ind_dict['ind_top_p1'][slc]] + tau_zz[:,ind_dict['ind_top_p1'][slc]])
#         max_in_gather    = np.max(gb[:,ind_dict['ind_top_p1'][slc]], axis=0)
#         max_p_now        = np.max(p, axis=0)
#         ratio            = max_in_gather/max_p_now
# 
#         if np.any(np.abs(ratio-1) > 0.05): 
#             raise Exception("Much larger empirical fix than expected. Probably good to halt here and investigate what happens")
# 
#         
#         tau_xx[:,ind_dict['ind_top_p1'][slc]] *= ratio
#         tau_zz[:,ind_dict['ind_top_p1'][slc]] *= ratio
#         tau_xz[:,ind_dict['ind_top_p1'][slc]] *= ratio
#     
#         #RIGHT
#         slc = 3 + np.arange(trunc_geom_dict['rec_v_n']+2) #slice (From outer injection boundary to outer injection boundary)
# 
#         #RIGHT LEFT
#         p                = 0.5*(tau_xx[:,ind_dict['ind_right_m1'][slc]] + tau_zz[:,ind_dict['ind_right_m1'][slc]])
#         max_in_gather    = np.max(gb[:,ind_dict['ind_right_m1'][slc]], axis=0)
#         max_p_now        = np.max(p, axis=0)
#         ratio            = max_in_gather/max_p_now
# 
#         if np.any(np.abs(ratio-1) > 0.05): 
#             raise Exception("Much larger empirical fix than expected. Probably good to halt here and investigate what happens")
#         
#         tau_xx[:,ind_dict['ind_right_m1'][slc]] *= ratio
#         tau_zz[:,ind_dict['ind_right_m1'][slc]] *= ratio
#         tau_xz[:,ind_dict['ind_right_m1'][slc]] *= ratio
# 
#         #RIGHT ON
#         p                = 0.5*(tau_xx[:,ind_dict['ind_right_0'][slc]] + tau_zz[:,ind_dict['ind_right_0'][slc]])
#         max_in_gather    = np.max(gb[:,ind_dict['ind_right_0'][slc]], axis=0)
#         max_p_now        = np.max(p, axis=0)
#         ratio            = max_in_gather/max_p_now
# 
#         if np.any(np.abs(ratio-1) > 0.05): 
#             raise Exception("Much larger empirical fix than expected. Probably good to halt here and investigate what happens")
#         
#         tau_xx[:,ind_dict['ind_right_0'][slc]] *= ratio
#         tau_zz[:,ind_dict['ind_right_0'][slc]] *= ratio
#         tau_xz[:,ind_dict['ind_right_0'][slc]] *= ratio        
#     
#         #RIGHT RIGHT
#         p                = 0.5*(tau_xx[:,ind_dict['ind_right_p1'][slc]] + tau_zz[:,ind_dict['ind_right_p1'][slc]])
#         max_in_gather    = np.max(gb[:,ind_dict['ind_right_p1'][slc]], axis=0)
#         max_p_now        = np.max(p, axis=0)
#         ratio            = max_in_gather/max_p_now
# 
#         if np.any(np.abs(ratio-1) > 0.05): 
#             raise Exception("Much larger empirical fix than expected. Probably good to halt here and investigate what happens")
#         
#         tau_xx[:,ind_dict['ind_right_p1'][slc]] *= ratio
#         tau_zz[:,ind_dict['ind_right_p1'][slc]] *= ratio
#         tau_xz[:,ind_dict['ind_right_p1'][slc]] *= ratio    
#     
#         #BOT
#         slc = 3+np.arange(trunc_geom_dict['rec_h_n']+2) #slice (From outer injection boundary to outer injection boundary)
# 
#         #BOT ABOVE
#         p                = 0.5*(tau_xx[:,ind_dict['ind_bot_m1'][slc]] + tau_zz[:,ind_dict['ind_bot_m1'][slc]])
#         max_in_gather    = np.max(gb[:,ind_dict['ind_bot_m1'][slc]], axis=0)
#         max_p_now        = np.max(p, axis=0)
#         ratio            = max_in_gather/max_p_now
# 
#         if np.any(np.abs(ratio-1) > 0.05): 
#             raise Exception("Much larger empirical fix than expected. Probably good to halt here and investigate what happens")
#         
#         tau_xx[:,ind_dict['ind_bot_m1'][slc]] *= ratio
#         tau_zz[:,ind_dict['ind_bot_m1'][slc]] *= ratio
#         tau_xz[:,ind_dict['ind_bot_m1'][slc]] *= ratio        
# 
#         #BOT ON
#         p                = 0.5*(tau_xx[:,ind_dict['ind_bot_0'][slc]] + tau_zz[:,ind_dict['ind_bot_0'][slc]])
#         max_in_gather    = np.max(gb[:,ind_dict['ind_bot_0'][slc]], axis=0)
#         max_p_now        = np.max(p, axis=0)
#         ratio            = max_in_gather/max_p_now
# 
#         if np.any(np.abs(ratio-1) > 0.05): 
#             raise Exception("Much larger empirical fix than expected. Probably good to halt here and investigate what happens")
#         
#         tau_xx[:,ind_dict['ind_bot_0'][slc]] *= ratio
#         tau_zz[:,ind_dict['ind_bot_0'][slc]] *= ratio
#         tau_xz[:,ind_dict['ind_bot_0'][slc]] *= ratio
# 
#         #BOT BELOW
#         p                = 0.5*(tau_xx[:,ind_dict['ind_bot_p1'][slc]] + tau_zz[:,ind_dict['ind_bot_p1'][slc]])
#         max_in_gather    = np.max(gb[:,ind_dict['ind_bot_p1'][slc]], axis=0)
#         max_p_now        = np.max(p, axis=0)
#         ratio            = max_in_gather/max_p_now
# 
#         if np.any(np.abs(ratio-1) > 0.05): 
#             raise Exception("Much larger empirical fix than expected. Probably good to halt here and investigate what happens")
#         
#         tau_xx[:,ind_dict['ind_bot_p1'][slc]] *= ratio
#         tau_zz[:,ind_dict['ind_bot_p1'][slc]] *= ratio
#         tau_xz[:,ind_dict['ind_bot_p1'][slc]] *= ratio
#         
#         #LEFT
#         slc = 3 + np.arange(trunc_geom_dict['rec_v_n']+2) #slice (From outer injection boundary to outer injection boundary)
#         
#         #LEFT LEFT
#         p                = 0.5*(tau_xx[:,ind_dict['ind_left_m1'][slc]] + tau_zz[:,ind_dict['ind_left_m1'][slc]])
#         max_in_gather    = np.max(gb[:,ind_dict['ind_left_m1'][slc]], axis=0)
#         max_p_now        = np.max(p, axis=0)
#         ratio            = max_in_gather/max_p_now
# 
#         if np.any(np.abs(ratio-1) > 0.05): 
#             raise Exception("Much larger empirical fix than expected. Probably good to halt here and investigate what happens")
#         
#         tau_xx[:,ind_dict['ind_left_m1'][slc]] *= ratio
#         tau_zz[:,ind_dict['ind_left_m1'][slc]] *= ratio
#         tau_xz[:,ind_dict['ind_left_m1'][slc]] *= ratio        
#         
#         #LEFT ON
#         p                = 0.5*(tau_xx[:,ind_dict['ind_left_0'][slc]] + tau_zz[:,ind_dict['ind_left_0'][slc]])
#         max_in_gather    = np.max(gb[:,ind_dict['ind_left_0'][slc]], axis=0)
#         max_p_now        = np.max(p, axis=0)
#         ratio            = max_in_gather/max_p_now
# 
#         if np.any(np.abs(ratio-1) > 0.05): 
#             raise Exception("Much larger empirical fix than expected. Probably good to halt here and investigate what happens")
#         
#         tau_xx[:,ind_dict['ind_left_0'][slc]] *= ratio
#         tau_zz[:,ind_dict['ind_left_0'][slc]] *= ratio
#         tau_xz[:,ind_dict['ind_left_0'][slc]] *= ratio        
# 
#         #LEFT RIGHT
#         p                = 0.5*(tau_xx[:,ind_dict['ind_left_p1'][slc]] + tau_zz[:,ind_dict['ind_left_p1'][slc]])
#         max_in_gather    = np.max(gb[:,ind_dict['ind_left_p1'][slc]], axis=0)
#         max_p_now        = np.max(p, axis=0)
#         ratio            = max_in_gather/max_p_now
# 
#         if np.any(np.abs(ratio-1) > 0.05): 
#             raise Exception("Much larger empirical fix than expected. Probably good to halt here and investigate what happens")
#         
#         tau_xx[:,ind_dict['ind_left_p1'][slc]] *= ratio
#         tau_zz[:,ind_dict['ind_left_p1'][slc]] *= ratio
#         tau_xz[:,ind_dict['ind_left_p1'][slc]] *= ratio        
        
    def gen_bdry_flds_arr(vx, vz, tau_xx_in, tau_xz_in, tau_zz_in, trunc_geom_dict, m):
        
        ###################################
        #In the EL code, for each timestep we first compute stress and then velocity. Then advance step. 
        #Here we use the PySIT CDA pressure field to compute the velocities at the same timesteps within the now elastic medium
        #Then we use these velocities to compute the stress, which is the next timestep to be consistent with the EL code  
        tau_xx  = np.zeros_like(tau_xx_in)
        tau_xz  = np.zeros_like(tau_xz_in)
        tau_zz  = np.zeros_like(tau_zz_in)
        
        tau_xx[1:, :] = tau_xx_in[0:-1, :]
        tau_xz[1:, :] = tau_xz_in[0:-1, :]
        tau_zz[1:, :] = tau_zz_in[0:-1, :]
        ###################################
        
        txx_off = 0
        tzz_off = 1
        txz_off = 2
        vx_off  = 3
        vz_off  = 4
        
        #NOW USE THESE COMPUTED VALUES TO STORE THE BOUNDARY_FIELDS, SAME AS IN fd2d_rec_wavefields.c 
        n_rows = int(np.round((trunc_geom_dict['rec_z_b'] - trunc_geom_dict['rec_z_t'])/m.z.delta)) + 1
        n_cols = int(np.round((trunc_geom_dict['rec_x_r'] - trunc_geom_dict['rec_x_l'])/m.x.delta)) + 1
                
        #at each cell we save 5 quantities. tauxx, tauzz, tauxz, vx, vz
        row_increment = 5*n_cols
        col_increment = 5*n_rows
                
        #We have two horizontal and two vertical sides of square recording surface.
        #On each side we save three rows or columns of cells 
        #We do this for each timestep
        nt              = vz.shape[0]
        nvals_bdry_each = (2*3*row_increment + 2*3*col_increment)
        nvals_bdry      = nvals_bdry_each*nt 
        
        bdry_flds       = np.zeros(nvals_bdry)
        bdry_flds_2d    = np.reshape(bdry_flds, (nt, nvals_bdry_each), 'F') #Temporarily reshape into 2D so i can more easily store stuff

        dx = m.x.delta
        dz = m.z.delta
        
        ################# STORE TOP ################# 
        
        #all x pos from 'on' boundary on left to 'on' boundary on right
        x_pos = trunc_geom_dict['rec_x_l'] + dx*np.arange(trunc_geom_dict['rec_h_n'])
        z_pos = (trunc_geom_dict['rec_z_t']-dz)* np.ones_like(x_pos) 

        #'above' gather indices
        gather_ind_above = [pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation = True) for (x,z) in zip(x_pos, z_pos)]
        
        #'on' gather indices
        z_pos += dz
        gather_ind_on    = [pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation = True) for (x,z) in zip(x_pos, z_pos)]
        
        #'below' gather indices
        z_pos += dz
        gather_ind_below = [pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation = True) for (x,z) in zip(x_pos, z_pos)]

        ####
        offset       = 0
        inds_stencil = 5*np.arange(n_cols)
        
        #STORE ABOVE
        bdry_flds_2d[:, offset + inds_stencil + txx_off] = tau_xx[:,gather_ind_above]
        bdry_flds_2d[:, offset + inds_stencil + tzz_off] = tau_zz[:,gather_ind_above]
        bdry_flds_2d[:, offset + inds_stencil + txz_off] = tau_xz[:,gather_ind_above]
        bdry_flds_2d[:, offset + inds_stencil +  vx_off] = vx[    :,gather_ind_above]
        bdry_flds_2d[:, offset + inds_stencil +  vz_off] = vz[    :,gather_ind_above]

        offset      += row_increment
                
        #STORE ON
        bdry_flds_2d[:, offset + inds_stencil + txx_off] = tau_xx[:,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil + tzz_off] = tau_zz[:,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil + txz_off] = tau_xz[:,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil +  vx_off] = vx[    :,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil +  vz_off] = vz[    :,gather_ind_on]

        offset      += row_increment        
        
        #STORE BELOW
        bdry_flds_2d[:, offset + inds_stencil + txx_off] = tau_xx[:,gather_ind_below]
        bdry_flds_2d[:, offset + inds_stencil + tzz_off] = tau_zz[:,gather_ind_below]
        bdry_flds_2d[:, offset + inds_stencil + txz_off] = tau_xz[:,gather_ind_below]
        bdry_flds_2d[:, offset + inds_stencil +  vx_off] = vx[    :,gather_ind_below]
        bdry_flds_2d[:, offset + inds_stencil +  vz_off] = vz[    :,gather_ind_below]

        offset      += row_increment        
        
        ################# STORE RIGHT #################
        
        #all z pos from 'on' boundary on top to 'on' boundary on bot
        z_pos = trunc_geom_dict['rec_z_t'] + dz*np.arange(trunc_geom_dict['rec_v_n'])
        x_pos = (trunc_geom_dict['rec_x_r'] + dx)*np.ones_like(z_pos) 

        #'right' gather indices
        gather_ind_right = [pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation = True) for (x,z) in zip(x_pos, z_pos)]
        
        #'on' gather indices
        x_pos -= dx
        gather_ind_on    = [pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation = True) for (x,z) in zip(x_pos, z_pos)]
        
        #'left' gather indices
        x_pos -= dx
        gather_ind_left  = [pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation = True) for (x,z) in zip(x_pos, z_pos)]

        inds_stencil = 5*np.arange(n_rows)
        
        #STORE RIGHT
        bdry_flds_2d[:, offset + inds_stencil + txx_off] = tau_xx[:,gather_ind_right]
        bdry_flds_2d[:, offset + inds_stencil + tzz_off] = tau_zz[:,gather_ind_right]
        bdry_flds_2d[:, offset + inds_stencil + txz_off] = tau_xz[:,gather_ind_right]
        bdry_flds_2d[:, offset + inds_stencil +  vx_off] = vx[    :,gather_ind_right]
        bdry_flds_2d[:, offset + inds_stencil +  vz_off] = vz[    :,gather_ind_right]
        
        offset      += col_increment
        
        #STORE ON
        bdry_flds_2d[:, offset + inds_stencil + txx_off] = tau_xx[:,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil + tzz_off] = tau_zz[:,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil + txz_off] = tau_xz[:,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil +  vx_off] = vx[    :,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil +  vz_off] = vz[    :,gather_ind_on]
        
        offset      += col_increment
        
        #STORE LEFT
        bdry_flds_2d[:, offset + inds_stencil + txx_off] = tau_xx[:,gather_ind_left]
        bdry_flds_2d[:, offset + inds_stencil + tzz_off] = tau_zz[:,gather_ind_left]
        bdry_flds_2d[:, offset + inds_stencil + txz_off] = tau_xz[:,gather_ind_left]
        bdry_flds_2d[:, offset + inds_stencil +  vx_off] = vx[    :,gather_ind_left]
        bdry_flds_2d[:, offset + inds_stencil +  vz_off] = vz[    :,gather_ind_left]
        
        offset      += col_increment

        ################# STORE BOT #################
        x_pos = trunc_geom_dict['rec_x_l'] + dx*np.arange(trunc_geom_dict['rec_h_n'])
        x_pos = x_pos[::-1] #go from right to left in numbering C code
        z_pos = (trunc_geom_dict['rec_z_b']+dz)* np.ones_like(x_pos) 

        #'below' gather indices
        gather_ind_below = [pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation = True) for (x,z) in zip(x_pos, z_pos)]
        
        #'on' gather indices
        z_pos -= dz
        gather_ind_on    = [pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation = True) for (x,z) in zip(x_pos, z_pos)]
        
        #'above' gather indices
        z_pos -= dz
        gather_ind_above = [pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation = True) for (x,z) in zip(x_pos, z_pos)]

        inds_stencil = 5*np.arange(n_cols)
        
        #STORE BELOW
        bdry_flds_2d[:, offset + inds_stencil + txx_off] = tau_xx[:,gather_ind_below]
        bdry_flds_2d[:, offset + inds_stencil + tzz_off] = tau_zz[:,gather_ind_below]
        bdry_flds_2d[:, offset + inds_stencil + txz_off] = tau_xz[:,gather_ind_below]
        bdry_flds_2d[:, offset + inds_stencil +  vx_off] = vx[    :,gather_ind_below]
        bdry_flds_2d[:, offset + inds_stencil +  vz_off] = vz[    :,gather_ind_below]
        
        offset      += row_increment
        
        #STORE ON
        bdry_flds_2d[:, offset + inds_stencil + txx_off] = tau_xx[:,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil + tzz_off] = tau_zz[:,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil + txz_off] = tau_xz[:,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil +  vx_off] = vx[    :,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil +  vz_off] = vz[    :,gather_ind_on]        
        
        offset      += row_increment
        
        #STORE ABOVE
        bdry_flds_2d[:, offset + inds_stencil + txx_off] = tau_xx[:,gather_ind_above]
        bdry_flds_2d[:, offset + inds_stencil + tzz_off] = tau_zz[:,gather_ind_above]
        bdry_flds_2d[:, offset + inds_stencil + txz_off] = tau_xz[:,gather_ind_above]
        bdry_flds_2d[:, offset + inds_stencil +  vx_off] = vx[    :,gather_ind_above]
        bdry_flds_2d[:, offset + inds_stencil +  vz_off] = vz[    :,gather_ind_above]
        
        offset      += row_increment
        
        ################# STORE LEFT #################
        #FIELDS ARE STORED FROM BOT TO TOP IN fd2d_rec_wavefields.c
        #NEED TO FLIP DIRECTION BY USING [::-1]
        
        #all z pos from 'on' boundary on top to 'on' boundary on bot
        z_pos = trunc_geom_dict['rec_z_t'] + dz*np.arange(trunc_geom_dict['rec_v_n'])
        z_pos = z_pos[::-1] #from bot to top
        x_pos = (trunc_geom_dict['rec_x_l'] - dx)*np.ones_like(z_pos) 

        #'left' gather indices
        gather_ind_left = [pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation = True) for (x,z) in zip(x_pos, z_pos)]
        
        #'on' gather indices
        x_pos += dx
        gather_ind_on    = [pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation = True) for (x,z) in zip(x_pos, z_pos)]
        
        #'right' gather indices
        x_pos += dx
        gather_ind_right = [pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation = True) for (x,z) in zip(x_pos, z_pos)]

        inds_stencil = 5*np.arange(n_rows)
        
        #STORE LEFT        
        bdry_flds_2d[:, offset + inds_stencil + txx_off] = tau_xx[:,gather_ind_left]
        bdry_flds_2d[:, offset + inds_stencil + tzz_off] = tau_zz[:,gather_ind_left]
        bdry_flds_2d[:, offset + inds_stencil + txz_off] = tau_xz[:,gather_ind_left]
        bdry_flds_2d[:, offset + inds_stencil +  vx_off] = vx[    :,gather_ind_left]
        bdry_flds_2d[:, offset + inds_stencil +  vz_off] = vz[    :,gather_ind_left]
        
        offset      += col_increment

        #STORE ON        
        bdry_flds_2d[:, offset + inds_stencil + txx_off] = tau_xx[:,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil + tzz_off] = tau_zz[:,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil + txz_off] = tau_xz[:,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil +  vx_off] = vx[    :,gather_ind_on]
        bdry_flds_2d[:, offset + inds_stencil +  vz_off] = vz[    :,gather_ind_on]
        
        offset      += col_increment

        #STORE RIGHT        
        bdry_flds_2d[:, offset + inds_stencil + txx_off] = tau_xx[:,gather_ind_right]
        bdry_flds_2d[:, offset + inds_stencil + tzz_off] = tau_zz[:,gather_ind_right]
        bdry_flds_2d[:, offset + inds_stencil + txz_off] = tau_xz[:,gather_ind_right]
        bdry_flds_2d[:, offset + inds_stencil +  vx_off] = vx[    :,gather_ind_right]
        bdry_flds_2d[:, offset + inds_stencil +  vz_off] = vz[    :,gather_ind_right]
        
        offset      += col_increment

        ################# RESHAPE BACK AND RETURN #################
        bdry_flds = np.reshape(bdry_flds_2d, (nvals_bdry,), 'C')
        return bdry_flds        
    
    ######################################################
    if np.any(gather_bdry[0,:] > 1e-14): #Green should be 0 at timestep 0, but due to convolution with source wavelet some values around numerical eps could be present
        raise Exception("Was the source placed within the integral zone? Probably will be no problem, but in this case it may no longer be valid to put txx, txz and tzz to 0.0 at t = 0 as I am currently doing. If this error is triggered and the source is not within integral region, it is probably safe to ignore it. Velocity at t_step = 0 will give stresses at t_step = 1, there is no velocity to give the stress at t_step = 0")
    
    amp_scale   = empirical_amplitude_fix_density_contrast(m, trunc_geom_dict, vp_2d, rho_2d, vs_2d, pos_x_src, pos_z_src, rho_src)
    #gather_bdry = amp_scale * gather_bdry (multiplying first seems to give worse results than multiplying after for some reason?)
    
    #First collect all the indices.
    try:
        ind_dict = compute_boundary_fields_cda_solver.ind_dict
    except:
        ind_dict = precompute_indices(m, trunc_geom_dict)
        compute_boundary_fields_cda_solver.ind_dict = ind_dict

    gb = gather_bdry #shorthand 

    if m.x.delta != m.z.delta:
        raise Exception("Assuming equal spacing...")
    
    #Use input recording times to get dt
    gb_dt   = ts_in[1]-ts_in[0] #This dt is different from the dt that was used by the solver if the gather is subsampled 
    gb_nt   = ts_in.size
    if not np.all(float_eq(ts_in/gb_dt, np.arange(gb_nt))):
        raise Exception("Input times array is not uniformly spaced") 

    #If a wavelet is provided, I will assume gb contains Green's functions which need to be convolved with the source wavelet.
    if wavelet:
        source_trace = wavelet._evaluate_time(ts_in)
        gb = recording_period*convolve_shotgather_with_wavelet(gb, source_trace)[0:gb_nt,:] #multiplication with recording period compensates amplitude for reduction in timesteps. Only take up to the input number of timesteps (convolution changes length) 

    #Dense times. These were the times the solver used when computing the wavefield before subsampling    
    dt_dense  = gb_dt / recording_period
    ts_dense  = dt_dense*np.arange(recording_period*(gb_nt-1) + 1)

    dtdx      = dt_dense/m.x.delta

    #COMPUTE VELOCITIES ASSUMING A CDA MEDIUM (RHO CONSTANT AND EQUAL TO TAPER VALUE, VS = 0)
    #If subsampled gather, the velocities will be correct at the subsampling points. 
    #Within 'compute_velocities' an interpolation will take place, so vx_gb and vz_gb have same size as ts_sub
    vx_gb, vz_gb = compute_velocities(gb, m, rho_2d, dtdx, ind_dict, trunc_geom_dict, recording_period, ts_in, ts_dense)
    
    #Use these (interpolated) velocities to compute the stresses in the medium with stiffness parameters lambda and mu obtained from vp_2d, rho and vs_2d
    tau_xx_gb, tau_zz_gb, tau_xz_gb = compute_stresses(vx_gb, vz_gb, m, vp_2d, rho_2d, vs_2d, dtdx, ind_dict, trunc_geom_dict)
    
    vx_gb *= amp_scale
    vz_gb *= amp_scale
    tau_xx_gb *= amp_scale
    tau_xz_gb *= amp_scale
    tau_zz_gb *= amp_scale
    
    #Now fill bdry_flds vector which will be used in the local solver
    bdry_flds = gen_bdry_flds_arr(vx_gb, vz_gb, tau_xx_gb, tau_xz_gb, tau_zz_gb, trunc_geom_dict, m)
    
    return bdry_flds
    
def compute_boundary_fields_cda_solver(complete_gather, m, dt, rho, trunc_geom_dict):

    #Precompute indices
    def precompute_indices(m, trunc_geom_dict):
        #COMPUTES INDICES FOR GATHER WITHOUT PHYSICAL RECEIVERS (ibg below). 
         
        
        #WE ALSO NEED TO CALCULATE VX AND VZ UPDATES FROM THE PRESSURE TRACES USING 
        #THE FOURTH ORDER DERIVATIVE STENCIL. DO STAGGERED GRID EVALUATIONS.
        #INSTEAD OF USING THREE LAYERS AROUND EACH BOUNDARY, NOW PAD WITH 2 LAYERS ON EACH SIDE.
        #EACH LAYER IS ALSO 2 WIDER
        #JUST AS IN fd2d_update_SSG.c
        dx = m.x.delta
        dz = m.z.delta
        ind_dict = dict()
        
        #LOTS OF REPETITION ON CODE BELOW. SHOULD BE ABLE TO MAKE MORE COMPACT
        
        ###################### TOP #############################################
        
        x_min = trunc_geom_dict['rec_x_l'] - 3*dx #pad by 3 for horizontal derivs
        x_max = trunc_geom_dict['rec_x_r'] + 3*dx #pad by 3 for horizontal derivs
        nx    = trunc_geom_dict['rec_h_n'] + 6 
        
        z_min = trunc_geom_dict['rec_z_t'] - 3*dz
        z_max = trunc_geom_dict['rec_z_t'] + 3*dz
        nz    = 7
    
        top_x = np.linspace(x_min, x_max, nx)
        top_z = np.linspace(z_min, z_max, nz)
        
        #Loop is not very efficient in python. But we cache the results, so only need to do once. 
        #Probably not necessary to optimize because of that
        name_arr = ['ind_top_m3',
                    'ind_top_m2',
                    'ind_top_m1',
                    'ind_top_0' ,
                    'ind_top_p1',
                    'ind_top_p2',
                    'ind_top_p3']
        iz = 0
        for z in top_z:
            name = name_arr[iz]
            ind_arr = np.zeros(nx, dtype='int64')
            ix = 0
            for x in top_x:
                ind_arr[ix] = pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict)
                ix         +=1
                
            ind_dict[name] = ind_arr
            iz            += 1
            
        ###################### END TOP #########################################
    
        ###################### RIGHT############################################
        x_min = trunc_geom_dict['rec_x_r'] - 3*dx #pad by 3 for horizontal derivs
        x_max = trunc_geom_dict['rec_x_r'] + 3*dx #pad by 3 for horizontal derivs
        nx    = 7 
        
        z_min = trunc_geom_dict['rec_z_t'] - 3*dz
        z_max = trunc_geom_dict['rec_z_b'] + 3*dz
        nz    = trunc_geom_dict['rec_v_n'] + 6 
    
        right_x = np.linspace(x_min, x_max, nx)
        right_z = np.linspace(z_min, z_max, nz)        
        
        name_arr = ['ind_right_m3',
                    'ind_right_m2',
                    'ind_right_m1',
                    'ind_right_0' ,
                    'ind_right_p1',
                    'ind_right_p2',
                    'ind_right_p3']        
        
        ix = 0
        for x in right_x:
            name = name_arr[ix]
            ind_arr = np.zeros(nz, dtype='int64')
            iz = 0
            for z in right_z:
                ind_arr[iz] = pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict)
                iz         +=1
                
            ind_dict[name] = ind_arr        
            ix            += 1
        ###################### END RIGHT########################################
        
        ###################### BOT #############################################
        
        x_min = trunc_geom_dict['rec_x_l'] - 3*dx #pad by 3 for horizontal derivs
        x_max = trunc_geom_dict['rec_x_r'] + 3*dx #pad by 3 for horizontal derivs
        nx    = trunc_geom_dict['rec_h_n'] + 6 
        
        z_min = trunc_geom_dict['rec_z_b'] - 3*dz
        z_max = trunc_geom_dict['rec_z_b'] + 3*dz
        nz    = 7
    
        bot_x = np.linspace(x_min, x_max, nx)
        bot_z = np.linspace(z_min, z_max, nz)
        
        #Loop is not very efficient in python. But we cache the results, so only need to do once. 
        #Probably not necessary to optimize because of that
        name_arr = ['ind_bot_m3',
                    'ind_bot_m2',
                    'ind_bot_m1',
                    'ind_bot_0' ,
                    'ind_bot_p1',
                    'ind_bot_p2',
                    'ind_bot_p3']
        iz = 0
        for z in bot_z:
            name = name_arr[iz]
            ind_arr = np.zeros(nx, dtype='int64')
            ix = 0
            for x in bot_x:
                ind_arr[ix] = pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict)
                ix         +=1
                
            ind_dict[name] = ind_arr
            iz            += 1      
            
        ###################### END BOT #########################################
        
        ###################### LEFT ############################################
        
        x_min = trunc_geom_dict['rec_x_l'] - 3*dx #pad by 3 for horizontal derivs
        x_max = trunc_geom_dict['rec_x_l'] + 3*dx #pad by 3 for horizontal derivs
        nx    = 7 
        
        z_min = trunc_geom_dict['rec_z_t'] - 3*dz
        z_max = trunc_geom_dict['rec_z_b'] + 3*dz
        nz    = trunc_geom_dict['rec_v_n'] + 6 
    
        left_x = np.linspace(x_min, x_max, nx)
        left_z = np.linspace(z_min, z_max, nz)        
        
        name_arr = ['ind_left_m3',
                    'ind_left_m2',
                    'ind_left_m1',
                    'ind_left_0' ,
                    'ind_left_p1',
                    'ind_left_p2',
                    'ind_left_p3']        
        
        ix = 0
        for x in left_x:
            name = name_arr[ix]
            ind_arr = np.zeros(nz, dtype='int64')
            iz = 0
            for z in left_z:
                ind_arr[iz] = pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict)
                iz         +=1
                
            ind_dict[name] = ind_arr        
            ix            += 1
        
        ###################### END LEFT ########################################
        return ind_dict
        
    nr_physical = get_nr_physical(complete_gather, trunc_geom_dict, additional_boundary_field = True)
    int_bdry_gather = complete_gather[:,nr_physical:]

    #First collect all the indices.
    try:
        ind_dict = compute_boundary_fields_cda_solver.ind_dict
    except:
        ind_dict = precompute_indices(m, trunc_geom_dict)
        compute_boundary_fields_cda_solver.ind_dict = ind_dict
    
    ibg = int_bdry_gather #shorthand 
    
    txx_off = 0
    tzz_off = 1
    txz_off = 2
    vx_off  = 3
    vz_off  = 4
    
    if m.x.delta != m.z.delta:
        raise Exception("Assuming equal spacing...")
    
    dtdx    = dt/m.x.delta
    
    coe1    =  9./8
    coe2    = -1./24
    ###################### TOP #######################
    slc        = 3+np.arange(trunc_geom_dict['rec_h_n'])  #slice
    
    #vertical deriv of tau_zz (pressure) evaluated at vz stagger points
    ztzz_above = coe1*(ibg[:, ind_dict['ind_top_0' ][slc]] - ibg[:, ind_dict['ind_top_m1'][slc]]) + \
                 coe2*(ibg[:, ind_dict['ind_top_p1'][slc]] - ibg[:, ind_dict['ind_top_m2'][slc]])
    ztzz_on    = coe1*(ibg[:, ind_dict['ind_top_p1'][slc]] - ibg[:, ind_dict['ind_top_0' ][slc]]) + \
                 coe2*(ibg[:, ind_dict['ind_top_p2'][slc]] - ibg[:, ind_dict['ind_top_m1'][slc]])
    ztzz_below = coe1*(ibg[:, ind_dict['ind_top_p2'][slc]] - ibg[:, ind_dict['ind_top_p1'][slc]]) + \
                 coe2*(ibg[:, ind_dict['ind_top_p3'][slc]] - ibg[:, ind_dict['ind_top_0' ][slc]])
    
    #horizontal deriv of tau_xx (pressure) evaluated at vx stagger points
    xtxx_above = coe1*(ibg[:, ind_dict['ind_top_m1'][slc  ]] - ibg[:, ind_dict['ind_top_m1'][slc-1]]) + \
                 coe2*(ibg[:, ind_dict['ind_top_m1'][slc+1]] - ibg[:, ind_dict['ind_top_m1'][slc-2]])
    xtxx_on    = coe1*(ibg[:, ind_dict['ind_top_0' ][slc  ]] - ibg[:, ind_dict['ind_top_0' ][slc-1]]) + \
                 coe2*(ibg[:, ind_dict['ind_top_0' ][slc+1]] - ibg[:, ind_dict['ind_top_0' ][slc-2]])
    xtxx_below = coe1*(ibg[:, ind_dict['ind_top_p1'][slc  ]] - ibg[:, ind_dict['ind_top_p1'][slc-1]]) + \
                 coe2*(ibg[:, ind_dict['ind_top_p1'][slc+1]] - ibg[:, ind_dict['ind_top_p1'][slc-2]])                                  
    
    #from fd2d_update_SSG.c, function 'update_V_SSG'
    vz_top_updates_above = ztzz_above*dtdx/rho
    vz_top_updates_on    = ztzz_on*dtdx/rho
    vz_top_updates_below = ztzz_below*dtdx/rho
      
    vx_top_updates_above = xtxx_above*dtdx/rho
    vx_top_updates_on    = xtxx_on*dtdx/rho
    vx_top_updates_below = xtxx_below*dtdx/rho    

    #axis = 0 causes cumulative sum of updates over time index
    vz_top_above = np.cumsum(vz_top_updates_above, axis=0)
    vz_top_on    = np.cumsum(vz_top_updates_on   , axis=0)
    vz_top_below = np.cumsum(vz_top_updates_below, axis=0)
    
    vx_top_above = np.cumsum(vx_top_updates_above, axis=0)
    vx_top_on    = np.cumsum(vx_top_updates_on   , axis=0)
    vx_top_below = np.cumsum(vx_top_updates_below, axis=0)    

    p_top_above  = ibg[:, ind_dict['ind_top_m1'][slc]]
    p_top_on     = ibg[:, ind_dict['ind_top_0' ][slc]]
    p_top_below  = ibg[:, ind_dict['ind_top_p1'][slc]]        

    ###################### RIGHT #######################
    slc        = 3+np.arange(trunc_geom_dict['rec_v_n'])  #slice
    
    #vertical deriv of tau_zz (pressure) evaluated at vz stagger points
    ztzz_right = coe1*(ibg[:, ind_dict['ind_right_p1'][slc+1]] - ibg[:, ind_dict['ind_right_p1'][slc-0]]) + \
                 coe2*(ibg[:, ind_dict['ind_right_p1'][slc+2]] - ibg[:, ind_dict['ind_right_p1'][slc-1]])
    ztzz_on    = coe1*(ibg[:, ind_dict['ind_right_0' ][slc+1]] - ibg[:, ind_dict['ind_right_0' ][slc-0]]) + \
                 coe2*(ibg[:, ind_dict['ind_right_0' ][slc+2]] - ibg[:, ind_dict['ind_right_0' ][slc-1]])
    ztzz_left  = coe1*(ibg[:, ind_dict['ind_right_m1'][slc+1]] - ibg[:, ind_dict['ind_right_m1'][slc-0]]) + \
                 coe2*(ibg[:, ind_dict['ind_right_m1'][slc+2]] - ibg[:, ind_dict['ind_right_m1'][slc-1]])         
    
    #horizontal deriv of tau_xx (pressure) evaluated at vx stagger points
    xtxx_right = coe1*(ibg[:, ind_dict['ind_right_p1'][slc]] - ibg[:, ind_dict['ind_right_0' ][slc]]) + \
                 coe2*(ibg[:, ind_dict['ind_right_p2'][slc]] - ibg[:, ind_dict['ind_right_m1'][slc]])
    xtxx_on    = coe1*(ibg[:, ind_dict['ind_right_0' ][slc]] - ibg[:, ind_dict['ind_right_m1'][slc]]) + \
                 coe2*(ibg[:, ind_dict['ind_right_p1'][slc]] - ibg[:, ind_dict['ind_right_m2'][slc]])
    xtxx_left  = coe1*(ibg[:, ind_dict['ind_right_m1'][slc]] - ibg[:, ind_dict['ind_right_m2'][slc]]) + \
                 coe2*(ibg[:, ind_dict['ind_right_0' ][slc]] - ibg[:, ind_dict['ind_right_m3'][slc]])    

    #from fd2d_update_SSG.c, function 'update_V_SSG'
    vz_right_updates_right = ztzz_right*dtdx/rho
    vz_right_updates_on    = ztzz_on*dtdx/rho
    vz_right_updates_left  = ztzz_left*dtdx/rho
      
    vx_right_updates_right = xtxx_right*dtdx/rho
    vx_right_updates_on    = xtxx_on*dtdx/rho
    vx_right_updates_left  = xtxx_left*dtdx/rho    

    #axis = 0 causes cumulative sum of updates over time index
    vz_right_right = np.cumsum(vz_right_updates_right, axis=0)
    vz_right_on    = np.cumsum(vz_right_updates_on   , axis=0)
    vz_right_left  = np.cumsum(vz_right_updates_left , axis=0)
    
    vx_right_right = np.cumsum(vx_right_updates_right, axis=0)
    vx_right_on    = np.cumsum(vx_right_updates_on   , axis=0)
    vx_right_left  = np.cumsum(vx_right_updates_left , axis=0)    

    p_right_right  = ibg[:, ind_dict['ind_right_p1'][slc]]
    p_right_on     = ibg[:, ind_dict['ind_right_0' ][slc]]
    p_right_left   = ibg[:, ind_dict['ind_right_m1'][slc]]   
    
    ###################### BOT #########################
    slc        = 3+np.arange(trunc_geom_dict['rec_h_n'])  #slice
    
    #vertical deriv of tau_zz (pressure) evaluated at vz stagger points
    ztzz_above = coe1*(ibg[:, ind_dict['ind_bot_0' ][slc]] - ibg[:, ind_dict['ind_bot_m1'][slc]]) + \
                 coe2*(ibg[:, ind_dict['ind_bot_p1'][slc]] - ibg[:, ind_dict['ind_bot_m2'][slc]])
    ztzz_on    = coe1*(ibg[:, ind_dict['ind_bot_p1'][slc]] - ibg[:, ind_dict['ind_bot_0' ][slc]]) + \
                 coe2*(ibg[:, ind_dict['ind_bot_p2'][slc]] - ibg[:, ind_dict['ind_bot_m1'][slc]])
    ztzz_below = coe1*(ibg[:, ind_dict['ind_bot_p2'][slc]] - ibg[:, ind_dict['ind_bot_p1'][slc]]) + \
                 coe2*(ibg[:, ind_dict['ind_bot_p3'][slc]] - ibg[:, ind_dict['ind_bot_0' ][slc]])
    
    #horizontal deriv of tau_xx (pressure) evaluated at vx stagger points
    xtxx_above = coe1*(ibg[:, ind_dict['ind_bot_m1'][slc  ]] - ibg[:, ind_dict['ind_bot_m1'][slc-1]]) + \
                 coe2*(ibg[:, ind_dict['ind_bot_m1'][slc+1]] - ibg[:, ind_dict['ind_bot_m1'][slc-2]])
    xtxx_on    = coe1*(ibg[:, ind_dict['ind_bot_0' ][slc  ]] - ibg[:, ind_dict['ind_bot_0' ][slc-1]]) + \
                 coe2*(ibg[:, ind_dict['ind_bot_0' ][slc+1]] - ibg[:, ind_dict['ind_bot_0' ][slc-2]])
    xtxx_below = coe1*(ibg[:, ind_dict['ind_bot_p1'][slc  ]] - ibg[:, ind_dict['ind_bot_p1'][slc-1]]) + \
                 coe2*(ibg[:, ind_dict['ind_bot_p1'][slc+1]] - ibg[:, ind_dict['ind_bot_p1'][slc-2]])     

    #from fd2d_update_SSG.c, function 'update_V_SSG'
    vz_bot_updates_above = ztzz_above*dtdx/rho
    vz_bot_updates_on    = ztzz_on*dtdx/rho
    vz_bot_updates_below = ztzz_below*dtdx/rho
      
    vx_bot_updates_above = xtxx_above*dtdx/rho
    vx_bot_updates_on    = xtxx_on*dtdx/rho
    vx_bot_updates_below = xtxx_below*dtdx/rho    

    #axis = 0 causes cumulative sum of updates over time index
    vz_bot_above = np.cumsum(vz_bot_updates_above, axis=0)
    vz_bot_on    = np.cumsum(vz_bot_updates_on   , axis=0)
    vz_bot_below = np.cumsum(vz_bot_updates_below, axis=0)
    
    vx_bot_above = np.cumsum(vx_bot_updates_above, axis=0)
    vx_bot_on    = np.cumsum(vx_bot_updates_on   , axis=0)
    vx_bot_below = np.cumsum(vx_bot_updates_below, axis=0)    

    p_bot_above  = ibg[:, ind_dict['ind_bot_m1'][slc]]
    p_bot_on     = ibg[:, ind_dict['ind_bot_0' ][slc]]
    p_bot_below  = ibg[:, ind_dict['ind_bot_p1'][slc]]  
                         
    ###################### LEFT ########################
    slc        = 3+np.arange(trunc_geom_dict['rec_v_n'])  #slice
    
    #vertical deriv of tau_zz (pressure) evaluated at vz stagger points
    ztzz_right = coe1*(ibg[:, ind_dict['ind_left_p1'][slc+1]] - ibg[:, ind_dict['ind_left_p1'][slc-0]]) + \
                 coe2*(ibg[:, ind_dict['ind_left_p1'][slc+2]] - ibg[:, ind_dict['ind_left_p1'][slc-1]])
    ztzz_on    = coe1*(ibg[:, ind_dict['ind_left_0' ][slc+1]] - ibg[:, ind_dict['ind_left_0' ][slc-0]]) + \
                 coe2*(ibg[:, ind_dict['ind_left_0' ][slc+2]] - ibg[:, ind_dict['ind_left_0' ][slc-1]])
    ztzz_left  = coe1*(ibg[:, ind_dict['ind_left_m1'][slc+1]] - ibg[:, ind_dict['ind_left_m1'][slc-0]]) + \
                 coe2*(ibg[:, ind_dict['ind_left_m1'][slc+2]] - ibg[:, ind_dict['ind_left_m1'][slc-1]])         
    
    #horizontal deriv of tau_xx (pressure) evaluated at vx stagger points
    xtxx_right = coe1*(ibg[:, ind_dict['ind_left_p1'][slc]] - ibg[:, ind_dict['ind_left_0' ][slc]]) + \
                 coe2*(ibg[:, ind_dict['ind_left_p2'][slc]] - ibg[:, ind_dict['ind_left_m1'][slc]])
    xtxx_on    = coe1*(ibg[:, ind_dict['ind_left_0' ][slc]] - ibg[:, ind_dict['ind_left_m1'][slc]]) + \
                 coe2*(ibg[:, ind_dict['ind_left_p1'][slc]] - ibg[:, ind_dict['ind_left_m2'][slc]])
    xtxx_left  = coe1*(ibg[:, ind_dict['ind_left_m1'][slc]] - ibg[:, ind_dict['ind_left_m2'][slc]]) + \
                 coe2*(ibg[:, ind_dict['ind_left_0' ][slc]] - ibg[:, ind_dict['ind_left_m3'][slc]])    
                     
    #from fd2d_update_SSG.c, function 'update_V_SSG'
    vz_left_updates_right = ztzz_right*dtdx/rho
    vz_left_updates_on    = ztzz_on*dtdx/rho
    vz_left_updates_left  = ztzz_left*dtdx/rho
      
    vx_left_updates_right = xtxx_right*dtdx/rho
    vx_left_updates_on    = xtxx_on*dtdx/rho
    vx_left_updates_left  = xtxx_left*dtdx/rho    

    #axis = 0 causes cumulative sum of updates over time index
    vz_left_right = np.cumsum(vz_left_updates_right, axis=0)
    vz_left_on    = np.cumsum(vz_left_updates_on   , axis=0)
    vz_left_left  = np.cumsum(vz_left_updates_left , axis=0)
    
    vx_left_right = np.cumsum(vx_left_updates_right, axis=0)
    vx_left_on    = np.cumsum(vx_left_updates_on   , axis=0)
    vx_left_left  = np.cumsum(vx_left_updates_left , axis=0)    

    p_left_right  = ibg[:, ind_dict['ind_left_p1'][slc]]
    p_left_on     = ibg[:, ind_dict['ind_left_0' ][slc]]
    p_left_left   = ibg[:, ind_dict['ind_left_m1'][slc]]       
    
    ###################### SAVE INTO BOUNDARY FIELDS ########################
    
    #NOW USE THESE COMPUTED VALUES TO STORE THE BOUNDARY_FIELDS, SAME AS IN fd2d_rec_wavefields.c 
    n_rows = int(np.round((trunc_geom_dict['rec_z_b'] - trunc_geom_dict['rec_z_t'])/m.z.delta)) + 1
    n_cols = int(np.round((trunc_geom_dict['rec_x_r'] - trunc_geom_dict['rec_x_l'])/m.x.delta)) + 1
            
    #at each cell we save 5 quantities. tauxx, tauzz, tauxz, vx, vz
    row_increment = 5*n_cols
    col_increment = 5*n_rows
            
    #We have two horizontal and two vertical sides of square recording surface.
    #On each side we save three rows or columns of cells 
    #We do this for each timestep
    nt              = ibg.shape[0]
    nvals_bdry_each = (2*3*row_increment + 2*3*col_increment)
    nvals_bdry      = nvals_bdry_each*nt 
    warnings.warn("""FOR NOW I WILL STORE THE BOUNDARY VALUES EXACTLY AS IN THE ELASTIC C CODE. 
                     THIS IS INEFFICIENT BECAUSE txz = 0 ALWAYS AND txx = tzz. """)
    
    bdry_flds       = np.zeros(nvals_bdry)
    bdry_flds_2d    = np.reshape(bdry_flds, (nt, nvals_bdry_each), 'F') #Temporarily reshape into 2D so i can more easily store stuff
    
    ################# STORE TOP ################# 
    offset       = 0
    inds_stencil = 5*np.arange(n_cols)
    
    #ABOVE
    bdry_flds_2d[:, offset + inds_stencil + txx_off] = p_top_above
    bdry_flds_2d[:, offset + inds_stencil + tzz_off] = p_top_above
    bdry_flds_2d[:, offset + inds_stencil + vx_off ] = vx_top_above
    bdry_flds_2d[:, offset + inds_stencil + vz_off ] = vz_top_above
    
    offset      += row_increment
    #ON
    bdry_flds_2d[:, offset + inds_stencil + txx_off] = p_top_on
    bdry_flds_2d[:, offset + inds_stencil + tzz_off] = p_top_on
    bdry_flds_2d[:, offset + inds_stencil + vx_off ] = vx_top_on
    bdry_flds_2d[:, offset + inds_stencil + vz_off ] = vz_top_on    

    offset      += row_increment
    #BELOW
    bdry_flds_2d[:, offset + inds_stencil + txx_off] = p_top_below
    bdry_flds_2d[:, offset + inds_stencil + tzz_off] = p_top_below
    bdry_flds_2d[:, offset + inds_stencil + vx_off ] = vx_top_below
    bdry_flds_2d[:, offset + inds_stencil + vz_off ] = vz_top_below
    
    offset      += row_increment
    
    ################# STORE RIGHT #################
    inds_stencil = 5*np.arange(n_rows)
    
    #RIGHT
    bdry_flds_2d[:, offset + inds_stencil + txx_off] = p_right_right
    bdry_flds_2d[:, offset + inds_stencil + tzz_off] = p_right_right
    bdry_flds_2d[:, offset + inds_stencil + vx_off ] = vx_right_right
    bdry_flds_2d[:, offset + inds_stencil + vz_off ] = vz_right_right    
    
    offset      += col_increment
    #ON
    bdry_flds_2d[:, offset + inds_stencil + txx_off] = p_right_on
    bdry_flds_2d[:, offset + inds_stencil + tzz_off] = p_right_on
    bdry_flds_2d[:, offset + inds_stencil + vx_off ] = vx_right_on
    bdry_flds_2d[:, offset + inds_stencil + vz_off ] = vz_right_on    
    
    offset      += col_increment
    #LEFT
    bdry_flds_2d[:, offset + inds_stencil + txx_off] = p_right_left
    bdry_flds_2d[:, offset + inds_stencil + tzz_off] = p_right_left
    bdry_flds_2d[:, offset + inds_stencil + vx_off ] = vx_right_left
    bdry_flds_2d[:, offset + inds_stencil + vz_off ] = vz_right_left    
    
    offset      += col_increment    
    
    ################# STORE BOT #################
    #FIELDS ARE STORED FROM RIGHT TO LEFT IN fd2d_rec_wavefields.c
    #NEED TO FLIP DIRECTION BY USING [::-1]
    inds_stencil = 5*np.arange(n_cols)
    
    #BELOW
    bdry_flds_2d[:, offset + inds_stencil + txx_off] = p_bot_below[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + tzz_off] = p_bot_below[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + vx_off ] = vx_bot_below[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + vz_off ] = vz_bot_below[:, ::-1]
    
    offset      += row_increment
    #ON
    bdry_flds_2d[:, offset + inds_stencil + txx_off] = p_bot_on[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + tzz_off] = p_bot_on[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + vx_off ] = vx_bot_on[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + vz_off ] = vz_bot_on[:, ::-1]
    
    offset      += row_increment    
    #ABOVE
    bdry_flds_2d[:, offset + inds_stencil + txx_off] = p_bot_above[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + tzz_off] = p_bot_above[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + vx_off ] = vx_bot_above[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + vz_off ] = vz_bot_above[:, ::-1]
    
    offset      += row_increment
    ################# STORE LEFT #################
    #FIELDS ARE STORED FROM BOT TO TOP IN fd2d_rec_wavefields.c
    #NEED TO FLIP DIRECTION BY USING [::-1]
    inds_stencil = 5*np.arange(n_rows)
    
    #LEFT
    bdry_flds_2d[:, offset + inds_stencil + txx_off] = p_left_left[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + tzz_off] = p_left_left[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + vx_off ] = vx_left_left[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + vz_off ] = vz_left_left[:, ::-1]
    
    offset      += col_increment
    #ON
    bdry_flds_2d[:, offset + inds_stencil + txx_off] = p_left_on[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + tzz_off] = p_left_on[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + vx_off ] = vx_left_on[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + vz_off ] = vz_left_on[:, ::-1]
    
    offset      += col_increment
    #RIGHT
    bdry_flds_2d[:, offset + inds_stencil + txx_off] = p_left_right[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + tzz_off] = p_left_right[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + vx_off ] = vx_left_right[:, ::-1]
    bdry_flds_2d[:, offset + inds_stencil + vz_off ] = vz_left_right[:, ::-1]
    
    offset      += col_increment
    ################# RESHAPE BACK AND RETURN #################
    bdry_flds = np.reshape(bdry_flds_2d, (nvals_bdry,), 'C')
    return bdry_flds
    
def pos_to_ind_int_bdry_gather(x,z,m, trunc_geom_dict, new_implementation = False): #Convenience function
    #COMPUTES INDICES FOR GATHER WITHOUT PHYSICAL RECEIVERS.
    #Only nodes corresponding to 'int' and 'bdry' points
    
    #new_implementation refers to the new CDA green's function precomputation where I store only pressure
    #In this setting I store more layers of pressure so I can later compute injection velocities from this based on mu and rho of the local medium
    #Backward compatability with the old formulation is not making this code easier to read
    
    def pos_to_ind_int_bdry_gather_compute(x,z,m, trunc_geom_dict, new_implementation): #local function
    
    
        if x < trunc_geom_dict['int_x_l'] or x > trunc_geom_dict['int_x_r']:
            raise Exception("x out of range")
          
        if z < trunc_geom_dict['int_z_t'] or z > trunc_geom_dict['int_z_b']:
            raise Exception("z out of range")
        
        #If within truncated domain
        if (x >= trunc_geom_dict['rec_x_l'] and x <= trunc_geom_dict['rec_x_r'] and
            z >= trunc_geom_dict['rec_z_t'] and z <= trunc_geom_dict['rec_z_b']):   
        
            #we are in the bdry section of the gather, offset by the number of traces in the 'int' part.
            offset = trunc_geom_dict['int_n'] 
            
            if new_implementation:
                x_pos, z_pos = get_additional_required_boundary_fields_positions_new(m, trunc_geom_dict)  
            else: #old
                x_pos, z_pos = get_additional_required_boundary_fields_positions(m, trunc_geom_dict)
            req_pos = (x,z)
            ind = 0
            for pos in zip(x_pos,z_pos):
                if float_eq(req_pos[0], pos[0]) and float_eq(req_pos[1], pos[1]):
                    return offset + ind
                
                ind += 1
                
            raise Exception("Position (%.2f, %.2f) was not in the list. Some logic error"%(x,z))            
            
            
        #If within integrate domain
        else:
            offset = 0
            
            x_pos, z_pos = get_boundary_integral_positions(m, trunc_geom_dict)
            req_pos = (x,z)
            ind = 0
            for pos in zip(x_pos,z_pos):
                if float_eq(req_pos[0], pos[0]) and float_eq(req_pos[1], pos[1]):
                    return offset + ind
                
                ind += 1
                
            raise Exception("Position (%.2f, %.2f) was not in the list. Some logic error"%(x,z))

    ###### CHECK IF WE DID NOT USE A DIFFERENT VALUE FOR 'new_implementation' PREVIOUSLY. COULD HAVE INCONSISTENT VALUES CACHED ###########
    try:
        prev_val_new_implementation = pos_to_ind_int_bdry_gather.prev_val_new_implementation
    except:
        pos_to_ind_int_bdry_gather.prev_val_new_implementation = new_implementation
        prev_val_new_implementation = new_implementation
        
    if prev_val_new_implementation != new_implementation:
        raise Exception("Not consistent in whether running old or new implementation in the code. ") 

    ###### END CHECK ##################

    #CHECK IF WE HAVE THE CACHE DICTIONARY CREATED
    try:
        prec_dict = pos_to_ind_int_bdry_gather.prec_dict
    except:
        prec_dict = dict()
        pos_to_ind_int_bdry_gather.prec_dict = prec_dict

    #GET INDEX FROM THE DICTIONARY FOR THE POSITION POS. IF NOT CACHED, COMPUTE IT.
    pos = (x,z)        
    try:
        ind = prec_dict[pos]
    except:
        ind = pos_to_ind_int_bdry_gather_compute(x,z,m, trunc_geom_dict, new_implementation=new_implementation)
        prec_dict[pos] = ind
        
    return ind
        
def precompute(shots, m, rec_boundary_geom, vp_2d, rho_2d, vs_2d, elastic_options_dict, save_prefix = ''):
    #Determine unique source and receiver locations
    #Loop over the unique positions and make a shot for each of them.
    #For each of these shots, put 5 boxes of receivers outside of S_i. 
    #That way we can compute both the inward and centered derivative and see which is best
    #Will overlap slightly which results in double storage with 'rec_boundary', could optimize later.
    #Could probably also bring the recording boundary closer so that it overlaps more with 'rec_boundary' to optimize reuse more
    #But this would result in a few pressure recordings which would be in places whre Vs != 0 and then I'm not sure if 
    #we actually could use our pressure boundary integral (assumes no Vs to make the elastic integral reduce to pressure integral)
    
    #Input:
    #shots: list of shot objects
    #m: mesh object with elastic PML objects
    #rec_boundary_geom: A dict describing the geometry of the truncated domain. 
    #                   includes keys such as rec_x_l, rec_x_r, rec_z_t, rec_z_b
    #vp_2d : 2d array of vp to generate Green's functions on
    #rho_2d: 2d array of vp to generate Green's functions on
    #vs_2d : 2d array of vp to generate Green's functions on
    rec_boundary_geom = elastic_options_dict['rec_boundary_geom']
    validate_truncation_params(m, rec_boundary_geom)

    trunc_geom_dict = get_trunc_geom_dict(m, rec_boundary_geom)
    
    #I'm just going to assume that it is cheapest to generate Green's functions from sources and receivers to the truncation box
    #In the 2D truncated Helmholtz solver I used a greedy algorithm.
    #The reason for not doing that here is that I don't just need pressure Green's functions.
    #I also need to record Vx, Vz, Tauxx, Tauzz and Tauxz at the recording boundary for source positions.
    #Greedy algorithm would need to be more complex. Not worth it right now
    
    x_pos_int, z_pos_int = get_boundary_integral_positions(m, trunc_geom_dict)
    
    #Only need to do 'rec_boundary' for shots corresponding to real source positions for the forward model. 
    #But if we ever need to compute the adjoint field we would also need these quantities for sources at the receiver positions.
    
    #Make new shot objects 
    shots_s = get_source_shots(shots, m, x_pos_int, z_pos_int)
    u_rec_pos_x, u_rec_pos_z = get_unique_receiver_positions(shots)
    shots_r = get_receiver_shots(m, u_rec_pos_x, u_rec_pos_z, x_pos_int, z_pos_int)

    #Some error check
    validate_shots(shots_s, trunc_geom_dict)
    validate_shots(shots_r, trunc_geom_dict)
        
    compute_greens_functions(shots_s, shots_r, m, vp_2d, rho_2d, vs_2d, elastic_options_dict, save_prefix)

def shot_indices_for_process(nshots):
    #GIVE BACK AN ARRAY OF NUMBERS WITH RANGE FROM 0 - nshots-1 TO EACH PROCESS
    nshots = int(nshots) #Should already be int
    
    #PARALLEL VARIABLES
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    min_nr_per_process = nshots / size #Does floor
    
    if min_nr_per_process > 0:
        nr_leftover_shots  =  nshots % (min_nr_per_process * size)
    else: #Prevent potential modulus 0 error when more processes than shots 
        nr_leftover_shots  =  nshots 
     
    nr_shots_this_process = min_nr_per_process #Initialize
    if rank < nr_leftover_shots: #Rank starts at 0
        nr_shots_this_process += 1    
    
    local_shot_indices = []
    for i in xrange(nr_shots_this_process):
        
        shot_index = i*size + rank
        local_shot_indices.append(shot_index)
        print "Assigning shot %i of %i to process %i"%(shot_index+1, nshots, rank)    

    return np.array(local_shot_indices)

def precompute_using_cda_solver(shots, m, rec_boundary_geom, vp_2d, cda_options_dict, save_prefix=''):
    #Same idea as 'precompute' where an elastic solver is used.
    #But here was use the pysit CDA solver
    
    validate_truncation_params(m, rec_boundary_geom)
    trunc_geom_dict = get_trunc_geom_dict(m, rec_boundary_geom)
    
    x_pos_int , z_pos_int  = get_boundary_integral_positions(m, trunc_geom_dict)
    x_pos_bdry, z_pos_bdry = get_additional_required_boundary_fields_positions(m, trunc_geom_dict) #We need a few extra receivers now to record the boundary pressure for the local solver 
    
    x_pos_additional = np.concatenate([x_pos_int, x_pos_bdry])
    z_pos_additional = np.concatenate([z_pos_int, z_pos_bdry])
    
    #make new shot objects
    shots_s = get_source_shots(shots, m, x_pos_additional, z_pos_additional)
    u_rec_pos_x, u_rec_pos_z = get_unique_receiver_positions(shots)
    shots_r = get_receiver_shots(m, u_rec_pos_x, u_rec_pos_z, x_pos_additional, z_pos_additional)  
    
    #Some error check
    validate_shots(shots_s, trunc_geom_dict)
    validate_shots(shots_r, trunc_geom_dict)   
    
    compute_greens_functions_cda_solver(shots_s, shots_r, m, vp_2d, trunc_geom_dict, cda_options_dict, save_prefix)
    
def precompute_p_greens_only_using_cda_solver(shots, m, rec_boundary_geom, vp_2d, cda_options_dict, save_prefix='', l_ind = None, r_ind = None):
    """This function is similar to precompute_using_cda_solver, except here we store extra pressure fields
       but we will not store the boundary_fields vector. With the extra pressures I can compute 
       the velocities and their spatial derivatives. With these velocity derivatives we can compute txx, txz and tzz 
       which we need at the injection boundary during the local solve.
       The reason for computing these stresses during the local solve is that it will depend on what value of mu we use in the taper. 
       For instance, when the taper mu = 0, the injection boundary is a liquid and txx = tzz = p and we could have used the p from pysit.
       If mu != 0, in general txx != tzz and txz != 0 even though a pressure wave is being injected.
       IN ESSENSE WE COMPUTE THE VELOCITY USING THE CDA PYSIT SOLVER. THEN WE CAN INJECT THIS VELOCITY INTO A MEDIUM WITH NONZERO MU.
       WITH THESE VELOCITY VALUES OF THE INCIDENT P-WAVE FRONT WE CAN THEN CALCULATE THE CORRECT VALUES FOR TAU_XX, TAU_ZZ AND TAU_XZ FOR THE LOCAL MEDIUM.
       That txz != 0 for a P-wave in certain propagation directions when txx != tzz is easily seen through Mohr's circle. 
       #When txx != tzz the radius of this circle is larger than zero, and certain coordinate frames will have nonzero shear stress txz.
       
       #l_ind and r_ind are optional and can be provided when you only want to compute a subset of the 'receiver shots'.
        This is useful when you want to split the load over multiple computers, as the receiver shots are the shots 
        we have most of in my examples. When splitting work the source shots will be recomputed which is wasteful.
        Would have to change code to do this more efficiently.
    """
    
    validate_truncation_params(m, rec_boundary_geom)
    trunc_geom_dict = get_trunc_geom_dict(m, rec_boundary_geom)
    
    x_pos_int , z_pos_int  = get_boundary_integral_positions(m, trunc_geom_dict)
    
    #add several layers of pressure receivers in PySIT so we can compute vx, vz at all the locations which we will later require when computing the velocity derivatives
    x_pos_bdry, z_pos_bdry = get_additional_required_boundary_fields_positions_new(m, trunc_geom_dict)  
    
    #concatenate these positions into arrays containing all required positions
    x_pos_additional = np.concatenate([x_pos_int, x_pos_bdry])
    z_pos_additional = np.concatenate([z_pos_int, z_pos_bdry])    
    
    #make new shot objects. We compute a bit too much right now. For the receivers we do not actually need the boundary_fields for injection,
    #since right now we only extrapolate to them. In the future it may be useful though if we want to to backpropagations
    #Also, for the physical shots we only need the boundary field for injection, we don't need green's functions to the integration points.
    #Not really efficient, could improve there.
    #make new shot objects

    u_rec_pos_x, u_rec_pos_z = get_unique_receiver_positions(shots)
    
    #SEE IF WE ARE GOING TO COMPUTE ONLY A SUBSET OF THE NUMBER OF RECEIVER SHOTS
    if type(l_ind) != type(None) and type(r_ind) != type(None):
        nr_r_shots_full = len(u_rec_pos_x)
        
        if r_ind > nr_r_shots_full:
            r_ind = nr_r_shots_full

        #use subset
        u_rec_pos_x = u_rec_pos_x[l_ind:r_ind]
        u_rec_pos_z = u_rec_pos_z[l_ind:r_ind]
    
    #number of source and receiver shots
    nr_s_shots = len(shots)
    nr_r_shots = len(u_rec_pos_x)
    
    nshots = nr_s_shots + nr_r_shots 
    
    #now distribute
    indices_for_process   = shot_indices_for_process(nshots)
    
    #get source and receiver indices this particular process will compute
    s_indices_for_process = indices_for_process[indices_for_process <  nr_s_shots]
    r_indices_for_process = indices_for_process[indices_for_process >= nr_s_shots] - nr_s_shots
    
    s_shots_for_process = [shots[i] for i in s_indices_for_process]
    u_rec_pos_x_for_process = np.array([u_rec_pos_x[i] for i in r_indices_for_process])
    u_rec_pos_z_for_process = np.array([u_rec_pos_z[i] for i in r_indices_for_process])
    
    shots_s_in_for_process = get_source_shots(s_shots_for_process, m, x_pos_additional, z_pos_additional, distr_shots = False)
    shots_r_in_for_process = get_receiver_shots(m, u_rec_pos_x_for_process, u_rec_pos_z_for_process, x_pos_additional, z_pos_additional, distr_shots = False) 
    

    shots_in_for_process = shots_s_in_for_process + shots_r_in_for_process 
    
    #some error check
    validate_shots(shots_in_for_process, trunc_geom_dict)
    
    compute_p_greens_functions_only_cda_solver(shots_in_for_process, m, vp_2d, trunc_geom_dict, cda_options_dict, save_prefix)
    

    
    