import copy
import warnings
import numpy as np
from scipy.signal import fftconvolve
from pysit.core import *
from scipy import interpolate
from pysit_extensions.elastic_solver.wrapping_functions import elastic_solve, full_to_truncated, Elastic_Code_PML, antideriv_wavelet

def float_eq(a, b, epsilon=0.00000001): #convenience function to test for float equality. Should really use numerical eps instead of the arbitrary number
    return abs(a - b) < epsilon

def interpolate_gather_in_time_2d(gather_2d, ts_in, ts_out):
    ntrace = gather_2d.shape[1]
    
    #interpolate
    gather_interpolate        = interpolate.interp2d(np.arange(ntrace), ts_in, gather_2d, kind='cubic')
    boundary_field_full_2d    = gather_interpolate(np.arange(ntrace), ts_out)    
    return boundary_field_full_2d

def change_PMLs(m, new_thickness):
    d0factor = m.domain.x.lbc.d0factor
    PPW0     = m.domain.x.lbc.PPW0
    p_power  = m.domain.x.lbc.p_power
    
    x_min = m.domain.x.lbound
    z_min = m.domain.z.lbound

    x_max = m.domain.x.rbound
    z_max = m.domain.z.rbound    
    
    pmlx = Elastic_Code_PML(new_thickness, d0factor, PPW0, p_power)
    pmlz = Elastic_Code_PML(new_thickness, d0factor, PPW0, p_power)

    x_config = (x_min, x_max, pmlx, pmlx)
    z_config = (z_min, z_max, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)
    
    uniform_spacing = m.x.delta
    nx = int((x_max-x_min)/uniform_spacing + 1)
    nz = int((z_max-z_min)/uniform_spacing + 1)

    return CartesianMesh(d, nx, nz)    

def reshape_boundary_fields_from_1d_to_2d(trunc_geom_dict, boundary_fields):
    nvals_each = trunc_geom_dict['nvals_each']
    if boundary_fields.size%nvals_each != 0:
        raise Exception("Some error")
    
    nt = boundary_fields.size/nvals_each
    boundary_fields_2d = np.reshape(boundary_fields, (nt, nvals_each), 'C')
    return boundary_fields_2d

def reshape_boundary_fields_from_2d_to_1d(boundary_fields_2d):
    nt, nvals_each = boundary_fields_2d.shape
    boundary_fields = np.reshape(boundary_fields_2d, (nt*nvals_each,), 'C')
    return boundary_fields

def compensate_boundary_fields_for_current_rho(boundary_fields, rho_boundary, rho_used_for_boundary_fields, trunc_geom_dict):
    """
    When computing the boundary wavefields with the CDA solver I needed to compute velocity from pressure gradients.
    This required me to use a value for density. When computing the boundary wavefields I used a single value of rho.
    When using a different value of rho in the local solver we have to correct the velocities for the rho values we 
    should have used. 
    """
    
    boundary_fields    = np.copy(boundary_fields) #Do not overwrite original
    
    boundary_fields_2d = reshape_boundary_fields_from_1d_to_2d(trunc_geom_dict, boundary_fields)
    vx_off = 3
    vz_off = 4    
    
    nvals_each = boundary_fields_2d.shape[1]
    
    #FOR NOW JUST CHANGE VX BY CHANGING THE DENSITY. INSTEAD WE MAY HAVE TO CHANGE THE PRESSURE ? HAVE TO THINK ABOUT HOW CDA RELATES TO VDA
    
    warnings.warn("IF IN THE FUTURE I ALLOW VARIABLE RHO ALONG BOUNDARY, MAKE SURE I THINK BETTER ABOUT WHETHER VELOCITY SHOULD BE FIXED OR PRESSURE WHEN GOING FROM CDA TO VDA. ALSO NEED TO MODIFY BOUNDARY INTEGRAL IN THAT CASE!!!!")
    
    #ASSUMING DENSITY IS CONSTANT ALONG BOUNDARY FOR NOW. IF NOT WE WILL NEED TO DO MORE SOPHISTICATED THINGS
    
    #Compensate vx
    n_inj_nodes = nvals_each/5
    vx_ind = 5*np.arange(n_inj_nodes) + vx_off
    boundary_fields_2d[:, vx_ind] *= rho_used_for_boundary_fields/rho_boundary
    
    #Compensate vz
    vz_ind = 5*np.arange(n_inj_nodes) + vz_off
    boundary_fields_2d[:, vz_ind] *= rho_used_for_boundary_fields/rho_boundary
    
    return boundary_fields
    
def compute_boundary_recordings_local_solver(shots_in, m_full, vp_2d_full, rho_2d_full, vs_2d_full, elastic_options, recorded_boundary_fields, n_pad_nodes = 5, PMLwidth = 100.0, ts_subsampled=None):
    #Will overwrite certain options from elastic solver 
    elastic_options_truncated = dict()
    elastic_options_truncated.update(elastic_options)
    elastic_options_truncated['snap_output']  = False
    elastic_options_truncated['rec_boundary'] = False
    elastic_options_truncated['local_solve']  = True
    elastic_options_truncated['traces_output']= True
    elastic_options_truncated['traces_mem']   = True

    rec_boundary_geom = elastic_options_truncated['rec_boundary_geom']
    trunc_geom_dict   = get_trunc_geom_dict(m_full, rec_boundary_geom)

    dx          = trunc_geom_dict['dx']
    dz          = trunc_geom_dict['dz']

    #Change PML thickness. Probably can do with much thinner PML.
    #Bit hacky...
    m_new = change_PMLs(m_full, PMLwidth)
     
    x_left_subm  = trunc_geom_dict['int_x_l'] - n_pad_nodes*dx
    x_right_subm = trunc_geom_dict['int_x_r'] + n_pad_nodes*dx 
    z_top_subm   = trunc_geom_dict['int_z_t'] - n_pad_nodes*dz
    z_bot_subm   = trunc_geom_dict['int_z_b'] + n_pad_nodes*dz

    if (x_left_subm < m_full.domain.x.lbound or x_right_subm > m_full.domain.x.rbound or 
        z_top_subm  < m_full.domain.z.lbound or z_bot_subm   > m_full.domain.z.rbound): 
        raise Exception("The extra pad nodes make domain exceed physical domain.") 

    vp_trunc_2d   = full_to_truncated( vp_2d_full, x_left_subm, x_right_subm, z_top_subm, z_bot_subm, dx)
    rho_trunc_2d  = full_to_truncated(rho_2d_full, x_left_subm, x_right_subm, z_top_subm, z_bot_subm, dx)
    vs_trunc_2d   = full_to_truncated( vs_2d_full, x_left_subm, x_right_subm, z_top_subm, z_bot_subm, dx)
    
    [nz_trunc, nx_trunc] = vp_trunc_2d.shape
    elastic_options_truncated['gx0'] = x_left_subm
    elastic_options_truncated['gz0'] = z_top_subm
    elastic_options_truncated['MM']  = nx_trunc
    elastic_options_truncated['KK']  = nz_trunc

    if 'traces_step_length' not in elastic_options_truncated.keys():
        elastic_options_truncated['traces_step_length'] = 1
    
    validate_shots(shots_in, trunc_geom_dict)
    
    x_pos_int, z_pos_int = get_boundary_integral_positions(m_new, trunc_geom_dict)
    
    shots_out = []
    for shot_in in shots_in:
        #Create a new shot for the local solver based on the full domain shot. 
        #Remove the full domain receivers, and replace with boundary integral receivers
        
        source = copy.deepcopy(shot_in.sources)
        receivers = ReceiverSet(m_new, [PointReceiver(m_new, (x, z), approximation='delta') for x,z in zip(x_pos_int, z_pos_int)])
        shots_out.append(Shot(source, receivers))        

    if len(recorded_boundary_fields) != len(shots_out):
        raise Exception("Should have a list with one for each shot!")

    if type(ts_subsampled) != type(None): #If we subsampled the boundary fields, interpolate now so that local solver will have something to inject at every timestep
        nshots = len(shots_in)
        for i in xrange(nshots):
            boundary_field_subsampled = recorded_boundary_fields[i]
            nt_full                   = elastic_options_truncated['itimestep']
            ts_full                   = np.arange(nt_full) * elastic_options_truncated['dt']

            boundary_field_subsampled_2d = reshape_boundary_fields_from_1d_to_2d(trunc_geom_dict, boundary_field_subsampled)
            
            boundary_field_full_2d    = interpolate_gather_in_time_2d(boundary_field_subsampled_2d, ts_subsampled, ts_full)
            boundary_field_full       = reshape_boundary_fields_from_2d_to_1d(boundary_field_full_2d)
            
            #Replace the boundary field with its interpolated one
            recorded_boundary_fields[i] = boundary_field_full
            
        

    retdict = elastic_solve(shots_out, 
                            m_new, 
                            vp_trunc_2d, 
                            rho_trunc_2d, 
                            vs_trunc_2d, 
                            elastic_options_truncated,
                            recorded_boundary_fields = recorded_boundary_fields
                            )

    return retdict['shotgathers'], retdict['shotgathers_times']

def validate_shots(shots, trunc_geom_dict):
    #make sure no shot (which could be a physical receiver) is within the integral (and also truncated domain as result). 
    #Would have to treat differently perhaps. To be safe, just don't allow 
    l = trunc_geom_dict['int_x_l']
    r = trunc_geom_dict['int_x_r']
    t = trunc_geom_dict['int_z_t']
    b = trunc_geom_dict['int_z_b']
    for shot in shots:
        xs, zs = shot.sources.position
        if xs >= l and xs <= r and zs >= t and zs <=b:
            raise Exception("Source within integral domain")
     

def global_geometry(m): #To save some space
    domain_global = m.domain
    
    x_min = domain_global.x['lbound']
    x_max = domain_global.x['rbound']
    
    z_min = domain_global.z['lbound']
    z_max = domain_global.z['rbound']

    mesh_global = m
    dx = mesh_global.x['delta']
    dz = mesh_global.z['delta']

    return [x_min, x_max, z_min, z_max, dx, dz, domain_global]

def get_trunc_geom_dict(m, rec_boundary_geom):
    trunc_geom_dict = dict()
    trunc_geom_dict.update(rec_boundary_geom)
    
    [x_min, x_max, z_min, z_max, dx, dz, domain_global] = global_geometry(m)
    trunc_geom_dict['rec_boundary_geom'] = rec_boundary_geom
    
    #nodes corresponding to boundary S_i. When rec_boundary flag is used the 5 elastic wavefield quantities
    #on S_i and on either side are saved for the local solver later on
    trunc_geom_dict['rec_x_l_n'] = int(np.round(rec_boundary_geom['rec_x_l']/dx))
    trunc_geom_dict['rec_x_r_n'] = int(np.round(rec_boundary_geom['rec_x_r']/dx))
    trunc_geom_dict['rec_z_t_n'] = int(np.round(rec_boundary_geom['rec_z_t']/dz))
    trunc_geom_dict['rec_z_b_n'] = int(np.round(rec_boundary_geom['rec_z_b']/dz))  
    trunc_geom_dict['rec_h_n']   = trunc_geom_dict['rec_x_r_n'] - trunc_geom_dict['rec_x_l_n'] + 1
    trunc_geom_dict['rec_v_n']   = trunc_geom_dict['rec_z_b_n'] - trunc_geom_dict['rec_z_t_n'] + 1 

    #Some convenience definitions for the integral
    trunc_geom_dict['int_h_n']   = trunc_geom_dict['rec_h_n']+2*5
    trunc_geom_dict['int_v_n']   = trunc_geom_dict['rec_v_n']+2*5 
    trunc_geom_dict['int_x_l']   = trunc_geom_dict['rec_x_l'] - 5 * dx
    trunc_geom_dict['int_x_r']   = trunc_geom_dict['rec_x_r'] + 5 * dx  
    trunc_geom_dict['int_z_t']   = trunc_geom_dict['rec_z_t'] - 5 * dz
    trunc_geom_dict['int_z_b']   = trunc_geom_dict['rec_z_b'] + 5 * dz

    #Somewhat unfortunate name the 4 quantities below. The 'n' is not a number, but a pix index. Changing name now might break a lot of things...
    trunc_geom_dict['int_x_l_n'] = trunc_geom_dict['rec_x_l_n'] - 5
    trunc_geom_dict['int_x_r_n'] = trunc_geom_dict['rec_x_r_n'] + 5    
    trunc_geom_dict['int_z_t_n'] = trunc_geom_dict['rec_z_t_n'] - 5
    trunc_geom_dict['int_z_b_n'] = trunc_geom_dict['rec_z_b_n'] + 5    
    
    trunc_geom_dict['nvals_each'] = 5*2*3*(trunc_geom_dict['rec_h_n'] +
                                           trunc_geom_dict['rec_v_n'])  #Size of boundary values for each timestep
    
    trunc_geom_dict['int_n']     = 2*5*(trunc_geom_dict['int_h_n']  + #Two horizontal sides plus corners
                                        trunc_geom_dict['rec_v_n'])   #plus the remaining vertical nodes on both sides
    
    trunc_geom_dict['dx'] = dx; trunc_geom_dict['dz'] = dz;
    
    #useful quantity when using CDA solver to generate green's functions in the old way 
    trunc_geom_dict['bdry_n']    = 2*4*((trunc_geom_dict['rec_h_n']  ) + #On top and bot we have 4 horizontal rows (boundary plus three interior)
                                        (trunc_geom_dict['rec_v_n']-8) ) #Then on both sides also 4 columns (boundary and three interior).

    #useful quantity when using CDA solver to generate green's functions in the new way.
    #the new way will store many pressure fields, but will not actually compute and store boundary fields
    #it will compute the boundary fields vector from the stored pressure green's functions before starting the local solve
    #In addition to the 5 integration layers we need 5 extra layers of pressure green's functions interior of the integration layers for this to work  
    trunc_geom_dict['bdry_extra_n'] = 2*5*((trunc_geom_dict['rec_h_n']  ) + #On top and bot we have 5 horizontal rows (boundary plus four interior)
                                           (trunc_geom_dict['rec_v_n']-10)) #Then on both sides also 5 columns (boundary and three interior).
    
    
    return trunc_geom_dict


def get_boundary_integral_positions(m, tr_dict):
    #We store the green's function traces one row at a time.
    #Start at the top left. 
    #We save 5 layers around S_i defined by tr_dict 

    def precompute_boundary_integral_positions(m, tr_dict): #local function
        [x_min, x_max, z_min, z_max, dx, dz, domain_global] = global_geometry(m)
    

        nvals = tr_dict['int_n']
    
        x_pos = np.zeros(nvals)
        z_pos = np.zeros(nvals)
         
        offset = 0                  #Tracks array offset
        cur_z  = tr_dict['int_z_t'] #We go down one row at a time, so can keep track of Z easily
        
        #Top 5 rows
        for i in xrange(5): #5 rows   
            x_pos[offset:offset+tr_dict['int_h_n']] = np.linspace(tr_dict['int_x_l'], tr_dict['int_x_r'], tr_dict['int_h_n'])
            z_pos[offset:offset+tr_dict['int_h_n']] = cur_z * np.ones(tr_dict['int_h_n'])
            offset += tr_dict['int_h_n']; 
            cur_z  += dz
            
        #The next tr_dict['rec_v_n'] rows have 5 cols on left and 5 on right
        for i in xrange(tr_dict['rec_v_n']):
            x_pos[offset    :offset+ 5] = np.linspace(tr_dict['int_x_l']     , tr_dict['int_x_l'] + 4*dx, 5)
            x_pos[offset + 5:offset+10] = np.linspace(tr_dict['rec_x_r'] + dx, tr_dict['rec_x_r'] + 5*dx, 5)
            z_pos[offset    :offset+10] = cur_z * np.ones(10)
            offset += 10
            cur_z  += dz
    
        #Bot 5 rows
        for i in xrange(5): #5 rows   
            x_pos[offset:offset+tr_dict['int_h_n']] = np.linspace(tr_dict['int_x_l'], tr_dict['int_x_r'], tr_dict['int_h_n'])
            z_pos[offset:offset+tr_dict['int_h_n']] = cur_z * np.ones(tr_dict['int_h_n'])
            offset += tr_dict['int_h_n']; 
            cur_z  += dz    
        
        return x_pos, z_pos

    try:
        x_pos = get_boundary_integral_positions.x_pos
        z_pos = get_boundary_integral_positions.z_pos
    except:
        [x_pos, z_pos] = precompute_boundary_integral_positions(m, tr_dict)
        get_boundary_integral_positions.x_pos = x_pos
        get_boundary_integral_positions.z_pos = z_pos
        
    return x_pos,z_pos

def get_additional_required_boundary_fields_positions(m, tr_dict):
    #This adds the boundary row and two interior rows.
    #Need these in addition to some of the positions of 'get_boundary_integral_positions' 
    #to compute the velocity on the boundary and one layer on each side through a derivative stencil. 

    #We cache the 3 layers positions     
    def precompute_boundary_fields_positions(m, tr_dict): #local function
        [x_min, x_max, z_min, z_max, dx, dz, domain_global] = global_geometry(m)
    
        nvals = tr_dict['bdry_n'] 
                
        x_pos = np.zeros(nvals)
        z_pos = np.zeros(nvals)
         
        offset = 0                  #Tracks array offset
        cur_z  = tr_dict['rec_z_t'] #We go down one row at a time, so can keep track of Z easily
        
        #Top 4 rows
        for i in xrange(4): #4 rows   
            x_pos[offset:offset+tr_dict['rec_h_n']] = np.linspace(tr_dict['rec_x_l'], tr_dict['rec_x_r'], tr_dict['rec_h_n'])
            z_pos[offset:offset+tr_dict['rec_h_n']] = cur_z * np.ones(tr_dict['rec_h_n'])
            offset += tr_dict['rec_h_n']; 
            cur_z  += dz
            
        #The next tr_dict['rec_v_n']-8 rows have 4 cols on left and 4 on right
        for i in xrange(tr_dict['rec_v_n']-8):
            x_pos[offset    :offset+4] = np.linspace(tr_dict['rec_x_l']       , tr_dict['rec_x_l'] + 3*dx, 4)
            x_pos[offset + 4:offset+8] = np.linspace(tr_dict['rec_x_r'] - 3*dx, tr_dict['rec_x_r']       , 4)
            z_pos[offset    :offset+8] = cur_z * np.ones(8)
            offset +=  8
            cur_z  += dz
    
        #Bot 4 rows
        for i in xrange(4): #4 rows   
            x_pos[offset:offset+tr_dict['rec_h_n']] = np.linspace(tr_dict['rec_x_l'], tr_dict['rec_x_r'], tr_dict['rec_h_n'])
            z_pos[offset:offset+tr_dict['rec_h_n']] = cur_z * np.ones(tr_dict['rec_h_n'])
            offset += tr_dict['rec_h_n']; 
            cur_z  += dz    
        
        return x_pos, z_pos

    try:
        x_pos = get_additional_required_boundary_fields_positions.x_pos
        z_pos = get_additional_required_boundary_fields_positions.z_pos
    except:
        [x_pos, z_pos] = precompute_boundary_fields_positions(m, tr_dict)
        get_additional_required_boundary_fields_positions.x_pos = x_pos
        get_additional_required_boundary_fields_positions.z_pos = z_pos
        
    return x_pos,z_pos
    
def get_additional_required_boundary_fields_positions_new(m, tr_dict):
    """ Similar to the function 'get_additional_required_boundary_fields_positions'
        But here we use slightly different positions since we want the velocity (vx, vz)
        at all the locations required to compute their spatial derivatives at the three boundary layers.
        Those derivatives will be needed to compute the strain rates in the three boundary layers at the 
        tau_xx, tau_zz, tau_xz locations. Those are used later for the local solver when we inject the velocity and then compute the corresponding stresses.
        
        In this function we add 5 layers to the 5 integral layer we already had.
        These new 5 layers are the layers directly to the inside of the boundary integral layers:
        Together we have 10 layers at which we record the pressure field. 
    """

    #Cache the results
    def precompute_extra_boundary_positions(m, tr_dict): #local function
        [x_min, x_max, z_min, z_max, dx, dz, domain_global] = global_geometry(m)
        
        #number of extra positions we need to compute all the required velocities
        nvals = tr_dict['bdry_extra_n']

        x_pos = np.zeros(nvals)
        z_pos = np.zeros(nvals)
        
        offset = 0                  #Tracks array offset
        cur_z  = tr_dict['rec_z_t'] #We go down one row at a time, so can keep track of Z easily
        
        #Top 5 rows
        for i in xrange(5): #5 rows   
            x_pos[offset:offset+tr_dict['rec_h_n']] = np.linspace(tr_dict['rec_x_l'], tr_dict['rec_x_r'], tr_dict['rec_h_n'])
            z_pos[offset:offset+tr_dict['rec_h_n']] = cur_z * np.ones(tr_dict['rec_h_n'])
            offset += tr_dict['rec_h_n']; 
            cur_z  += dz
            
        #The next tr_dict['rec_v_n']-10 rows have 5 cols on left and 5 on right
        for i in xrange(tr_dict['rec_v_n']-10):
            x_pos[offset    :offset+ 5] = np.linspace(tr_dict['rec_x_l']       , tr_dict['rec_x_l'] + 4*dx, 5)
            x_pos[offset + 5:offset+10] = np.linspace(tr_dict['rec_x_r'] - 4*dx, tr_dict['rec_x_r']       , 5)
            z_pos[offset    :offset+10] = cur_z * np.ones(10)
            offset += 10
            cur_z  += dz
    
        #Bot 5 rows
        for i in xrange(5): #5 rows   
            x_pos[offset:offset+tr_dict['rec_h_n']] = np.linspace(tr_dict['rec_x_l'], tr_dict['rec_x_r'], tr_dict['rec_h_n'])
            z_pos[offset:offset+tr_dict['rec_h_n']] = cur_z * np.ones(tr_dict['rec_h_n'])
            offset += tr_dict['rec_h_n']; 
            cur_z  += dz    
        
        return x_pos, z_pos

    try:
        x_pos = get_additional_required_boundary_fields_positions_new.x_pos
        z_pos = get_additional_required_boundary_fields_positions_new.z_pos
    except:
        [x_pos, z_pos] = precompute_extra_boundary_positions(m, tr_dict)
        get_additional_required_boundary_fields_positions_new.x_pos = x_pos
        get_additional_required_boundary_fields_positions_new.z_pos = z_pos
        
    return x_pos,z_pos
        
def temp_naive_convolve(trace_1, trace_2): #This is the naive way to convolve, this is what we want through the 'convolve_traces' method
    n1 = trace_1.size
    n2 = trace_2.size
    n_out = n1 + n2 - 1
    trace_out = np.zeros(n_out)
    for i in xrange(n2):
        trace_out[i:i+n1] += trace_2[i]*trace_1
        
    return trace_out

def convolve_traces(trace_1, trace_2): #Are there 2D version that can take the entire shotgather?
    #Example:
    #Trace 1: Green's function 
    #Trace 2: Source wavelet
    
    #Traces don't need the same length, but I assume the 'dt' is the same
    return fftconvolve(trace_1, trace_2)

def convolve_shotgather_with_wavelet(shotgather, trace):
    #poorly chosen function name. Actually convolving with a trace (numpy ndarray). 
    #changing now would break all old functions...
    #made a function for wavelet object as well
    nt, nr = shotgather.shape
    nt_wavelet = trace.size
    
    warnings.warn("If this loop is too slow, find 2D routine, use MPI or interface with a optimized C library ? ")
    outgather = np.zeros((nt + nt_wavelet - 1, nr))
    for i in xrange(nr):
        outgather[:,i] = convolve_traces(shotgather[:,i], trace)
        
    return outgather

def convolve_shotgather_with_wavelet_object(shotgather, wavelet, ts):
    if shotgather.shape[0] != ts.size:
        raise Exception("Should not really pose a problem if the wavelet we convolve with has different length, but for now I assume they are the same and that they have the same 'dt'")

    trace = wavelet._evaluate_time(ts)
    shotgather = convolve_shotgather_with_wavelet(shotgather, trace)
    shotgather = shotgather[:ts.size, :] #convolution results in shotgather with more timesteps than we started with. Take only the part which has the same size we started with
    return shotgather

def convolve_boundary_fields(greens_boundary_fields, trace, itimestep):
    nvals_boundary  = greens_boundary_fields.size #As defined in wrapping_functions.py
    
    if nvals_boundary % itimestep != 0:
        raise Exception("Passing wrong size information")
    
    number_traces = nvals_boundary / itimestep 
    
    greens_boundary_fields_2d = np.reshape(greens_boundary_fields, (itimestep, number_traces), 'C')
    boundary_fields_2d        = convolve_shotgather_with_wavelet(greens_boundary_fields_2d, trace)
    boundary_fields           = np.reshape(boundary_fields_2d[:itimestep,:] , (itimestep*number_traces), 'C')
    return boundary_fields

def convolve_trace_pairs(gather1, gather2):
    ntr1 = gather1.shape[1]
    ntr2 = gather2.shape[1]
    if ntr1 != ntr2:
        raise Exception("Not equal number of traces in both gathers") 

    for i in xrange(ntr1):
        trace1 = gather1[:,i]
        trace2 = gather2[:,i]
        outtrace = convolve_traces(trace1, trace2)
        if i == 0:
            nt_out = outtrace.size
            out    = np.zeros((nt_out, ntr1))
             
        out[:,i] = outtrace

    return out

def normal_derivative(trunc_geom_dict, field, deriv):
    n_h_int = trunc_geom_dict['int_h_n']
    n_v_int = trunc_geom_dict['int_v_n']
    
    nt, nrec = field.shape #nrec should be equal to 10*n_h_int + 10*(n_v_int - 10)
    if nrec != 10*n_h_int + 10*(n_v_int - 10): raise Exception("Something is not consistent")
    
    if deriv   == 'inward': #Boundary is at the outermost layer of the 5 integral recording layers
        #Computing normal derivative using a directional stencil 
        out = np.zeros((nt, 2*(n_h_int + n_v_int)))
        
        #4th order directional stencil
        c0  = -25./12
        c1  =     4.0
        c2  =    -3.0
        c3  =    4./3
        c4  =   -1./4
        
        off_out    = 0
        directions = ['top', 'right', 'bot', 'left']
        strides    = [n_h_int, -1, -n_h_int, 1] 
        for direction, stride in zip(directions,strides):
            boundary_ind = get_boundary_ind(trunc_geom_dict, direction, deriv)
            
            contr_0 = field[:,boundary_ind + 0*stride]
            contr_1 = field[:,boundary_ind + 1*stride]
            contr_2 = field[:,boundary_ind + 2*stride]
            contr_3 = field[:,boundary_ind + 3*stride]
            contr_4 = field[:,boundary_ind + 4*stride]
            
            increment = boundary_ind.size
            addition  = c0*contr_0 + c1*contr_1 + c2*contr_2 + c3*contr_3 + c4*contr_4
            
            #Corner node contributions count half as much
            #The surface area of the integral the corner node contributes to a side is half that of a normal point
            #A corner node will appear in two sides though 
            addition[:, 0]*=0.5
            addition[:,-1]*=0.5
            
            out[:, off_out:off_out+increment] = addition  
            off_out   += increment

    elif deriv == 'center': #Boundary is at the middle layer of the 5 integral recording layers
        #Computing normal derivative using a central stencil
        out = np.zeros((nt, 2*(n_h_int-4 + n_v_int-4))) #-4 because we skip 2 nodes on either side in central integral
        
        #4th order central stencil
        c1 =  2./3
        c2 = -1./12
        
        off_out    = 0
        directions = ['top', 'right', 'bot', 'left']
        strides    = [n_h_int, -1, -n_h_int, 1] 
        for direction, stride in zip(directions,strides):
            boundary_ind = get_boundary_ind(trunc_geom_dict, direction, deriv)
            
            contr_p1 = field[:,boundary_ind + 1*stride]
            contr_m1 = field[:,boundary_ind - 1*stride]
            contr_p2 = field[:,boundary_ind + 2*stride]
            contr_m2 = field[:,boundary_ind - 2*stride]
            
            increment = boundary_ind.size
            addition  = c1*(contr_p1 - contr_m1) + c2*(contr_p2 - contr_m2)
            
            #Corner node contributions count half as much
            #The surface area of the integral the corner node contributes to a side is half that of a normal point
            #A corner node will appear in two sides though 
            addition[:, 0]*=0.5
            addition[:,-1]*=0.5            
            
            out[:, off_out:off_out+increment] = addition
            off_out   += increment

    #Out is the derivative into the local domain. This is the one we want.
    #The volume integral was over volume V, which was the exterior. So the outward normal from V
    #is the inward normal in the local domain.
                                                     
    out = out / trunc_geom_dict['dx'] #derivative requires division by grid spacing. Should probably divide individual edges of square by both dx and dz
    return out    

def get_boundary_ind(trunc_geom_dict, direction, deriv):
    n_h_int = trunc_geom_dict['int_h_n']
    n_v_int = trunc_geom_dict['int_v_n']    
    nrec    = 10*n_h_int + 10*(n_v_int - 10)
    if deriv   == 'inward':
        if direction == 'top':
            ind_stencil = np.arange(0, n_h_int)
        elif direction == 'right':
            n = n_v_int
            ind_stencil_1 = n_h_int * np.arange(1,6) - 1 #1*n_h_int - 1, ..., 5*n_h_int - 1
            ind_stencil_2 = 5*n_h_int + 10*np.arange(n-10) + 9
            ind_stencil_3 = (nrec-1) - n_h_int*np.arange(4,-1,-1)
            ind_stencil   = np.concatenate([ind_stencil_1, ind_stencil_2, ind_stencil_3])
        elif direction == 'bot':
            ind_stencil   = (nrec-1) - np.arange(0,n_h_int)
        elif direction == 'left':
            n = n_v_int
            ind_stencil_1 = nrec - n_h_int*np.arange(1,6)
            ind_stencil_2 = nrec - 5*n_h_int - 10*(np.arange(n - 10)+1)
            ind_stencil_3 = n_h_int*np.arange(4,-1,-1)
            ind_stencil   = np.concatenate([ind_stencil_1, ind_stencil_2, ind_stencil_3])
        else:
            raise Exception("Invalid direction provided")            
    elif deriv == 'center':
        if direction == 'top':
            ind_stencil   = 2*n_h_int + np.arange(2,n_h_int-2)
        elif direction == 'right':
            n = n_v_int - 4
            ind_stencil_1 = np.array([2*n_h_int + n_h_int-3,
                                      3*n_h_int + n_h_int-3, 
                                      4*n_h_int + n_h_int-3])              #Three nodes that are above recording boundary 
            ind_stencil_2 = 5*n_h_int + 10*np.arange(n - 6) + 7            #The nodes a horizontal offset away from the recording boundary
            ind_stencil_3 = np.array([nrec-3-4*n_h_int, nrec-3-3*n_h_int, nrec-3-2*n_h_int]) #Three nodes that are below recording boundary
            ind_stencil   = np.concatenate([ind_stencil_1,ind_stencil_2,ind_stencil_3])              
        elif direction == 'bot':
            n             = n_h_int - 4
            ind_stencil   = nrec-3 - 2 * n_h_int - np.arange(n)
        elif direction == 'left':
            n             = n_v_int - 4
            ind_stencil_1 = np.array([nrec + 2 - 3 * n_h_int,nrec + 2 - 4 * n_h_int, nrec + 2 - 5 * n_h_int]) #Three nodes that are below recording boundary
            ind_stencil_2 = nrec-5*n_h_int - 8 - 10*np.arange(n - 6)                           #The nodes a horizontal offset away from the recording boundary
            ind_stencil_3 = np.array([4*n_h_int + 2, 3*n_h_int + 2, 2*n_h_int + 2])
            ind_stencil   = np.concatenate([ind_stencil_1, ind_stencil_2, ind_stencil_3])            
        else:
            raise Exception("Invalid direction provided")
    else:
        raise Exception("Invalid deriv option provided")

    return ind_stencil

def compute_acoustic_boundary_integral(trunc_geom_dict, greens_int_array, pressure_int_array, deriv = 'inward', greens_el = False, dt_green=0, dt_pressure=0, multiplier = None):
    #The green's array has spatial ordering prescribed by 'get_boundary_integral_positions' in precompute_greens.py
    #The pressure array is assumed to follow the same spatial pattern here
    #The trace length does not necessarily need to be the same for pressure vs green
    warnings.warn("When propagating to multiple receivers, should cache gp_dn and boundary pressure somehow. Only need to compute once of course...")
    
    warnings.warn('Right now assuming uniform spacing. Need to use both dx and dz for integral if variable spacing...')
    
    if greens_el: 
        #PySIT and EL solver match if you take antiderivative of source wavelet for EL.
        #The following full domain simulations match: 
        #1: convolve EL green's with antideriv of source wavelet W(t)
        #2: convolve PySIT green's with source wavelet W(t)

        #The 'pressure_int_arrays' of pysit and EL match already in the current function
        #because i compensated (integrated) the source wavelets. The green's functions differ.
        #Will compensate by taking integral of green's  
        
        nt_green, ntrace = greens_int_array.shape
        new_greens_int_array = np.zeros_like(greens_int_array)
        for i in xrange(ntrace):
            in_trace = greens_int_array[:,i]
            t_arr_green = dt_green*np.arange(nt_green)
            new_greens_int_array[:,i] = antideriv_wavelet(in_trace, t_arr_green, initial = 0)
    
        greens_int_array = new_greens_int_array
        print "Finished with integrating."
    
    #Potentially subsample pressure array to the same rate as the green's functions are subsampled
    if not float_eq(dt_pressure, dt_green) and dt_pressure != 0 and dt_green !=0: #If both set and not equal
        print "subsampling pressure"
        recording_period = int(np.round(dt_green/dt_pressure))
        if not float_eq(dt_green, recording_period*dt_pressure):
            raise Exception("Timesteps not integer multiple")
        
        nt_pressure = pressure_int_array.shape[0]
        
        subsample_index = np.arange(0, nt_pressure, recording_period)
        pressure_int_array = pressure_int_array[subsample_index,:]
    else:
        recording_period = 1 #No subsampling
    
    dg_dn = normal_derivative(trunc_geom_dict,   greens_int_array, deriv) 
    dp_dn = normal_derivative(trunc_geom_dict, pressure_int_array, deriv) 
    
    g_top   = greens_int_array[:, get_boundary_ind(trunc_geom_dict,   'top', deriv)]
    g_right = greens_int_array[:, get_boundary_ind(trunc_geom_dict, 'right', deriv)]
    g_bot   = greens_int_array[:, get_boundary_ind(trunc_geom_dict,   'bot', deriv)]
    g_left  = greens_int_array[:, get_boundary_ind(trunc_geom_dict,  'left', deriv)] 
    g       = np.concatenate([g_top, g_right, g_bot, g_left], axis=1)

    p_top   = pressure_int_array[:, get_boundary_ind(trunc_geom_dict,   'top', deriv)]
    p_right = pressure_int_array[:, get_boundary_ind(trunc_geom_dict, 'right', deriv)]
    p_bot   = pressure_int_array[:, get_boundary_ind(trunc_geom_dict,   'bot', deriv)]
    p_left  = pressure_int_array[:, get_boundary_ind(trunc_geom_dict,  'left', deriv)] 
    p       = np.concatenate([p_top, p_right, p_bot, p_left], axis=1)
    
    #CDA, for VDA you need to multiply by 1/rho at the boundary integrand location
    
    integrand  = convolve_trace_pairs(g, dp_dn) - convolve_trace_pairs(p, dg_dn)
    integrand *= recording_period #Without this the amplitude is lower when subsampling (fewer samples, larger dt) 
    
    if multiplier != None: #to allow reuse of this boundary integral routine when using variable density. This will just scale the integrand
        #multiplier is a 1D array of size equal to the number of columns in integrand
        integrand *= multiplier
    
    integral   = trunc_geom_dict['dx']*np.sum(integrand, axis = 1)
     
    
    #Scattered is a time trace
    scattered = integral 
    return scattered
    
def compute_acoustic_boundary_integral_cda_green_var_dens(trunc_geom_dict, greens_int_array, pressure_int_array, vp_2d, rho_loc_2d, vs_loc_2d,rec_x, rec_z, m, deriv = 'inward', greens_el = False, dt_green=0, dt_pressure=0, rho_real_2d = None, vs_real_2d = None):
    #Here I am trying to implement a boundary integral for the CDA green's function case when the local domain has variable density and variable Vs along the injection boundary
    #I want to minimize the incorrect amplitude and want to prevent scattering to happen at the injection boundary itself.
    #To correct the amplitude I am using scaling factors to make the amplitude of the CDA green's functions similar to what the EL code implements
    #This will involve ratios of density and ratios of vp**2-vs**2

    #The local solver only used part of the p velocity, density and shear velocity models (the part within the local solver)
    #In some cases we tapered the model to a boundary value in the density and shear velocity arrays passed to the local solver
    #When compensating amplitudes based on density and shear velocity at the receiver positions, we would incorrectly use the taper values if we used these arrays
    #therefore I allow the user to pass the real density and shear velocity arrays we want to model the wavefield on (we will extract the reciever density and shear velocities from these not tapered arrays) 
    #ONLY USEFUL WHEN THE LOCAL SOLVER USED A TAPERED INPUT ARRAY. OTHERWISE THE rho_loc_2d and rho_real_2d THE SAME (ALSO VS CASE) 
    

    #In this function I will do the integral
    n_h_int = trunc_geom_dict['int_h_n']
    n_v_int = trunc_geom_dict['int_v_n']    
    
    uniform_spacing = trunc_geom_dict['dx'] 
    
    #Get density along the boundary of the integral
    if deriv == 'inward':
        boundary_vp = np.zeros(2*(n_h_int + n_v_int))
        boundary_rho= np.zeros(2*(n_h_int + n_v_int))
        boundary_vs = np.zeros(2*(n_h_int + n_v_int))
        
        x_min_trunc = trunc_geom_dict['int_x_l']
        x_max_trunc = trunc_geom_dict['int_x_r']
        z_min_trunc = trunc_geom_dict['int_z_t']
        z_max_trunc = trunc_geom_dict['int_z_b']
        
    elif deriv == 'center':
        boundary_vp = np.zeros(2*(n_h_int-4 + n_v_int-4))
        boundary_rho= np.zeros(2*(n_h_int-4 + n_v_int-4))
        boundary_vs = np.zeros(2*(n_h_int-4 + n_v_int-4))
        
        x_min_trunc = trunc_geom_dict['int_x_l'] + 2*uniform_spacing
        x_max_trunc = trunc_geom_dict['int_x_r'] - 2*uniform_spacing
        z_min_trunc = trunc_geom_dict['int_z_t'] + 2*uniform_spacing
        z_max_trunc = trunc_geom_dict['int_z_b'] - 2*uniform_spacing
        
    else:
        raise Exception("Wrong integral type")
    
    nhor  = np.int(np.round((x_max_trunc - x_min_trunc)/uniform_spacing) + 1)
    nver  = np.int(np.round((z_max_trunc - z_min_trunc)/uniform_spacing) + 1)
    
    #top
    [x_min, _, z_min, _, _, _, _] = global_geometry(m)
    
    nx          = m.x.n
    nz          = m.z.n
    
    top_pos_x   = np.linspace(x_min_trunc, x_max_trunc, nhor)
    right_pos_z = np.linspace(z_min_trunc, z_max_trunc, nver) 
    bot_pos_x   = np.linspace(x_max_trunc, x_min_trunc, nhor)
    left_pos_z  = np.linspace(z_max_trunc, z_min_trunc, nver)
    
    top_pos_z   = z_min_trunc*np.ones_like(top_pos_x)
    right_pos_x = x_max_trunc*np.ones_like(right_pos_z)
    bot_pos_z   = z_max_trunc*np.ones_like(bot_pos_x)
    left_pos_x  = x_min_trunc*np.ones_like(left_pos_z)

    top_ind     = (np.round(nz*(top_pos_x   - x_min)/uniform_spacing + (top_pos_z   - z_min)/uniform_spacing)).astype('int32')
    right_ind   = (np.round(nz*(right_pos_x - x_min)/uniform_spacing + (right_pos_z - z_min)/uniform_spacing)).astype('int32')
    bot_ind     = (np.round(nz*(bot_pos_x   - x_min)/uniform_spacing + (bot_pos_z   - z_min)/uniform_spacing)).astype('int32')
    left_ind    = (np.round(nz*(left_pos_x  - x_min)/uniform_spacing + (left_pos_z  - z_min)/uniform_spacing)).astype('int32')
    
    rec_ind     = np.round(nz*(rec_x   - x_min)/uniform_spacing + (rec_z   - z_min)/uniform_spacing).astype('int32')
    
    if type(rho_real_2d) == type(None):
        rho_real_2d = rho_loc_2d

    if type(vs_real_2d) == type(None):
        vs_real_2d  = vs_loc_2d
    
    vp          = vp_2d.flatten(order='F')
    
    rho_loc     = rho_loc_2d.flatten(order='F')
    vs_loc      = vs_loc_2d.flatten(order='F')
    
    rho_real    = rho_real_2d.flatten(order='F')
    vs_real     = vs_real_2d.flatten(order='F')    
    
    #Use 'real' versions, this is the non-tapered version of the array
    vp_rec      = vp[rec_ind]
    rho_rec     = rho_real[rec_ind]
    vs_rec      = vs_real[rec_ind]
    
    offset      = 0
    
    #Use the 'loc' versions, the arrays where the local wavefield has been generated on. Could potentially be tapered array
    
    #top
    boundary_vp[ offset: offset+nhor] = vp[ top_ind];
    boundary_rho[offset: offset+nhor] = rho_loc[top_ind]; 
    boundary_vs[ offset: offset+nhor] = vs_loc[ top_ind];   offset += nhor
    
    #right
    boundary_vp[ offset: offset+nver] = vp[ right_ind];
    boundary_rho[offset: offset+nver] = rho_loc[right_ind]; 
    boundary_vs[ offset: offset+nver] = vs_loc[ right_ind]; offset += nver
    
    #bot
    boundary_vp[ offset: offset+nhor] = vp[ bot_ind];
    boundary_rho[offset: offset+nhor] = rho_loc[bot_ind];
    boundary_vs[ offset: offset+nhor] = vs_loc[ bot_ind];   offset += nhor
    
    #left
    boundary_vp[ offset: offset+nver] = vp[ left_ind];
    boundary_rho[offset: offset+nver] = rho_loc[left_ind];
    boundary_vs[ offset: offset+nver] = vs_loc[ left_ind]; offset += nver
    
    multiplier_rho = np.sqrt(rho_rec/boundary_rho)
    #multiplier_vs  = (vp_rec**2-vs_rec**2)/(boundary_vp**2-boundary_vs**2)
    multiplier_vs  = boundary_vp**2/(boundary_vp**2-boundary_vs**2)
    multiplier     = multiplier_rho * multiplier_vs
    return compute_acoustic_boundary_integral(trunc_geom_dict, greens_int_array, pressure_int_array, deriv = deriv, greens_el = greens_el, dt_green=dt_green, dt_pressure=dt_pressure, multiplier = multiplier)
    
    