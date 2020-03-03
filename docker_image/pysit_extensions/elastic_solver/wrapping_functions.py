from ctypes import *
from structs import *
from pysit.core import DomainBC
from scipy import integrate
import numpy as np
import time 
import warnings
import os

#PASSING STRUCT
#http://stackoverflow.com/questions/4351721/python-ctypes-passing-a-struct-to-a-function-as-a-pointer-to-get-back-data

#Passing string
#http://stackoverflow.com/questions/12500069/ctypes-how-to-pass-string-from-python-to-c-function-and-how-to-return-string

#PASSING NDARRAY

#POTENTIALLY PASSING AS 2D ARRAY
#http://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes

def correct_unphysical_vs(vp_2d, vs_2d):
    """
    When we construct the tapered velocity models, sometimes the extension of a high Vs can go into a low Vp zone.
    The resulting velocity model has elastic parameter 'lambda' smaller than 0 at certain pixels.
    This is something that must be prevented. 
    Here I will just ensure there are no negative elastic parameters by making sure vp**2 - 2vs**2 = lambda/rho > 0
    So if vp**2-2vs**2 <= 0, set vs**2 = 0.5 vp**2
    This will result in lambda = 0, which is not realistic either. But at least there is no negative elastic parameter   
    """

    vals = vp_2d**2 - 2*vs_2d**2
    vs_2d_new= np.copy(vs_2d)
    vs_2d_new[vals < 0] = 1./np.sqrt(2)*vp_2d[vals < 0]
    return vs_2d_new

def suggest_compatible_dt(min_spacing, max_c): 
    #Get a dt that will satisfy both the pysit and the EL requirement. 
    #Helpful to have same timestep when linking pysit and EL solvers for local solver.
    
    dt_pysit = 1./6*min_spacing/max_c
    dt_el    = 1./3*min_spacing/(np.sqrt(2)*max_c*(9./8 +1./24)) 
    
    return min(dt_pysit, dt_el)

def full_to_truncated(arr, x_left_subm, x_right_subm, z_top_subm, z_bot_subm, dx):
    nx_trunc = int((x_right_subm - x_left_subm)/dx) + 1
    nz_trunc = int((z_bot_subm - z_top_subm)/dx) + 1    
    ind_x_left_trunc = int(x_left_subm/dx)
    ind_z_top_trunc  = int(z_top_subm/dx)
    
    return np.copy(arr[ ind_z_top_trunc:ind_z_top_trunc+nz_trunc, ind_x_left_trunc:ind_x_left_trunc + nx_trunc])    

def get_ss_param(extradict, mesh):
   
    dx = mesh.x.delta

    n_pml_left = mesh.x.lbc.n
    n_pml_top  = mesh.z.lbc.n
    
    ix = int((extradict['ss_param']['left'] - extradict['gx0'])/dx)
    iz = int((extradict['ss_param']['top']  - extradict['gz0'])/dx)
    
    ss_m0 = ix + n_pml_left
    ss_k0 = iz + n_pml_top
    
    ss_M = int((extradict['ss_param']['right'] - extradict['ss_param']['left'])/dx) + 1
    ss_K = int((extradict['ss_param']['bot']   - extradict['ss_param']['top'])/dx) + 1
    
    ss_dm = 1
    ss_dk = 1
    
    ss = ssparams(extradict['ss_param']['ssvar'],
                  ss_m0,
                  ss_dm,
                  ss_M,
                  ss_k0,
                  ss_dk,
                  ss_K,                  
                  ss_M,
                  ss_K
                  )    

    return ss

def determine_array_size(extradict):
    #Using the same setup as in the C code
    n_recording_occasions = 0
    n_snapshot_occasions = 0
    for jj in xrange(extradict['itimestep']):
        time = jj*extradict['dt']
        
        if extradict['traces_output'] and extradict['traces_mem'] and (jj >= extradict['traces_iter_start']) and (jj <= extradict['traces_iter_end']) and (jj%extradict['traces_step_length'])==0:
            n_recording_occasions = n_recording_occasions + 1;        
        
        if extradict['snap_output'] and extradict['snaps_mem'] and (jj >= extradict['snap_iter_start']) and (jj <= extradict['snap_iter_end']) and (jj%extradict['snap_step_length'])==0:
            n_snapshot_occasions = n_snapshot_occasions + 1;

    return n_recording_occasions, n_snapshot_occasions

def reshape_shotgather(shotgather, n_vars, nt, nrcv):
    shotgathers = []
    for i in xrange(n_vars):
        n_per_var = nt*nrcv 
        offset = i*n_per_var
        shotgather_2d = np.reshape(shotgather[offset:offset + n_per_var], (nt,nrcv), 'F')
        shotgathers.append(shotgather_2d)
        
    return shotgathers

def reshape_wavefield(wavefield_from_elastic, n_vars, nt, nz, nx, shape = 'pysit'):
    tt = time.time()
    if shape == '1d':
        raise Exception("Untested. probably wrong order. Must use trick as in shape='pysit'?")
        ret = np.reshape(wavefield_from_elastic, (n_vars, nt, nz*nx))
    elif shape == 'pysit': #Return in same shape pysit stores wavefield. A wavefield is a list with nt 1d-ndarrays. Expensive?
        wavefields = []
        for i in xrange(n_vars):
            wavefield = []
            offset_var = i * (nt*nz*nx)
            for j in xrange(nt):
                offset_time = j * (nz*nx)
                offset = offset_var + offset_time
                snap_2d = np.reshape(wavefield_from_elastic[offset:(offset+nz*nx)], (nz,nx), 'C')
                snap_1d = np.reshape(snap_2d, (nz*nx,), 'F')
                wavefield.append(snap_1d) #for each timestep a slice of nz*nx
                
            wavefields.append(wavefield) #for each var
            
        ret = wavefields
    elif shape =='2d':
        ret = np.reshape(wavefield_from_elastic, (n_vars, nt, nz, nx))
    else:
        raise Exception("wrong shape provided")
    
    print "Reshaping took %e seconds\n"%(time.time()-tt)
    return ret 

def antideriv_wavelet(wavelet, t_arr, initial = 0):
    #Want to divide by 1/omega. For simplicity I'm using the integrate functionality.
    #May have to do something about 0 frequency component?
    return integrate.cumtrapz(wavelet, t_arr, initial=initial)

def get_source_wavelet(shot, t_arr, match_pysit, initial = 0):
    #In Elastic C code the force is added directly at a pixel.
    #In PySIT we apply the force in the right hand side. 
    #It takes dt seconds to propagate from the RHS of the equation to the wavefield.
    #Therefore, in the elastic C code there is a ~dt shift plus other differences caused by elastic vs acoustic implementation
    
    wavelet = shot.sources.w._evaluate_time(t_arr)
    if match_pysit: 
        #Want to get antiderivative of wavelet because the elastic solver effectively takes time derivative of wavelet. 
        wavelet = antideriv_wavelet(wavelet, t_arr, initial)
        warnings.warn("Should probably make sure it returns to 0. Right now it keeps a minor residual value sometimes")
        
    return wavelet

def elastic_solve(shots, mesh, vp, rho, vs, optionsdict, recorded_boundary_fields = None, **kwargs):
     
    ######################SETTING DEFAULT VALUES IN THE DICTIONARY WHICH WILL BE UPDATED WITH OPTIONSDICT AND KWARGS ################################
    
    maxint = np.iinfo(c_int).max
    extra_options = dict()
    
    #traces_mem and snaps_mem determine whether the shotgather and wavefields should be stored in memory.
    extra_options['traces_mem']   = False 
    extra_options['snaps_mem']    = False
    extra_options['rec_boundary'] = False
    extra_options['local_solve']  = False
    
    #If traces or snaps is False, then disk will be used. The output directory is given as default below, but can be updated by kwargs of course    
    extra_options['traces_output_dir'] = "./"
    extra_options['snaps_output_dir'] = "./"
    
    #if rec_boundary or local_solve are True, then we want know the geometry. 
    #Here I will just declare default values which should be overwritten
    extra_options['rec_boundary_geom'] = { 'rec_x_l': 0.0,  'rec_x_r': 0.0,  'rec_z_t': 0.0, 'rec_z_b': 0.0}
    #The submesh boundary geometry will be obtained from gx0, gz0, and M, K, nabs etc. 
    #Same as for a normal solver setup. 'rec_boundary_geom' needs to be provided as well!
    
    extra_options['screen_output'] = 1; extra_options['screen_iter_start'] = 0; extra_options['screen_iter_end'] = maxint; extra_options['screen_step_length'] = 100
    extra_options['snap_output'] = 0; extra_options['snap_iter_start'] = 0; extra_options['snap_iter_end'] = maxint ; extra_options['snap_step_length'] = 10
    extra_options['traces_output'] = 1; extra_options['traces_iter_start'] = 0; extra_options['traces_iter_end'] = maxint ; extra_options['traces_step_length'] = 1 #traces = hist in c_code
    extra_options['gx0'] = mesh.domain.x.lbound; extra_options['gz0'] = mesh.domain.z.lbound
    extra_options['MM'] = mesh.x.n; extra_options['KK'] = mesh.z.n;
    
    extra_options['ss_param'] = {'left':mesh.domain.x.lbound, 'right': mesh.domain.x.rbound, 'top': mesh.domain.z.lbound, 'bot': mesh.domain.z.rbound}
    
    extra_options['iwavelet'] = 0 #Ricker
    extra_options['match_pysit'] = True #Used when iwavelet = 3, reading source wavelet from shot. If True, integrate wavelet
    extra_options['amp0'] = 1.0
    extra_options['freq0'] = 1.0
    extra_options['itimestep'] = 1 #only do 1 timestep by default
    extra_options['snap_wavefield_return_shape'] = '1d'
    
    
    #%================= output options ========================%
    #% var[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  --> 
    #%  [vx, vz, pr, div, curl, txx, tzz, txz, vpx, vpz, vsx, vsz]       
    
    TwelveIntegers = c_int * 12
    
    extra_options['histvar']             = TwelveIntegers(0,0,1,0,0,0,0,0,0,0,0,0) #pr
    extra_options['ss_param']['ssvar']   = TwelveIntegers(0,0,1,0,0,0,0,0,0,0,0,0) #pr

    extra_options.update(optionsdict)
    extra_options.update(kwargs) #update the default options
    
    retval = dict()
    saving_wavefield_to_mem = extra_options['snap_output'] and extra_options['snaps_mem']
    saving_shotgather_to_mem = extra_options['traces_output'] and extra_options['traces_mem']
    saving_boundary_fields = extra_options['rec_boundary']
    
    ######################PREPARE RETURN VALS DEPENDING ON INPUT ################################
    if saving_wavefield_to_mem:
        retval['wavefields'] = []
        retval['wavefields_times'] = []

    if saving_shotgather_to_mem:
        retval['shotgathers'] = []
        retval['shotgathers_times'] = []
    
    if saving_boundary_fields:
        retval['boundary_fields'] = []
        retval['boundary_fields_times'] = []
    
    ######################START LOOP OVER SHOTS###################################################
    
    
    
    for i in xrange(len(shots)): #Right now the C code only supports one shot at a time
        print "Shot number %i: "%i
        shot = shots[i]
        
        #pre-allocate src and rcv position arrays here. Cannot create within create_info_struct function
        #If you do and create pointer, the arrays pointed to go out of scope in python and are collected by garbage collector sometimes.
        #Result is garbage for positions.        
        
        nsrc = 1 #For now we work with 1 shot per istruct. C code is also assuming 1 shot at a time. Slightly inefficient since for each shot the solver is reinitialized. But insignificant
        srcx = np.zeros(nsrc)
        srcz = np.zeros(nsrc)
        nrcv = shot.receivers.receiver_count #later confirm all shots same receivers. May later adapt this. Right now no problem since we make info struct for one shot at a time
        rcvx = np.zeros(nrcv)
        rcvz = np.zeros(nrcv)        
        
        istruct = create_info_struct(shot, mesh, extra_options, srcx, srcz, rcvx, rcvz)

        n_vars_snap = np.sum(extra_options['ss_param']['ssvar'])
        n_vars_shotgather = np.sum(extra_options['histvar'])

        n_recording_occasions, n_snapshot_occasions = determine_array_size(extra_options) #may contain useless values if not recording, but then these values will not be used anyway so OK
        
        ################## PRE ALLOCATE SOME ARRAYS DEPENDING ON WHAT WE WILL RECORD ##############################
        if saving_wavefield_to_mem: 
            wavefield = np.zeros(n_vars_snap*n_snapshot_occasions*istruct.ss.M*istruct.ss.K)
            wavefield_times = np.zeros(n_snapshot_occasions)
        else:
            wavefield = np.zeros(0)
            wavefield_times = np.zeros(0)
            
        if saving_shotgather_to_mem:
            shotgather = np.zeros(n_vars_shotgather*nrcv*n_recording_occasions)
            shotgather_times = np.zeros(n_recording_occasions)
        else:
            shotgather = np.zeros(0)
            shotgather_times = np.zeros(0)
            
        if saving_boundary_fields:
            n_rows = int(np.round((extra_options['rec_boundary_geom']['rec_z_b'] - extra_options['rec_boundary_geom']['rec_z_t'])/mesh.z.delta)) + 1
            n_cols = int(np.round((extra_options['rec_boundary_geom']['rec_x_r'] - extra_options['rec_boundary_geom']['rec_x_l'])/mesh.x.delta)) + 1
            
            #at each cell we save 5 quantities. tauxx, tauzz, tauxz, vx, vz
            row_increment = 5*n_cols 
            col_increment = 5*n_rows
            
            #We have two horizontal and two vertical sides of square recording surface.
            #On each side we save three rows or columns of cells 
            #We do this for each timestep
            nvals_boundary = (2*3*row_increment + 2*3*col_increment)*extra_options['itimestep']
            boundary_fields = np.zeros(nvals_boundary)
            boundary_fields_times = np.zeros(extra_options['itimestep'])
            print "nvals_boundary " + str(nvals_boundary) + ":\n"
        else:
            boundary_fields = np.zeros(0)
            boundary_fields_times = np.zeros(0)
            
        if extra_options['local_solve']:
            if type(recorded_boundary_fields) == type(None):
                raise Exception("Forgot to pass along the recorded boundary field")
            if len(recorded_boundary_fields) != len(shots):
                raise Exception("Boundary fields are incorrectly shaped")
            
            boundary_fields = recorded_boundary_fields[i] #Will pass the recording along
            
        if extra_options['iwavelet'] == 3: #If we pass wavelet from PySIT directly to C code
            #For now not so flexible with t_min and t_max. Assuming we start at t = 0 and tmax is related to the number of timesteps
            t_arr = extra_options['dt'] * np.arange(extra_options['itimestep']-1)
            source_arr = get_source_wavelet(shot, t_arr, extra_options['match_pysit'])
        else:
            source_arr = np.zeros(extra_options['itimestep']) #Will not be used, dummy
            
        ###########################################DO SOLVE ####################################################
        print "Entering solve\n"
        initiate_solve(istruct, vp, rho, vs, wavefield, wavefield_times, shotgather, shotgather_times, boundary_fields, boundary_fields_times, source_arr)
        ################## POTENTIALLY RESHAPING AND STORING OF THE OUTPUT RESULTS THAT WERE PASSED BACK TO PYSIT ##############################
        
        if saving_wavefield_to_mem:
            wavefield = reshape_wavefield(wavefield, n_vars_snap, n_snapshot_occasions, istruct.ss.K, istruct.ss.M, shape = extra_options['snap_wavefield_return_shape'])
            retval['wavefields'].append(wavefield[0])
            retval['wavefields_times'].append(wavefield_times)
            print "change wavefields storage. Seperate different variables, and maybe make convenience functions for reshaping?"
            
        if saving_shotgather_to_mem:
            shotgather = reshape_shotgather(shotgather, n_vars_shotgather, n_recording_occasions, nrcv)
            retval['shotgathers'].append(shotgather[0])
            retval['shotgathers_times'].append(shotgather_times)

        if saving_boundary_fields:
            retval['boundary_fields'].append(boundary_fields)
            retval['boundary_fields_times'].append(boundary_fields_times)
            
        print "Finished with shot\n"
    return retval

def create_info_struct(shot, mesh, extradict, srcx, srcz, rcvx, rcvz): #extract most info from shots and mesh, but put the rest in 'extradict'
    warnings.warn("In the future I should cache certain checks so they are not performed for each iteration?")
    warnings.warn("ALSO: I AM ASSUMING I COULD PASS MULTIPLE SHOTS TO THIS FUNCTION RIGHT NOW. BUT THE ELASTIC SOLVER ONLY ALLOWS ONE AT A TIME...")
    warnings.warn("As quick fix I am calling this function only for one shot at a time.")
    
    d = mesh.domain
    nabs = mesh.x.lbc.n
    if nabs != mesh.x.rbc.n or nabs != mesh.z.lbc.n or nabs != mesh.z.rbc.n:
        raise Exception("Right now I expect PML on all sides. Will have to change wrapper to allow free surface")

    if type(d.x.lbc) != Elastic_Code_PML or type(d.x.rbc) != Elastic_Code_PML or type(d.z.lbc) != Elastic_Code_PML or type(d.z.rbc) != Elastic_Code_PML:
        raise Exception("Not using elastic PML objects")
    else:
        okay = 1
        PPW0 = d.x.lbc.PPW0
        if PPW0 != d.x.rbc.PPW0 or PPW0 != d.z.lbc.PPW0 or PPW0 != d.z.rbc.PPW0:
            okay = 0
        
        d0factor = d.x.lbc.d0factor
        if d0factor != d.x.rbc.d0factor or d0factor != d.z.lbc.d0factor or d0factor != d.z.rbc.d0factor:
            okay = 0
        
        p_power =  d.x.lbc.p_power
        if p_power != d.x.rbc.p_power or p_power != d.z.lbc.p_power or p_power != d.z.rbc.p_power:
            okay = 0
                     
        if not okay:
            raise Exception("Something wrong with PML")
        
    if mesh.x.delta == mesh.z.delta:
        uniform_spacing = mesh.x.delta
    else:
        raise Exception("Not uniformly spaced")  
       
       

    if shot.sources.approximation != 'delta':
        raise Exception('Elastic code only delta right now')
        
    srcx[0], srcz[0] = shot.sources.position
    
    nrcv = len(rcvx)
    if nrcv != len(rcvz): raise Exception("Mismatch")
    
    if shot.receivers.receiver_count != nrcv:
        raise Exception("Not fixed spread for sure")
        
    for j in xrange(nrcv):
        rcvx[j], rcvz[j] = shot.receivers.receiver_list[j].position 
                
    srcx = np.ascontiguousarray(srcx)
    srcz = np.ascontiguousarray(srcz)
    rcvx = np.ascontiguousarray(rcvx)
    rcvz = np.ascontiguousarray(rcvz)

    srcx_ptr = srcx.ctypes.data_as(POINTER(c_double))
    srcz_ptr = srcz.ctypes.data_as(POINTER(c_double))
    rcvx_ptr = rcvx.ctypes.data_as(POINTER(c_double))
    rcvz_ptr = rcvz.ctypes.data_as(POINTER(c_double))
        
    nhist = nrcv
         
    saving_wavefield_to_mem = extradict['snap_output'] and extradict['snaps_mem']
    saving_shotgather_to_mem = extradict['traces_output'] and extradict['traces_mem']
    
    screen = outparams(extradict['screen_output'],extradict['screen_iter_start'],extradict['screen_iter_end'],extradict['screen_step_length'])
    snap = outparams(extradict['snap_output'],extradict['snap_iter_start'],extradict['snap_iter_end'],extradict['snap_step_length'])
    hist = outparams(extradict['traces_output'],extradict['traces_iter_start'],extradict['traces_iter_end'],extradict['traces_step_length'])
    
    ss = get_ss_param(extradict, mesh)
    
    bparams = boundary_params(extradict['rec_boundary'],
                              extradict['local_solve'],
                              extradict['rec_boundary_geom']['rec_x_l'],
                              extradict['rec_boundary_geom']['rec_x_r'],
                              extradict['rec_boundary_geom']['rec_z_t'],
                              extradict['rec_boundary_geom']['rec_z_b']
                              )
    
    istruct = info_struct(extradict['MM'], 
                          extradict['KK'], 
                          nabs, #nabs
                          c_int(0), #ifscb, RIGHT NOW ONLY SUPPORTING PML IN WRAPPER
                          d0factor, 
                          PPW0, 
                          p_power,
                          uniform_spacing, #dx
                          extradict['dt'],
                          extradict['itimestep'],
                          extradict['gx0'],
                          extradict['gz0'],
                          extradict['iwavelet'], #iwavelet by default 0 for now (Ricker). Later change so that you will read source wavelet from shot object
                          extradict['amp0'],     
                          extradict['freq0'],    #will only be used when iwavelet = 0 (Ricker)
                          0, #isourcecomp. Right now just use 0, point explosion. But 1 (moment tensor) or 2 (vector force) may be good as well
                          1, #nsrc (currently hardcoded 1)
                          srcx_ptr, 
                          srcz_ptr,
                          screen,
                          snap,
                          hist,
                          extradict['histvar'],
                          bparams,
                          ss,
                          nhist,
                          rcvx_ptr,
                          rcvz_ptr,
                          saving_shotgather_to_mem, 
                          saving_wavefield_to_mem,
                          extradict['traces_output_dir'],
                          extradict['snaps_output_dir']
                          )
    return istruct
    
def initiate_solve(istruct, vp, rho, vs, wavefield, wavefield_times, shotgather, shotgather_times, boundary_fields, boundary_fields_times, source_arr):
    def load_func(): #Will be called only once. Load the function from the so
        lib = cdll.LoadLibrary(os.path.dirname(__file__) +'/elastic_c_code/_elastic_solver.so') #os.path.dirname(__file__) gets path of this wrapper function
        func = lib.solve
        func.restype  = c_int
        func.argtypes = [POINTER(info_struct), 
                         np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"), #vp
                         np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"), #rho
                         np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"), #vs
                         np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"), #wavefield
                         np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"), #wavefield_times
                         np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"), #shotgather
                         np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"), #shotgather_times
                         np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"), #boundary_fields
                         np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"), #boundary_fields_times
                         np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS")] #source_arr
        return func
    
    try: #See if we already loaded the shared lib and got func setup
        func = initiate_solve.func
    except: #If not, load and setup
        func = load_func()
        initiate_solve.func = func
    
    try:
        func(istruct, np.ascontiguousarray(vp), np.ascontiguousarray(rho), np.ascontiguousarray(vs), np.ascontiguousarray(wavefield), np.ascontiguousarray(wavefield_times), np.ascontiguousarray(shotgather), np.ascontiguousarray(shotgather_times), np.ascontiguousarray(boundary_fields), np.ascontiguousarray(boundary_fields_times), np.ascontiguousarray(source_arr))
    except:
        raise Exception('something went wrong')

def suggest_d0_factor(p_power, uniform_spacing, pml_thickness, cp):
    #From paper: 'Unsplit complex frequency-shifted PML implementation using auxiliary differential equations for seismic wave modeling'
    #Did not really read paper. Just read a few recommendations 
    #cp is some scalar vp measure around the boundary
    
    N = pml_thickness/uniform_spacing
    eq_39_paper     = -(np.log10(N)-1)/np.log10(2) - 3
    d0factor        = -(p_power+1)*cp/(2*pml_thickness)*eq_39_paper  
    d0factor       *= 3 #paper says it is good to multiply by 2 or 3 to get better results for grazing waves    
    return d0factor

class Elastic_Code_PML(DomainBC):
    """ A glorified storage unit

    """

    type = 'pml'

    def __init__(self, length, d0factor, PPW0, p_power):
        # Length is currently in physical units.
        self.length = length
        self.d0factor = d0factor
        self.PPW0 = PPW0
        self.p_power = p_power
        self.boundary_type = type

    def evaluate(self, nval, dummyval): #dummy function
        return np.zeros(nval) 
    
