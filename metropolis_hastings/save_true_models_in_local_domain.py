#import all libraries and modules
import time
import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import pickle 
import scipy.linalg
import math

from scipy.stats import norm
from pysit import *
from pysit.gallery import marmousi
from pysit.gallery import marmousi2
from collections import deque
from common_funcs.load_data import load_data, load_inverted_baseline_data
from pysit_extensions.petsc4py_complex_convenience.petsc_solver_truncated_complex import petsc_wrapper_truncated
from pysit_extensions.truncated_domain_helmholtz_solver.lil_pos_ut import lil_pos_ut, read_lil_pos_ut_from_file
from pysit_extensions.truncated_domain_helmholtz_solver.truncated_helper_tools import *
from pysit_extensions.truncated_domain_helmholtz_solver.truncate_domain import truncate_domain, truncate_array, truncated_back_to_full #horrendously long


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "4"
    #   Load or generate true wave speed we can add the compact operator
    #   flag it will speed up the resolution

    nz = 176
    nx = 651


    indict = spio.loadmat("indata/marm_true_models_dct.mat")
    uniform_spacing       = indict['uniform_spacing'][0][0]
    marm_base_true_2d     = indict['marm_base_true_2d']
    marm_moni_true_2d     = indict['marm_moni_true_2d']
    
    x_min   = 0.0
    x_max   = (nx-1)*uniform_spacing
    z_min   = 0.0
    z_max   = (nz-1)*uniform_spacing

    PMLwidth = 40 * uniform_spacing #assumes that dx = dz
    PMLamp = 100.0

    x_lbc = PML(PMLwidth, PMLamp)
    x_rbc = PML(PMLwidth, PMLamp)
    z_lbc = PML(PMLwidth, PMLamp)
    z_rbc = PML(PMLwidth, PMLamp)

    x_config = (x_min, x_max, x_lbc, x_rbc)
    z_config = (z_min, z_max, z_lbc, z_rbc)

    d = RectangularDomain(x_config, z_config)
    m = CartesianMesh(d, nx, nz)

    marm_base_1d = np.reshape(marm_base_true_2d, (nz*nx,1), 'F')
    marm_moni_1d = np.reshape(marm_moni_true_2d, (nz*nx,1), 'F')
    
    freq = 8.0 
    freqs = [freq]
     
    #load green's functions for this frequency
    print "Loading green's functions from file"
    path_local = "outdata/truncated_solver_components/"
    sparse_matrix_filename = path_local + "sparse_greens_matrix_" + str(freq) + "_hz.mat.mtx"
    sparse_greens_matrix = read_lil_pos_ut_from_file(m, sparse_matrix_filename) #constructs the sparse green's matrix from the csr mat stored in the file given by greens_mat_string
    
    #load truncated mesh
    truncated_mesh_collection_string = path_local + "mesh_" + str(freq) +  "_hz.pickle"
    print "loading truncated_mesh from file: " + truncated_mesh_collection_string + " \n"
    fileObject = open(truncated_mesh_collection_string,'r')
    truncated_mesh_collection = pickle.load(fileObject)
    mesh_collection = truncated_mesh_collection
    domain_collection = truncated_mesh_collection.domain_collection
    fileObject.close()
        
	#DEFINE LOCAL SOLVER
    wrapper = petsc_wrapper_truncated(truncated_mesh_collection, sparse_greens_matrix, petsc='mumps')
    truncated_solver = wrapper.solver

    #Get the baseline model in the local domain
    truncated_baseline_velocity_1d = truncate_array(marm_base_1d, m, domain_collection)
    truncated_baseline_velocity_2d = truncate_array(np.reshape(marm_base_1d, (nz, nx), 'F'), m, domain_collection)
    
    #Get the monitor model in the local domain
    truncated_monitor_velocity_1d = truncate_array(marm_moni_1d, m, domain_collection)
    truncated_monitor_velocity_2d = truncate_array(np.reshape(marm_moni_1d, (nz, nx), 'F'), m, domain_collection)
    
    #Get the time lapse model in the local domain
    truncated_timelapse_velocity_2d = truncated_monitor_velocity_2d - truncated_baseline_velocity_2d 
    
    print np.shape(truncated_baseline_velocity_2d)
    print np.shape(marm_base_true_2d)
    
    results = dict()
    results['truncated_baseline_velocity_2d']  = truncated_baseline_velocity_2d
    results['truncated_monitor_velocity_2d']   = truncated_monitor_velocity_2d
    results['truncated_timelapse_velocity_2d'] = truncated_timelapse_velocity_2d
    
    spio.savemat('outdata/dct_components/true_models_in_local_domain.mat', results)




    
 
    
