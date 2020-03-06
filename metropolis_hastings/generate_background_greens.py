#generate the true baseline and monitor data using a full domain solver
# Std import block
import time
import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

from pysit import *
from pysit.gallery import marmousi
from pysit.gallery import marmousi2
from pysit_extensions.petsc4py_complex_convenience.petsc_solver_complex import petsc_wrapper
from pysit_extensions.truncated_domain_helmholtz_solver.truncate_domain import truncate_domain

import scipy.io as spio
import pickle


if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "4"
    #   Load or generate true wave speed we can add the compact operator
    #   flag it will speed up the resolution

    nz = 176
    nx = 651
        
    indict = spio.loadmat("indata/marm_true_models_dct.mat")
    marm_base_true_2d     = indict['marm_base_true_2d']
    uniform_spacing       = indict['uniform_spacing'][0][0]
                
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

    marm_base_true_1d = np.reshape(marm_base_true_2d, (nz*nx,1), 'F')
    
    freqs  = [3.0, 4.0, 5.0, 6.5, 8.0, 10.0]
    for freq in freqs:
        print "Starting for freq %f \n"%freq

        nshots = 64
        peakfreq = 6.0
        wavelet = RickerWavelet(peakfreq)
    
        src_depth = uniform_spacing
        rec_depth = uniform_spacing
        # Set up shots for baseline
        shots_base = equispaced_acquisition(m,
                                            wavelet,
                                            sources=nshots,
                                            source_depth=src_depth,
                                            source_kwargs={'approximation':'delta'},
                                            receivers='max',
                                            receiver_depth=rec_depth,
                                            receiver_kwargs={'approximation':'delta'},
                                            )
    
        print "CHANGE TO PETSC SOLVER"        
        spatial_accuracy_order = 2
        wrapper = petsc_wrapper(m, model_parameters={'C': marm_base_true_1d}, spatial_shifted_differences=True, spatial_accuracy_order=spatial_accuracy_order) #I think we don't need to pass the model parameters here. Nothing happens with it yet.
        solver = wrapper.solver
    
        base_model = solver.ModelParameters(m,{'C': marm_base_true_1d})
        solver.model_parameters = base_model #This builds the operators for the first time (but does not decompose them!)
    
        truncation_params_list = []
        
        print "In 'truncate_domain', the function validate_truncation_param should make sure the domains do not intersect."
        
        #####################################
        # DEFINE TRUNCATED REGIONS
        #####################################
                
        #REGION 1
        truncation_params = dict()
        truncation_params['xpos_top_left_corner'] = 5940.0
        truncation_params['zpos_top_left_corner'] = 1860.0
        truncation_params['width'] = 860.0
        truncation_params['height'] = 480.0    
        truncation_params_list.append(truncation_params)
             
        #####################################
        # FINISHED DEFINING TRUNCATED REGIONS
        #####################################    
    
        x_pos_sources = []
        z_pos_sources = []
        for shot in shots_base:
            (pos_x, pos_z) = shot.sources.position
            x_pos_sources.append(pos_x)
            z_pos_sources.append(pos_z)
    
        #all shots have same receiver coordinates
        x_pos_receivers = []
        z_pos_receivers = []
        for receiver in shot.receivers.receiver_list:
            (pos_x, pos_z) = receiver.position
            x_pos_receivers.append(pos_x)
            z_pos_receivers.append(pos_z)
    
        truncated_solver = truncate_domain(solver, shots_base, truncation_params_list, freq, wrapper_type = 'petsc')
        
        ########
        sparse_matrix_filename = "sparse_greens_matrix_" + str(freq) + "_hz.mat"
        mesh_filename = "mesh_" + str(freq) + "_hz.pickle"
            
        #additional data that is useful
        extra_data_filename = "extra_data_" + str(freq) + "_hz.mat"
        path = "outdata/truncated_solver_components/"
        
        #save the sparse greens matrix 
        spio.mmwrite(path+sparse_matrix_filename, truncated_solver.sparse_greens_matrix)
        
        #pickle the mesh object
        fileObject = open(path+mesh_filename,'wb')
        pickle.dump(truncated_solver.mesh_collection, fileObject, 2)
        fileObject.close()

        #extra data    
        extra_data_dict = dict()
        extra_data_dict['x_pos_sources'] = x_pos_sources
        extra_data_dict['z_pos_sources'] = z_pos_sources
        extra_data_dict['x_pos_receivers'] = x_pos_receivers
        extra_data_dict['z_pos_receivers'] = z_pos_receivers
        extra_data_dict['PMLwidth'] = PMLwidth
        extra_data_dict['PMLamp'] = PMLamp
        extra_data_dict['nx'] = nx
        extra_data_dict['nz'] = nz
        extra_data_dict['dx'] = uniform_spacing
        extra_data_dict['dz'] = uniform_spacing
        extra_data_dict['freqs'] = freqs
        spio.savemat(path+extra_data_filename, extra_data_dict)

        #Don't think I need to do this
        del truncated_solver
