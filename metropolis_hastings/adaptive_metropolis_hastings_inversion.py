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
    
    ###############################################################################################################################################################
    ####################                     LOAD COVARIANCE, BACKGROUND MODEL, NOISY TIME-LAPSE DATA                ##############################################
    ###############################################################################################################################################################
    
    indict            = spio.loadmat("indata/marm_true_models_dct.mat")
    uniform_spacing   = indict['uniform_spacing'][0][0]
    marm_base_true_2d = indict['marm_base_true_2d']
    
       
    indict2           = spio.loadmat("outdata/measuredField_at_shot_32_with_freq_8/Covariance_matrix.mat")
    covMatrix         = indict2['covMatrix']
    covMatrix         = np.asarray(covMatrix)
    
    indict3           = spio.loadmat("outdata/measuredField_at_shot_32_with_freq_8/Covariance_inverse.mat")
    covInv            = indict3['covInv']
    covInv            = np.asarray(covInv)
    
    indict4           = spio.loadmat("outdata/measuredField_at_shot_32_with_freq_8/Noisy_measured_field_2D.mat")
    measuredField     = indict4['measuredField']
    measuredField     = np.asarray(measuredField)
    
    ###############################################################################################################################################################
    ###############################################################################################################################################################
    ###############################################################################################################################################################
    
    indict5           = spio.loadmat("outdata/dct_components/phi_matrix.mat")
    phi               = indict5['phi']
    phi               = np.asarray(phi)
    
    indict6           = spio.loadmat("outdata/dct_components/taper_2d.mat")
    taper_2d          = indict6['taper_2d']
    taper_2d          = np.asarray(taper_2d)
    
    ###############################################################################################################################################################
    ###############################################################################################################################################################
    ###############################################################################################################################################################
    
    marm_base_1d = np.reshape(marm_base_true_2d, (nz*nx,1), 'F')
    
    dof = 20 # number of degrees of freedom
    
    #define the dimensions of the local domain
    nz_local = 25
    nx_local = 44
        
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

    nshots = 64 #gives source spacing coinciding with grid points
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
                                            

    shots_moni         = copy.deepcopy(shots_base)
    shots_predicted    = copy.deepcopy(shots_base)
      
    #For simplicity we do it only for one frequency now; Will be easy to change to more later on
    freq = 8.0 
    freqs = [freq]

    #Now populate the shots; ill still use the 64 shots 
    
    #first load true base and monitor data
    path_data = 'indata/'
    load_data(freqs, path_data, shots_base, shots_moni)
      
    #Now choose only one shot calculation for simplicity
    shot_nr           = 32
    shot_moni         = shots_moni[shot_nr]
    shot_base         = shots_base[shot_nr]
    shot_predicted    = shots_predicted[shot_nr]
    
    num_receivers     = len(shot_moni.receivers.receiver_list)
    
    # parameters for the Metropolis Hastings inversions 
    sigma   = 0.0056      # step size, aka initial C_0
    maxIter = 100000      # total number of iterations
    t0      = 1000.0      # N_c --> number of iterations with a fixed step size
    Dx      = 20.0        # 20 DOF --> Dx = 20.0
    Sd      = (2.4**2)/Dx # value from Haario (2001)
    eps     = 1.0e-10     # value from Haario (2001)
    
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
    
    start_time = time.time()
    
    #Information that needs to be stored
    x                  = []
    accept_history     = []
    likelihood_history = []
    alpha_history      = []
       
    # DEFINE THE LIMITS AND RANGES IN MODEL SPACE FOR ALL ALPHAS
    min_a = -20000.0
    max_a =  20000.0
    a_Limits = [min_a, max_a] 
    a_range = max_a - min_a
    
    #Initialize the x and normalize it by x init -> x[0] = (x[0]-np.mean(a1_Limits))/a1_range*2.0
    x_init = np.ones(dof)
    x_init = np.reshape(x_init, [dof,1], 'F')
    x_init = np.subtract(x_init, np.mean(a_Limits))
    x_init = np.divide(x_init, a_range*2.0)
    x.append(x_init)
    
    #initialize the alpha vectors
    alphavec = np.ones(dof)
    alphavec = np.reshape(alphavec, [dof,1], 'F')
    alpha_history.append(alphavec)
     
    #initialize the new monitor in full domain
    marm_perturb_2d = np.zeros((nz, nx))
    marm_perturb_1d = np.reshape(marm_perturb_2d, (nz*nx,1), 'F')
    
    #Initialize the delta_m in local domain
    deltam_2d = np.zeros((nz_local,nx_local))
    deltam_1d = np.reshape(deltam_2d, (nz_local*nx_local,1), 'F')
    
    #Get the delta by phi*alpha
    deltam_1d = np.dot(phi,alphavec)
    deltam_2d = np.reshape(deltam_1d , (nz_local, nx_local), 'F')
    
    #Apply the taper
    deltam_2d_tapered = np.zeros((nz_local,nx_local))
    deltam_2d_tapered = np.multiply(deltam_2d, taper_2d)
    deltam_1d_tapered = np.reshape(deltam_2d_tapered, (nz_local*nx_local,1), 'F')
    
    #convert to the full domain
    deltam_full_2d   = np.zeros((nz, nx))
    deltam_full_2d[92:117, 296:340] = deltam_2d_tapered
      
    #Create the the monitor by adding the deltam_full to the full domain baseline model    
    deltam_full_1d  = np.reshape(deltam_full_2d, (nz*nx,1), 'F')
    marm_perturb_1d = marm_base_1d + deltam_full_1d
    
    #Get the model in the local domain
    truncated_velocity_perturbed_list    = truncate_array(marm_perturb_1d, m, domain_collection)
    truncated_velocity_perturbed_2d_list = truncate_array(np.reshape(marm_perturb_1d, (nz, nx), 'F'), m, domain_collection)
    
    # Calculate the predicted data for the new monitor model
    truncated_model_perturbed         = truncated_solver.ModelParameters(truncated_solver.mesh_collection, {'C':truncated_velocity_perturbed_list})
    truncated_solver.model_parameters = truncated_model_perturbed
    
    total_field_at_receivers = truncated_solver.total_field_at_receivers_from_shot(shot_predicted, nu=freq)
    
    receiver_dict = dict()
    num_receivers = len(shot_predicted.receivers.receiver_list)
    shot_predicted.receivers.data_dft[freq] = np.zeros((1, num_receivers), dtype='complex128')
    
    for n in xrange(num_receivers):
    	receiver = shot_predicted.receivers.receiver_list[n]
    	pos = receiver.position
    	receiver_dict[pos] = n
    	
    for pos in total_field_at_receivers:
    	index = receiver_dict[pos]
    	shot_predicted.receivers.data_dft[freq][0][index] = total_field_at_receivers[pos]
    
    # get the residual delta_d	     
    residuals_predicted = np.zeros(num_receivers,dtype='complex128')
    estimatedField      = np.zeros((num_receivers*2,1))
    dMisfit				= np.zeros((num_receivers*2,1))
    
    for rec2 in xrange(num_receivers):
    	residuals_predicted[rec2] = shot_predicted.receivers.data_dft[freq][0][rec2] - shot_base.receivers.data_dft[freq][0][rec2]
    	 
    estimatedField = np.concatenate((residuals_predicted.real, residuals_predicted.imag ))
    estimatedField = np.reshape(estimatedField, [1302,1], 'F')
    
    # evaluate the likelihood
    dMisfit = estimatedField - measuredField
    dMisfit = np.reshape(dMisfit, [1302,1], 'F')
    c1 = np.dot(covInv, dMisfit)
    c2 = np.dot(np.transpose(dMisfit), c1)
    pStar= (-0.5 * c2)
    pStar = pStar.real
    likelihood_history.append(pStar)
 #########################################################################
    
    for i in range(1, int(maxIter)):
     	print "Iteration %i of %i"%(i+1, maxIter) #i is zero indexed
     	
     	# Ct : covariance for proposal distribution
     	if i<t0:
 			Ct = sigma
     	elif i == t0:
     		y = np.array([np.array(xi) for xi in x[1:i-1]])
     		Ct  = Sd*np.cov(np.squeeze(y).T) + Sd*eps*np.identity(dof)
     	else:
     		y = np.array([np.array(xi) for xi in x[1:i-1]])
     		Ct  = Sd*np.cov(np.squeeze(y).T) + Sd*eps*np.identity(dof)
     		    		    	
    	# Define the xCur and pCur
        xCur  = x[i-1]
        pCur  = likelihood_history[i-1]
        aCur  = alpha_history[i-1]
         
        #if np.isnan(xCur):
        if np.any(xCur==np.nan):
        	break
         
        #get the new proposed model
        if i<t0: 	
        	xStar = np.random.normal(xCur, Ct)
        elif i==t0:
        	xStar = np.random.multivariate_normal(xCur.flatten(), Ct)
        else:
        	xStar = np.random.multivariate_normal(xCur.flatten(), Ct)

        
        #translate to alpha proposals
        xStar = np.reshape(xStar, [dof,1], 'F')
        aStar = np.multiply(xStar, a_range/2.0)
        aStar = np.add(aStar, np.mean(a_Limits))
                        
        #initialize the marm monitor
        marm_perturb_2d = np.zeros((nz, nx))
        marm_perturb_1d = np.reshape(marm_perturb_2d, (nz*nx,1), 'F')
        
        #Initialize new time lapse change delta_m
        deltam_2d = np.zeros((nz_local,nx_local))
        deltam_1d = np.reshape(deltam_2d, (nz_local*nx_local,1), 'F')
        
        # Get the time lapse change
        deltam_1d = np.dot(phi,np.reshape(aStar, [dof,1], 'F'))
        deltam_2d = np.reshape(deltam_1d , (nz_local, nx_local), 'F')
          
        #Apply the taper
        deltam_2d_tapered = np.zeros((nz_local,nx_local))
        deltam_2d_tapered = np.multiply(deltam_2d, taper_2d)
        deltam_1d_tapered = np.reshape(deltam_2d_tapered, (nz_local*nx_local,1), 'F')
        
        #convert to the full domain
        deltam_full_2d   = np.zeros((nz, nx))
        deltam_full_2d[92:117, 296:340] = deltam_2d_tapered
        
        #Create the proposed monitor by adding the proposed perturbation to the baseline model
        deltam_full_1d = np.reshape(deltam_full_2d, (nz*nx,1), 'F')
        marm_perturb_1d = marm_base_1d + deltam_full_1d
        
        truncated_velocity_perturbed_list    = truncate_array(marm_perturb_1d, m, domain_collection)
        truncated_velocity_perturbed_2d_list = truncate_array(np.reshape(marm_perturb_1d, (nz, nx), 'F'), m, domain_collection)
        truncated_model_perturbed            = truncated_solver.ModelParameters(truncated_solver.mesh_collection, {'C':truncated_velocity_perturbed_list})
        truncated_solver.model_parameters    = truncated_model_perturbed
        
        total_field_at_receivers = truncated_solver.total_field_at_receivers_from_shot(shot_predicted, nu=freq)
        
        receiver_dict = dict()
        num_receivers = len(shot_predicted.receivers.receiver_list)
        shot_predicted.receivers.data_dft[freq] = np.zeros((1, num_receivers), dtype='complex128')
        
        for n1 in xrange(num_receivers):
        	receiver = shot_predicted.receivers.receiver_list[n1]
        	pos = receiver.position
        	receiver_dict[pos] = n1
        	
        for pos1 in total_field_at_receivers:
        	index = receiver_dict[pos1]
        	shot_predicted.receivers.data_dft[freq][0][index] = total_field_at_receivers[pos1]
        	
        residuals_predicted = np.zeros(num_receivers,dtype='complex128')
        estimatedField      = np.zeros((num_receivers*2,1))
        dMisfit				= np.zeros((num_receivers*2,1))
        
        for rec3 in xrange(num_receivers):
        	residuals_predicted[rec3] = shot_predicted.receivers.data_dft[freq][0][rec3] - shot_base.receivers.data_dft[freq][0][rec3]
        	
        if np.any(np.abs(xStar)>1.0):
     		pStar = float("-inf")
     	else:
     		estimatedField = np.concatenate((residuals_predicted.real, residuals_predicted.imag ))
     		estimatedField = np.reshape(estimatedField, [1302,1], 'F')
     		
     		# evaluate likelihood
     		dMisfit = estimatedField - measuredField
     		dMisfit = np.reshape(dMisfit, [1302,1], 'F')
     		c1 = np.dot(covInv, dMisfit)
     		c2 = np.dot(np.transpose(dMisfit), c1)
     		pStar= (-0.5 * c2)
     		pStar = pStar.real
     		
        # Calculate acceptable probability
        pQ    = pStar - pCur
        alpha = min(1.0, np.exp(pQ))
        u     = np.random.uniform(0,1)
        
        if u<alpha:
        	x.append(xStar)
        	accept_history.append(1.0)
        	likelihood_history.append(pStar)
        	alpha_history.append(aStar)
        else:
        	x.append(xCur)
        	accept_history.append(0.0)
        	likelihood_history.append(pCur)
        	alpha_history.append(aCur)
        	
        	

    #Turn them to a numpy array
    acceptance_history    = np.array(accept_history)
    pStar_hist            = np.array(likelihood_history)
    alpha                 = np.array(alpha_history)

    #print "Finished MH loop; saving the data now"
    print("--- %s seconds ---" % (time.time() - start_time))

    # Save them in a dictionary
    results = dict()
    results['acceptance_history'] = acceptance_history
    results['pStar_hist']         = pStar_hist
    results['alpha']              = alpha
    spio.savemat('outdata/mcmc_results/Chain1_freq8_initalphas1_truebaseback_iter_%s_initial_sigma_%s.mat'%(maxIter, sigma), results)




    
 
    
