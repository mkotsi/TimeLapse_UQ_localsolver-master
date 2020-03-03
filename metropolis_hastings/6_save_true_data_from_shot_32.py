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


    indict          = spio.loadmat("indata/marm_true_models_dct.mat")
    uniform_spacing = indict['uniform_spacing'][0][0]
        
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
    
    num_receivers            = len(shot_moni.receivers.receiver_list)
    residuals_recorded       = np.zeros(num_receivers,dtype='complex128')
       
    for rec1 in xrange(num_receivers):
    	residuals_recorded[rec1] = shot_moni.receivers.data_dft[freq][0][rec1] - shot_base.receivers.data_dft[freq][0][rec1]
    
    moni_data = shot_moni.receivers.data_dft[freq][0]
    base_data = shot_base.receivers.data_dft[freq][0]
    
    # Save them in a dictionary
    spio.savemat('outdata/measuredField_at_shot_32_with_freq_8/residuals_recorded.mat', {'residuals_recorded': residuals_recorded})
    spio.savemat('outdata/measuredField_at_shot_32_with_freq_8/monitor_data.mat', {'moni_data':moni_data})
    spio.savemat('outdata/measuredField_at_shot_32_with_freq_8/baseline_data.mat', {'base_data':base_data})

    
 
    
