#generate the true baseline and monitor data using a full domain solver
# Std import block
import time
import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio

from pysit import *
from pysit.gallery import marmousi2

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "4"
    #   Load or generate true wave speed we can add the compact operator
    #   flag it will speed up the resolution
    
    indict = spio.loadmat("indata/marm_true_models_dct.mat")
    marm_base_true_2d     = indict['marm_base_true_2d']
    marm_moni_true_2d     = indict['marm_moni_true_2d']
    uniform_spacing       = indict['uniform_spacing'][0][0]
    
    nz,nx  = marm_base_true_2d.shape 
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
    marm_moni_true_1d = np.reshape(marm_moni_true_2d, (nz*nx,1), 'F')
    
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

    # Set up shots for monitor (same as for baseline here)
    shots_moni = equispaced_acquisition(m,
                                        wavelet,
                                        sources=nshots,
                                        source_depth=src_depth,
                                        source_kwargs={'approximation':'delta'},
                                        receivers='max',
                                        receiver_depth=rec_depth,
                                        receiver_kwargs={'approximation':'delta'},
                                        )

    # Define and configure the wave solver. Use 2nd order acc
    solver = ConstantDensityHelmholtz(m, spatial_accuracy_order=2, spatial_shifted_differences=True)
    freqs  = [3.0, 4.0, 5.0, 6.5, 8.0, 10.0]

    # Generate synthetic Seismic data
    print('Generating baseline data...')
    base_model = solver.ModelParameters(m,{'C': marm_base_true_1d})
    tt = time.time()
    generate_seismic_data(shots_base, solver, base_model, frequencies=freqs, petsc='mumps')
    print 'Baseline data generation: {0}s'.format(time.time()-tt)

    print('Generating monitor data...')
    moni_model = solver.ModelParameters(m,{'C': marm_moni_true_1d})
    tt = time.time()
    generate_seismic_data(shots_moni, solver, moni_model, frequencies=freqs, petsc='mumps')
    print 'monitor data generation: {0}s'.format(time.time()-tt)
    
    shotnr = 0 
    for shot_base, shot_moni in zip(shots_base, shots_moni):
        base_data_dict = shot_base.receivers.data_dft
        moni_data_dict = shot_moni.receivers.data_dft
    
        for freq in freqs:
            base_data_dict[freq].flatten().tofile('indata/shot_%i_base_freq_%.2f.bin'%(shotnr, freq))
            moni_data_dict[freq].flatten().tofile('indata/shot_%i_moni_freq_%.2f.bin'%(shotnr, freq))
            
        shotnr += 1
        
    geom_dict = dict()
    geom_dict['nx'] = nx
    geom_dict['nz'] = nz
    geom_dict['uniform_spacing'] = uniform_spacing
    geom_dict['PMLwidth'] = PMLwidth
    geom_dict['PMLamp']   = PMLamp
    geom_dict['rec_depth'] = rec_depth
    geom_dict['src_depth'] = src_depth
    
    spio.savemat('indata/true_geom.mat', geom_dict)
