# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit.gallery import horizontal_reflector
from pysit_extensions.elastic_solver.wrapping_functions import *
from pysit_extensions.ximage.ximage import ximage

if __name__ == '__main__':
    # Setup

    p_power=2
    d0factor=1
    PPW0=10

    pml_thickness = 300.0

    #   Define Domain
    pmlx = Elastic_Code_PML(pml_thickness, d0factor, PPW0, p_power)
    pmlz = Elastic_Code_PML(pml_thickness, d0factor, PPW0, p_power)

    x_min = 0.0
    z_min = 0.0

    x_max = 2000
    z_max = 500 #different value so I can test if I do symmetry correctly

    x_config = (x_min, x_max, pmlx, pmlx)
    z_config = (z_min, z_max, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    uniform_spacing = 10.0
    nx = int((x_max-x_min)/uniform_spacing + 1)
    nz = int((z_max-z_min)/uniform_spacing + 1)

    m = CartesianMesh(d, nx, nz)
    
    # Set up shots
    nshots = 1
    nreceivers = nx
    z_pos = z_min + 2*uniform_spacing
    #x_pos_sources = np.linspace(x_min,x_max,nshots)
    x_pos_sources = (x_max - x_min)/2*np.ones(nshots)
    x_pos_receivers = np.linspace(x_min,x_max,nreceivers)
    
    print "Should I assure that source positions are exactly on grid ? "
    
    shots = []
    for i in xrange(nshots):
        # Define source location and type
        print "Freq passed to ricker wavelet here has no influence. The ricker object itself is not even used right now"
        source = PointSource(m, (x_pos_sources[i], z_pos), RickerWavelet(10.0), approximation='delta') 
        
        # Define set of receivers
        receivers = ReceiverSet(m, [PointReceiver(m, (x, z_pos), approximation='delta') for x in x_pos_receivers])
    
        # Create and store the shot
        shot = Shot(source, receivers)
        shots.append(shot)

    vp = 2000
    rho = 1000
    vs = 1000
    
    vp_1d  =  vp*np.ones(nx*nz)
    vs_1d  =  vs*np.ones(nx*nz)
    rho_1d = rho*np.ones(nx*nz)
    
    #I will do fortran ordering here just like in normal pysit. Internally still C ordering though. Will also pass as contiguous C array to lib
    vp_2d = np.reshape(vp_1d,(nz,nx),'F')
    vp_2d[nz*2/3:,:] *= 2 #Should give reflection so I can see if I do correctly. C order, F order etc. 
    rho_2d = np.reshape(rho_1d,(nz,nx),'F')
    vs_2d = np.reshape(vs_1d,(nz,nx),'F')

    vmax = np.max([vp_2d,rho_2d,vs_2d])    
    dt_max = uniform_spacing/(np.sqrt(2)*vmax*(9./8 +1./24))
    dt = 1./3*dt_max 
    
    t_max = 1.5
    itimestep = int(np.floor(t_max/dt)); #number of timesteps
    ############DO TEST############
    amp0  = 1.0
    freq0 = 10.0
    vmin = np.min([vp,rho,vs])
    dx = uniform_spacing
    #critical values used in input script for normal elastic solver
    ddx=vmin/(1.5*freq0)/6
    ddt=dx/(np.sqrt(2.)*vmax*(9./8+1./24));     
    
    if ddx<uniform_spacing:
        print 'WARNING: unstable dx = %f, should be < %f'%(dx,ddx)
    else:
        print'dx = %f (<%f) is stable'%(dx,ddx)
    
    if ddt<dt:
        print 'WARNING: unstable dt = %f, should be < %f'%(dt,ddt)
    else:
        print 'dt = %f (<%f) is stable'%(dt,ddt)
    
    ############END DO TEST############
    
    #call elastic code
    snaps_mem = True
    traces_mem = True
    elastic_options = {'dt': dt,
                       'amp0': amp0,
                       'freq0': freq0,
                       'itimestep':itimestep, 
                       'snap_output': 1,
                       'snap_step_length': 7,
                       'snaps_mem': snaps_mem,
                       'traces_output': 1,
                       'traces_step_length': 7,
                       'traces_mem': traces_mem,
                       'snap_wavefield_return_shape': 'pysit'
                       }
    
    print "CHANGE WRAPPER SO THAT I CAN TELL DISTANCE PIXEL RANGE WHERE I WANT WAVEFIELD. THEN WRAPPER WILL CALCULATE APPROPRIATE PIXEL CONVERSIONS"
    print "Also incorporate the storage directory locations when saving to disk?"
    
    retval = elastic_solve(shots, m, vp_2d, rho_2d, vs_2d, elastic_options)
    wavefield_times = retval['wavefields_times'][0] #taking from shot 0
    shotgathers_times = retval['shotgathers_times'][0] #taking from shot 0
    print "Finished solve"

    wavefield = retval['wavefields'][0]
    vis.animate(wavefield, m, display_rate=1,pause=1, scale=None, show=True)
    ximage(retval['shotgathers'][0], perc=90)
