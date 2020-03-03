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

    x_max = 1500.0
    z_max = 800.0 #different value so I can test if I do symmetry correctly

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
    #z_pos = z_min + 2*uniform_spacing
    z_pos = 530.0
    #x_pos_sources = (x_max - x_min)/2*np.ones(nshots)
    #x_pos_sources = 1220.0*np.ones(nshots);
    x_pos_sources = 280.0*np.ones(nshots);
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

    vp  = 3000
    rho = 1000
    vs  = 1000
    vp_1d  =  vp*np.random.rand(nx*nz) + vp
    vs_1d  =  vs*np.ones(nx*nz) + 0.5*vs
    rho_1d =  rho*np.random.rand(nx*nz) + rho
    
    #I will do fortran ordering here just like in normal pysit. Internally still C ordering though. Will also pass as contiguous C array to lib
    vp_2d = np.reshape(vp_1d,(nz,nx),'F')
    rho_2d = np.reshape(rho_1d,(nz,nx),'F')
    vs_2d = np.reshape(vs_1d,(nz,nx),'F')
    
    dt = 1./6 * uniform_spacing/np.max(vp_2d)
    
    t_max = 1002*dt
    itimestep = int(np.floor(t_max/dt)); #number of timesteps
    ############DO TEST############
    amp0  = 1.0
    freq0 = 10.0
    vmin = np.min([vp,rho,vs])
    vmax = np.max([vp,rho,vs])
    dx = uniform_spacing
    #critical values used in input script for normal elastic solver
    ddx=vmin/(1.5*freq0)/6
    print "Is this the correct stability number?"
    ddt=dx/(np.sqrt(2.)*vmax*(9/8+1/24));     
    
    if ddx<uniform_spacing:
        print 'WARNING: unstable dx = %f, should be < %f'%(dx,ddx)
    else:
        print'dx = %f (<%f) is stable'%(dx,ddx)
    
    if ddt<dt:
        print 'WARNING: unstable dt = %f, should be < %f'%(dt,ddt)
    else:
        print 'dt = %f (<%f) is stable'%(dt,ddt)
    
    ############END DO TEST############
    
    ############DO FULL DOMAIN SOLVE############
    x_left_record  = x_min + 30*dx
    x_right_record = x_max - 30*dx
    z_top_record   = z_min + 10*dx
    z_bot_record   = z_max - 10*dx
    
    x_left_subm     = x_min + 10*dx
    x_right_subm    = x_max - 10*dx
    z_top_subm      = z_min  +  5*dx
    z_bot_subm      = z_max  -  5*dx    
    
    #call elastic code
    snaps_mem = True
    traces_mem = False
    rec_boundary = True
    local_solve  = False
    elastic_options_full = {'dt': dt,
                            'amp0': amp0,
                            'freq0': freq0,
                            'itimestep':itimestep, 
                            'snap_output': 1,
                            'snap_step_length': 1,    #Not my focus in this test. But want to keep on so I can verify it still doesnt segfault
                            'snaps_mem': snaps_mem,
                            'traces_output': 1,
                            'traces_step_length': 100,  #Not my focus in this test. But want to keep on so I can verify it still doesnt segfault
                            'traces_mem': traces_mem,
                            'snap_wavefield_return_shape': 'pysit',
                            'rec_boundary': rec_boundary,
                            'rec_boundary_geom': { 'rec_x_l': x_left_record,  'rec_x_r': x_right_record,  'rec_z_t': z_top_record, 'rec_z_b': z_bot_record},
                            'ss_param': {'left':x_left_subm, 'right': x_right_subm, 'top': z_top_subm, 'bot': z_bot_subm, 'ssvar': (c_int * 12)(0,0,1,0,0,0,0,0,0,0,0,0)}, #In this full domain solve, save snapshots in the submesh. The 12 ints refer to the following vals: [vx, vz, pr, div, curl, txx, tzz, txz, vpx, vpz, vsx, vsz]
                            'local_solve': local_solve 
                            }
    
    retval_full = elastic_solve(shots, m, vp_2d, rho_2d, vs_2d, elastic_options_full)
    recorded_boundary_fields = retval_full['boundary_fields'] #shot 0
    print "Finished with full domain solve. Prepare for truncated domain solve"
    ############DO TRUNCATED DOMAIN SOLVE############
    rec_boundary = False
    local_solve  = True
    record_traces= False

    vp_trunc_2d   = full_to_truncated( vp_2d, x_left_subm, x_right_subm, z_top_subm, z_bot_subm, dx)
    rho_trunc_2d  = full_to_truncated(rho_2d, x_left_subm, x_right_subm, z_top_subm, z_bot_subm, dx)
    vs_trunc_2d   = full_to_truncated( vs_2d, x_left_subm, x_right_subm, z_top_subm, z_bot_subm, dx)
    
    [nz_trunc, nx_trunc] = vp_trunc_2d.shape

    elastic_options_truncated = {'gx0': x_left_subm,
                                 'gz0': z_top_subm,
                                 'MM' : nx_trunc,
                                 'KK' : nz_trunc,
                                 'dt': dt,
                                 'amp0': amp0,
                                 'freq0': freq0,
                                 'itimestep':itimestep, 
                                 'snap_output': 1,
                                 'snap_step_length': 1,    #Not my focus in this test. But want to keep on so I can verify it still doesnt segfault
                                 'snaps_mem': snaps_mem,
                                 'traces_output': record_traces,
                                 'traces_step_length': 100,  #Not my focus in this test. But want to keep on so I can verify it still doesnt segfault
                                 'traces_mem': traces_mem,
                                 'snap_wavefield_return_shape': 'pysit',
                                 'rec_boundary': rec_boundary,
                                 'rec_boundary_geom': { 'rec_x_l': x_left_record,  'rec_x_r': x_right_record,  'rec_z_t': z_top_record, 'rec_z_b': z_bot_record},
                                 'ss_param': {'left':x_left_subm, 'right': x_right_subm, 'top': z_top_subm, 'bot': z_bot_subm, 'ssvar': (c_int * 12)(0,0,1,0,0,0,0,0,0,0,0,0)}, #In this full domain solve, save snapshots in the submesh. The 12 ints refer to the following vals: [vx, vz, pr, div, curl, txx, tzz, txz, vpx, vpz, vsx, vsz]
                                 'local_solve': local_solve 
                                 }    
   
    retval_truncated = elastic_solve(shots, 
                                     m, 
                                     vp_trunc_2d, 
                                     rho_trunc_2d, 
                                     vs_trunc_2d, 
                                     elastic_options_truncated,
                                     recorded_boundary_fields = recorded_boundary_fields
                                     )
    
    full_domain_wavefield = retval_full['wavefields'][0]
    truncated_domain_wavefield = retval_truncated['wavefields'][0]
   
    t = 1000*dt
    snap_step = int(t/dt)
    full_domain_snap_2d = np.reshape(full_domain_wavefield[snap_step],(nz_trunc,nx_trunc), 'F')
    truncated_domain_snap_2d = np.reshape(truncated_domain_wavefield[snap_step],(nz_trunc,nx_trunc), 'F')
    difference_snap_2d = np.reshape(truncated_domain_snap_2d-full_domain_snap_2d, (nz_trunc,nx_trunc), 'F')
    
    plt.figure(1); plt.imshow(full_domain_snap_2d, interpolation='nearest'); plt.colorbar()
    plt.figure(2); plt.imshow(truncated_domain_snap_2d, interpolation='nearest'); plt.colorbar()
    plt.figure(3); plt.imshow(difference_snap_2d, interpolation='nearest'); plt.colorbar()
    plt.figure(4);
    plt.plot(truncated_domain_snap_2d[:,111], 'b') #Should be scattered field, and zero.
    plt.plot(truncated_domain_snap_2d[:,110], 'r') #Should be scattered field, and zero.
    plt.plot(truncated_domain_snap_2d[:,109], 'k')
         
    #plt.plot(truncated_domain_snap_2d[65,:], 'b') #Should be scattered field, and zero.
    #plt.plot(truncated_domain_snap_2d[66,:], 'r') #Should be scattered field, and zero.
    #plt.plot(truncated_domain_snap_2d[67,:], 'k') 
    plt.show()