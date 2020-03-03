import numpy as np
from pysit_extensions.elastic_solver.wrapping_functions import Elastic_Code_PML, suggest_d0_factor
from pysit.gallery.marmousi2 import marmousi2
from pysit import *

def compute_shot_and_mesh(type='EL'):
    vp_background   = 2000.0
    pml_thickness   = 1000.0 #For now kind of excessive.
    uniform_spacing = 10.0
    
    p_power=2
    PPW0=10
    d0factor=suggest_d0_factor(p_power, uniform_spacing, pml_thickness, vp_background)

    #   Define Domain
    if type == 'EL':
        pmlx = Elastic_Code_PML(pml_thickness, d0factor, PPW0, p_power)
        pmlz = Elastic_Code_PML(pml_thickness, d0factor, PPW0, p_power)
    elif type == 'PySIT':
        pmlx = PML(pml_thickness, 50)
        pmlz = PML(pml_thickness, 50)        
    else:
        raise Exception("Mistakes were made...")

    x_min = 0.0
    z_min = 0.0

    x_max = 800.0
    z_max = 800.0 #different value so I can test if I do symmetry correctly

    x_config = (x_min, x_max, pmlx, pmlx)
    z_config = (z_min, z_max, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    nx = int((x_max-x_min)/uniform_spacing + 1)
    nz = int((z_max-z_min)/uniform_spacing + 1)

    m = CartesianMesh(d, nx, nz)
    
    # Set up shots
    nshots          = 3
    nreceivers      = 9 
    z_pos_source    = 10.0
    x_pos_sources   = np.linspace(x_min, x_max, nshots);
    x_pos_receivers = np.linspace(x_min,x_max,nreceivers) #100, 200m, 300m, ... , 800m
    z_pos_receivers = 30.0
    print "Should I assure that source positions are exactly on grid ? "
    
    if z_pos_receivers%uniform_spacing != 0 or z_pos_source%uniform_spacing != 0:
        raise Exception("not integer number of points")
    
    shots = []
    for i in xrange(nshots):
        # Define source location and type
        if type == 'EL':
            print "Freq passed to ricker wavelet here has no influence. The ricker object itself is not even used right now"
            
        source = PointSource(m, (x_pos_sources[i], z_pos_source), RickerWavelet(6.0), approximation='delta') 
        
        # Define set of receivers
        receivers = ReceiverSet(m, [PointReceiver(m, (x, z_pos_receivers), approximation='delta') for x in x_pos_receivers])
    
        # Create and store the shot
        shot = Shot(source, receivers)
        shots.append(shot)

    return m, shots

def suggest_dt(m, vp_2d, rho_2d, vs_2d):
    vmax_in_model = np.max([vp_2d,rho_2d,vs_2d])
    vmax_allowed  = 8000.0
    if vmax_in_model > vmax_allowed:
        raise Exception("""We would exceed the maximum velocity! 
                           We use this max allowed velocity to have allow for uniform dt in green's functions and the perturbed model. 
                           Otherwise we would have to interpolate which does not work well. 
                           Gives artificial scattered field due to interpolation differences.""")
    
    dt_max = m.x.delta/(np.sqrt(2)*vmax_allowed*(9./8 +1./24)) #assuming equal spacing
    dt = 1./3*dt_max         
    return dt

def give_background_models(m):
    nx = m.x.n
    nz = m.z.n
    
    #To get a realisic inhomegeneous background model i just grab a small part of the marmousi model
    C, C0_dummy, m_dummy, d_dummy = marmousi2(patch='fracture-square')
    C_2d = np.reshape(C, (m_dummy.z.n, m_dummy.x.n), 'F')

    C_2d = np.copy(C_2d[:nz, :nx])
    C_2d[:,2*nx/3:]*= 1.2 #create some horizontal variation to make sure scaling applies to the source pixel
    C = np.reshape(C_2d, (nz*nx,),'F')
    #Doing some extra reshapes here. For some reason I had some problems calling the c code otherwise. Maybe related to contiguous in memory?
    
    dens_background = 1000.0
    vp_1d           = C*np.ones(nx*nz)
    rho_1d          = dens_background*np.ones_like(vp_1d)
    vs_1d           = np.zeros_like(vp_1d)
    
    vp_2d = np.reshape(vp_1d,(nz,nx),'F')
    rho_2d = np.reshape(rho_1d,(nz,nx),'F')
    vs_2d = np.reshape(vs_1d,(nz,nx),'F')
    
    dt = suggest_dt(m, vp_2d, rho_2d, vs_2d)
        
    return vp_2d, rho_2d, vs_2d, dt
    
def give_perturbed_models(m, amp):
    vp_2d, rho_2d, vs_2d, dt_background = give_background_models(m)

    nx = m.x.n
    nz = m.z.n    

    vp_2d[(nz-1)/2,(nx-1)/2] += amp
    dt_perturbed = suggest_dt(m, vp_2d, rho_2d, vs_2d)
    
    if dt_perturbed != dt_background:
        print "Dt in perturbed and background model not the same. Need to be careful with handling..."
        
    return vp_2d, rho_2d, vs_2d, dt_perturbed