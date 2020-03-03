from pysit import *

from pysit_extensions.convenient_plot_functions.plot_functions import *
from pysit_extensions.elastic_solver.wrapping_functions import *
from pysit_extensions.ximage.ximage import ximage
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import scipy.io as spio
import copy

multiplier = 1.0
dx = 12.5*multiplier #Same as in ss
dz = 12.5*multiplier

nx = int(round(1000/multiplier + 1)) #different sized model though
nz = int(round(50/multiplier + 1))

x_min = 0.0
x_max = (nx-1)*dx

z_min = 0.0
z_max = (nz-1)*dz

background_vel = 1500.0

C_2d = background_vel*np.ones((nz,nx))
C  = np.reshape(C_2d  ,(nz*nx,1), 'F')


#   Define Domain
pmlx = PML(125*dx/multiplier, 50)
pmlz = PML(125*dz/multiplier, 50)

x_config = (x_min, x_max, pmlx, pmlx)
z_config = (z_min, z_max, pmlz, pmlz)

d = RectangularDomain(x_config, z_config)

m = CartesianMesh(d, nx, nz)

# Set up shots

peakfreq = 6.0
depth_source_receiver = dz

x_pos_source = 1500.0 #A little bit away from the PML
z_pos_source = depth_source_receiver

x_pos_receivers = np.linspace(x_min, x_max, nx)
z_pos_receivers = depth_source_receiver #I guess it will always record 0.0 because that is what this Dirichlet boundary is 

shots_cda = []
source_approx = 'delta'
receiver_approx = 'delta'
source = PointSource(m, (x_pos_source, z_pos_source), RickerWavelet(peakfreq, threshold=0.0), approximation = source_approx)
receivers = ReceiverSet(m, [PointReceiver(m, (x, z_pos_receivers), approximation = receiver_approx) for x in x_pos_receivers])
shot = Shot(source, receivers)
shots_cda.append(shot)
shots_el = copy.deepcopy(shots_cda)

# Define and configure the wave solver
trange = (0.0,1.0)

cfl_safety = 1./12 #default is 1./6. Smaller should give better results?
solver = ConstantDensityAcousticWave(m,
                                     spatial_accuracy_order=4,
                                     trange=trange,
                                     cfl_safety = cfl_safety, #smaller timestep
                                     kernel_implementation='cpp')


# Generate synthetic Seismic data
sys.stdout.write('Generating data...')
base_model = solver.ModelParameters(m,{'C': C})
solver.model_parameters = base_model #This will set dt

tt = time.time()
generate_seismic_data(shots_cda, solver, base_model)

shotgather_cda = shots_cda[0].receivers.data
ts_cda = solver.ts()

###################### ELASTIC #########################
print "NOW DOING ELASTIC"

shots_el[0].sources.w.t_shift = shots_cda[0].sources.w.t_shift + solver.dt
print "Reducing time shift of ricker by %e seconds. Reason is that in code Xinding the pressure is applied immediately at the pixel. In pysit it takes dt to propagate from the right hand side. We compensate before letting C code read the source wavelet"%solver.dt

pml_thickness = 125*dx/multiplier
p_power=2
PPW0=10
d0factor=suggest_d0_factor(p_power, dx, pml_thickness, background_vel)

pmlx = Elastic_Code_PML(pml_thickness, d0factor, PPW0, p_power)
pmlz = Elastic_Code_PML(pml_thickness, d0factor, PPW0, p_power)

x_config = (x_min, x_max, pmlx, pmlx)
z_config = (z_min, z_max, pmlz, pmlz)

d = RectangularDomain(x_config, z_config)
m = CartesianMesh(d, nx, nz)

vp_2d  = C_2d
rho_2d = 1000.0*np.ones_like(vp_2d)
vs_2d  = 0.0*np.ones_like(vp_2d)

vmax = np.max([vp_2d,rho_2d,vs_2d])

dt_max = dx/(np.sqrt(2)*vmax*(9./8 +1./24))
dt = 1./3*dt_max #levander 1988 shows that 1./2 is a good step. I'm just a little extra safe since PVA depends on good phase and i saw that timesteps can influence dispersion a bit 
itimestep = int(np.floor(trange[1]/dt)); #number of timesteps
#call elastic code
snaps_mem = True
traces_mem = True
rec_boundary = False
local_solve  = False
amp0  = 1.0
elastic_options = {'dt': dt,
                   'amp0': amp0,
                   'iwavelet': 3, #Use the PySIT wavelet from the source
                   'itimestep':itimestep, 
                   'snap_output': 1,
                   'snap_step_length': 1,    #Not my focus in this test. But want to keep on so I can verify it still doesnt segfault
                   'snaps_mem': snaps_mem,
                   'traces_output': 1,
                   'traces_step_length': 1,  #Not my focus in this test. But want to keep on so I can verify it still doesnt segfault
                   'traces_mem': traces_mem,
                   'snap_wavefield_return_shape': 'pysit',
                   'rec_boundary': rec_boundary,
                   'local_solve': local_solve 
                   }

retval = elastic_solve(shots_el, m, vp_2d, rho_2d, vs_2d, elastic_options)
shotgather_el = retval['shotgathers'][0]
ts_el = retval['shotgathers_times'][0]

###################### PLOT RESULTS #########################

wavefield = retval['wavefields'][0]
vis.animate(wavefield, m, display_rate=1,pause=1, scale=None, show=True)
ximage(retval['shotgathers'][0], perc=90)

source_pixel = int(np.round(x_pos_source/dx))

offset = 300.0 #plot traces at this offset
offset_pixel = int(np.round(offset/dx))
trace_delta_cda = shotgather_cda[:, source_pixel + offset_pixel].flatten()
trace_delta_el  = shotgather_el[ :, source_pixel + offset_pixel].flatten()

dt_cda = ts_cda[1]-ts_cda[0]
dt_el  =  dt

energy_cda   = np.sum(dt_cda * trace_delta_cda**2)
energy_el    = np.sum(dt_el  *  trace_delta_el**2)

energy_ratio = energy_cda/energy_el

#As long as Vs = 0, the square root of the energy ratio is basically the background velocity squared
print np.sqrt(energy_ratio), background_vel**2, np.sqrt(energy_ratio)/background_vel**2

#trace_delta_el *= np.sqrt(energy_ratio)

fig = plt.figure(1)
plt.plot(ts_cda, trace_delta_cda, 'r', lw = 3, label='CDA')
plt.plot(ts_el,  background_vel**2*trace_delta_el , 'b', lw = 3, label='EL * $c^2$')
plt.legend(fontsize = 16)
plt.xlabel('time', fontsize = 16)
plt.show()