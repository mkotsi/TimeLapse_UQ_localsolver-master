from pysit import *
from pysit_extensions.convenient_plot_functions.plot_functions import *
from pysit_extensions.elastic_solver.wrapping_functions import *
from pysit_extensions.calculate_phase.calc_phase import calc_phase
from pysit_extensions.calculate_phase.window import window_direct_wave, window_reflected_wave
from pysit_extensions.ximage.ximage import ximage
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import scipy.io as spio
import copy

peakfreq = 25.0
vp_top_layer = 2000.0
vs_top_layer = 880.0
rho_top_layer= 2000.0
vp_bot_layer = 4000.0
vs_bot_layer = 1540.0
rho_bot_layer= 2300.0

x_min = 0.0
x_max = 8100.0 #8000m is what I want. But put source at x=100,z=100
    
z_min = 0.0
z_max = 1600.0

#uniform_spacing = 2.0
uniform_spacing = 2.5

nx = (x_max-x_min)/uniform_spacing + 1
nz = (z_max-z_min)/uniform_spacing + 1

#these are in F ordering
vp_2d   = vp_bot_layer  * np.ones((nz,nx))
vs_2d   = vs_bot_layer  * np.ones((nz,nx))
dens_2d = rho_bot_layer * np.ones((nz,nx))

depth_reflector = 1100.0 #real depth is 1000m of the layer. But I put source and receiver at 100m depth, just to be sure PML is no problem
z_pixel_reflector = int(np.round(depth_reflector/uniform_spacing))
    
vp_2d[0:z_pixel_reflector,:] = vp_top_layer
vs_2d[0:z_pixel_reflector,:] = vs_top_layer
dens_2d[0:z_pixel_reflector,:] = rho_top_layer

#I have another script which runs the EL code standalone. There I used PML thickness of 1500.0m. Probably excessive
pml_thickness = 750.0 
nabs = int(round(pml_thickness/uniform_spacing)) 
ifsbc=0       # =1: free surface on top; =0: no free surface
p_power=2
d0factor=1
PPW0=10       

pmlx = Elastic_Code_PML(pml_thickness, d0factor, PPW0, p_power)
pmlz = Elastic_Code_PML(pml_thickness, d0factor, PPW0, p_power)
x_config = (x_min, x_max, pmlx, pmlx)
z_config = (z_min, z_max, pmlz, pmlz)
d = RectangularDomain(x_config, z_config)
m = CartesianMesh(d, nx, nz)

source_x = 100.0
depth    = 100.0
spacing_receivers = 25.0
x_pos_receivers = np.arange(x_min, x_max, spacing_receivers)
source = PointSource(m, (source_x, depth), RickerWavelet(peakfreq, threshold=0.0), approximation = 'delta')
receivers = ReceiverSet(m, [PointReceiver(m, (x, depth), approximation = 'delta') for x in x_pos_receivers])
shot = Shot(source, receivers)
shots = [shot]

vmax = np.max([vp_2d,dens_2d,vs_2d])
dt_max = uniform_spacing/(np.sqrt(2)*vmax*(9./8 +1./24))
dt = 1./3*dt_max # 1/2 also gives more or less the same result for this horizontal spacing. But 1./3 slightly better. Worth the price?
tmax = 5.0
itimestep = int(np.floor(tmax/dt)); #number of timesteps

snaps_mem = True
traces_mem = True
rec_boundary = False
local_solve  = False
amp0  = 1.0
elastic_options = {'dt': dt,
                   'amp0': amp0,
                   'freq0': peakfreq,
                   'itimestep':itimestep, 
                   'snap_output': 0,
                   'snap_step_length': 1,    #Not my focus in this test. But want to keep on so I can verify it still doesnt segfault
                   'snaps_mem': snaps_mem,
                   'traces_output': 1,
                   'traces_step_length': 1,  #Not my focus in this test. But want to keep on so I can verify it still doesnt segfault
                   'traces_mem': traces_mem,
                   'snap_wavefield_return_shape': 'pysit',
                   'rec_boundary': rec_boundary,
                   'local_solve': local_solve 
                   }

retval = elastic_solve(shots, m, vp_2d, dens_2d, vs_2d, elastic_options)
shotgather_pr_2d = retval['shotgathers'][0]
ts = retval['shotgathers_times'][0]

########################### NOW USE THE LOGIC FROM BEFORE IN plot_subsampled_shotgather_and_trace #######################
x_min_km = x_min / 1000.0; x_max_km = x_max / 1000.0; t_min = 0.0; t_max = tmax
extents = {'x_min':x_min_km, 'x_max':x_max_km, 'z_min':t_min, 'z_max':t_max}

#SEE PLOT_2005_GATHERS IN cop-gen-paper-figures FOLDER FOR EXAMPLE HOW TO PLOT MULTIPLE SHOTGATHERS IN A GRID

fignr = 0
fig_dir = 'figures/'
savedpi=600.0
ftsize = 8
axes_pad=0.3
cmap_greys=plt.get_cmap('Greys')
##########################################
fignr += 1
figsize = (3.33, 4.5)

list_of_2d_ndarrays = [shotgather_pr_2d/shotgather_pr_2d[110:,:].max()]
list_of_titles = ['Shotgather pr', 'Shotgather vz']
n_rows = 2
n_cols = 1

cbar_min = -2e-1
cbar_max = 2e-1
cbar_title = "whatever, cbar_mode is None"
#cbar_title = None

ylabel = 'Time (s)'
fig = plot_in_grid(fignr, list_of_2d_ndarrays, list_of_titles, n_rows, n_cols, extents, axes_pad=axes_pad, cbar_mode = None, cbar_min = cbar_min, cbar_max = cbar_max, ylabel = ylabel, figsize=figsize,cmap = cmap_greys, x_align_val_cbar = 0.0, cbar_title = cbar_title, ftsize= ftsize)
fig.tight_layout(pad = 0.0, rect = [0.0, 0.0, 1.0, 1.0])
#fig.tight_layout(pad = 0.0, rect = [0.0085, 0.0, 1.0, 1.0])

plt.figure(2)
#ts = np.linspace(t_min, t_max, nt)
sx = 100.0
trace_offset = 5000.0
trace_x = sx + trace_offset
trace_pixel = int(np.round(trace_x/spacing_receivers))
plt.plot(ts, shotgather_pr_2d[:,trace_pixel],'r', label ='%e m spacing'%uniform_spacing)
#plt.plot(ts, shotgather_pr_2d[:,7*trace_pixel],'b')
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.legend()
plt.title('Offset %e m'%trace_offset)

print "Both the 2.0m spacing and the 1.0m spacing simulations give results that are practically identical. Using the 2.0m spacing results is just fine."

#DIRECT WAVE

len_window = 0.15
v_top_layer = 2000.0
#offset_arr = np.linspace(x_min, x_max, nx) - sx
offset_arr = x_pos_receivers - sx
windowed_direct_wave_shotgather = window_direct_wave(shotgather_pr_2d, offset_arr, ts, len_window, v_top_layer)

fignr = 3
plt.figure(fignr)
extents = {'x_min':x_min_km, 'x_max':x_max_km, 'z_min':0.0, 'z_max':len_window}
list_of_2d_ndarrays = [windowed_direct_wave_shotgather / windowed_direct_wave_shotgather.max()]
n_rows = 1
fig = plot_in_grid(fignr, list_of_2d_ndarrays, list_of_titles, n_rows, n_cols, extents, axes_pad=axes_pad, cbar_mode = None, cbar_min = cbar_min, cbar_max = cbar_max, ylabel = ylabel, figsize=figsize,cmap = cmap_greys, x_align_val_cbar = 0.0, cbar_title = cbar_title, ftsize= ftsize)
fig.tight_layout(pad = 0.0, rect = [0.0, 0.0, 1.0, 1.0])

#For each trace in gather, probably need to window first (step 1, locate the wavelet)
phases_direct = calc_phase(windowed_direct_wave_shotgather) #first sec of trace at offset 1km only contains direct wave

#NOW FOR REFLECTION
len_window = 0.15
reflector_depth = 1000.0
windowed_reflected_wave_shotgather = window_reflected_wave(shotgather_pr_2d, offset_arr, ts, len_window, v_top_layer, reflector_depth)

fignr = 4
plt.figure(fignr)
extents = {'x_min':x_min_km, 'x_max':x_max_km, 'z_min':0.0, 'z_max':len_window}
list_of_2d_ndarrays = [windowed_reflected_wave_shotgather / windowed_reflected_wave_shotgather.max()]
n_rows = 1
fig = plot_in_grid(fignr, list_of_2d_ndarrays, list_of_titles, n_rows, n_cols, extents, axes_pad=axes_pad, cbar_mode = None, cbar_min = cbar_min, cbar_max = cbar_max, ylabel = ylabel, figsize=figsize,cmap = cmap_greys, x_align_val_cbar = 0.0, cbar_title = cbar_title, ftsize= ftsize)
fig.tight_layout(pad = 0.0, rect = [0.0, 0.0, 1.0, 1.0])

#For each trace in gather, probably need to window first (step 1, locate the wavelet)
phases_reflected = calc_phase(windowed_reflected_wave_shotgather) #first sec of trace at offset 1km only contains direct wave
phases_difference = phases_reflected-phases_reflected[int(np.round(sx/uniform_spacing))] #subtract zero offset reflection phase
phases_difference[np.where(phases_difference<0)] += 360 #Otherwise wraps in difficult to interpret way
plt.figure(5)
plt.plot(offset_arr, phases_direct, 'r', label='direct')
plt.plot(offset_arr, phases_reflected, 'b', label='reflected')
plt.plot(offset_arr, phases_difference, 'k', label='difference with zero-offset reflection')
plt.legend()

angle_arr = np.arctan((offset_arr/2)/reflector_depth)*180./np.pi
#same as figure 5, now plot angle on horizontal axis
plt.figure(6)
plt.plot(angle_arr, phases_direct, 'r', label='direct')
plt.plot(angle_arr, phases_reflected, 'b', label='reflected')
plt.plot(angle_arr, phases_difference, 'k', label='difference with zero-offset reflection')
plt.legend()

plt.show()