# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit_extensions.elastic_solver.wrapping_functions import elastic_solve
from pysit_extensions.elastic_solver.precompute_greens import precompute
from shared_routines import *

if __name__ == '__main__':

    m, shots = compute_shot_and_mesh()

    vp_2d, rho_2d, vs_2d, dt = give_background_models(m)

    t_max = 1.0
    itimestep = int(np.floor(t_max/dt) + 1); #number of timesteps
    
    rec_boundary_geom = { 'rec_x_l': 200.0,  'rec_x_r': 600.0,  'rec_z_t': 200.0, 'rec_z_b': 600.0}

    elastic_options_dict = {'dt': dt,
                            'itimestep':itimestep, 
                            'iwavelet': 9, #Green's function calculation 
                            'rec_boundary_geom': rec_boundary_geom,
                            }
    
    precompute(shots, m, rec_boundary_geom, vp_2d, rho_2d, vs_2d, elastic_options_dict, save_prefix = 'greens_out/step1_')
    
    