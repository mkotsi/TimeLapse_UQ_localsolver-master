# Std import block
import time

import numpy as np
import matplotlib.pyplot as plt

from pysit import *
from pysit_extensions.elastic_solver.precompute_greens import precompute_using_cda_solver
from shared_routines import *

if __name__ == '__main__':

    m, shots = compute_shot_and_mesh('PySIT')

    vp_2d, rho_2d, vs_2d, dt = give_background_models(m)

    t_min = 0.0
    t_max = 1.0
    itimestep = int(np.floor(t_max/dt) + 1); #number of timesteps
    
    rec_boundary_geom = { 'rec_x_l': 200.0,  'rec_x_r': 650.0,  'rec_z_t': 200.0, 'rec_z_b': 500.0}
    
    cda_options_dict          = dict()
    cda_options_dict['t_min'] = t_min
    cda_options_dict['t_max'] = t_max
    cda_options_dict['rho']   = np.mean(rho_2d) #rho_2d should be uniform anyway in this test. We need a constant rho to turn pressure into velocity for boundary field computation using the CDA solver 

    precompute_using_cda_solver(shots, m, rec_boundary_geom, vp_2d, cda_options_dict, save_prefix = 'greens_out/step1_')


    
    