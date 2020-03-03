import numpy as np
import math

__all__ = ['set_solver_dt']

def set_solver_dt(solver,dt_suggested):
    #based on _process_mp_reset(self, *args, **kwargs): in constant_density_acoustic_time_base
    
    CFL = solver.cfl_safety
    t0, tf = solver.trange
    min_deltas = np.min(solver.mesh.deltas)
    C = solver._mp.C
    max_C = max(abs(C.min()), C.max())
    max_dt = CFL*min_deltas / max_C
    
    if dt_suggested > max_dt:
        raise Exception('suggested timestep is larger than prescribed by CFL!')
    
    #from this point on we know that dt_suggested is safe
    dt = dt_suggested
    nsteps = int(math.ceil((tf - t0)/dt))
    solver.dt = dt
    solver.nsteps = nsteps
    solver._rebuild_operators()
    assert(solver.dt == dt) #rebuild operator will not change it I think, just making sure.