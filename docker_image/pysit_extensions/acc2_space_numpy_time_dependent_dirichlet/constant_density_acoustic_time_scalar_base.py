from pysit.solvers.constant_density_acoustic.time.constant_density_acoustic_time_base import *
from pysit.solvers.solver_data import SolverDataTimeBase

from pysit.util.solvers import inherit_dict
import numpy as np

__all__ = ['ConstantDensityAcousticTimeScalarBase']

__docformat__ = "restructuredtext en"


class _ConstantDensityAcousticTimeScalar_SolverData(SolverDataTimeBase):

    def __init__(self, solver, time_dir_bc_data, temporal_accuracy_order, **kwargs):

        self.solver = solver

        self.temporal_accuracy_order = temporal_accuracy_order

        # self.us[0] is kp1, [1] is k or current, [2] is km1, [3] is km2, etc
        self.us = [solver.WavefieldVector(solver.mesh, dtype=solver.dtype) for x in xrange(3)]
        self.step = 0
        self.time_dir_bc = None 

        
        nx = self.solver.mesh.x.n
        nz = self.solver.mesh.z.n

        self.time_dir_bc = time_dir_bc_data[0].copy()  
        self.node_order = time_dir_bc_data[1] #should be in order
        self.n_boundary_nodes = self.node_order.size

    def advance(self):
        self.step += 1 #increment current time step. Do this before updating boundary, so you will correct the wavefield at the boundary for the first time at the first step (at step 0 your wavefield should be good, which is uniform 0)
        self.update_wavefield_boundary()
        self.update_dir_bc_future()

        self.us[-1] *= 0
        self.us.insert(0, self.us.pop(-1))
        


    def update_wavefield_boundary(self):
        self.us[0].primary_wavefield[self.node_order,0] = self.time_dir_bc[:,self.step]

    def update_dir_bc_future(self):
        print "Should update the future boundary conditions here"

    @property
    def kp1(self):
        return self.us[0]

    @kp1.setter
    def kp1(self, arg):
        self.us[0] = arg

    @property
    def k(self):
        return self.us[1]

    @k.setter
    def k(self, arg):
        self.us[1] = arg

    @property
    def km1(self):
        return self.us[2]

    @km1.setter
    def km1(self, arg):
        self.us[2] = arg
        
    @property
    def step_index(self):
        return self.step
    
    @step_index.setter
    def step_index(self, arg):
        self.step = arg
    
    @property
    def time_dirichlet_bc_array(self):
        return self.time_dir_bc
    
    @time_dirichlet_bc_array.setter
    def time_dirichlet_bc_array(self, arg):
        self.time_dir_bc = arg

@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticTimeScalarBase(ConstantDensityAcousticTimeBase):

    _local_support_spec = {'equation_formulation': 'scalar',
                           'temporal_integrator': 'leap-frog',
                           'temporal_accuracy_order': 2}

    def __init__(self, mesh, **kwargs):

        self.A_km1 = None
        self.A_k   = None
        self.A_f   = None

        self.temporal_accuracy_order = 2

        ConstantDensityAcousticTimeBase.__init__(self, mesh, **kwargs)

    def time_step(self, solver_data, rhs_k, rhs_kp1):
        u_km1 = solver_data.km1
        u_k   = solver_data.k
        u_kp1 = solver_data.kp1

        f_bar = self.WavefieldVector(self.mesh, dtype=self.dtype)
        f_bar.u += rhs_k

        u_kp1 += self.A_k*u_k.data + self.A_km1*u_km1.data + self.A_f*f_bar.data

    _SolverData = _ConstantDensityAcousticTimeScalar_SolverData

    def SolverData(self, time_dir_bc_data, *args, **kwargs):
        return self._SolverData(self, time_dir_bc_data, self.temporal_accuracy_order, **kwargs)
