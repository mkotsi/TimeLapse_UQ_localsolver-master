import numpy as np
import scipy.sparse as spsp
from pysit.core.domain import Dirichlet

from pysit.solvers.wavefield_vector import *
from pysit_extensions.acc2_space_numpy_time_dependent_dirichlet.constant_density_acoustic_time_scalar_base import * #the solver base did not change, but the solver_data includes the current time step and the time/space dependent boundary. I do not want the default PySIT ones right now.

from pysit.util import Bunch
from pysit.util import PositiveEvenIntegers
from pysit.util.derivatives import build_derivative_matrix
from pysit.util.matrix_helpers import build_sigma, make_diag_mtx

from pysit.util.solvers import inherit_dict

from pysit.solvers.constant_density_acoustic.time.scalar.constant_density_acoustic_time_scalar_cpp import (
    constant_density_acoustic_time_scalar_2D_2os,
    constant_density_acoustic_time_scalar_2D_4os,
    constant_density_acoustic_time_scalar_2D_6os,
    constant_density_acoustic_time_scalar_2D_8os)


__all__ = ['ConstantDensityAcousticTimeScalar_2D_numpy',
           'ConstantDensityAcousticTimeScalar_2D_cpp']

__docformat__ = "restructuredtext en"


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticTimeScalar_2D(ConstantDensityAcousticTimeScalarBase):

    _local_support_spec = {'spatial_discretization': 'finite-difference',
                           'spatial_dimension': 2,
                           'boundary_conditions': ['pml', 'pml-sim', 'dirichlet']}

    def __init__(self, mesh, spatial_accuracy_order, **kwargs):

        self.operator_components = Bunch()

        self.spatial_accuracy_order = spatial_accuracy_order

        ConstantDensityAcousticTimeScalarBase.__init__(self,
                                                       mesh,
                                                       spatial_accuracy_order=spatial_accuracy_order,
                                                       **kwargs)

       


    def _rebuild_operators(self):

        oc = self.operator_components

        built = oc.get('_base_components_built', False)

        # build the static components
        if not built:
            oc.sx = build_sigma(self.mesh, self.mesh.x)
            oc.sz = build_sigma(self.mesh, self.mesh.z)

            oc.sxPsz = oc.sx + oc.sz
            oc.sxsz = oc.sx * oc.sz

            oc._base_components_built = True

    class WavefieldVector(WavefieldVectorBase):

        aux_names = [] #No auxillary wavefields phix/phiz in my case since dirichley 


@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticTimeScalar_2D_numpy(ConstantDensityAcousticTimeScalar_2D):

    _local_support_spec = {'kernel_implementation': 'numpy',
                           'spatial_accuracy_order': PositiveEvenIntegers,
                           'precision': ['single', 'double']}

    def __init__(self, mesh, spatial_accuracy_order=4, **kwargs):
        #check is second order accuracy is used. that is all i can do right now. maybe add higher orders later but i would need more green's functions.
        if spatial_accuracy_order != 2:
            raise Exception('This time dependent dirichlet boundary condition implementation only works with second order accuracy right now.') 

        #Check if all Dirichlet
        if type(mesh.domain.x['lbc']) != Dirichlet:
            raise Exception('Need Dirichlet BC')
        if type(mesh.domain.x['rbc']) != Dirichlet:
            raise Exception('Need Dirichlet BC')
        if type(mesh.domain.z['lbc']) != Dirichlet:
            raise Exception('Need Dirichlet BC')
        if type(mesh.domain.z['rbc']) != Dirichlet:
            raise Exception('Need Dirichlet BC')
        
        ConstantDensityAcousticTimeScalar_2D.__init__(self,
                                                      mesh,
                                                      spatial_accuracy_order=spatial_accuracy_order,
                                                      **kwargs)
        #THIS MAY NOT WORK ? 
        if 'dt_factor' in kwargs.keys(): #Right now I use second order accuracy, so maybe I want to use smaller timesteps. In the end it may be more efficient to just do 4th order accurate and compute some extra Green's functions.
            self.cfl_safety = self.cfl_safety*kwargs['dt_factor'] #This will change dt with the same factor later when a model is set and the corresponding timestep is determined. Defined in 'ConstantDensityAcousticTimeBase'
        
    def _zero_mat_rows(self, mat,nodes):    #Basically, the rows corresponding to the dirichlet boundaries will be turned into 0 to indicate that the final value on the boundary should be changed.
            print "Zeroing parts of matrix, this current approach is not super efficient"
            
            #Not sure if this approach is super efficient.
            if type(mat) is spsp.csr_matrix: 
                ret_mat = spsp.csr_matrix(mat) #copy input)
            
                #set all elements to zero in the required rows
                for i in nodes: #not super efficient, looping 
                    row = ret_mat.getrow(i)
                    for j in row.indices: #number of nonzero elements in the row that needs to be put to zero 
                        ret_mat[i,j] = 0.0
            
                ret_mat.eliminate_zeros() #remove the entries from the sparse matrix that are 0.0. Will result in slightly faster matrix-vector products
                
            elif type(mat) is spsp.dia_matrix:
                ret_mat = spsp.csr_matrix(mat) #copy input into intermediate csr_matrix (dia_matrix does not allow assignment)
            
                #set all elements to zero in the required rows
                for i in nodes: #not super efficient, looping 
                    row = ret_mat.getrow(i)
                    for j in row.indices: #number of nonzero elements in the row that needs to be put to zero 
                        ret_mat[i,j] = 0.0
            
                ret_mat.eliminate_zeros() #remove the entries from the sparse matrix that are 0.0. Will result in slightly faster matrix-vector products
                ret_mat.todia()           #this seems to fill-in diagonal zeros in the sparse matrix for some reason
            else:
                raise Exception('Unknown input data type')
        
            return ret_mat
        
    def _rebuild_operators(self):

        ConstantDensityAcousticTimeScalar_2D._rebuild_operators(self)


        dof = self.mesh.dof(include_bc=True)

        oc = self.operator_components

        built = oc.get('_numpy_components_built', False)

        # build the static components
        if not built:
            nx = self.mesh.x.n
            nz = self.mesh.z.n

            #We actually don't need to correct the operators for the boundary nodes. But it will make the final matrix a little more sparse and therefore faster. Also, it makes is more clear that something needs to happen to the boundary. Any failure to do so can easily be seen by eye.
            self.left_boundary_nodes =  np.arange(0,nz)             #includes left corner dirichlet nodes
            self.right_boundary_nodes = np.arange((nx-1)*nz,nx*nz)  #uncludes right corner dirichlet nodes
            self.top_boundary_nodes = np.arange(nz,(nx-1)*nz,nx)    #does not include corner dirichlet nodes
            self.bot_boundary_nodes = np.arange(2*nz-1,nx*nz-1,nx)  #does not include corner dirichlet nodes
            self.all_boundary_nodes = np.concatenate([self.left_boundary_nodes, self.right_boundary_nodes, self.top_boundary_nodes, self.bot_boundary_nodes])
            
            # build laplacian
            oc.L = build_derivative_matrix(self.mesh,
                                           2,
                                           self.spatial_accuracy_order)

            
            #Set the rows corresponding to the boundary nodes in oc.L to zero (could be any junk value). This is not necessary, but will indicate that I will have to correct the boundary values at the end of the time increment.
            oc.L = self._zero_mat_rows(oc.L, self.all_boundary_nodes)
            
            # build other useful things
            oc.I = spsp.eye(dof, dof)
            oc.empty = spsp.csr_matrix((dof, dof))

            oc._numpy_components_built = True
        
        C = self.model_parameters.C #just wavespeed. Don't confuse with the matrix C that used to be here for a while
        oc.m = make_diag_mtx((C**-2).reshape(-1,))

        K = -oc.L

        M = oc.m / self.dt**2
        M_bc_zero = self._zero_mat_rows(M,self.all_boundary_nodes) #put boundary values to zero, not really necessary because boundary values will be overwritten anyway. But easier to see that something must happen when the algorithm will default them to zero so you know you need to correct them.
        Stilde_inv = M
        Stilde_inv.data = 1./Stilde_inv.data

        self.A_k   = Stilde_inv*(2*M_bc_zero - K)
        self.A_km1 = -1*Stilde_inv*(M_bc_zero)
        self.A_f   = Stilde_inv

    def time_step(self, solver_data, rhs_k, rhs_kp1):
        u_km1 = solver_data.km1
        u_k   = solver_data.k
        u_kp1 = solver_data.kp1

        f_bar = self.WavefieldVector(self.mesh, dtype=self.dtype)
        f_bar.u += rhs_k

        u_kp1 += self.A_k*u_k.data + self.A_km1*u_km1.data + self.A_f*f_bar.data

@inherit_dict('supports', '_local_support_spec')
class ConstantDensityAcousticTimeScalar_2D_cpp(ConstantDensityAcousticTimeScalar_2D):

    _local_support_spec = {'kernel_implementation': 'cpp',
                           'spatial_accuracy_order': [2, 4, 6, 8],
                           'precision': ['single', 'double']}

    _cpp_funcs = {2: constant_density_acoustic_time_scalar_2D_2os,
                  4: constant_density_acoustic_time_scalar_2D_4os,
                  6: constant_density_acoustic_time_scalar_2D_6os,
                  8: constant_density_acoustic_time_scalar_2D_8os}

    def time_step(self, solver_data, rhs_k, rhs_kp1):

        lpmlx = self.mesh.x.lbc.sigma if self.mesh.x.lbc.type is 'pml' else np.array([])
        rpmlx = self.mesh.x.rbc.sigma if self.mesh.x.rbc.type is 'pml' else np.array([])

        lpmlz = self.mesh.z.lbc.sigma if self.mesh.z.lbc.type is 'pml' else np.array([])
        rpmlz = self.mesh.z.rbc.sigma if self.mesh.z.rbc.type is 'pml' else np.array([])

        nx, nz = self.mesh.shape(include_bc=True, as_grid=True)

        self._cpp_funcs[self.spatial_accuracy_order](solver_data.km1.u,
                                                     solver_data.k.Phix,
                                                     solver_data.k.Phiz,
                                                     solver_data.k.u,
                                                     self.model_parameters.C,
                                                     rhs_k,
                                                     lpmlx, rpmlx,
                                                     lpmlz, rpmlz,
                                                     self.dt,
                                                     self.mesh.x.delta,
                                                     self.mesh.z.delta,
                                                     nx, nz,
                                                     solver_data.kp1.Phix,
                                                     solver_data.kp1.Phiz,
                                                     solver_data.kp1.u)
