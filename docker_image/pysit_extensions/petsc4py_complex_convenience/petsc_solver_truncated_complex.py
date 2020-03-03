#Simple modification of umfpack wrapper. Right now only a single shot is passed to petsc at a time. In reality you want to pass multiple at the same time. Will require a bit of rewriting. See pysit frequency example where petsc is used and multiple shots are passed.
#same idea as umf_solver_complex. That routine works with the general factory datastructure for helmholtz solvers. But the truncated solver is different and needs a slightly different approach.
from pysit_extensions.truncated_domain_helmholtz_solver.truncated_helmholtz_solver import ConstantDensityAcousticFrequencyScalar_2D_truncated_domain
from pysit.util import ConstructableDict
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from pysit.util.wrappers.petsc import PetscWrapper
import numpy as np
import warnings

class petsc_wrapper_truncated:
    petscdict = None #initial value
    
    def __init__(self, truncated_mesh_collection, sparse_greens_matrix, petsc='mumps'):
        #The petsc argument takes the type petsc solver you want
        
        #Create the solver the normal way. This will not generate the correct constructabledict 'self.solvers'. So we need to overwrite that. Hacky...
        self.solver = ConstantDensityAcousticFrequencyScalar_2D_truncated_domain(truncated_mesh_collection, sparse_greens_matrix)
        self.solver.solvers = ConstructableDict(self.factorize_at_freq) #This function will be used. 
        self.petsc = petsc
        
    def factorize_at_freq(self, freq):
        #This function follows the same idea as the 'factorized' routine of scipy.sparse.linalg.dsolve.linsolve
        
        if self.petscdict is None: #umfdict contains the umf objects for every frequency.
            self.petscdict = dict()
        
        #The matrix
        A = self.solver.linear_operators[freq]
        print "Factorizing matrix"
        wrapper = PetscWrapper()
        linear_solver = wrapper.factorize(A, self.petsc, PETSc.COMM_WORLD)        
        
        self.petscdict[freq] = linear_solver
        
        def solve(b):
            warnings.warn('Right now only solving for one shot at a time. Petsc can take many RHS at the same time. Need to change code...')
            
            #b is a complex128 vector. The solve routine of umfpack expects a vector with real and imaginary part (two float64 vectors).
            if b.dtype != "complex128":
                raise Exception("Efficient only for complex128 systems. Need to write more optimal routines for other dtypes.")

            ndof = b.size #Assuming b is length of single shot RHS vector

            # creating the B rhs Matrix. Borrowed from the standard CDA solver, its petsc solve routine            
            B = PETSc.Mat().createDense([ndof, 1]) #right now 1 refers to us passing only 1 shot at a time
            B.setUp()
            B.setValues(range(0, ndof), [0], b) #only set up one shot for now
            B.assemblyBegin()
            B.assemblyEnd()
            
            x = self.petscdict[freq](B.getDenseArray())
            
            return x.flatten() #NOW THAT I AM USING ONE SHOT AT A TIME I AM STILL EXPECTING A FLAT RETURN VECTOR
        
        return solve 
        
    
    def clear_factors(self): #In case you need the factors cleaned immediately. The garbage collector sometimes keeps too much in memory.
        print "Trying to clear the petsc factors. The solver object itself will eventually be collected by the garbarge collector. But it will have minimal memory consumption in the meantime.\n"
#         for freq in self.umfdict.keys():
#             self.umfdict[freq].clear_arrays() #Clears both the factorsand all the temporary arrays allocated (related to indexing of the sparse matrix that was factorized). 
        
        #I think factors should be caught by themselves
        self.petscdict = None
        self.solver.solvers.clear()