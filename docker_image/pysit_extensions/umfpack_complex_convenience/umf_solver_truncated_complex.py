#same idea as umf_solver_complex. That routine works with the general factory datastructure for helmholtz solvers. But the truncated solver is different and needs a slightly different approach.
from pysit_extensions.truncated_domain_helmholtz_solver.truncated_helmholtz_solver import ConstantDensityAcousticFrequencyScalar_2D_truncated_domain
from pysit.util import ConstructableDict
from pysparse.direct import umfpack
from convert_csr_mat import csr_scp_lil_pysp
import numpy as np

class umfpack_wrapper_truncated:
    umfdict = None #initial value
    
    def __init__(self, truncated_mesh_collection, sparse_greens_matrix):
        #Create the solver the normal way. This will not generate the correct constructabledict 'self.solvers'. So we need to overwrite that. Hacky...
        self.solver = ConstantDensityAcousticFrequencyScalar_2D_truncated_domain(truncated_mesh_collection, sparse_greens_matrix)

        self.solver.solvers = ConstructableDict(self.factorize_at_freq) #This function will be used. 
        
    def factorize_at_freq(self, freq):
        #This function follows the same idea as the 'factorized' routine of scipy.sparse.linalg.dsolve.linsolve
        
        if self.umfdict is None: #umfdict contains the umf objects for every frequency.
            self.umfdict = dict()
        
        print "Converting matrix at freq %f \n"%freq
        [A_real_pysp, A_imag_pysp] = csr_scp_lil_pysp(self.solver.linear_operators[freq]) #Takes the csr matrix, returned from the function _build_helmholtz_operator
        
        #If memory consumption of the sparse unfactorized matrices becomes limiting, investigate if it is possible to remove solver.linear_operators[nu] now.
        #(And the K, C, M matrices it is made from?).  
        
        print "Starting factorizing using UMFPACK at freq %f \n"%freq
        self.umfdict[freq] = umfpack.factorize_complex(A_real_pysp, A_imag_pysp)
        print "Finished factorization"
        
        del A_real_pysp, A_imag_pysp #Not sure if they are actually removed that easily. They should be deleted automatically when they go out of scope. Whether that happens, I'm not sure. This del may not result in their deletion either.
        
        def solve(b):
            #b is a complex128 vector. The solve routine of umfpack expects a vector with real and imaginary part (two float64 vectors).
            if b.dtype != "complex128":
                raise Exception("Efficient only for complex128 systems. Need to write more optimal routines for other dtypes.")
            
            #have to make the arrays contiguous in memory. So can't just pass in complex_vector.real and complex_vector.imag (if you do complex_vector.real.flags or complex_vector.imag.flags you see it is not contiguous)
            x_real = np.zeros(b.size, dtype='float64')
            x_imag = np.zeros(b.size, dtype='float64')
            
            b_real = np.ascontiguousarray(np.copy(b.real))
            b_imag = np.ascontiguousarray(np.copy(b.imag))
            
            self.umfdict[freq].solve(b_real, b_imag, x_real, x_imag)
            
            x = np.zeros(b.size, dtype='complex128')
            x.real = x_real
            x.imag = x_imag
            
            return x
        
        return solve 
        
    
    def clear_factors(self): #In case you need the factors cleaned immediately. The garbage collector sometimes keeps too much in memory.
        print "Trying to clear the umfpack factors. The solver object itself will eventually be collected by the garbarge collector. But it will have minimal memory consumption in the meantime.\n"
        for freq in self.umfdict.keys():
            self.umfdict[freq].clear_arrays() #Clears both the factorsand all the temporary arrays allocated (related to indexing of the sparse matrix that was factorized). 
        
        self.umfdict = None
        self.solver.solvers.clear()