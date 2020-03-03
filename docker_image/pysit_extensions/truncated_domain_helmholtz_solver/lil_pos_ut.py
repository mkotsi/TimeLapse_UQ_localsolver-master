import scipy.sparse as spsp
import numpy as np

#need for overloaded __getitem__ routine
#FROM A PROGRAMMING POINT OF VIEW THIS SELECTIVE OVERLOADING OF THE SPARSE LIL_MATRIX IS REALLY NOT SUCH A GREAT IDEA.
#IT IS TOO SENSITIVE TO CHANGES IN IMPLEMENTATION OF SCIPY'S LIL_MATRIX IMPLEMENTATION...
#AT THE TIME OF CODING IT WAS THE EASIEST THING TO DO THOUGH.
from scipy.sparse.sputils import getdtype, isshape, issequence, isscalarlike 
from warnings import warn
from scipy.sparse.base import SparseEfficiencyWarning
from scipy.sparse import _csparsetools
import scipy.io as spio

class lil_pos_ut(spsp.lil_matrix): #upper triangular. respect reciprocity
    def __init__(self, full_mesh, dtype = 'float64', do_sanity_checks = True):

        #Should make some functions that regenerate x_min, x_max, z_min, z_max, dx, dz etc when full mesh and full domain change, although I see no reason why you would change the full domain and full mesh
        self.full_mesh = full_mesh
        self.full_domain = full_mesh.domain
        self.do_sanity_checks = do_sanity_checks
        
        if self.full_mesh.dim != 2:
            raise Exception('For now I only work with 2D meshes')        
        
        self.x_min = self.full_domain.x['lbound']
        self.x_max = self.full_domain.x['rbound']
        self.z_min = self.full_domain.z['lbound']
        self.z_max = self.full_domain.z['rbound']        
        
        self.n_nodes_x = self.full_mesh.x['n']
        self.n_nodes_z = self.full_mesh.z['n']
        self.dx = (self.x_max - self.x_min)/(self.n_nodes_x-1)
        self.dz = (self.z_max - self.z_min)/(self.n_nodes_z-1)
        
        shape = (self.n_nodes_z*self.n_nodes_x, self.n_nodes_z*self.n_nodes_x)
        spsp.lil_matrix.__init__(self, shape, dtype = dtype)
        
    def turn_node_nr_into_position(self, node_nr): #Input is node number on grid, which is row number in matrix. Works for single entry of array of node_nr. Returns tuple or list of tuples.
        grid_col_nr = np.floor(node_nr/self.n_nodes_z)
        grid_row_nr = node_nr - grid_col_nr*self.n_nodes_z
        
        x_pos_rel = grid_col_nr*self.dx
        z_pos_rel = grid_row_nr*self.dz
        
        x_pos = self.x_min + x_pos_rel
        z_pos = self.z_min + z_pos_rel
    
        if (type(x_pos) != list and type(x_pos) != np.ndarray) or (type(z_pos) != list and type(z_pos) != np.ndarray):
            ret = (x_pos, z_pos)
        else:
            ret = zip(x_pos,z_pos)

        return ret
    
    def get_connecting_positions(self, pos_from): #this gives all the positions that have a connection with pos_from
        [row,row] = self._positions_to_entry(pos_from, pos_from)
        [rows, cols] = self[row,(row+1):].nonzero() #rows will be [0,0, .... ,0], but cols contain indices where there is a connection. (row+1): to skip the diagonal. Will be added when doing columns
        cols = cols + (row+1) #because the columns of the matrix slice '(row+1):' start counting at 0 while the first entry is actually (row+1)  
        positions_from_row = self.turn_node_nr_into_position(cols)
        
        #THE COLUMN SLICE IS NOT EFFICIENT IN LIL. IF THIS TURNS OUT TO BE A SPEED BOTTLENECK, THEN I SHOULD CONVERT THE lil_pos_ut MATRIX INTO A csr_pos_ut MATRIX.
        #THE COMMAND BELOW WILL GENERATE A WARNING 
        [rows, cols] = self[:,row].nonzero() #cols will be [0,0, .... ,0], but rows contain indices where there is a connection.
        positions_from_col = self.turn_node_nr_into_position(rows)
        
        positions_from_col.extend(positions_from_row) #Start with col, so increasing node number. Makes bookkeeping easier in some cases. List of tuples!
        #Extending changing the original object. So the result is stored in positions_from_col
        return positions_from_col 
    
    def get_value_by_position(self, pos_from, pos_to):
        index = self._positions_to_entry(pos_from, pos_to)
        return self[index]
            
    def set_value_by_position(self, pos_from, pos_to, val):
        index = self._positions_to_entry(pos_from, pos_to)
        self[index] = val

    def position_to_node_nr(self, pos):
        x_index = int(round((pos[0]-self.x_min)/self.dx))
        z_index = int(round((pos[1]-self.z_min)/self.dz))
        full_mesh_nodenr =int( self.n_nodes_z*(x_index) + z_index)
        return full_mesh_nodenr
                
    def _positions_to_entry(self, pos_from, pos_to):
        if self.do_sanity_checks:
            self._sanity_check(pos_from, pos_to) #expensive to do at every lookup, but I am willing to do it for now. Can remove this later
                
        x_index_from = int(round((pos_from[0]-self.x_min)/self.dx))
        z_index_from = int(round((pos_from[1]-self.z_min)/self.dz))

        x_index_to = int(round((pos_to[0]-self.x_min)/self.dx))
        z_index_to = int(round((pos_to[1]-self.z_min)/self.dz))
        
        full_mesh_nodenr_from = self.position_to_node_nr(pos_from)
        full_mesh_nodenr_to = self.position_to_node_nr(pos_to)   

        ret = (int(0), int(0))
        
        #give upper triangular index (including diagonal)        
        if full_mesh_nodenr_from <= full_mesh_nodenr_to:
            ret = (full_mesh_nodenr_from, full_mesh_nodenr_to)
        else:
            ret = (full_mesh_nodenr_to, full_mesh_nodenr_from)
        
        return ret
     
    def _sanity_check(self, pos_from, pos_to):
        #check if x_from is in range
        x_from = pos_from[0]
        z_from = pos_from[1]

        x_to = pos_to[0]
        z_to = pos_to[1]
        
        if x_from < self.x_min or x_from > self.x_max:
            raise Exception("x_from outside domain")
            
        if x_to < self.x_min or x_to > self.x_max:
            raise Exception("x_to outside domain")
            
        if z_from < self.z_min or z_from > self.z_max:
            raise Exception("z_from outside domain")
            
        if z_to < self.z_min or z_to > self.z_max:
            raise Exception("z_to outside domain")
        
        #Not sure if below is sensitive to rounding errors. Want to make sure that position coincides with node position. I suspect it is. If it gives problems and I remove this, do the same in truncate_domain.py
        
        #modulus is not so great for large numbers. Even with double precision errors seem to occur sometimes. In truncate_domain I first brought down the value to the range of 0 - spacing. Then I take the modulus and use an epsilon
        eps = 1e-3 #allow some slack up to a millimeter or so. Sometimes there are some rounding errors
        
        x_from_remainder = x_from - int(np.floor((x_from-self.x_min)/self.dx))*self.dx - self.x_min
        z_from_remainder = z_from - int(np.floor((z_from-self.z_min)/self.dz))*self.dz - self.z_min
        
        x_to_remainder = x_to - int(np.floor((x_to-self.x_min)/self.dx))*self.dx - self.x_min
        z_to_remainder = z_to - int(np.floor((z_to-self.z_min)/self.dz))*self.dz - self.z_min

        if np.mod(x_from_remainder + eps, self.dx) > eps and np.mod(x_from_remainder, self.dx) > eps:
            raise Exception("the provided x_from position does not coincide with a node pair")
        if np.mod(z_from_remainder + eps, self.dz) > eps and np.mod(z_from_remainder, self.dz) > eps:
            raise Exception("the provided z_from position does not coincide with a node pair")
        if np.mod(x_to_remainder + eps, self.dx) > eps and np.mod(x_to_remainder, self.dx) > eps:
            raise Exception("the provided x_to position does not coincide with a node pair")
        if np.mod(z_to_remainder + eps, self.dz) > eps and np.mod(z_to_remainder, self.dz) > eps:
            raise Exception("the provided z_to position does not coincide with a node pair")



#IN NEW SCIPY VERSION THE __GETITEM__ ROUTINE NO LONGER CAUSED PROBLEMS. DID NOT HAVE TO OVERLOAD ANYMORE.
#I LEAVE THE OLD HACK DOWN HERE COMMENTED OUT, SO I CAN USE IT AGAIN IF PROBLEM REAPPEARS AT SOME POINT IN THE FUTURE.
#GOOD EXAMPLE OF WHY INHERITING FROM LIL_MATRIX WAS NOT A GOOD IDEA...
         
#     def __getitem__(self, index): #overloaded from lil. When slicing it called __getitem__ from lil. This called self.__class__ which returned lil_pos_ut objects instead of lil_matrix. The problem with that is that self[:,i] would fail because it would generate a new lil_pos_ut matrix but would not pass the right values for mesh etc to the constructor.
#         """Return the element(s) index=(i, j), where j may be a slice.
#         This always returns a copy for consistency, since slices into
#         Python lists return copies.
#         """
# 
#         # Scalar fast path first
#         if isinstance(index, tuple) and len(index) == 2:
#             i, j = index
#             # Use isinstance checks for common index types; this is
#             # ~25-50% faster than isscalarlike. Other types are
#             # handled below.
#             if ((isinstance(i, int) or isinstance(i, np.integer)) and
#                 (isinstance(j, int) or isinstance(j, np.integer))):
#                 v = _csparsetools.lil_get1(self.shape[0], self.shape[1],
#                                            self.rows, self.data,
#                                            i, j)
#                 return self.dtype.type(v)
# 
#         # Utilities found in IndexMixin
#         i, j = self._unpack_index(index)
# 
#         # Proper check for other scalar index types
#         if isscalarlike(i) and isscalarlike(j):
#             v = _csparsetools.lil_get1(self.shape[0], self.shape[1],
#                                        self.rows, self.data,
#                                        i, j)
#             return self.dtype.type(v)
# 
#         i, j = self._index_to_arrays(i, j)
#         if i.size == 0:
#             return spsp.lil_matrix(i.shape, dtype=self.dtype)
# 
#         new = spsp.lil_matrix(i.shape, dtype=self.dtype)
# 
#         i, j = _csparsetools.prepare_index_for_memoryview(i, j)
#         _csparsetools.lil_fancy_get(self.shape[0], self.shape[1],
#                                     self.rows, self.data,
#                                     new.rows, new.data,
#                                     i, j)
#         return new

def read_lil_pos_ut_from_file(full_mesh, mat_filename):
    
    #read mat
    mat_csr = spio.mmread(mat_filename)
    mat_lil = mat_csr.tolil()
    
    mat_dtype = mat_lil.dtype
    mat_lil_pos_ut = lil_pos_ut(full_mesh, dtype=mat_dtype)
    mat_lil_pos_ut[mat_lil.nonzero()[0], mat_lil.nonzero()[1]] = mat_lil[mat_lil.nonzero()[0], mat_lil.nonzero()[1]]
    
    return mat_lil_pos_ut
  
#     def __getitem__(self, index): #overloaded from lil. When slicing it called __getitem__ from lil. This called self.__class__ which returned lil_pos_ut objects instead of lil_matrix. The problem with that is that self[:,i] would fail because it would generate a new lil_pos_ut matrix but would not pass the right values for mesh etc to the constructor.
#         """Return the element(s) index=(i, j), where j may be a slice.
#         This always returns a copy for consistency, since slices into
#         Python lists return copies.
#         """
#         try:
#             i, j = index
#         except (AssertionError, TypeError):
#             raise IndexError('invalid index')
# 
#         if not np.isscalar(i) and np.isscalar(j):
#             warn('Indexing into a lil_matrix with multiple indices is slow. '
#                  'Pre-converting to CSC or CSR beforehand is more efficient.',
#                  SparseEfficiencyWarning)
# 
#         if np.isscalar(i):
#             if np.isscalar(j):
#                 return self._get1(i, j)
#             if isinstance(j, slice):
#                 j = self._slicetoseq(j, self.shape[1])
#             if issequence(j):
#                 return spsp.lil_matrix([[self._get1(i, jj) for jj in j]])
#         elif issequence(i) and issequence(j):
#             return spsp.lil_matrix([[self._get1(ii, jj) for (ii, jj) in zip(i, j)]])
#         elif issequence(i) or isinstance(i, slice):
#             if isinstance(i, slice):
#                 i = self._slicetoseq(i, self.shape[0])
#             if np.isscalar(j):
#                 return spsp.lil_matrix([[self._get1(ii, j)] for ii in i])
#             if isinstance(j, slice):
#                 j = self._slicetoseq(j, self.shape[1])
#             if issequence(j):
#                 return spsp.lil_matrix([[self._get1(ii, jj) for jj in j] for ii in i])
#         else:
#             raise IndexError