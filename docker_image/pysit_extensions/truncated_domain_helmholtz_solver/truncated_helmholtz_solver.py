import scipy.sparse as spsp

from pysit.solvers.wavefield_vector import *
from pysit.core.sources import PointSource

from pysit.solvers.constant_density_acoustic.frequency.constant_density_acoustic_frequency_scalar_2D import *
from pysit_extensions.truncated_domain_helmholtz_solver.collection_objects import *

from pysit.util import Bunch
from pysit.util import PositiveEvenIntegers
from pysit.util.matrix_helpers import build_sigma, make_diag_mtx

from pysit.util.solvers import inherit_dict

import numpy as np
import copy

##########################################################################################################
# INSTEAD OF A MESH I PASS IN A MESH_COLLECTION OBJECT. MESH_COLLECTION HAS A DOMAIN_COLLECTION MEMBER.
# THIS MEMBER CAN BE CALLED THROUGH MESH_COLLECTION.DOMAIN_COLLECTION OR THROUGH MESH_COLLECTION.DOMAIN.
# THIS WAY SOLVER_BASE DOES NOT HAVE TO BE CHANGED. I SOMEHOW NEED TO OVERRIDE 
# 'ModelParameters = ConstantDensityAcousticParameters' AS IS DONE IN ConstantDensityAcousticBase.
# I NEED TO CHANGE IT TO A MODELPARAMETER_COLLECTION, WHICH IS COMPOSED OF ALL THE MODELPARAMETER OBJECTS.
# WHEN ADDING TWO MODELPARAMETER_COLLECTION OBJECTS TOGETHER (OR OTHER OPERATIONS), INVOKE THE MEMBER
# OPERATIONS FOR ALL THE SUBDOMAINS INDIVIDUALLY!
##########################################################################################################

class ConstantDensityAcousticFrequencyScalar_2D_truncated_domain(ConstantDensityAcousticFrequencyScalar_2D):

    _local_support_spec = {'spatial_discretization': 'finite-difference',
                           'spatial_dimension': 2,
                           'spatial_accuracy_order': PositiveEvenIntegers,
                           'boundary_conditions': ['pml', 'pml-sim', 'dirichlet'],
                           'precision': ['single', 'double']}

    def __init__(self,
                 mesh_collection,
                 sparse_greens_matrix,
                 spatial_accuracy_order=2,
                 spatial_shifted_differences=False,
                 do_sanity_checks = True,
                 **kwargs):
        
        if type(mesh_collection) == CartesianMesh: #In this case we are dealing with a single mesh and domain.
            mesh = mesh_collection
            domain_collection = DomainCollection(domain_list = [mesh.domain])
            mesh_collection = MeshCollection(domain_collection, mesh_list = [mesh])

        
        #I should add a frequency data field to sparse_greens_matrix and check if the 'nu' passed to solve matches that frequency when using self.solve()
        
        if do_sanity_checks:
            print "Doing sanity checks on positions. Perhaps slows things down?"
        
        self.do_sanity_checks = do_sanity_checks #True can be expensive, but perhaps good when running for the first time in a problem to catch some errors
        self.sparse_greens_matrix = sparse_greens_matrix
        
        ########################################
        # PRECOMPUTED MAPPING FUNCTIONALITY
        ########################################
        self.have_we_precomputed_mappings_between_pos_and_regular_node_nr = False #REGULAR HERE REFERS TO REGULAR NUMBERING SCHEME (DEPTH FIRST, COLUMN AT A TIME) BUT NOW APPLIED TO THE TRUNCATED MESH(ES)
        self.precomputed_pos_to_regular_node_nr = dict()
        self.precomputed_regular_node_nr_to_pos = dict()
        
        self.have_we_precomputed_mappings_between_pos_and_spiraling_node_nr = False
        self.precomputed_pos_to_spiraling_node_nr = dict()
        self.precomputed_spiraling_node_nr_to_pos = dict()
        
        self.have_we_precomputed_mappings_between_regular_node_nr_and_spiraling_node_nr = False
        self.precomputed_spiraling_node_nr_to_regular_node_nr = None #will be an array
        self.precomputed_regular_node_nr_to_spiraling_node_nr = None #will be an array
        
        self.have_we_precomputed_node_nr_list_of_lists_for_scattered_field_computation = False
        

        self.conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers = dict()
        self.conveniently_stored_greens_functions_for_efficient_rhs_calculation = dict()

        self.have_we_precomputed_conversion_arrays_regular_node_nr_global_2d_indices = False
        
        ########################################
        # END PRECOMPUTED MAPPING FUNCTIONALITY
        ########################################
        
        self.nodes_per_mesh = None
        self.first_reg_node_nr_mesh = None 
        self.nr_outer_layer_nodes_per_mesh = None #number of nodes outer layer for every mesh
        self.nr_inner_layer_nodes_per_mesh = None #number of nodes one layer to the interior for every mesh
        
        #needs green's functions from the shots and the receivers to the outer boundary. Could reuse stuff from self.conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers, but not doing so. That dict() does not include corner nodes for instance. Also only has receiver nodes
        self.mesh_collection = mesh_collection #has the truncated meshes.
        

        print "The truncated Helmholtz solver should only be used for one frequency right now ! For multiple frequencies, the sparse greens function and the way operator K is computed may have to be changed?"

        if spatial_accuracy_order != 2:
            raise Exception('Spatial accuracy should be 2')

        if spatial_shifted_differences == True:
            raise Exception('No shifted spatial differences will be used on truncated domain in order to stay consistent with the interior of the full domain where the normal Laplacian stencil is used') 
        
        #The standard constant density solver got changed. It now specifically checks for "if self.mesh.x.lbc.type == 'pml'"
        #To see if the solver uses a compact PML. First of all, we are not interested in inheriting this part of the behavior 
        #of the standard frequency domain solver. But self.mesh for us is a mesh_collection, and it does not have the 'x' attribute.
        #Therefore the call will fail. To avoid this useless check (for us) from failing the program I will just give the mesh collection
        #a useless attribute 'x'.
        mesh_collection.x = mesh_collection.get_list()[0].x #JUST A HACK SO THAT THE TEST AT THE END OF THE NEXT CONSTRUCTOR WILL NOT FAIL
        
        #Inheriting some of the behavior of a normal frequency domain solver. Overloading some other behavior. In hindsight this is really not clean at all...
        ConstantDensityAcousticFrequencyScalar_2D.__init__(self,
                                                           mesh_collection,
                                                           spatial_accuracy_order=spatial_accuracy_order,
                                                           spatial_shifted_differences=spatial_shifted_differences,
                                                           **kwargs)
        
        self.ModelParameters = ModelParameterCollection #Use model parameter collections now!

    def _rebuild_operators(self):

        #dof = self.mesh.dof(include_bc=True)

        oc = self.operator_components

        built = oc.get('_numpy_components_built', False)

        # build the static components
        if not built:
            # build laplacian
            oc.L = self._build_second_order_laplacian()
            
            self._assemble_K(oc) #The green's function blocks do not change when model changes so should only build once
            self._assemble_C(oc) #empty matrix here
            
            oc._numpy_components_built = True

        oc.m = self._build_model_part_of_matrix()

        self._assemble_M(oc)




        #The final matrix will be constructed by:    -(omega**2)*self.M + omega*1j*self.C + self.K

    def get_nodes_per_mesh(self):
        if not self.have_we_precomputed_mappings_between_pos_and_regular_node_nr: #We need info about the number of nodes per mesh, is computed in self._precompute_regular_truncated_node_number(). 
            self._precompute_mappings_between_pos_and_regular_node_nr()

        return self.nodes_per_mesh

    def get_nr_outer_layer_nodes_per_mesh(self):
        if not self.have_we_precomputed_mappings_between_pos_and_spiraling_node_nr: #The part where the spiraling node numbers are computed, the number of nodes in the outer layer and one layer to the interior are also computed and stored in arrays. Since we need to compute the spiraling node numbers at some point anyway, we might as well do it now if it has not been done before.
            self._precompute_mappings_between_pos_and_spiraling_node_nr()

        return self.nr_outer_layer_nodes_per_mesh

    def convert_pos_to_regular_node_nr(self, pos):
        #pos has the shape (x, z)
        #very similar to the member functions of lil_pos_ut, but cannot easily reuse since they use the grid assigned to that matrix. Perhaps I can move these member functions to a different file so that it can be used here as well
        
        if self.do_sanity_checks:
            in_range = self._check_if_pos_is_on_a_mesh(pos) #OVERHEAD
            if not in_range:
                raise Exception('Pos is not on a mesh!')
 
        if not self.have_we_precomputed_mappings_between_pos_and_regular_node_nr:
            self._precompute_mappings_between_pos_and_regular_node_nr() 
            
        return self.precomputed_pos_to_regular_node_nr[pos]

    def convert_regular_node_nr_to_pos(self, node_nr):
        if not self.have_we_precomputed_mappings_between_pos_and_regular_node_nr:
            self._precompute_mappings_between_pos_and_regular_node_nr()

        return self.precomputed_regular_node_nr_to_pos[node_nr]

    def _precompute_mappings_between_pos_and_regular_node_nr(self):
        if not self.have_we_precomputed_mappings_between_pos_and_regular_node_nr: #only do work if we have not precomputed before
        
            #Storing the number of nodes in each mesh, and the first node number for each mesh
            mesh_list = self.mesh_collection.get_list()
            n_meshes = len(mesh_list)
            self.nodes_per_mesh = np.zeros(n_meshes, dtype='int32')
            self.first_reg_node_nr_mesh = np.zeros(n_meshes, dtype='int32')
        
            #lots of recomputing. Should do this just once. Whenever mesh gets changed (mesh setter function of solver, like ModelParameters), just update them. Future...
            first_reg_node_nr_mesh = 0
            for mesh in mesh_list:
                index = mesh_list.index(mesh)
                x_min = mesh.domain.x['lbound']
                x_max = mesh.domain.x['rbound']
                z_min = mesh.domain.z['lbound']
                z_max = mesh.domain.z['rbound']        
        
                n_nodes_x = mesh.x['n']
                n_nodes_z = mesh.z['n']
            
                n_nodes_mesh = n_nodes_x*n_nodes_z
                self.nodes_per_mesh[index] = n_nodes_mesh
                self.first_reg_node_nr_mesh[index] = first_reg_node_nr_mesh 

                x_vals = np.linspace(x_min,x_max,n_nodes_x)
                z_vals = np.linspace(z_min,z_max,n_nodes_z) 
        
                for ix in xrange(x_vals.size):
                    for iz in xrange(z_vals.size):
                        x = x_vals[ix]
                        z = z_vals[iz]
                
                        node_nr = first_reg_node_nr_mesh + n_nodes_z*ix + iz
                        pos = (x,z)
                        
                        self.precomputed_pos_to_regular_node_nr[pos] = node_nr
                        self.precomputed_regular_node_nr_to_pos[node_nr] = pos
                    
                first_reg_node_nr_mesh += n_nodes_mesh #update for next iteration
            
            self.have_we_precomputed_mappings_between_pos_and_regular_node_nr = True
            print "Finished precomputing the mapping between pos and regular node nr."

    def convert_pos_to_spiraling_node_nr(self, pos):
        #pos has the shape (x, z)
        #useful function to have for making the laplacian with this node numbering
        #can just march over the grid with location and address the neighbours through this function.
        
        if self.do_sanity_checks:
            in_range = self._check_if_pos_is_on_a_mesh(pos) #OVERHEAD
            if not in_range:
                raise Exception('Pos is not on a mesh!')
        
        if not self.have_we_precomputed_mappings_between_pos_and_spiraling_node_nr: #if not yet precomputed and stored in dict
            self._precompute_mappings_between_pos_and_spiraling_node_nr()
            
        #at this point it has been precomputed
        return self.precomputed_pos_to_spiraling_node_nr[pos]

    def convert_spiraling_node_nr_to_pos(self, node_nr):
        if not self.have_we_precomputed_mappings_between_pos_and_spiraling_node_nr:
            self._precompute_mappings_between_pos_and_spiraling_node_nr()

        return self.precomputed_spiraling_node_nr_to_pos[node_nr]

    def _precompute_mappings_between_pos_and_spiraling_node_nr(self):
        if not self.have_we_precomputed_mappings_between_pos_and_spiraling_node_nr:
            
            mesh_list = self.mesh_collection.get_list()
            n_meshes = len(mesh_list)
            self.nr_outer_layer_nodes_per_mesh = np.zeros(n_meshes, dtype='int32') #number of nodes outer layer for every domain
            self.nr_inner_layer_nodes_per_mesh = np.zeros(n_meshes, dtype='int32') #number of nodes one layer to the interior for every domain
            
            #The numbering will go like this. We start at the top left of the first domain, and count all the nodes anti-clockwise on the outer layer. Those are the first nodes. Then we continue numbering the outer layer in the same way in all the other domains. Then we do the same thing one layer to the interior in every domain. Finally, we spiral inwards all the way, one domain at a time.
            #First fill the nr_outer_layer_nodes_per_mesh and nr_inner_layer_nodes_per_mesh arrays. With that knowledge we can use the above mentioned numbering scheme while being able to move over one grid at a time. 
            for mesh in mesh_list:
                index = mesh_list.index(mesh)
                n_nodes_x = mesh.x['n']
                n_nodes_z = mesh.z['n']
                
                if n_nodes_x <= 3 or n_nodes_z <= 3:
                    raise Exception("Counting the number of inner nodes in this case does not work according to the equation below. Maybe other complications as well ? Better to abort.")
                
                self.nr_outer_layer_nodes_per_mesh[index] = 2*(n_nodes_x + n_nodes_z - 2)
                self.nr_inner_layer_nodes_per_mesh[index] = 2*((n_nodes_x-2) + (n_nodes_z-2) - 2) 
                
            for mesh in mesh_list:
                
                index = mesh_list.index(mesh)
                
                x_min = mesh.domain.x['lbound']
                x_max = mesh.domain.x['rbound']
                z_min = mesh.domain.z['lbound']
                z_max = mesh.domain.z['rbound']        
            
                n_nodes_x = mesh.x['n']
                n_nodes_z = mesh.z['n']
       
                
                nr_outer_nodes = self.nr_outer_layer_nodes_per_mesh[index]
                nr_inner_nodes = self.nr_inner_layer_nodes_per_mesh[index]
            
                #Use this grid to find out where you can walk in spiraling shape. Size of real grid with a boundary of 'False' around it that will prevent you from going there
                checkgrid = np.ones((n_nodes_z+2, n_nodes_x+2),dtype=bool) 
                checkgrid[0,:] = False; checkgrid[-1,:] = False; checkgrid[:,0] = False; checkgrid[:,-1] = False;
            
                move_direction = 'down'
      
                spiral_node_nr_this_mesh = 0 #this is the node number I count in my spiraling numbering scheme
            
                #checkgrid has a 1 layer padding of False around it. Start at the top-left point in this grid that is True, which is 1,1
            
                #START TOP LEFT AND THEN SPIRAL INWARD
                #left grid: initial checkgrid, right, numbers corresponding to algorithm below
            
                #F F F F F                    * * * * *
                #F T T T F                    * 0 7 6 *
                #F T T T F                    * 1 8 5 *
                #F T T T F                    * 2 3 4 *
                #F F F F F                    * * * * *
            
                row_index = 1
                col_index = 1
            
                #settings for down
                increment_row = 1
                increment_col = 0
            
                x_vals = np.linspace(x_min,x_max,n_nodes_x)
                z_vals = np.linspace(z_min,z_max,n_nodes_z) 
            
                x = x_vals[0]
                z = z_vals[0]
    
                busy = True
            
            
                #Since we are now dealing with multiple grids we have a mode complicated node numbering scheme as described above the first 'for mesh in mesh_list:' loop in this function.
                #Therefore we need to add some offsets to node_nr. Within the outer layer, we need to offset the node number with the number of outer layer nodes in all the previous domains.
                #When counting within the layer that is one node into the interior, we need to offset the spiral_node_nr_this_mesh by the number of nodes in all the outer layers of all the domains and all the nodes in the layers one node to the interior of all the domains before.
                
                offset_start_outer = np.sum(self.nr_outer_layer_nodes_per_mesh[0:index])
                offset_start_inner = np.sum(self.nr_outer_layer_nodes_per_mesh [:]) + np.sum(self.nr_inner_layer_nodes_per_mesh[0:index])
                offset_start_interior = np.sum(self.get_nodes_per_mesh()[0:index]) +  np.sum(self.nr_outer_layer_nodes_per_mesh[index:]) + np.sum(self.nr_inner_layer_nodes_per_mesh[index:]) #All the nodes in previous meshes plus the number of nodes on the outer layer and one node to the interior of the other meshes (including the current one) 
            
                #I am sure there is a more elegant way... 
                while busy:
                    #Precompute current position to node number
                    pos = (x,z)
                    
                    #Depending on whether we are on outer layer, inner layer or interior, use the corresponding offsets. We need to do this because we have multiple domains now and a numbering scheme as described at the start of this function
                    
                    if spiral_node_nr_this_mesh < nr_outer_nodes: #We are in outer layer
                        spiraling_node_nr = offset_start_outer + spiral_node_nr_this_mesh
                        self.precomputed_pos_to_spiraling_node_nr[pos] = spiraling_node_nr
                        self.precomputed_spiraling_node_nr_to_pos[spiraling_node_nr] = pos
                        
                    elif spiral_node_nr_this_mesh >= nr_outer_nodes and spiral_node_nr_this_mesh < nr_outer_nodes + nr_inner_nodes: #We are in the layer one node to the interior
                        spiraling_node_nr = offset_start_inner + (spiral_node_nr_this_mesh - nr_outer_nodes)
                        self.precomputed_pos_to_spiraling_node_nr[pos] = spiraling_node_nr  
                        self.precomputed_spiraling_node_nr_to_pos[spiraling_node_nr] = pos
                        
                    elif spiral_node_nr_this_mesh >= nr_outer_nodes + nr_inner_nodes:
                        spiraling_node_nr = offset_start_interior + (spiral_node_nr_this_mesh - (nr_outer_nodes + nr_inner_nodes) )
                        self.precomputed_pos_to_spiraling_node_nr[pos] = spiraling_node_nr
                        self.precomputed_spiraling_node_nr_to_pos[spiraling_node_nr] = pos
                    
                    checkgrid[row_index,col_index] = False
                
                    if checkgrid[row_index+increment_row,col_index+increment_col] == False: #This conditional is triggered when the direction has to be changed.
                        if move_direction == 'down':
                            move_direction = 'right'
                            increment_row = 0
                            increment_col = 1
                        elif move_direction == 'right':
                            move_direction = 'up'
                            increment_row = -1
                            increment_col = 0
                        elif move_direction == 'up':
                            move_direction = 'left'
                            increment_row = 0
                            increment_col = -1
                        elif move_direction == 'left':
                            move_direction = 'down'
                            increment_row = 1
                            increment_col = 0
                        else:
                            raise Exception('something went wrong')
                
                        #This was just a suggested direction. If the node in this direction is also 'False' on checkgrid, we have reached the end (the spiral ended up on itself)
                        if checkgrid[row_index+increment_row,col_index+increment_col] == False:
                            busy = False
                            break #I guess I don't even need to set busy then
                
                    spiral_node_nr_this_mesh += 1
                
                    #The row_index and col_index are referring to the checkgrid which is padded by one layer. So the x_vals and z_vals are offset by 1. Therefore the -1.
                    x = x_vals[col_index-1 + increment_col]
                    z = z_vals[row_index-1 + increment_row]
                
                    row_index += increment_row
                    col_index += increment_col
                    
            self.have_we_precomputed_mappings_between_pos_and_spiraling_node_nr = True
            print "Finished precomputing the mapping between pos and spiraling node nr."

    def convert_spiraling_node_nr_to_regular_node_nr(self, node_nr):
        if not self.have_we_precomputed_mappings_between_regular_node_nr_and_spiraling_node_nr:
            self._precompute_mappings_between_regular_node_nr_and_spiraling_node_nr()
            
        return self.precomputed_spiraling_node_nr_to_regular_node_nr[node_nr]
        
    def convert_regular_node_nr_to_spiraling_node_nr(self, node_nr):
        if not self.have_we_precomputed_mappings_between_regular_node_nr_and_spiraling_node_nr:
            self._precompute_mappings_between_regular_node_nr_and_spiraling_node_nr()
            
        return self.precomputed_regular_node_nr_to_spiraling_node_nr[node_nr]

    def return_all_spiraling_node_nr_to_regular_node_nr(self): # !!!! As in the rest of this file, regular refers to regular numbering scheme within truncated region. Not the global numbering scheme.
        if not self.have_we_precomputed_mappings_between_regular_node_nr_and_spiraling_node_nr:
            self._precompute_mappings_between_regular_node_nr_and_spiraling_node_nr()

        return self.precomputed_spiraling_node_nr_to_regular_node_nr #return the entire array here.

    def _precompute_mappings_between_regular_node_nr_and_spiraling_node_nr(self):
        if not self.have_we_precomputed_mappings_between_regular_node_nr_and_spiraling_node_nr:
            mesh_list = self.mesh_collection.get_list()
            total_nodes = np.sum(self.get_nodes_per_mesh())
            
            self.precomputed_spiraling_node_nr_to_regular_node_nr = np.zeros(total_nodes, dtype='int32')
            self.precomputed_regular_node_nr_to_spiraling_node_nr = np.zeros(total_nodes, dtype='int32')
            
            for mesh in mesh_list:
                x_min = mesh.domain.x['lbound']
                x_max = mesh.domain.x['rbound']
                z_min = mesh.domain.z['lbound']
                z_max = mesh.domain.z['rbound']                    
    
                n_nodes_x = mesh.x['n']
                n_nodes_z = mesh.z['n']
                
                x_vals = np.linspace(x_min,x_max,n_nodes_x)
                z_vals = np.linspace(z_min,z_max,n_nodes_z)
                
                for x in x_vals:
                    for z in z_vals:
                        pos = (x,z)
                        spiraling_node_nr = self.convert_pos_to_spiraling_node_nr(pos)
                        regular_node_nr = self.convert_pos_to_regular_node_nr(pos)         
                         
                        self.precomputed_spiraling_node_nr_to_regular_node_nr[spiraling_node_nr] = regular_node_nr 
                        self.precomputed_regular_node_nr_to_spiraling_node_nr[regular_node_nr] = spiraling_node_nr
                
            self.have_we_precomputed_mappings_between_regular_node_nr_and_spiraling_node_nr = True
            print "Finished precomputing the mapping between regular node nr and spiraling node nr."

    def return_outer_spiraling_node_numbers_as_ndarray(self):
        if not self.have_we_precomputed_node_nr_list_of_lists_for_scattered_field_computation:
            self._precompute_node_nr_list_of_lists_for_scattered_field_computation()        

        return self.outer_spiraling_node_numbers_as_ndarray
    
    def return_inner_spiraling_node_numbers_as_ndarray(self):
        if not self.have_we_precomputed_node_nr_list_of_lists_for_scattered_field_computation:
            self._precompute_node_nr_list_of_lists_for_scattered_field_computation()        

        return self.inner_spiraling_node_numbers_as_ndarray
        
    def return_outer_spiraling_node_numbers(self):
        if not self.have_we_precomputed_node_nr_list_of_lists_for_scattered_field_computation:
            self._precompute_node_nr_list_of_lists_for_scattered_field_computation()        
            
        return self.outer_spiraling_node_numbers
    
    def return_inner_spiraling_node_numbers(self):
        if not self.have_we_precomputed_node_nr_list_of_lists_for_scattered_field_computation:
            self._precompute_node_nr_list_of_lists_for_scattered_field_computation()        
            
        return self.inner_spiraling_node_numbers        

    def return_outer_global_node_numbers(self):
        if not self.have_we_precomputed_node_nr_list_of_lists_for_scattered_field_computation:
            self._precompute_node_nr_list_of_lists_for_scattered_field_computation()        
            
        return self.outer_global_node_numbers
    
    def return_inner_global_node_numbers(self):
        if not self.have_we_precomputed_node_nr_list_of_lists_for_scattered_field_computation:
            self._precompute_node_nr_list_of_lists_for_scattered_field_computation()        
            
        return self.inner_global_node_numbers

    
    def _precompute_node_nr_list_of_lists_for_scattered_field_computation(self): 
        #Gets the spiraling node numbers and the global node numbers for the nodes on the boundary that are involved in calculating the scattered field
        #at the boundary of the truncated domain and at the receivers. There would be other ways of gettings these lists as well. 
        #I walk over the sides of every mesh, and exclude the boundary nodes. For the nodes that I encounter I also take the corresponding nodes on the inner boundary.
        #For each side of every mesh I append a list to the list of lists. When constructing the matrix K I need to have the sides separated from each other.
        #The reason for this is that the total field on corner nodes on the layer 'one node to the interior' will be multiplied by two green's functions and then added.
        #So I need to add two green's functions and put them in the matrix for unknown 'total wavefield on interior at corner node'.

        if not self.have_we_precomputed_node_nr_list_of_lists_for_scattered_field_computation:

            mesh_list = self.mesh_collection.get_list()
            
            self.outer_spiraling_node_numbers = []
            self.inner_spiraling_node_numbers = []            
            
            self.outer_global_node_numbers = []
            self.inner_global_node_numbers = []
            
            for mesh in mesh_list:
                x_min = mesh.domain.x['lbound']
                x_max = mesh.domain.x['rbound']
                z_min = mesh.domain.z['lbound']
                z_max = mesh.domain.z['rbound']
    
                n_nodes_x = mesh.x['n']
                n_nodes_z = mesh.z['n']
    
                

                #Generate in the same way as everyone else in this file to make sure no rounding differences occur.
                x_nodes = np.linspace(x_min,x_max,n_nodes_x)
                z_nodes = np.linspace(z_min,z_max,n_nodes_z)
    
                #######################################
                #REQUIRED POSITIONS TO COMPUTE THE SCATTERED FIELD
                #######################################
                outer_boundary_pos = []
                inner_boundary_pos = []
    
                #left boundary nodes required for field
                outer_boundary_pos.append(zip(x_min*np.ones(n_nodes_z-2), z_nodes[1:-1])) #This way I exclude z_max
                inner_boundary_pos.append(zip(x_nodes[1]*np.ones(n_nodes_z-2), z_nodes[1:-1])) 
                
                #right boundary nodes required for field
                outer_boundary_pos.append(zip(x_max*np.ones(n_nodes_z-2), z_nodes[1:-1]))
                inner_boundary_pos.append(zip(x_nodes[-2]*np.ones(n_nodes_z-2), z_nodes[1:-1]))
                
                #bottom boundary nodes required for field
                outer_boundary_pos.append(zip(x_nodes[1:-1], z_max*np.ones(n_nodes_x-2)))
                inner_boundary_pos.append(zip(x_nodes[1:-1], z_nodes[-2]*np.ones(n_nodes_x-2)))             
    
                #top boundary nodes required for field
                outer_boundary_pos.append(zip(x_nodes[1:-1], z_min*np.ones(n_nodes_x-2)))
                inner_boundary_pos.append(zip(x_nodes[1:-1], z_nodes[1]*np.ones(n_nodes_x-2)))
    
                for side_out, side_in in zip (outer_boundary_pos, inner_boundary_pos):
                    len_side = len(side_out) #same as length of side_in
                    
                    side_contribution_outer_spiraling = []
                    side_contribution_inner_spiraling = []
                    side_contribution_outer_global = []
                    side_contribution_inner_global = []
    
                    #Get spiraling node numbers for nodes on the boundaries
                    for i in xrange(len_side):
                        side_contribution_outer_spiraling.append(self.convert_pos_to_spiraling_node_nr(side_out[i]))
                        side_contribution_inner_spiraling.append(self.convert_pos_to_spiraling_node_nr(side_in[i]))
                    
                    #For all the boundary nodes, store their global node numbers (these will be used to index the sparse matrix
                    for i in xrange(len_side):
                        side_contribution_outer_global.append(self.sparse_greens_matrix.position_to_node_nr(side_out[i]))
                        side_contribution_inner_global.append(self.sparse_greens_matrix.position_to_node_nr(side_in[i]))
                
                    self.outer_spiraling_node_numbers.append(side_contribution_outer_spiraling)
                    self.inner_spiraling_node_numbers.append(side_contribution_inner_spiraling)
                    self.outer_global_node_numbers.append(side_contribution_outer_global)
                    self.inner_global_node_numbers.append(side_contribution_inner_global)

            #Make one long array for efficient computation
            self.outer_spiraling_node_numbers_as_ndarray = np.array([], dtype='int')
            self.inner_spiraling_node_numbers_as_ndarray = np.array([], dtype='int')

            for side_nr in xrange(len(self.outer_spiraling_node_numbers)):                     
                self.outer_spiraling_node_numbers_as_ndarray = np.append(self.outer_spiraling_node_numbers_as_ndarray,  self.outer_spiraling_node_numbers[side_nr])
                self.inner_spiraling_node_numbers_as_ndarray = np.append(self.inner_spiraling_node_numbers_as_ndarray,  self.inner_spiraling_node_numbers[side_nr])

            self.have_we_precomputed_node_nr_list_of_lists_for_scattered_field_computation = True

    def _precompute_and_return_efficient_greens_array_for_propagation_to_receivers(self, receiver_positions):
        if type(receiver_positions) == tuple:
            receiver_positions = [receiver_positions] #iterable now
        
        not_yet_precomputed_receiver_positions = []
        
        for pos in receiver_positions: #DETERMINE IF WE STILL NEED TO DO SOME PRECOMPUTATIONS
            if pos not in self.conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers.keys():
                not_yet_precomputed_receiver_positions.append(pos)
        
        
        if len(not_yet_precomputed_receiver_positions) > 0: #ONLY DO WORK IF WE HAVE NOT SET IT BEFORE
            outer_global_node_numbers = self.return_outer_global_node_numbers()
            inner_global_node_numbers = self.return_inner_global_node_numbers()
            
            #THE LISTS ABOVE ARE LISTS OF LISTS. UNLIKE THE MATRIX CASE WHERE THIS WAS REQUIRED TO ACCOMODATE THE THE CORNERS OF THE INNER LAYER APPEARING TWICE (AND THEREFORE NEEDING ENTRIES SUMMED IN THE MATRIX) 
            #WE DO NOT HAVE TO WORRY ABOUT THIS CASE HERE. WHEN MULTIPLYING THE CONVENIENT GREENS ARRAY THAT WE CONSTRUCT HERE
            #WITH U_INNER, WE JUST HAVE THE CORNER NODES OF THE INNER LAYER APPEAR TWICE IN U_INNER. JUST MAKE ONE BIG LIST HERE 
            
            nr_outer_layer_nodes_per_mesh = self.get_nr_outer_layer_nodes_per_mesh()
            n_meshes = len(nr_outer_layer_nodes_per_mesh)
            n_sides = 4 * n_meshes
    
            for pos in not_yet_precomputed_receiver_positions: 
                if self.do_sanity_checks:
                    in_range = self._check_if_pos_is_on_a_mesh(pos)
                    if in_range: #This check is not perfect. The pos could be inside the domains, but not on a mesh. The check would return 'false' and the code would continue when it should stop.
                        raise Exception('We should not have a receiver in the truncated domains. Although on the boundary would technically work.') 
                    
                self.conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers[pos] = dict()
                global_node_nr_pos = self.sparse_greens_matrix.position_to_node_nr(pos)
    
                self.conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers[pos]["outer"] = np.array([], dtype='complex128')
                self.conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers[pos]["inner"] = np.array([], dtype='complex128')
                
                #loop over the sides, and just append the contributions of each side to form a very long list as mentioned in comments at the start of this routine
                for side_nr in xrange(n_sides):
                    #INSTEAD OF DOING THIS '2 CONTRIBUTION AND CHECK FOR DIAGONAL THING' WHEN TRYING TO RETRIEVE A ROW FROM THE UPPER TRIANGULAR MATRIX, I SHOULD WRITE THIS CAPABILITY IN THE SPARSE GREEN'S MATRIX CLASS. MY CURRENT APPROACH MAKES IT MESSY AND I ALSO COPY THE CODE IN THE PART WHERE I MAKE 'K'
                    
                    outer_greens_array_contribution1 = self.sparse_greens_matrix[global_node_nr_pos, outer_global_node_numbers[side_nr]].toarray()[0,:]
                    outer_greens_array_contribution2 = self.sparse_greens_matrix[outer_global_node_numbers[side_nr], global_node_nr_pos].toarray()[:,0]

                    if global_node_nr_pos in outer_global_node_numbers[side_nr]: #Diagonal is present (pretty sure this will never happen for this function though! I think all the 'pos' will be receivers and they are not on the box so there will be no green's function to itself
                        entry = outer_global_node_numbers[side_nr].index(global_node_nr_pos)
                        outer_greens_array_contribution2[entry] = 0 #So you dont double it by adding addition1 and 2        
                    
                    self.conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers[pos]["outer"] = np.append(self.conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers[pos]["outer"], (outer_greens_array_contribution1 + outer_greens_array_contribution2))
    
                    inner_greens_array_contribution1 = self.sparse_greens_matrix[global_node_nr_pos, inner_global_node_numbers[side_nr]].toarray()[0,:]
                    inner_greens_array_contribution2 = self.sparse_greens_matrix[inner_global_node_numbers[side_nr], global_node_nr_pos].toarray()[:,0]
                    self.conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers[pos]["inner"] = np.append(self.conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers[pos]["inner"], (inner_greens_array_contribution1 + inner_greens_array_contribution2)) #Don't need to check for diagonal here. We know that the position is not inside the box (see check above), so global_node_nr_pos cannot be in inner_global_node_numbers                
            
        return self.conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers

    def _precompute_and_return_efficient_greens_array_for_rhs(self, shots):
        if type(shots) is not list:
            shots = [shots]

        total_outer_layer_nodes = np.sum(self.get_nr_outer_layer_nodes_per_mesh())
        

        for shot in shots: #Check if everything has been precomputed. Does this cost a lot ?
            if type(shot.sources) is not PointSource:
                raise Exception("Only working with point sources for now. Should not be hard to extend")
            
            pos = shot.sources.position
            if pos not in self.conveniently_stored_greens_functions_for_efficient_rhs_calculation: #if not in there, we need to prepare the green's functions
                self.conveniently_stored_greens_functions_for_efficient_rhs_calculation[pos] = np.zeros(total_outer_layer_nodes, dtype="complex128")
                for spiraling_node_nr_outer_boundary in xrange(total_outer_layer_nodes):
                    pos_outer = self.convert_spiraling_node_nr_to_pos(spiraling_node_nr_outer_boundary)
                    self.conveniently_stored_greens_functions_for_efficient_rhs_calculation[pos][spiraling_node_nr_outer_boundary] = self.sparse_greens_matrix.get_value_by_position(pos, pos_outer)        
        
        return self.conveniently_stored_greens_functions_for_efficient_rhs_calculation

    def _precompute_and_return_efficient_conversion_arrays_all_regular_node_nr_global_2d_indices(self):
        if not self.have_we_precomputed_conversion_arrays_regular_node_nr_global_2d_indices: #PRECOMPUTE
            print "Precomputing and storing in order the global 2d indices corresponding to regular node numbers. Useful for 'convert_solution_vector_to_list_of_2d_arrays'"


            print "POORLY DEBUGGED FUNCTION. Also should reset somehow if mesh is changed. Future..."


            mesh_global = self.sparse_greens_matrix.full_mesh
            dx = self.sparse_greens_matrix.dx
            dz = self.sparse_greens_matrix.dz
            x_min_global = mesh_global.domain.x['lbound']
            z_min_global = mesh_global.domain.z['lbound']
            
            total_truncated_nodes = np.sum(self.nodes_per_mesh)
            
            self.global_x_indices_corresponding_to_regular_node_nrs = np.zeros(total_truncated_nodes, dtype='int32') 
            self.global_z_indices_corresponding_to_regular_node_nrs = np.zeros(total_truncated_nodes, dtype='int32')
            
            for truncated_regular_node_nr in xrange(total_truncated_nodes):
                pos = self.convert_regular_node_nr_to_pos(truncated_regular_node_nr)

                x = pos[0]
                z = pos[1]
        
                index_x_global = int(np.round((x-x_min_global)/dx))
                index_z_global = int(np.round((z-z_min_global)/dz))                

                self.global_x_indices_corresponding_to_regular_node_nrs[truncated_regular_node_nr] = index_x_global
                self.global_z_indices_corresponding_to_regular_node_nrs[truncated_regular_node_nr] = index_z_global
            
            self.have_we_precomputed_conversion_arrays_regular_node_nr_global_2d_indices = True
            
        return self.global_x_indices_corresponding_to_regular_node_nrs, self.global_z_indices_corresponding_to_regular_node_nrs
            
        
    def _check_if_pos_is_on_a_mesh(self, pos):
        #check if x_from is in range
        
        mesh_list = self.mesh_collection.get_list()
        domain_list = self.mesh_collection.domain_collection.get_list()
        in_range = False #initialize
        
        for domain, mesh in zip(domain_list, mesh_list): #Check each time if in range of one of the domains. As soon as we verified the position falls within one of the domains we can stop checking.
            x_min = domain.x['lbound']
            x_max = domain.x['rbound']
            z_min = domain.z['lbound']
            z_max = domain.z['rbound']        
        
            x = pos[0]
            z = pos[1]

            if x >= x_min and x <= x_max and z >= z_min and z <= z_max: #in range for this domain! Check if we match the grid up to a reasonable accuracy
                
                #Not sure if below is sensitive to rounding errors. Want to make sure that position coincides with node position. I suspect it is. If it gives problems and I remove this, do the same in truncate_domain.py
                n_nodes_x = mesh.x['n']
                n_nodes_z = mesh.z['n']
                dx = (x_max - x_min)/(n_nodes_x-1)
                dz = (z_max - z_min)/(n_nodes_z-1)
    
        
                #allow some slack up to a millimeter or so. Sometimes there are some rounding errors. Also, the mod of large numbers seems to be a bit buggy. So bring the number down to the range 0 - spacing
                eps = 1.0e-3        
    
                x_remainder = x - int(np.floor((x-x_min)/dx))*dx - x_min
                z_remainder = z - int(np.floor((z-z_min)/dz))*dz - z_min
        
                if np.mod(x_remainder + eps, dx) > eps and np.mod(x_remainder, dx) > eps:
                    raise Exception('The x does not correspond with the x of a node')
                if np.mod(z_remainder + eps, dz) > eps and np.mod(z_remainder, dz) > eps:
                    raise Exception('The z does not correspond with the z of a node')
                
                #We are in a domain, and we fall on the mesh. No longer need to search other meshes and domains in the collection.
                in_range = True
                break

        return in_range

    

    def _build_second_order_laplacian(self): #The laplacian needs to be constructed for the interior of the mesh. It will be wider than it is tall because of this
        
        nodes_per_mesh = self.get_nodes_per_mesh() #Calling this function will make sure that all the required precomputations have been done. Results in a centralized location where the checks are handled.
        nr_outer_layer_nodes_per_mesh = self.get_nr_outer_layer_nodes_per_mesh() #Same idea
        
        total_nodes = np.sum(nodes_per_mesh)
        total_nodes_in_outer_layers = np.sum(nr_outer_layer_nodes_per_mesh)
        total_interior_nodes = total_nodes - total_nodes_in_outer_layers
        
        L = spsp.lil_matrix((total_interior_nodes, total_nodes))
        mesh_list = self.mesh_collection.get_list()
        
        total_number_of_outer_nodes = np.sum(nr_outer_layer_nodes_per_mesh)
        
        for mesh in mesh_list:
            x_min = mesh.domain.x['lbound']
            x_max = mesh.domain.x['rbound']
            z_min = mesh.domain.z['lbound']
            z_max = mesh.domain.z['rbound']        
        
            n_nodes_x = mesh.x['n']
            n_nodes_z = mesh.z['n']
            dx = (x_max - x_min)/(n_nodes_x-1)
            dz = (z_max - z_min)/(n_nodes_z-1)
        
            if n_nodes_x <= 3 or n_nodes_z <= 3:#not sure if algorithm still works then. perhaps best to stop in this case
                raise Exception('truncated grid too small ? ') 
                    
            #March over the interior of the grid. Do the same linspace command as in 'precompute spiraling node numbers' so no rounding differences will occur.
            x_nodes = np.linspace(x_min,x_max,n_nodes_x)
            z_nodes = np.linspace(z_min,z_max,n_nodes_z)
        
            floating_point_slack = 1e-6 #Bit hacky, and 1e-6 is somewhat arbitrary
            if dx - dz > floating_point_slack:
                raise Exception('I need to define Laplacian slightly differently then. No longer ~1/h^2') 

            h = dx #is dz for equal spacing I assume here 
        
            #MOVE OVER THE INTERIOR NODES ONLY         
            for ix in xrange(1,x_nodes.size-1):
                for iz in xrange(1,z_nodes.size-1):
               
                    node_nr = self.convert_pos_to_spiraling_node_nr((x_nodes[ix],z_nodes[iz]))
                    top_node_nr = self.convert_pos_to_spiraling_node_nr((x_nodes[ix],z_nodes[iz-1]))
                    bot_node_nr = self.convert_pos_to_spiraling_node_nr((x_nodes[ix],z_nodes[iz+1]))
                    left_node_nr = self.convert_pos_to_spiraling_node_nr((x_nodes[ix-1],z_nodes[iz]))
                    right_node_nr = self.convert_pos_to_spiraling_node_nr((x_nodes[ix+1],z_nodes[iz]))
    
                    #need to subtract the number of nodes in the outer layer from row number since those nodes do not have rows
                    L[node_nr-total_number_of_outer_nodes,node_nr] = -4/(h**2)
                    L[node_nr-total_number_of_outer_nodes,top_node_nr] = 1/(h**2)
                    L[node_nr-total_number_of_outer_nodes,bot_node_nr] = 1/(h**2)
                    L[node_nr-total_number_of_outer_nodes,left_node_nr] = 1/(h**2)
                    L[node_nr-total_number_of_outer_nodes,right_node_nr] = 1/(h**2)
    
    
                
        return L.tocsr() #positive Laplacian. 

    def _build_model_part_of_matrix(self):
        nodes_per_mesh = self.get_nodes_per_mesh()
        outer_layer_nodes_per_mesh = self.get_nr_outer_layer_nodes_per_mesh()
        n_nodes_outer_layer = np.sum(outer_layer_nodes_per_mesh)
        n_nodes_interior = np.sum(nodes_per_mesh) - n_nodes_outer_layer
        m = spsp.lil_matrix((n_nodes_interior,n_nodes_interior))
        
        mesh_list = self.mesh_collection.get_list()
        model_parameter_list = self.model_parameters.get_list()        
        for mesh, model_parameters in zip(mesh_list, model_parameter_list):
            mesh_index = mesh_list.index(mesh)
                    
            x_min = mesh.domain.x['lbound']
            x_max = mesh.domain.x['rbound']
            z_min = mesh.domain.z['lbound']
            z_max = mesh.domain.z['rbound']        
            
            n_nodes_x = mesh.x['n']
            n_nodes_z = mesh.z['n']
            
            dx = (x_max - x_min)/(n_nodes_x-1)
            dz = (z_max - z_min)/(n_nodes_z-1)
            
            floating_point_slack = 1e-6 #Bit hacky, and 1e-6 is somewhat arbitrary
            if dx - dz > floating_point_slack:
                raise Exception('I need to define Laplacian slightly differently then. No longer ~1/h^2')
                        
            x_nodes = np.linspace(x_min,x_max,n_nodes_x)
            z_nodes = np.linspace(z_min,z_max,n_nodes_z)
                        
            C = model_parameters.C #Each model parameters in the list has a 'C' array
        
            regular_node_nr_offset = np.sum(nodes_per_mesh[0:mesh_index])
            #Move only over the interior nodes        
            for ix in xrange(1,n_nodes_x-1):
                for iz in xrange(1,n_nodes_z-1):
                    x = x_nodes[ix]
                    z = z_nodes[iz]
                    
                    regular_node_nr = self.convert_pos_to_regular_node_nr((x,z))
                    spiraling_node_nr = self.convert_pos_to_spiraling_node_nr((x,z))
    
                    wavespeed = C[regular_node_nr-regular_node_nr_offset,0] #has 1 as second dimension instead of ,
                    
                    #The n_nodes_outer_layer nodes on the outer layer are not part of this square block. need to subtract
                    m[spiraling_node_nr - n_nodes_outer_layer, spiraling_node_nr - n_nodes_outer_layer] = 1.0/(wavespeed**2) 
                
        return m.tocsr()
    
    def _assemble_K(self,oc):
        outer_spiraling_node_numbers = self.return_outer_spiraling_node_numbers()
        inner_spiraling_node_numbers = self.return_inner_spiraling_node_numbers()
        outer_global_node_numbers = self.return_outer_global_node_numbers()
        inner_global_node_numbers = self.return_inner_global_node_numbers()        
        
        
        
        nr_outer_layer_nodes_per_mesh = self.get_nr_outer_layer_nodes_per_mesh()
        n_meshes = len(nr_outer_layer_nodes_per_mesh)
        n_sides = 4 * n_meshes
        
        nr_outer_layer_nodes = np.sum(nr_outer_layer_nodes_per_mesh)
        nr_inner_layer_nodes = nr_outer_layer_nodes - 8*n_meshes #The inner layer always has 8 nodes less than the outer layer except if one dimension is 3 nodes or less. But at another place there is a check for that case and an exception should be throwed.

        nr_nodes_per_mesh = self.get_nodes_per_mesh() 
        nr_nodes_total = np.sum(nr_nodes_per_mesh)
        nr_nodes_not_in_outer_layer = nr_nodes_total - nr_outer_layer_nodes
        nr_nodes_not_in_outer_and_inner_layer = nr_nodes_total - nr_outer_layer_nodes - nr_inner_layer_nodes
        
        #The blocks are basically dense
        outer_greens_block = np.zeros((nr_outer_layer_nodes, nr_outer_layer_nodes), dtype='complex128')
        inner_greens_block = np.zeros((nr_outer_layer_nodes, nr_inner_layer_nodes), dtype='complex128')
        ############################################################################################################################################
        #FILL BOTH MATRIX BLOCKS
        ############################################################################################################################################
        
        
        
        #The outer block represents the total pressure variables on the outer boundary and their product with the green's function on the corresponding inner boundary
        #The inner block represents the total pressure variables on the inner boundary and their product with the green's function on the corresponding outer boundary. Corner nodes on the inner boundary will have a sum of two green's functions because they have two neighbouring points on the outer boundary!
        
        for spiral_node_nr_row in xrange(nr_outer_layer_nodes): #fill row for every node on outer boundary.
            pos_row = self.convert_spiraling_node_nr_to_pos(spiral_node_nr_row)
            global_node_nr_row = self.sparse_greens_matrix.position_to_node_nr(pos_row)
            for side_nr in xrange(n_sides):
                
                #Keep in mind that the sparse greens matrix is upper triangular. Our logic would like us to do the following two commented lines 
                #outer_greens_block[spiral_nr_row, spiraling_node_nr_outer[side_nr]] = self.sparse_greens_matrix[global_nr_row, global_node_nr_corresponding_inner_node[side_nr]].toarray()[0,:]
                #inner_greens_block[spiral_nr_row, (spiraling_node_nr_corresponding_inner_node[side_nr] - n_nodes_outer)] += (-1.0*self.sparse_greens_matrix[global_nr_row, global_node_nr_outer[side_nr] ].toarray()[0,:])                
                
                #This is not possible however, because then the column index is less than the row index (global node numbering), we get a zero for a node combination that has a nonzero green's function.
                #As solution we need to add the column to the row and make sure we don't count any possible diagonal element twice.
                
                #IT IS PROBABLY BEST TO MOVE THIS LOGIC INTO THE SPARSE GREEN'S MATRIX OBJECT AT SOME POINT.
                
                #outer block addition contains green's functions inner boundary to outer boundary. These contributions will not extract diagonal elements from the sparse greens matrix.
                outer_greens_block_row_addition = self.sparse_greens_matrix[global_node_nr_row, inner_global_node_numbers[side_nr]].toarray()[0,:]
                outer_greens_block_row_addition += self.sparse_greens_matrix[inner_global_node_numbers[side_nr], global_node_nr_row].toarray()[:,0]
                
                inner_greens_block_row_addition1 = self.sparse_greens_matrix[global_node_nr_row, outer_global_node_numbers[side_nr] ].toarray()[0,:]
                inner_greens_block_row_addition2 = self.sparse_greens_matrix[outer_global_node_numbers[side_nr], global_node_nr_row].toarray()[:,0]
                if global_node_nr_row in outer_global_node_numbers[side_nr]: #Diagonal is present
                    entry = outer_global_node_numbers[side_nr].index(global_node_nr_row)
                    inner_greens_block_row_addition2[entry] = 0 #So you dont double it by adding addition1 and 2
                    
                inner_greens_block_row_addition = inner_greens_block_row_addition1 + inner_greens_block_row_addition2
                
                outer_greens_block[spiral_node_nr_row, outer_spiraling_node_numbers[side_nr]] = outer_greens_block_row_addition
                inner_greens_block[spiral_node_nr_row, (inner_spiraling_node_numbers[side_nr] - nr_outer_layer_nodes)] += (-1.0 * inner_greens_block_row_addition) #+= because the corners of the inner nodes will be multiplied by green's functions on the outer layer twice (two nodes on outer boundary border it). So these green's functions have to be added.
            
        I_square = spsp.eye(nr_outer_layer_nodes).tocsr()
        large_zero_block = spsp.csr_matrix((nr_outer_layer_nodes,nr_nodes_not_in_outer_layer))
        small_zero_block = spsp.csr_matrix((nr_outer_layer_nodes,nr_nodes_not_in_outer_and_inner_layer ))
        #Now make the K matrix
        blockrow_1 = spsp.bmat([[-I_square, I_square,large_zero_block]])
        blockrow_2 = spsp.bmat([[I_square, outer_greens_block, inner_greens_block, small_zero_block]])
        blockrow_3 = spsp.bmat([[large_zero_block.transpose(), -1.0*oc.L]])
        
        self.K = spsp.bmat([[blockrow_1],[blockrow_2],[blockrow_3]])
    
    def _assemble_M(self,oc):
        n_nodes_outer_layer = np.sum(self.get_nr_outer_layer_nodes_per_mesh())
        n_nodes_interior = np.sum(self.get_nodes_per_mesh()) - n_nodes_outer_layer   
        
        top_left_zero_block = spsp.csr_matrix((2*n_nodes_outer_layer,2*n_nodes_outer_layer))
        rectangular_zero_block = spsp.csr_matrix((2*n_nodes_outer_layer, n_nodes_interior ))
        
        blockrow_1 = spsp.bmat([[top_left_zero_block, rectangular_zero_block]])
        blockrow_2 = spsp.bmat([[rectangular_zero_block.transpose(), oc.m]])
        
        self.M = spsp.bmat([[blockrow_1],[blockrow_2]]) #will be multiplied by - omega^2 later on        

    def _assemble_C(self,oc):
        n_nodes_outer_layer = np.sum(self.get_nr_outer_layer_nodes_per_mesh())
        n_nodes_total = np.sum(self.get_nodes_per_mesh())   
        
        N = n_nodes_outer_layer + n_nodes_total
        self.C = spsp.csr_matrix((N,N))      

    def total_field_at_receivers_from_shot(self, shot, nu, return_params = 'only_field_at_receivers'):
        #should include a check if 'nu' agrees with the 'nu' in the sparse green's matrix
        rhs = self.build_rhs(shot,nu)
        [u_spiraling_truncated_region, u_without_scattered_on_normal_truncated_grid] = self.solve_truncated(rhs, nu) #Since LU decomposition on small domain has already been done this should be very cheap to recompute
        
        receivers = shot.receivers.receiver_list
        receiver_positions = []
        for receiver in receivers:
            receiver_positions.append(receiver.position)
        
        scattered_field = self.propagate_scattered_field_to_positions_from_solution(receiver_positions, u_spiraling_truncated_region)
        
        #get background field
        intensity = shot.sources.intensity
        w = shot.sources.w(nu=nu)
        source_pos = shot.sources.position

        background_field = dict()            
        for receiver_pos in receiver_positions:
            background_field[receiver_pos] = intensity*w*self.sparse_greens_matrix.get_value_by_position(source_pos, receiver_pos)
        
        #Add total field and scattered field
        total_field = dict()
        for key in scattered_field:
            total_field[key] = scattered_field[key] + background_field[key]

        if return_params == 'only_field_at_receivers':
            ret = total_field
        elif return_params == 'field_at_receivers_and_in_target':
            ret = [total_field, u_spiraling_truncated_region, u_without_scattered_on_normal_truncated_grid]
        else:
            raise Exception('wrong return params provided!')
        
        return ret
    
    
     
    def propagate_scattered_field_to_positions_from_solution(self, receiver_positions, u_spiraling_truncated_region): 
        #Use this to get the scattered field outside of the target.   
        #u_spiraling_truncated_region should be the solution from solve. This has the scattered field on the boundary, although that part is not used to propagate the scattered field
        #positions can be a tuple or a list of tuples.
        
        #print "MAKE SURE THAT IN NEW IMPLEMENTATION JUST AS EFFICIENT. SHOULD COST NEGLIGIBLE TIME COMPARED TO THE ACTUAL SOLVE."
        
        #It will check all the receiver positions every time. If the receiver position is not in the precomputed list then precompute it. Otherwise it skips all the work.
        conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers = self._precompute_and_return_efficient_greens_array_for_propagation_to_receivers(receiver_positions) 
                 
        #Store boundary wavefields. Skip first n_boundary_nodes entries because those are the scattered wavefield entries
        #print "ARE THESE THE OUTER AND INNER SPIRALING NODE NUMBERS I HAVE IN MIND? THE SAME THAT I DEFINE IN OTHER FUNCTION?"
        #Need the following two arrays in ndarray form, because those objects allow addition with an integer.
        outer_spiraling_node_numbers = self.return_outer_spiraling_node_numbers_as_ndarray()
        inner_spiraling_node_numbers = self.return_inner_spiraling_node_numbers_as_ndarray()
        
        n_nodes_outer_layer = np.sum(self.get_nr_outer_layer_nodes_per_mesh())
        
        u_outer = u_spiraling_truncated_region.data[n_nodes_outer_layer + outer_spiraling_node_numbers] 
        u_inner = u_spiraling_truncated_region.data[n_nodes_outer_layer + inner_spiraling_node_numbers]
        
        #We now know that the green's functions on the outer boundary are in self.conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers[pos]["outer"] and the other ones are in ["inner"].
        #This allows us to propagate the wavefields very cheaply to every receiver. For each receiver it is just two vector inner products. np.sum(self.conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers[pos]["outer"] * u_inner) and np.sum(self.conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers[pos]["inner"] * u_outer)
        #Eventually if we need even more speed we can propagate the wavefield to all the recievers at the same time by doing a matrix-vector product type of operation instead of looping over all receiver positions. But it seems like propagating the wavefield to the receivers is no longer a bottlebeck.
         
        scattered_fields = dict()
        
        for pos in receiver_positions:
            scattered_fields[pos] = np.sum(conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers[pos]["outer"] * u_inner) - np.sum(conveniently_stored_greens_functions_for_efficient_wavefield_propragation_to_receivers[pos]["inner"] * u_outer)

        return scattered_fields

    def solve(self, solver_data, rhs, nu, *args, **kwargs):
        raise Exception('Not implemented in a way consistent with rest of PySIT formulation. Getting rhs as a full grid stencil does not work. I need to get the shot')

    def solve_truncated(self, rhs, nu):
         
        if type(rhs) is not self.BoxLayersWavefieldVector:
            raise Exception('something is wrong. rhs should be a BoxLayersWavefieldVector')

        _rhs = rhs.data
        u = self.solvers[nu](_rhs) 

        u = self.BoxLayersWavefieldVector(self, u)
                
        u_without_scattered_on_normal_grid = u.return_full_field_regular_truncated_node_numbering()

        return [u, u_without_scattered_on_normal_grid]

        #WHAT TO DO WITH RECEIVER SAMPLING ? Perhaps start with only doing forward model and then add the sampling part later 

    def build_rhs(self, shots, nu): 
        #I require shot object, but this is normally not passed along in rest of pysit code. Only a force grid and a vector where the result should be put in. I PROBABLY JUST HAVE TO ABANDON THE IDEA OF USING THE REST OF THE PYSIT CODE FOR NOW AND JUST EXTRACT THE PARTS THAT I NEED TO GENERATE A WAVEFIELD. WHICH IS JUST A RHS VECTOR CONSTRUCTED FROM SHOT. PERHAPS JUST MAKE A TRUNCATED_SOLVE() FUNCTION WHICH TAKES A SHOT INSTEAD OF AN RHS VECTOR AND LET THE NORMAL SOLVE RETURN AN EXCEPTION SO I KNOW THAT I CANNOT USE IT FOR NOW 
        #if shots is a list, then add the 'rhs' contributions. This allows us to do an adjoint wavefield simulation efficiently.

        if type(shots) is not list:
            shots = [shots]

        conveniently_stored_greens_functions_for_efficient_rhs_calculation = self._precompute_and_return_efficient_greens_array_for_rhs(shots)

        #For every shot the green's functions have been precomputed and stored in spiraling node numbering order. Just need to multiply by the intensity and the wavelet!
        n_nodes_outer_layer = np.sum(self.get_nr_outer_layer_nodes_per_mesh())
        n_nodes_total = np.sum(self.get_nodes_per_mesh())
        data = np.zeros(n_nodes_outer_layer + n_nodes_total, dtype = 'complex128') #scattered field has the same number of entries as the outer layer. Then the rest is complete wavefield
        for shot in shots: #Sum over shots (when doing adjoint wavefield calculation)
            pos = shot.sources.position
            intensity = shot.sources.intensity #The shot objects I create for the adjoint wavefield will set this to the complex conjugate of the residual. 
            w = shot.sources.w(nu=nu) #If we pass a real shot object, this will contain the wavelet. 
            data[0:n_nodes_outer_layer] += intensity*w*conveniently_stored_greens_functions_for_efficient_rhs_calculation[pos] #The first n_nodes_outer entries will contain the background wavefield calculated this way
            
        #The rest of the vector (n_nodes_x * n_nodes_z entries) contains only zeros
        return self.BoxLayersWavefieldVector(self, data)

    class WavefieldVector(WavefieldVectorBase):

        aux_names = [] #could define the scattered field on the boundary box an auxillary field I try to solve for, but will make things more complicated
    
    class BoxLayersWavefieldVector(object): #Should eventually inherit stuff from WavefieldVector, but right now I'm scared that I by only overriding a part of the functions I will get inconsistencies. Better to get a warning that the method does not exist

        aux_names = [] 
        
        def __init__(self, solver, data):
            self._data = data
            self.solver = solver
            
        @property
        def data(self): return self._data
        @data.setter
        def data(self, arg): self._data[:] = arg
                
        def return_full_field_regular_truncated_node_numbering(self): # I am using private functions of the solver here. Not really nice. Perhaps move this function to the solver as a convenience function.        
            n_nodes_outer_layer = np.sum(self.solver.get_nr_outer_layer_nodes_per_mesh())
            n_nodes_total = np.sum(self.solver.get_nodes_per_mesh())
            
            data = self._data[n_nodes_outer_layer:] #don't use the scattered field part
            ret = np.zeros((n_nodes_total,),dtype='complex128')

            ret[self.solver.return_all_spiraling_node_nr_to_regular_node_nr()] = data[0:n_nodes_total]
                    
            return ret