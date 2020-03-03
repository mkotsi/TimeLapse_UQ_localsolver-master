import numpy as np
from pysit_extensions.truncated_domain_helmholtz_solver.lil_pos_ut import lil_pos_ut
from pysit_extensions.truncated_domain_helmholtz_solver.truncated_helmholtz_solver import ConstantDensityAcousticFrequencyScalar_2D_truncated_domain
from pysit_extensions.truncated_domain_helmholtz_solver.collection_objects import *
from pysit.core import *
from pysit_extensions.impulse_time.impulse_time import ImpulseTimeWavelet as Delta
from pysit import generate_seismic_data


#this will return a new solver and a new domain (remember to use the correct positions and not start at 0).
#for now everything is in 2D

def truncate_domain(solver, shots, truncation_params_list, freq, wrapper_type = 'umfpack', petsc='petsc'):
    #boundary params is either a dict if there is one domain, or it is a list of dicts if there are multiple domains
    
    #wrapper_type is either 'umfpack' or 'petsc'
    #petsc is a string containing the type of petsc solver. Is only used when petsc is the actual solver
    
    if type(truncation_params_list) == dict:
        truncation_params_list = [truncation_params_list]
        
    if type(truncation_params_list) != list:
        raise Exception("For a single region, boundary_params can be a dict or a list of a single dict. For multiple regions, a list of dicts is required.")
    
    #boundary_params is a list of dicts now. loop over the dicts to generate truncated domains and truncated meshes for each region.
    mesh_global = solver.mesh

    dx = mesh_global.x['delta']
    dz = mesh_global.z['delta']
    
    truncated_d_list = []
    truncated_m_list = []
    for truncation_params in truncation_params_list:
        x_lbc = Dirichlet()
        x_rbc = Dirichlet()
        z_lbc = Dirichlet()
        z_rbc = Dirichlet()        
    
        xpos_top_left_corner = truncation_params['xpos_top_left_corner']
        zpos_top_left_corner = truncation_params['zpos_top_left_corner']
    
        width = truncation_params['width']
        height = truncation_params['height']
    
        x_min = xpos_top_left_corner
        x_max = xpos_top_left_corner + width
    
        z_min = zpos_top_left_corner
        z_max = zpos_top_left_corner + height

        truncated_x_config = (x_min, x_max, x_lbc, x_rbc)
        truncated_z_config = (z_min, z_max, z_lbc, z_rbc)
    
        truncated_n_nodes_x = int(np.round((x_max - x_min) / dx + 1))
        truncated_n_nodes_z = int(np.round((z_max - z_min) / dz + 1))
    
        truncated_d = RectangularDomain(truncated_x_config, truncated_z_config)
        truncated_m = CartesianMesh(truncated_d, truncated_n_nodes_x, truncated_n_nodes_z)
        
        truncated_d_list.append(truncated_d)
        truncated_m_list.append(truncated_m)
    
    domain_collection = DomainCollection(domain_list = truncated_d_list)
    mesh_collection = MeshCollection(domain_collection, mesh_list = truncated_m_list)

    
    sparse_greens_matrix = generate_sparse_greens_matrix(solver, shots, domain_collection, freq) #upper triangular sparse lil_matrix indexed by position
    
    truncated_velocity_list = truncate_array(solver.model_parameters.without_padding().data, mesh_global, domain_collection)
    #truncated_solver = ConstantDensityAcousticFrequencyScalar_2D_truncated_domain(truncated_m, sparse_greens_matrix) #NEED TO CHANGE THIS SO IT WILL USE UMFPACK
    
    if wrapper_type == 'umfpack':
        from pysit_extensions.umfpack_complex_convenience.umf_solver_truncated_complex import umfpack_wrapper_truncated
    
        truncated_solver_umfpack_wrapper = umfpack_wrapper_truncated(mesh_collection, sparse_greens_matrix) #NEED TO CHANGE THIS SO IT WILL USE UMFPACK
        truncated_solver = truncated_solver_umfpack_wrapper.solver
    elif wrapper_type == 'petsc':
        from pysit_extensions.petsc4py_complex_convenience.petsc_solver_truncated_complex import petsc_wrapper_truncated

        truncated_solver_petsc_wrapper = petsc_wrapper_truncated(mesh_collection, sparse_greens_matrix, petsc = petsc) #NEED TO CHANGE THIS SO IT WILL USE UMFPACK
        truncated_solver = truncated_solver_petsc_wrapper.solver
    
    truncated_model = truncated_solver.ModelParameters(mesh_collection,{'C': truncated_velocity_list}) 
    truncated_solver.model_parameters = truncated_model
    
    return truncated_solver

def truncate_array(input_array_global, mesh_global, domain_collection, return_shape = "", input_shape=""):
    if input_shape != "":
        import warnings
        warnings.simplefilter('once', UserWarning)
        warnings.warn("The input parameter 'input_shape' is no longer required. Derived from shape of input array.", PendingDeprecationWarning)
    
    domain_global = mesh_global.domain
    if domain_global.dim != 2:
        raise Exception("Right now I only work with 2D")
    
    x_min_global = domain_global.x['lbound']
    z_min_global = domain_global.z['lbound']

    dx = mesh_global.x['delta']
    dz = mesh_global.z['delta']
    
    n_nodes_x_global = mesh_global.x['n']
    n_nodes_z_global = mesh_global.z['n']

    input_shape_tuple = input_array_global.shape
    input_array_global_2d = np.reshape(input_array_global, (n_nodes_z_global, n_nodes_x_global),'F') #If it was already 2D, it will remain unchanged by this! But a (n_1d, 1) or (n_1d,) array will be changed into the correct shape.

    return_array_list = []
    #Cycle over the truncated domains and take out the corresponding sections from complete_array_2d
    truncated_d_list = domain_collection.get_list()
    for truncated_d in truncated_d_list:
        x_min_truncated = truncated_d.x['lbound']
        x_max_truncated = truncated_d.x['rbound']
        z_min_truncated = truncated_d.z['lbound']
        z_max_truncated = truncated_d.z['rbound']

        col_left = int(round((x_min_truncated - x_min_global)/dx))
        col_right = int(round((x_max_truncated - x_min_global)/dx))
    
        row_top = int(round((z_min_truncated - z_min_global)/dz))
        row_bot = int(round((z_max_truncated - z_min_global)/dz))

        array_2d_truncated = input_array_global_2d[row_top:row_bot+1, col_left:col_right+1] #+1 because of how the : works. I need row_bot and col_right as well
        [n_nodes_z_truncated, n_nodes_x_truncated] = array_2d_truncated.shape 

        #instead of mesh and domain lists, perhaps I should make a DomainCollection and MeshCollection object at some point?
        
        if return_shape == "": #if not set, return in same shape as input
            if len(input_shape_tuple) == 1:
                return_shape_tuple = (n_nodes_z_truncated * n_nodes_x_truncated,)
            elif len(input_shape_tuple) == 2:
                if input_shape_tuple[1] == 1: #Basically a 1D array with with the first dimension having all the entries and the second index is always 1.
                    return_shape_tuple = (n_nodes_z_truncated*n_nodes_x_truncated, 1)
                else:
                    return_shape_tuple = (n_nodes_z_truncated, n_nodes_x_truncated)
            else:
                raise Exception("Unhandled input type")

        elif return_shape == "one_dim_flat":
            return_shape_tuple = (n_nodes_z_truncated * n_nodes_x_truncated,)
            
        elif return_shape == "one_dim":
            return_shape_tuple = (n_nodes_z_truncated*n_nodes_x_truncated, 1)
        elif return_shape == "two_dim":
            return_shape_tuple = (n_nodes_z_truncated, n_nodes_x_truncated)
        else:
            return Exception("Error with the array reshaping.")
        
        truncated_array = np.reshape(array_2d_truncated, return_shape_tuple, 'F') #if it was 2D and we are reshaping it into 2D, it will not cause artifacts.
        return_array_list.append(truncated_array)

    if len(return_array_list) == 1:
        return return_array_list[0] #legacy code expects one array back
    else:
        return return_array_list

def truncate_array_2d(complete_array_2d, mesh, domain_collection): #NO LONGER NECESSARY WITH THE NEW WAY THAT truncate_array WORKS, BUT LEFT IN HERE FOR BACKWARDS COMPATABILITY.
    nx = mesh.x.n
    nz = mesh.z.n
    
    complete_array_1d = np.reshape(complete_array_2d, (nz,nx), 'F')
    truncated_array_2d = truncate_array(complete_array_1d, mesh, domain_collection, return_shape = "two_dim")
    return truncated_array_2d

def truncated_back_to_full(truncated_array_2d_list, truncated_mesh_collection, mesh_global, pad_val=0.0):
    if type(truncated_array_2d_list) == np.ndarray: #In case only one domain.
        truncated_array_2d_list = [truncated_array_2d_list]
        
    dx = mesh_global.x['delta']
    dz = mesh_global.z['delta']

    if dz != dz:
        raise Exception("spacing not equal? In other parts I may have made that assumption")
    
    uniform_spacing = dx

    n_nodes_x_global = mesh_global.x.n
    n_nodes_z_global = mesh_global.z.n    

    dtype = truncated_array_2d_list[0].dtype
    for truncated_array_2d in truncated_array_2d_list:
        if dtype != truncated_array_2d.dtype:
            raise Exception('arrays in list have different dtypes?')
        
    converted_array_global = np.ones((n_nodes_z_global ,n_nodes_x_global), dtype=dtype)*pad_val
    
    domain_global = mesh_global.domain    

    x_min_global = domain_global.x['lbound']
    z_min_global = domain_global.z['lbound']
    
    mesh_list = truncated_mesh_collection.get_list()
    for mesh, truncated_array_2d in zip(mesh_list, truncated_array_2d_list):

        truncated_x_min = mesh.domain.x['lbound']
        truncated_z_min = mesh.domain.z['lbound']
    
        n_nodes_x_truncated = mesh.x.n
        n_nodes_z_truncated = mesh.z.n
    
        startnode_x = int(np.round((truncated_x_min-x_min_global)/uniform_spacing))
        startnode_z = int(np.round((truncated_z_min-z_min_global)/uniform_spacing))
        converted_array_global[startnode_z:startnode_z+n_nodes_z_truncated, startnode_x:startnode_x+n_nodes_x_truncated] = truncated_array_2d
        
    return converted_array_global

def generate_sparse_greens_matrix(solver, shots, domain_collection, freq): 
    #This should figure out what the cheapest way (in number of solves) is to fill the matrix
    #The greens matrix will be used to form matrices S and T.
    #In the future I may use an approach that approximates S and T so I only need the green's functions from source to receiver! 
    
    validate_truncation_params(solver,domain_collection) #Sanity check on input
    
    sparse_connection_matrix = lil_pos_ut(solver.mesh, dtype='bool') #only need to store if there is a connection
        
    fill_sparse_connection_matrix(solver, sparse_connection_matrix, shots, domain_collection)
    
    connection_count_list = generate_connection_count_list(sparse_connection_matrix) #For now don't use linked list. The cost of deleting an item in normal python list is O(s) where 's' is the length of the array. This is less than O(n), with n being the number of nodes in fulll domain and negligible to the cost of doing an LU backsubstitution O(n^2)

    sparse_greens_matrix = lil_pos_ut(solver.mesh, dtype='complex128')
    iter = 0 #remove this counter later. Just for debugging right now    
    while connection_count_list != []: #as long as there are still connections left
        print "Generating Green's function %i."%(iter+1)
        max_index = np.argmax(connection_count_list, axis=0)
        max_index = max_index[1] #the second index gives the row where the second column (i.e. the connection count) is the largest.
        position_max_connections = connection_count_list[max_index][0] #(xpos,zpos)
        
        connecting_positions = sparse_connection_matrix.get_connecting_positions(position_max_connections)
        
        shot = generate_shot(solver, position_max_connections, connecting_positions)
        
        forward_model_and_store(solver, shot, sparse_greens_matrix, freq)
        
        update_connection_count_list(sparse_connection_matrix, connection_count_list, position_max_connections, connecting_positions) #remove the item, and decrease value of some by -1 and if they become 0 also remove those
        iter = iter + 1

    print "finished with greens matrix"
    return sparse_greens_matrix

def get_greens_by_position(from_x, from_z, to_x, to_z):
    #this will index the sparse green's matrix by position (maybe move this function to new solver)
    print "test"
    
def get_greens_by_node_nr(from_nr, to_nr):
    print "test"
    
def validate_truncation_params(solver, domain_collection): #make sure your box does not exceed the boundaries of the full domain
    #Verify global domain
    domain_global = solver.domain
    if domain_global.dim != 2:
        raise Exception("Right now I only work with 2D")
    
    x_min = domain_global.x['lbound']
    x_max = domain_global.x['rbound']
    
    z_min = domain_global.z['lbound']
    z_max = domain_global.z['rbound']

    mesh_global = solver.mesh
    dx = mesh_global.x['delta']
    dz = mesh_global.z['delta']
    
    #now check the truncated domains against this
    for truncated_d in domain_collection.get_list():
        xpos_top_left_corner = truncated_d.x['lbound']
        xpos_top_right_corner = truncated_d.x['rbound']
        zpos_top_left_corner = truncated_d.z['lbound']
        zpos_bot_left_corner = truncated_d.z['rbound']
    
        width = xpos_top_right_corner - xpos_top_left_corner
        height = zpos_bot_left_corner - zpos_top_left_corner 
    
        if xpos_top_left_corner + width > x_max:
            raise Exception('Truncated domain exceeds right boundary full domain!')
    
        if zpos_top_left_corner + height > z_max:
            raise Exception('Truncated domain exceeds bottom boundary of full domain!')
    
        #allow some slack up to a millimeter or so. Sometimes there are some rounding errors
        eps = 1.0e-3
    
        #Modulus is buggy sometimes for large numbers (even double precision). So I am first getting the numbers to be between 0 and spacing.
        left_remainder = xpos_top_left_corner - int(np.floor((xpos_top_left_corner-x_min)/dx))*dx - x_min
        top_remainder  = zpos_top_left_corner - int(np.floor((zpos_top_left_corner-z_min)/dz))*dz - z_min
        width_remainder = width - np.floor(width/dx)*dx
        height_remainder = height - np.floor(height/dz)*dz
    
        if np.mod(left_remainder + eps, dx) > eps and np.mod(left_remainder, dx) > eps:
            raise Exception('The left part of one of the truncated grids would not align with nodes on the full grid')
        if np.mod(top_remainder + eps, dz) > eps and np.mod(top_remainder, dz) > eps:
            raise Exception('The top part of one of the truncated grids would not align with nodes on the full grid') 
        if np.mod(width_remainder + eps, dx) > eps and np.mod(width_remainder, dx) > eps:
            raise Exception('The width of one of the truncated grids is not an integral number of nodes')    
        if np.mod(height_remainder + eps, dz) > eps and np.mod(height_remainder, dz) > eps:
            raise Exception('The height of one of the truncated grids is not an integral number of nodes')    
    
        print "DID NOT YET WRITE A CHECK TO MAKE SURE THE DOMAINS DO NOT INTERSECT. \n"
    
def fill_sparse_connection_matrix(solver, mat, shots, domain_collection):
    print "filling sparse connection matrix\n"

    #DOES NOT YET WORK FOR SOURCESETS!!!
    dx = solver.mesh.x['delta']
    dz = solver.mesh.z['delta']    

    #FIND ALL THE UNIQUE SOURCE POSITIONS
    print "finding unique source locations.\n"
    unique_source_pos = []
    for shot in shots:
        if shot.sources.approximation != 'delta':
            raise Exception('Working with delta sources is required for now')        
    
        source_pos = shot.sources.position #assuming single source
        if source_pos not in unique_source_pos:
            unique_source_pos.append(source_pos)
        
    #FIND ALL THE UNIQUE RECEIVER POSITIONS
    print "finding unique receiver locations.\n"
    unique_receiver_pos = []    
    for shot in shots:
        for receiver in shot.receivers.receiver_list:
            if receiver.approximation != 'delta':
                raise Exception('Working with delta receivers is required for now')
            
            receiver_pos = receiver.position
            if receiver_pos not in unique_receiver_pos:
                unique_receiver_pos.append(receiver_pos)

    outer_box_node_positions_list = [] #one per box. Need to store temporarily in order to connect boxes to each other
    inner_box_node_positions_list = [] #one per box. Need to store temporarily in order to connect boxes to each other
    
    truncated_d_list = domain_collection.get_list()
    
    for truncated_d in truncated_d_list:
        index = truncated_d_list.index(truncated_d)
        
        x_min = truncated_d.x['lbound']
        x_max = truncated_d.x['rbound']
        z_min = truncated_d.z['lbound']
        z_max = truncated_d.z['rbound']        
    
        truncated_n_nodes_x = int(np.round((x_max - x_min) / dx + 1))
        truncated_n_nodes_z = int(np.round((z_max - z_min) / dz + 1))
    
        x_vals = np.linspace(x_min,x_max,truncated_n_nodes_x)
        z_vals = np.linspace(z_min,z_max,truncated_n_nodes_z) 
    
        #GET ALL THE POSITIONS ON THE TWO BOUNDARIES. 
        #GET THE X AND Z VALUES THE SAME WAY AS IN TRUNCATED_HELMHOLTZ_SOLVER SO NO MINUTE FLOATING POINT DIFFERENCES WILL OCCUR
        outer_box_node_positions = get_position_nodes_on_boundary(solver, x_vals, z_vals)
        inner_box_node_positions = get_position_nodes_on_boundary(solver, x_vals[1:-1], z_vals[1:-1])
        
    
        print "Connecting the unique source locations to the outer boundary for domain %i out of %i.\n"%(index+1, len(truncated_d_list))
        for source_pos in unique_source_pos:
            set_connection_node_to_box(mat,source_pos,outer_box_node_positions)
    
        print "Connecting the unique receiver locations to the outer boundary for domain %i out of %i.\n"%(index+1, len(truncated_d_list))
        for receiver_pos in unique_receiver_pos:
            set_connection_node_to_box(mat,receiver_pos,outer_box_node_positions)    
    
        print "Connecting the unique receiver locations to the inner boundary for domain %i out of %i.\n"%(index+1, len(truncated_d_list))
        for receiver_pos in unique_receiver_pos:
            set_connection_node_to_box(mat,receiver_pos,inner_box_node_positions)    
        
        #Do all the connections requires for the nodes on the inner/outer boundary (outer->outer and inner->outer)
        print "Connecting the outer box to the inner box and itself for domain %i out of %i.\n"%(index+1, len(truncated_d_list))
        n_outer_box_nodes = len(outer_box_node_positions)
        for i in xrange(n_outer_box_nodes):
            set_connection_node_to_box(mat, outer_box_node_positions[i], outer_box_node_positions)
            set_connection_node_to_box(mat, outer_box_node_positions[i], inner_box_node_positions)

        outer_box_node_positions_list.append(outer_box_node_positions)
        inner_box_node_positions_list.append(inner_box_node_positions)


    print "Connecting the boxes to each other. First connect the outer nodes to the inner nodes of all other domains. \n" #We need the scattered field at the outside of every box. The outer layer of every box needs to be connected to the inner and outer layer of every other box. 
    for outer_box_node_positions in outer_box_node_positions_list:
        index = outer_box_node_positions_list.index(outer_box_node_positions)
        print "Connecting domain %i out of %i to other domains."%(index+1, len(truncated_d_list))
        domain_indices_to_connect_with = range(len(outer_box_node_positions_list))
        domain_indices_to_connect_with.remove(index) #remove domain 'index' from the list, because we already connected it to itself
        for domain_index in domain_indices_to_connect_with:
            inner_box_node_positions_to = inner_box_node_positions_list[domain_index]
            for i in xrange(len(outer_box_node_positions)):
                set_connection_node_to_box(mat, outer_box_node_positions[i], inner_box_node_positions_to)

    print "Now connect the outer nodes to the outer nodes of all other boxes. Exploit symmetry! \n"
    for outer_box_node_positions in outer_box_node_positions_list:
        index = outer_box_node_positions_list.index(outer_box_node_positions)
        print "Connecting domain %i out of %i to other domains."%(index+1, len(truncated_d_list))
        domain_indices_to_connect_with = range(index+1, len(outer_box_node_positions_list)) #index+1 because we do not want to connect to outer nodes of itself. We already did that. If from-to would be represented as a 2D matrix, you do the upper part this way which is sufficient due to symmetry.
        for domain_index in domain_indices_to_connect_with:
            outer_box_node_positions_to = outer_box_node_positions_list[domain_index]
            for i in xrange(len(outer_box_node_positions)):
                set_connection_node_to_box(mat, outer_box_node_positions[i], outer_box_node_positions_to)
        
    print "Connecting the sources to the receivers. \n"
    for shot in shots:
        source_pos = shot.sources.position
        for receiver in shot.receivers.receiver_list:
            receiver_pos = receiver.position
            mat.set_value_by_position(source_pos, receiver_pos, True)

    
    print "Finished with generating the sparse connection matrix!\n"

def set_connection_node_to_box(mat,pos,box_node_positions):
    n_box_nodes = len(box_node_positions)
    
    for i in xrange(n_box_nodes):
        mat.set_value_by_position(pos, box_node_positions[i], True) #Maybe see if I can rewrite lil_pos_ut in such a way that I can pass array 'box_node_positions' instead. More efficient probably.

def get_position_nodes_on_boundary(solver, x_nodes, z_nodes): #box param contains the geometrical configuration of the square boundary along which the node positions are required. Can be the outer boundary or one layer inside for instance      
    positions = []
    
    n_nodes_x = x_nodes.size
    n_nodes_z = z_nodes.size
    
    #get x and z values from array and don't keep adding dx and dz. Floating point differences may accumulate. Do it the same way as in truncated_helmholtz_solver
    
    #left boundary
    x_left_arr = np.zeros(n_nodes_z); x_left_arr[:] = x_nodes[0]
    z_left_arr = z_nodes[:]
    pos_left = zip(x_left_arr,z_left_arr)
    
    #right boundary
    x_right_arr = np.zeros(n_nodes_z); x_right_arr[:] = x_nodes[-1]
    z_right_arr = z_nodes[:]
    pos_right = zip(x_right_arr,z_right_arr)
    
    #bot boundary
    x_bot_arr = x_nodes[1:-1]
    z_bot_arr = np.zeros(n_nodes_x-2); z_bot_arr[:] = z_nodes[-1]
    pos_bot = zip(x_bot_arr,z_bot_arr)
    
    #top boundary
    x_top_arr = x_nodes[1:-1]
    z_top_arr = np.zeros(n_nodes_x-2); z_top_arr[:] = z_nodes[0]
    pos_top = zip(x_top_arr,z_top_arr)    
    
    positions.extend(pos_left)
    positions.extend(pos_right)
    positions.extend(pos_bot)
    positions.extend(pos_top)
    
    return positions

def generate_connection_count_list(sparse_connection_matrix):
    #first go through the matrix once to find all the positions that are there.
    print "Generating connection count list. \n" 
    [node_nr_from, node_nr_to] = sparse_connection_matrix.nonzero()
    
    #get all unique node numbers. Since upper triangular, the lowest numbers are in. Store in dict. Will set some nodes to 'present' several times, but at least I don't have to search through a list each time to find out if the node has been recorded already
    nodes_with_connections_dict = dict()
    all_nodes_with_connections_including_duplicates = np.append(node_nr_from, node_nr_to)
    for node in all_nodes_with_connections_including_duplicates:
        nodes_with_connections_dict[node] = 0
        
    #we now have recorded all the nodes that have connections. Turf all the connections
    for i in xrange(node_nr_from.size):
        node_from = node_nr_from[i]  
        node_to = node_nr_to[i]
        
        if (node_from == node_to): #don't count double
            nodes_with_connections_dict[node_from] += 1
        else:
            nodes_with_connections_dict[node_from] += 1
            nodes_with_connections_dict[node_to]   += 1
    
    #now fill the connection_count_list according to the specification I use in other functions. In retrospect a dict may have been easier to handle, but it is not a real problem.
    connection_count_list = []
    sorted_node_numbers = sorted(nodes_with_connections_dict) 
    for node in sorted_node_numbers: #sorted_node_numbers is a sorted list. So I will start with the lower node numbers and then work my way up. The ordering of the nodes in the connection_count_list is like expected, following the node number
        pos = sparse_connection_matrix.turn_node_nr_into_position(node) 
        n_con = nodes_with_connections_dict[node]
        connection_count_list.append([pos,n_con])
    
    print "Finished generating connection count list. \n"
    
    return connection_count_list
    #sorted [[(key1, key2), key3], [(key1, key2), key3], [(key1, key2), key3], .... ] will sort on key1. For equal matches, key2 decides. And then key3. So this will generate a list with increasing position x which I use everywhere else as well 

def generate_shot(solver, position_max_connections, connecting_positions):
    mesh = solver.mesh
    
    source = PointSource(mesh, position_max_connections, Delta(), approximation = 'delta')
    receivers = ReceiverSet(mesh, [PointReceiver(mesh, p, approximation = 'delta') for p in connecting_positions], approximation = 'delta') #Not sure if second pointreceiver actually does anything
    shot = Shot(source, receivers)
    return shot

def forward_model_and_store(solver, shot, sparse_greens_matrix, freq):
    generate_seismic_data([shot], solver, solver.model_parameters.without_padding(), frequencies = freq)
    
    n_receivers = len(shot.receivers.receiver_list) 
    source_pos = shot.sources.position
    for i in xrange(n_receivers):
        receiver = shot.receivers.receiver_list[i]
        receiver_pos = receiver.position
        
        recording = shot.receivers.data_dft[freq][0][i]
        sparse_greens_matrix.set_value_by_position(source_pos, receiver_pos, recording) 
            

def update_connection_count_list(sparse_connection_matrix, connection_count_list, position_max_connections, connecting_positions): 
    #remove the item, and decrease value of some by -1 and if they become 0 also remove those
    
    #connecting_positions should have been ordered so that the positions are in order of increasing node number. This should also be the case in connection_count_list
    
    #Set corresponding entries in sparse_connecting_matrix to 0. Not necessary, but will result in fewer receiver objects being generated in the future.
    for i in xrange(len(connecting_positions)):
        sparse_connection_matrix.set_value_by_position(position_max_connections, connecting_positions[i], 0)
    
    #update connection_count_list by decreasing values by -1. Assuming connection_count_list and connecting_positions have been ordered by node_number (so same ordering)
    j = 0
    for i in xrange(len(connection_count_list)):
        if connection_count_list[i][0] == connecting_positions[j]:
            connection_count_list[i][1] = connection_count_list[i][1] - 1
            j = j + 1
            
        if j == len(connecting_positions): #before reaching the end of the connection_count_list, we already went through the entire 'connecting_positions' array. The next evaluation of if connection_count_list[i][0] == connecting_positions[j] would result in a crash.  
            break 
    
    #finished decreasing by -1. Now remove the position_max_connections entry from connection_count_list and remove any value at 0
    delete_positions = []
    for i in xrange(len(connection_count_list)):
        if connection_count_list[i][0] == position_max_connections: #the node for which we just computed all connections
            delete_positions.append(i)
        elif connection_count_list[i][1] <= 0: #if no connections remaining
            delete_positions.append(i)
    
    #we now have a list of entries to delete from low to high. Start with the highest one, because the index of all later values will change but we dont care in that case
    delete_positions.reverse()
    for i in delete_positions:
        del connection_count_list[i]
        
    