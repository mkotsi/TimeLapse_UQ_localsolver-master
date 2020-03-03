#Generate seismic data (uses sample_wavefield_at_receivers)
#sample_wavefield_at_receivers (returns wavefield at receivers)
#truncated_solve (returns wavefield in target) 
#Adjoint 
#Gradient
import numpy as np
import copy
from pysit_extensions.impulse_time.impulse_time import ImpulseTimeWavelet as Delta
from pysit_extensions.truncated_domain_helmholtz_solver.truncate_domain import truncate_array, truncated_back_to_full
from pysit.core import *
from pysit_extensions.truncated_domain_helmholtz_solver.collection_objects import ModelPerturbationCollection

def truncated_solve(shot, truncated_solver, model, nu, rhs = None):
    #Should work both when model is a single model and if it is a collection
    truncated_solver.model_parameters = model
    
    if rhs == None: #If no RHS is provided, make one
        rhs = truncated_solver.build_rhs(shot, nu)
        
    [u, u_without_scattered_on_normal_grid] = truncated_solver.solve_truncated(rhs,nu)
    return [u, u_without_scattered_on_normal_grid]
    
def generate_seismic_data_truncated(shots, truncated_solver, model, nu):
    truncated_solver.model_parameters = model
    for shot in shots:
        total_field_at_receivers = truncated_solver.total_field_at_receivers_from_shot(shot, nu)
    
        #find out what positions the receivers have in the receiver list
        receiver_dict = dict()
        num_receivers = len(shot.receivers.receiver_list)
        shot.receivers.data_dft[nu] = np.zeros((1,num_receivers), dtype='complex128')
        for i in xrange(num_receivers):
            receiver = shot.receivers.receiver_list[i]
            pos = receiver.position
            receiver_dict[pos] = i
            
        for pos in total_field_at_receivers:
            index = receiver_dict[pos]
            shot.receivers.data_dft[nu][0][index] = total_field_at_receivers[pos]

def adjoint_contribution_shot(shot, truncated_solver, total_field_at_receivers, nu):
    #RIGHT NOW I ASSUME THAT total_field_at_receivers CONTAINS DATA OF ONLY A SINGLE FREQUENCY (CORRESPONDING TO NU ! HAVE TO EXTEND IF I WILL USE MULTIPLE FREQUENCIES.
    
    num_receivers = len(shot.receivers.receiver_list)
    
    residuals = np.zeros(num_receivers, dtype='complex128')
    

    
    n_nodes_outer_layer = np.sum(truncated_solver.get_nr_outer_layer_nodes_per_mesh())
    n_nodes_total = np.sum(truncated_solver.get_nodes_per_mesh())
    
    n_entries_rhs = n_nodes_outer_layer + n_nodes_total
    
    rhs = truncated_solver.BoxLayersWavefieldVector(truncated_solver, np.zeros(n_entries_rhs, dtype='complex128'))

    #compute residuals and make a shot for each
    adjoint_shots = []
    domain_list = truncated_solver.mesh_collection.domain_collection.get_list()
    junk_domain = 0
    temp_junk_pos = (domain_list[junk_domain].x['lbound'], domain_list[junk_domain].z['lbound']) #An arbitrary acceptable position
    temp_junk_mesh = truncated_solver.mesh_collection.get_list()[junk_domain] 
    for i in xrange(num_receivers):

        receiver = shot.receivers.receiver_list[i]
        pos = receiver.position
        recorded_data = shot.receivers.data_dft[nu][0][i]
        predicted_data = total_field_at_receivers[pos]
        residuals[i] = recorded_data - predicted_data 
    
        
        source = PointSource(temp_junk_mesh, temp_junk_pos, Delta(), approximation = 'delta')
        source.position = pos #Can not do directly because it falls outside of the truncated mesh and I get a crash. This method is a bit hacky though. Ideally I should make a truncated_solver.build_rhs function that does not require a shot to get the incident wavefield on the target boundary 
        junk_receiver = PointReceiver(temp_junk_mesh, temp_junk_pos, approximation = 'delta') #Will not do anything
        temp_shot = Shot(source, junk_receiver)
        
        temp_shot.sources.intensity = np.conj(residuals[i]) #We propagate the complex conjugate of the residual to the boundary of the truncated domain. So we will use that as source strength. Normally you would like to use a wavelet, but here I will just abuse 'intensity' and make it a complex valued value
        adjoint_shots.append(temp_shot)
        

    rhs_contribution = truncated_solver.build_rhs(adjoint_shots, nu) #This will give incident field for delta at the receiver location. So basically background green's functions to all boundary nodes
    rhs.data += rhs_contribution.data
    
    #The contribution of all the residuals to the background field on the boundary has now been computed. (background is what we want because the waves move through the unperturbed medium to the target)
    [adj, adj_on_normal_grid] = truncated_solve(shot, truncated_solver, truncated_solver.model_parameters, nu, rhs) #adj also contains scattered

    #here I have computed G*conj(residual). But the adjoint field is conj(G)*residual. So need to take conjugate
    adj.data = np.conj(adj.data)
    adj_on_normal_grid = np.conj(adj_on_normal_grid)
         
    return adj_on_normal_grid
        
def get_least_squares_objective_contribution_shot_from_precomputed_wavefield(shot, total_field_at_receivers, nu):
    
    num_receivers = len(shot.receivers.receiver_list)
    residuals_at_shot = np.zeros(num_receivers, dtype='complex128')
        
    for i in xrange(num_receivers):

        receiver = shot.receivers.receiver_list[i]
        pos = receiver.position
        recorded_data = shot.receivers.data_dft[nu][0][i]
        predicted_data = total_field_at_receivers[pos]
        residuals_at_shot[i] = recorded_data - predicted_data #The order does not matter here, because we will square it for the objective function 
        
        
    obj_val_contrib = 0.5*np.real(np.sum(residuals_at_shot*np.conj(residuals_at_shot))) #Is the 0.5 consistent with the pysit definition ? Does not really matter though because it will not change the direction of the search vector. The value has zero imaginary part even without the real statement, but would still be represented as a complex128 which gives some problems with plotting etc (throwing warnings). Therefore a real statement here as well.
    return obj_val_contrib

def get_least_squares_objective(shots, truncated_solver, model, nu):
    truncated_solver.model_parameters = model
    #do forward model (gives dictionary indexed by pos)    
    obj_val = 0.0
    for shot in shots:
        total_field_at_receivers = truncated_solver.total_field_at_receivers_from_shot(shot, nu, return_params = 'only_field_at_receivers')
        obj_val += get_least_squares_objective_contribution_shot_from_precomputed_wavefield(shot, total_field_at_receivers, nu)
        
    return obj_val

            
def migration_contribution_shot(shot, truncated_solver, model, nu, optional_return_dict = dict(), positive_gradient = False):
    truncated_solver.model_parameters = model
    
    
    #do forward model (gives dictionary indexed by pos)
    [total_field_at_receivers, u, u_without_scattered_on_normal_truncated_grid] = truncated_solver.total_field_at_receivers_from_shot(shot, nu, return_params = 'field_at_receivers_and_in_target') 
    
    obj_val_contribution = get_least_squares_objective_contribution_shot_from_precomputed_wavefield(shot, total_field_at_receivers, nu)
    
    #do adjoint model
    adjoint_field_on_normal_truncated_grid = adjoint_contribution_shot(shot, truncated_solver, total_field_at_receivers, nu)
    
    omega = 2*np.pi*nu
    #cross-correlate
    migration_contribution = omega**2 * u_without_scattered_on_normal_truncated_grid * np.conj(adjoint_field_on_normal_truncated_grid)
    
    #NEED TO PUT BOUNDARIES TO ZERO SO WE WON'T UPDATE THEM ACCIDENTALLY
    nodes_per_mesh = truncated_solver.get_nodes_per_mesh()
    truncated_mesh_list = truncated_solver.mesh_collection.get_list()
    for n_nodes, mesh in zip(nodes_per_mesh, truncated_mesh_list):
        index = truncated_mesh_list.index(mesh)
        offset = np.sum(nodes_per_mesh[0:index]) 
        velocity_in_this_mesh = migration_contribution[offset:(offset+n_nodes)]
        
        n_nodes_z = mesh.z['n']
        n_nodes_x = mesh.x['n']
    
        velocity_in_this_mesh = np.reshape(velocity_in_this_mesh, (n_nodes_z, n_nodes_x), 'F')
        velocity_in_this_mesh[0,:] = 0.0; velocity_in_this_mesh[-1,:] = 0.0; velocity_in_this_mesh[:,0] = 0.0; velocity_in_this_mesh[:,-1] = 0.0
        velocity_in_this_mesh = np.reshape(velocity_in_this_mesh, (n_nodes_z * n_nodes_x,), 'F')
        
        migration_contribution[offset:(offset+n_nodes)] = velocity_in_this_mesh
    
    if positive_gradient:
        migration_contribution = -1.0*migration_contribution #With the definition of the residual, we need to multiply by -1 to get the positive gradient. But in most cases you want to have the negative gradient (steepest descent).
    
    if 'obj_val' in optional_return_dict.keys():
        optional_return_dict['obj_val'] = obj_val_contribution
    
    if 'adjointfield' in optional_return_dict.keys():
        optional_return_dict['adjointfield'] = adjoint_field_on_normal_truncated_grid
    
    if 'wavefield' in optional_return_dict.keys():
        optional_return_dict['wavefield'] = u_without_scattered_on_normal_truncated_grid
        
    return migration_contribution

def get_gradient(shots, truncated_solver, model, nu, optional_return_dict = dict(), positive_gradient = False):
    truncated_solver.model_parameters = model
    #If positive_gradient is false, the negative gradient is returned. This is a more useful search direction.

    n_nodes_total = np.sum(truncated_solver.get_nodes_per_mesh())

    gradient = np.zeros((n_nodes_total,), dtype='float64')    
    obj_val = 0.0
    u_list = []
    adjoint_list = []
    for shot in shots:
        migration_contrib = migration_contribution_shot(shot, truncated_solver, model, nu, optional_return_dict, positive_gradient = positive_gradient)
        gradient +=  np.real(migration_contrib) #The real part is required
        
        if 'obj_val' in optional_return_dict.keys():
            obj_val_contribution = optional_return_dict['obj_val']
            obj_val += obj_val_contribution
            
        if 'wavefield' in optional_return_dict.keys():
            u_list.append(optional_return_dict['wavefield'])
            
        if 'adjointfield' in optional_return_dict.keys():
            adjoint_list.append(optional_return_dict['adjoint_field'])
    
    if 'obj_val' in optional_return_dict.keys():
        optional_return_dict['obj_val'] = obj_val
    
    if 'wavefield' in optional_return_dict.keys():
        optional_return_dict['wavefield'] = u_list

    if 'adjointfield' in optional_return_dict.keys():
        optional_return_dict['adjointfield'] = adjoint_list
            
    return gradient

def get_gradient_as_model_perturbation_collection(shots, mesh_global, truncated_solver, model, nu, optional_return_dict = dict(), positive_gradient = False):
    grad_as_1d_flat_arr = get_gradient(shots, truncated_solver, model, nu, optional_return_dict = optional_return_dict, positive_gradient = positive_gradient)
    
    #should do something more efficient than this
    
    #first make list of 2d arrays
    grad_as_list_2d_arrs = convert_solution_vector_to_list_of_2d_arrays(grad_as_1d_flat_arr, truncated_solver, mesh_global)
    grad_as_full_2d_arr  = truncated_back_to_full(grad_as_list_2d_arrs, truncated_solver.mesh_collection, mesh_global)
    grad_as_full_1d_arr  = np.reshape(grad_as_full_2d_arr, (mesh_global.z.n * mesh_global.x.n, 1), 'F')
    
    grad_as_list_1d_arrs = truncate_array(grad_as_full_1d_arr, mesh_global, truncated_solver.domain)
    
    #now get perturbation
    return ModelPerturbationCollection(truncated_solver.mesh_collection, inputs=grad_as_list_1d_arrs)
    
def convert_solution_vector_to_list_of_2d_arrays(sol_vec, truncated_solver, mesh_global):
    #CONVERT ALL THE REGULAR NODE NUMBERS INTO GLOBAL POS. THEN FILL A GLOBAL SIZED 2D ARRAY WITH THE CORRESPONDING DATA. THEN CALL TRUNCATE_ARRAY
    truncated_mesh_collection = truncated_solver.mesh_collection
    truncated_domain_collection = truncated_mesh_collection.domain_collection
    
    x_min_global = mesh_global.domain.x['lbound']
    z_min_global = mesh_global.domain.z['lbound']

    n_nodes_x_global = mesh_global.x.n
    n_nodes_z_global = mesh_global.z.n

    dx = mesh_global.x.delta
    dz = mesh_global.z.delta
    
    array_2d_global = np.zeros((n_nodes_z_global, n_nodes_x_global), dtype=sol_vec.dtype)
    
    [indices_x_global, indices_z_global] = truncated_solver._precompute_and_return_efficient_conversion_arrays_all_regular_node_nr_global_2d_indices()
    
    all_regular_node_nrs = np.arange(len(sol_vec))
    
    array_2d_global[indices_z_global, indices_x_global] = sol_vec[all_regular_node_nrs] 
    
#     
#     for truncated_regular_node_nr in xrange(len(sol_vec)):
#         pos = truncated_solver.convert_regular_node_nr_to_pos(truncated_regular_node_nr)
#         
#         x = pos[0]
#         z = pos[1]
#         
#         index_x_global = int(np.round((x-x_min_global)/dx))
#         index_z_global = int(np.round((z-z_min_global)/dz))
#         
#         array_2d_global[index_z_global,index_x_global] = sol_vec[truncated_regular_node_nr] 

    list_of_2d_arrays = truncate_array(array_2d_global, mesh_global, truncated_domain_collection)
    return list_of_2d_arrays

def correct_boundary_nodes(truncated_velocity_list, truncated_mesh_collection, velocity_2d_global_with_correct_boundary_data, mesh_global): #MAKE SURE THEY ARE NOT PERTURBED COMPARED TO 'full_velocity_2d_with_correct_boundary_data'
    if type(truncated_velocity_list) != list: #If it was a single 2d array. Otherwise the loop would fail.
        truncated_velocity_list = [truncated_velocity_list] 

    mesh_list = truncated_mesh_collection.get_list()
    domain_collection = truncated_mesh_collection.domain_collection
    domain_list = domain_collection.get_list()
    truncated_velocity_list = copy.deepcopy(truncated_velocity_list)
    
    dx = mesh_global.x.delta
    dz = mesh_global.z.delta
    
    for mesh, domain in zip(mesh_list, domain_list):
        index = domain_list.index(domain)
        input_shape_tuple = truncated_velocity_list[index].shape 
        
        n_nodes_x_truncated = mesh.x.n
        n_nodes_z_truncated = mesh.z.n
        
        truncated_velocity_2d = np.reshape(truncated_velocity_list[index], (n_nodes_z_truncated,n_nodes_x_truncated), 'F')
        
        x_min_global = mesh_global.domain.x.lbound
        z_min_global = mesh_global.domain.z.lbound
    
        x_min_truncated = domain.x.lbound
        x_max_truncated = domain.x.rbound
        z_min_truncated = domain.z.lbound
        z_max_truncated = domain.z.rbound    

        x_node_left  = int(np.round((x_min_truncated-x_min_global)/dx))
        x_node_right = int(np.round((x_max_truncated-x_min_global)/dx))
        z_node_top   = int(np.round((z_min_truncated-z_min_global)/dz))
        z_node_bot   = int(np.round((z_max_truncated-z_min_global)/dz))
    
        #correct left
        truncated_velocity_2d[: , 0]  = velocity_2d_global_with_correct_boundary_data[z_node_top:z_node_bot+1, x_node_left] #+1 because otherwise the bottom node is not included    
        #correct right
        truncated_velocity_2d[: ,-1]  = velocity_2d_global_with_correct_boundary_data[z_node_top:z_node_bot+1, x_node_right] #+1 because otherwise the bottom node is not included
        #correct top
        truncated_velocity_2d[0 , :]  = velocity_2d_global_with_correct_boundary_data[z_node_top,x_node_left:x_node_right+1] #+1 because otherwise the right node is not included
        #correct bot
        truncated_velocity_2d[-1, :]  = velocity_2d_global_with_correct_boundary_data[z_node_bot,x_node_left:x_node_right+1] #+1 because otherwise the right node is not included
        
        #Reformat back to the input shape
        if len(input_shape_tuple) == 1:
            return_shape_tuple = (n_nodes_z_truncated * n_nodes_x_truncated,)
        elif len(input_shape_tuple) == 2:
            if input_shape_tuple[1] == 1: #Basically a 1D array with with the first dimension having all the entries and the second index is always 1.
                return_shape_tuple = (n_nodes_z_truncated*n_nodes_x_truncated, 1)
            else:
                return_shape_tuple = (n_nodes_z_truncated, n_nodes_x_truncated)
        else:
            raise Exception("Unhandled input type")
    
        truncated_velocity_list[index] = np.reshape(truncated_velocity_2d, return_shape_tuple, 'F')
    
    return truncated_velocity_list
