from pysit.core import *
from pysit.solvers.model_parameter import *
import numpy as np
import numbers
#DomainList and MeshList do not inherit from their respective pysit base classes, because the functionality seems specific to one region and not a set of disconnected ones as I have here.

class DomainCollection(object): #Right now it just contains a list of domain objects. If needed I can add special functionality here. 
    pass
    def __init__(self, domain_list = None):
        if domain_list == None:
            self.domain_list = []
        elif type(domain_list) == list:
            self.domain_list = domain_list
        else:
            raise Exception("Wrong parameter passed for domain_list")
        
    def set_domain_list(self, domain_list):
        self.domain_list = domain_list
        
    def get_list(self):
        return self.domain_list

class MeshCollection(object): #Right now very basic functionality.
    def __init__(self, domain_collection, mesh_list = None):
        self.domain = domain_collection #solver_base in pysit needs to make a call to the member .domain in order to store the pointer to it.
        self.domain_collection = domain_collection
        if mesh_list == None:
            self.mesh_list = []
        else:
            self.mesh_list = mesh_list
        
    def set_mesh_list(self, mesh_list):
        self.mesh_list = mesh_list

    def get_list(self):
        return self.mesh_list

class ModelPerturbationCollection(object):
    #Not inheriting from the base model parameter class, because behavior is very different 
    #Will overload addition, subtraction etc. It will call the corresponding operations on the ModelParameter objects it contains for each mesh within the collection.
    #I should have implemented it like in PySIT where the perturbation is a subclass of the model parameters. 
    #Now they are next to each other, not as nice        
    def __init__(self, mesh_collection, inputs=None, padded=False):  
        #For now hardcoded to work with constant density acoustic
        #In my case padded doesn't really do anything. But I will handle padding by invoking the padding of the modelparameter objects within the collection.
        self.mesh_collection = mesh_collection
        self.model_perturbation_list = []
        self.padded = padded
        
        if inputs != None: #If it is none, then don't initialize the modelperturbations from inputs. They will be added later
            inputs_list = inputs
            
            if type(inputs_list) != list: #It could in theory be a single ndarray if there is a single domain, and not a list of ndarrays as would be the case for multiple domains. In truncate_domain a single truncated velocity ndarray is returned for instance if there is only one domain.
                inputs_list = [inputs_list]
    
            for mesh, perturbation_array in zip(mesh_collection.get_list(), inputs_list):
                model_parameter = ConstantDensityAcousticParameters.Perturbation(mesh, inputs=perturbation_array, padded=padded) 
                self.model_perturbation_list.append(model_parameter)
            
            #I should now have an array with model_perturbations for each subdomain!

    def inner_product(self, other):
        if type(other) is type(self):
            #do inner product by using inner product functionality of member perturbations
            ip = 0.0
            for perturbation_self, perturbation_other in zip(self.model_perturbation_list, other.model_perturbation_list):
                #assuming mesh collections are same, could double check
                ip += perturbation_self.inner_product(perturbation_other)
        else:
            raise ValueError('Perturbation inner product is only defined for perturbations.')
        return ip

    def add_model_perturbation_to_list(self, perturbation):
        self.model_perturbation_list.append(perturbation)
    
    def get_list(self):
        return self.model_perturbation_list    

    @property #getter.  
    def data(self):
        ret = np.array([])
        for model_perturbation in self.model_perturbation_list: #Make one big array with the data of all the regions appended to each other.
            ret = np.append(ret, model_perturbation.data)
                
        return ret
    
    def __mul__(self, rhs):
        #This multiplies the squared slowness!
        if type(rhs) == ModelPerturbationCollection:
            raise Exception("Multiplication with other model perturbation collection not yet implemented")
        
        mesh_collection = self.mesh_collection
        result = type(self)(mesh_collection)
        
        #if a number
        if isinstance(rhs, numbers.Real):
            model_perturbation_list = self.model_perturbation_list
            for perturbation_self in model_perturbation_list:
                perturbation_res = perturbation_self*rhs 
                result.add_model_perturbation_to_list(perturbation_res)
        
        return result    

    def __rmul__(self, lhs):
        return self.__mul__(lhs) 
       
    def __add__(self, rhs):

        if type(rhs) != ModelPerturbationCollection:
            raise Exception("Can only add 'ModelPerturbationCollection' to 'ModelPerturbationCollection' for now.")

        mesh_collection = self.mesh_collection
        if len(mesh_collection.get_list()) != len(rhs.mesh_collection.get_list()):
            raise Exception("different number of regions.")
                
        result = type(self)(mesh_collection) #Will generate an empty one
        
        if type(rhs) == ModelPerturbationCollection:
            for perturbation_self, perturbation_rhs in zip(self.get_list(), rhs.get_list()):
                perturbation_res = perturbation_self + perturbation_rhs
                result.add_model_perturbation_to_list(perturbation_res)
        else:
            raise Exception("Should not get here")        
        
        return result
    
    def __radd__(self,lhs):
        raise Exception("TEST WHAT THIS DOES FIRST")
        return self.__add__(lhs)
        
        
class ModelParameterCollection(object): 
    #Not inheriting from the base model parameter class, because behavior is very different 
    #Will overload addition, subtraction etc. It will call the corresponding operations on the ModelParameter objects it contains for each mesh within the collection.        
    def __init__(self, mesh_collection, inputs=None, padded=False):  
        #For now hardcoded to work with constant density acoustic
        #In my case padded doesn't really do anything. But I will handle padding by invoking the padding of the modelparameter objects within the collection.
        self.mesh_collection = mesh_collection
        self.model_parameter_list = []
        self.padded = padded
        
        if inputs != None: #If it is none, then don't initialize the modelparameters from inputs. They will be added later
            
            if type(inputs) != dict:
                raise Exception('expect a dict!')
        
            keys = inputs.keys()
            if len(keys) != 1:
                raise Exception('dict should have only 1 key!')
        
            key = keys[0]
            if key[0] != 'C':
                raise Exception("Only works with acoustic wavespeed for now.")
            
            inputs_list = inputs[key]
        
            if type(inputs_list) != list: #It could in theory be a single ndarray if there is a single domain, and not a list of ndarrays as would be the case for multiple domains. In truncate_domain a single truncated velocity ndarray is returned for instance if there is only one domain.
                inputs_list = [inputs_list]
    
            for mesh, velocity_array in zip(mesh_collection.get_list(), inputs_list):
                model_parameter = ConstantDensityAcousticParameters(mesh, {'C': velocity_array}, padded=padded) 
                self.model_parameter_list.append(model_parameter)
            
            
            
            #I should now have an array with model_parameters for each subdomain!

    def add_model_parameters_to_list(self, model_parameters):
        self.model_parameter_list.append(model_parameters)
    
    def set_model_parameter_list(self, model_parameter_list):
        padded = model_parameter_list[0].padded #take the first one, and then for consistency check if the others in the list have the same padded property
        
        for model_parameter in model_parameter_list:
            if type(model_parameter) != ConstantDensityAcousticParameters:
                raise Exception("Passing incorrect input")
            
            if model_parameter.padded != padded:
                raise Exception("Mix of padded/unpadded in elements in model_parameter_list")
            
        self.model_parameter_list = model_parameter_list
        self.padded = padded
    
    def get_list(self):
        return self.model_parameter_list
    
    def with_padding(self, **kwargs):
        if self.padded:
            return self
        
        #if it was not padded, then pad it.
        new_model_parameter_list = []
        for model in self.model_parameter_list:
            new_model_parameter_list.append(model.with_padding(**kwargs))
            
        result = type(self)(self.mesh_collection) #Empty modelparametercollection. Would be easier if the constructor can also be generated from a list of modelparameters. But then too many special cases.
        result.set_model_parameter_list(new_model_parameter_list) #will assume the padded or unpadded property of the meshes in the list. 
        
        return result
    
    def without_padding(self):
        if not self.padded:
            return self
    
        new_model_parameter_list = []
        for model in self.model_parameter_list:
            new_model_parameter_list.append(model.without_padding())
            
        result = type(self)(self.mesh_collection) #Empty modelparametercollection. Would be easier if the constructor can also be generated from a list of modelparameters. But then too many special cases.
        result.set_model_parameter_list(new_model_parameter_list) #will assume the padded or unpadded property of the meshes in the list.
    
        return result
    
    @property #getter. Solver_base requires an array mp.data to determine if the model has changed since the last time. 
    def data(self):
        ret = np.array([])
        for model_parameters in self.model_parameter_list: #Make one big array with the data of all the regions appended to each other.
            ret = np.append(ret, model_parameters.data)
                
        return ret
    
    def __mul__(self, rhs):
        #This multiplies the squared slowness!
        if type(rhs) == ModelParameterCollection:
            raise Exception("Multiplication with other model parameter collection not yet implemented")
        
        mesh_collection = self.mesh_collection
        result = type(self)(mesh_collection)
        
        #if a number
        if isinstance(rhs, numbers.Real):
            model_parameter_list = self.model_parameter_list
            for model_self in model_parameter_list:
                model_res = model_self*rhs 
                result.add_model_parameters_to_list(model_res)
        
        return result
    
    def __rmul__(self, lhs):
        return self.__mul__(lhs)        
    
    def __add__(self, rhs):

        if type(rhs) != ModelParameterCollection and type(rhs) != ModelPerturbationCollection:
            raise Exception("Can only add 'ModelParameterCollection' or 'ModelPerturbationCollection' to 'ModelParameterCollection'.")

        mesh_collection = self.mesh_collection
        if len(mesh_collection.get_list()) != len(rhs.mesh_collection.get_list()):
            raise Exception("different number of regions.")
                
        result = type(self)(mesh_collection) #Will generate an empty one
        
        if type(rhs) == ModelParameterCollection:
            for model_self, model_rhs in zip(self.get_list(), rhs.get_list()):
                model_res = model_self + model_rhs
                result.add_model_parameters_to_list(model_res)
        elif type(rhs) == ModelPerturbationCollection:
            for model_self, perturbation_rhs in zip(self.get_list(), rhs.get_list()):
                model_res = model_self + perturbation_rhs
                result.add_model_parameters_to_list(model_res)
        else:
            raise Exception("Should not get here")        
        
        return result
    
    def __radd__(self,lhs):
        raise Exception("TEST WHAT THIS DOES FIRST")
        return self.__add__(lhs)
    
    def __sub__(self, rhs):
        if type(rhs) != ModelParameterCollection:
            raise Exception("Can only subtract a 'ModelParameterCollection' from a 'ModelParameterCollection'.")

        mesh_collection = self.mesh_collection
        if len(mesh_collection.get_list()) != len(rhs.mesh_collection.get_list()):
            raise Exception("different number of regions.")
        
        #result = type(self)(mesh_collection)
                
        #In normal pysit you just call the perturbation routine of the model parameter, but here not implemented nicely like that
        #Will manually call the modelperturbationcollection class, assuming constantdensityacoustic
        
        result = ModelPerturbationCollection(mesh_collection)
        
        if type(rhs) == ModelParameterCollection:
            for model_self, model_rhs in zip(self.get_list(), rhs.get_list()):
                perturbation_res = model_self - model_rhs
                result.add_model_perturbation_to_list(perturbation_res)
        
    
        return result
        