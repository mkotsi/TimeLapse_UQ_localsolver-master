
#POORLY IMPLEMENTED. NO GOOD CHECKING (size of inputs etc)
#The linerarizing and unlinearizing of the data is done by the model parameter members.

__all__=['JointModel', 'JointPerturbation']
import numpy as np


class JointModel(object):
    def __init__(self, m_0, m_1):
        self.m_0 = m_0
        self.m_1 = m_1

    def __add__(self, rhs):
        if type(rhs) == JointModel:
            res_m_0 = self.m_0 + rhs.m_0
            res_m_1 = self.m_1 + rhs.m_1
        elif type(rhs) == JointPerturbation:
            res_m_0 = self.m_0 + rhs.p_0
            res_m_1 = self.m_1 + rhs.p_1            
        
        result = type(self)(res_m_0, res_m_1)
        return result

    def __radd__(self,lhs):
        return self.__add__(lhs)
    
    def __iadd__(self, rhs):
        if type(rhs) == JointModel:
            self.m_0 += rhs.m_0
            self.m_1 += rhs.m_1
        elif type(rhs) == JointPerturbation:
            self.m_0 += rhs.p_0
            self.m_1 += rhs.p_1
        return self
    
    def __sub__(self, rhs):
        if type(rhs) == JointModel: 
            res_p_0 = self.m_0 - rhs.m_0 #returns perturbation
            res_p_1 = self.m_1 - rhs.m_1 #returns perturbation
        elif type(rhs) == JointPerturbation:
            res_p_0 = self.m_0 - rhs.p_0 #returns perturbation
            res_p_1 = self.m_1 - rhs.p_1 #returns perturbation
        
        result = JointPerturbation(res_p_0, res_p_1)
        return result        
        
    def __mul__(self, rhs):
        if type(rhs) == JointModel:
            res_m_0 = self.m_0 * rhs.m_0
            res_m_1 = self.m_1 * rhs.m_1
        elif type(rhs) == JointPerturbation:
            res_m_0 = self.m_0 * rhs.p_0
            res_m_1 = self.m_1 * rhs.p_1            
        # any other sort of legal product (usually a single scalar) will return a new model instance
        else:
            res_m_0 = rhs * self.m_0
            res_m_1 = rhs * self.m_1 

        result = type(self)(res_m_0, res_m_1)
        return result        

    def __rmul__(self,lhs):
        return self.__mul__(lhs)

    def validate(self):
        if self.m_0.validate() and self.m_1.validate():
            return True
        else:
            return False

class JointPerturbation(object):
    def __init__(self, p_0, p_1):
        self.p_0 = p_0
        self.p_1 = p_1
    
    #For now i will just allow the perturbation to be added to another perturbation.
    #When adding the perturbation to a model, we should do model + perturbation, so that __add__ from model is invoked and a model is returbed.
    #We don't want to return a perturbation when doing perturbation + model (although I could specifically call __add__ from the model (which is rhs in that case) instead in that case)'
    def __add__(self, rhs):
        if type(rhs) == JointPerturbation:
            res_p_0 = self.p_0 + rhs.p_0
            res_p_1 = self.p_1 + rhs.p_1         
    
        else:
            raise Exception("not handled right now")
    
        result = type(self)(res_p_0, res_p_1)
        return result
    #I MAKE THE SAME ASSUMPTION FOR THE OTHER OPERATORS. I assume that when a model and a perturbation are involved, the routines from JointModel are called
        
    def __radd__(self,lhs): #Doesn't actually help anything at all with the assumptions we are currently making about the order of operations
        return self.__add__(lhs)
    
    def __iadd__(self, rhs):
        if type(rhs) == JointPerturbation:
            self.p_0 += rhs.p_0
            self.p_1 += rhs.p_1         
        
        return self
    
    def __sub__(self, rhs):
        if type(rhs) == JointPerturbation:
            res_p_0 = self.p_0 - rhs.p_0
            res_p_1 = self.p_1 - rhs.p_1         
    
        else:
            raise Exception("not handled right now")
    
        result = type(self)(res_p_0, res_p_1)        
        return result
    
    def __isub__(self, rhs):
        if type(rhs) == JointPerturbation:
            self.p_0 -= rhs.p_0
            self.p_1 -= rhs.p_1         
        
        return self

    def __mul__(self, rhs):
        if type(rhs) == JointModel:
            raise Exception("not supposed to happen")
        elif type(rhs) == JointPerturbation:
            res_p_0 = self.p_0 * rhs.p_0
            res_p_1 = self.p_1 * rhs.p_1            
        # any other sort of legal product (usually a single scalar) will return a new model instance
        else: #EITHER RHS IS A SCALAR OR IT IS A NDARRAY THE SIZE OF p_0 and p_1
            res_p_0 = rhs * self.p_0
            res_p_1 = rhs * self.p_1 

        result = type(self)(res_p_0, res_p_1)
        return result        

    def __rmul__(self,lhs):
        return self.__mul__(lhs)
    
    def __imul__(self,rhs):
        if type(rhs) == JointModel:
            raise Exception("not supposed to happen")
        elif type(rhs) == JointPerturbation:
            self.p_0 *= rhs.p_0
            self.p_1 *= rhs.p_1            
        # any other sort of legal product (usually a single scalar) will return a new model instance
        else:
            self.p_0 *= rhs
            self.p_1 *= rhs 

        return self        
    
    def inner_product(self, other):
        if type(other) is type(self):
            ip = 0.0
            ip += self.p_0.inner_product(other.p_0)
            ip += self.p_1.inner_product(other.p_1)
        else:
            raise ValueError('Perturbation inner product is only defined for perturbations.')
        return ip         
    
    def norm(self, ord=None):
        nm = 0.0
        nm_1 = self.p_0.norm(ord=ord)
        nm_2 = self.p_1.norm(ord=ord)
        nm = np.sqrt(nm_1**2 + nm_2**2)
        return nm   