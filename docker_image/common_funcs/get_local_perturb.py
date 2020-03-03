import numpy as np
import scipy.io as spio
def get_update_2d(fact):
    #return fact times the true timelapse as the perturbation. 
    #Will certainly fall within the truncated domain region
    indict = spio.loadmat("indata/marm_multiple_perturbations_model.mat")
    marm_base_true_2d     = indict['marm_base_true_2d']
    marm_moni_true_2d     = indict['marm_moni_true_2d']
    
    marm_tlps_true_2d    = marm_moni_true_2d - marm_base_true_2d
    
    return fact * marm_tlps_true_2d    