import numpy as np

__all__ = ['get_average_energy_ratio']
__docformat__ = "restructuredtext en"

def get_average_energy_ratio(data_shots, synthetic_shots, pwrap):
    


    #first make sure 1 shot per process. Otherwise I'd have to loop. Also I'd have to get nshots in a different way 
    assert(len(data_shots) == 1)
    assert(len(synthetic_shots) == 1)
    
    data_ts = data_shots[0].receivers.ts
    data_nt = len(data_ts)
    data_dt = (data_ts[-1]-data_ts[0])/data_nt

    synthetic_ts = synthetic_shots[0].receivers.ts
    synthetic_nt = len(synthetic_ts)
    synthetic_dt = (synthetic_ts[-1]-synthetic_ts[0])/(synthetic_nt-1)
    
    energy_in_data_shot      = np.sum(data_dt*data_shots[0].receivers.data**2)
    energy_in_synthetic_shot = np.sum(synthetic_dt*synthetic_shots[0].receivers.data**2)
    
    nshots = pwrap.size
    
    energy_ratio = energy_in_synthetic_shot / energy_in_data_shot
    sum_of_energy_ratio = np.array(0.0)
    if pwrap.size > 1: 
        pwrap.comm.Allreduce(np.array(energy_ratio), sum_of_energy_ratio)
    elif pwrap.size == 1:
        sum_of_energy_ratio = energy_ratio
    else:
        raise Exception("unexpected result in if-clause")
        
    avg_energy_ratio = sum_of_energy_ratio[()]/nshots #goofy way to index 0d array
    if pwrap.rank == 0:
        print "energy ratio and average over shots:" + str(energy_ratio) + str(avg_energy_ratio)

    return avg_energy_ratio
    
