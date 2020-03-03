import numpy as np
import copy

def load_data(freqs, path, shots_base, shots_moni=None):
    """freqs:      a list of frequencies
       path:       string giving the path to the place where the true data is stored. 
                   the last character should be the forward slash /
       shots_base: a list of baseline shot objects
       shots_moni: a list of monitor shot objects (optional, not needed for baseline inversion)
                   
       I'm assuming the shot order is the same as in the file where the true data is generated.
       I'm also assuming the surveys are the same in this test, number of shots are the same
       Here we will load the data, and then assign it to the individual shot objects so that
       it exactly resembles the way it is stored after using the pysit data generation routines
    """
    
    if type(shots_moni) == type(None): 
        #what I do now is wasteful.
        #in most cases I will need both baseline and monitor shots to be populated
        #so I wrote the code for that. But for baseline inversion I only need baseline true data
        #In order to not have to change the code I will just create dummy monitor shots which will be populated
        #They will then just be discarded after as they are not needed 
        shots_moni = copy.deepcopy(shots_base)
    
    nshots_base = len(shots_base)
    nshots_moni = len(shots_moni)
    
    if nshots_base != nshots_moni:
        raise Exception("Inconsistent baseline and monitor shots")
    
    #surveys are the same
    nshots = nshots_base
    
    shotnr = 0
    for shot_base, shot_moni in zip(shots_base, shots_moni):
        #generate the dict for the shots
        shot_base.receivers.data_dft = dict()
        shot_moni.receivers.data_dft = dict()
        
        for freq in freqs:
            filename_base = path + 'shot_%i_base_freq_%.2f.bin'%(shotnr, freq)
            filename_moni = path + 'shot_%i_moni_freq_%.2f.bin'%(shotnr, freq)
        
            base_data_arr = np.fromfile(filename_base, dtype='complex128', count=-1)
            moni_data_arr = np.fromfile(filename_moni, dtype='complex128', count=-1)
        
            nr_base = base_data_arr.size
            nr_moni = moni_data_arr.size
        
            if nr_base != nr_moni:
                raise Exception("Assuming equal acquisition here")
        
            #surveys are the same
            nr = nr_base
        
            #data has shape (1,nr) normally, will shape it as such here
            base_data_arr = np.reshape(base_data_arr, (1,nr))
            moni_data_arr = np.reshape(moni_data_arr, (1,nr))

            #assign data for this frequency to the dictionary of the shot        
            shot_base.receivers.data_dft[freq] = base_data_arr
            shot_moni.receivers.data_dft[freq] = moni_data_arr
        
        shotnr += 1

def load_inverted_baseline_data(freqs, path, shots_inv):
    nshots = len(shots_inv)
    shotnr = 0
    for shot_inv in shots_inv:
        #generate the dict for the shots
        shot_inv.receivers.data_dft = dict()
        
        for freq in freqs:
            filename_base = path + 'shot_%i_base_freq_%.2f.bin'%(shotnr, freq)
        
            inv_data_arr = np.fromfile(filename_base, dtype='complex128', count=-1)
        
            nr = inv_data_arr.size
        
            #data has shape (1,nr) normally, will shape it as such here
            inv_data_arr = np.reshape(inv_data_arr, (1,nr))

            #assign data for this frequency to the dictionary of the shot        
            shot_inv.receivers.data_dft[freq] = inv_data_arr
        
        shotnr += 1
    