import numpy as np

def window_direct_wave(shotgather_2d, offset_arr, t_arr, len_t_window, v):
    #shotgather_2d   : 2D shotgather
    #offset_arr      : 1D array containing the offset value of each trace
    #offset_arr      : 1D array containing the time values (vertical axis shotgathers)
    #len_t_window    : length of window in time
    #v               : velocity
    
    dt = t_arr[1] - t_arr[0]
    [nt, nr] = shotgather_2d.shape
    if nr != offset_arr.size or nt != t_arr.size:
        raise Exception('Incompatible input')
    
    arrival_time      = np.abs(offset_arr) / v
    arrival_time_index= np.floor(arrival_time/dt)
    window_length     = int(np.round(len_t_window/dt)) + 1
    windowed_array    = apply_window(shotgather_2d, arrival_time_index, nr, window_length)
    return windowed_array

def window_reflected_wave(shotgather_2d, offset_arr, t_arr, len_t_window, v, reflector_depth, neg_offset = 0.0):
    #neg_offset. Value larger than 0. Positive values will start the window a little earlier than the calculated arrival time
    #the reason is that for low frequencies, the rotation can sometimes push part of the wavelet earlier than the expected arrival time it seems 
    
    dt = t_arr[1] - t_arr[0]
    [nt, nr] = shotgather_2d.shape
    if nr != offset_arr.size or nt != t_arr.size:
        raise Exception('Incompatible input')
    
    arrival_time      = 2*np.sqrt((0.5*offset_arr)**2 + reflector_depth**2) / v
    arrival_time     -= neg_offset

    arrival_time_index= np.floor(arrival_time/dt)
    window_length     = int(np.round(len_t_window/dt)) + 1
    windowed_array    = apply_window(shotgather_2d, arrival_time_index, nr, window_length)
    return windowed_array
    
def apply_window(shotgather_2d, arrival_time_index, nr, window_length):    
    windowed_array    = np.zeros((window_length, nr))
    for ir in xrange(nr):
        windowed_array[:,ir] = shotgather_2d[arrival_time_index[ir]:(arrival_time_index[ir]+window_length), ir]     
        
    return windowed_array