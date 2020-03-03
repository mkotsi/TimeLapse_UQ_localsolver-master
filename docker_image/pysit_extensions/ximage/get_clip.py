import numpy as np

def get_clip(arr, perc=99.9):
    #Based on the matlab code that winston lewis sent to me
    arr_1d = np.reshape(arr, (arr.size,), 'F')
    
    P = perc/100.0;
    S = np.sort(np.abs(arr_1d))
    N = np.size(S)
    
    x = (np.array(range(0,N)) + 0.5)/N
    x = np.insert(x,0,0)                    #First entry will be a zero
    x = np.append(x, 1)                     #Last  entry will be a one
    
    y = S[:]
    y = np.insert(y,0,S[0])
    y = np.append(y,S[-1])
    
    Y = np.interp(P, x, y)
    
    clp = Y
    return clp