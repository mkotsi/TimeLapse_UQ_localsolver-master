import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import random

def getDCTcoeff(m,n,k,l):
    # m,n size of matrix
    # k,l X and Y degrees for DCT
    myOut = np.ones((m,n))
    #y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
    for i in range(0,n): 
        for j in range(0,m):
            tempX = np.cos((np.pi)/2*k*(2*j+1)/(2*m))
            tempY = np.cos((np.pi)/2*l*(2*i+1)/(2*n))
            
            # handle case if values are zero
            if k == 0:
                tempX = 1
            if l == 0:
                tempY = 1
            
            
            myOut[j,i] = tempX * tempY
            
    myOut = myOut/np.linalg.norm(myOut.flatten())
        
    return myOut
    # end of function
      
outputs = []

#size of the local domain in grid points
nz = 25
nx = 44

for k in range(1, nz+1):
	for l in range(1,nx+1):
		out = getDCTcoeff(nz,nx,k,l)
		outputs.append(out)
	
outputs = np.asarray(outputs)
spio.savemat('outdata/dct_components/outputs.mat', {'outputs':outputs})
	
     
