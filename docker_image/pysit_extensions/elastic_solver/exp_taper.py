#The 1D exp taper comes from Alison Malcolm's exp taper matlab function
from boundary_integral_helper import float_eq
import numpy as np

def exp_tap_1d(pos_arr, te1x, ts2x, tlen_l, tlen_r):
    #pos_arr: the positions of the nodes on the 1D grid. Assuming equal spacing (which will be enforced)
    #te1x   : leftmost position which is left unchanged
    #ts2x   : rightmost position which is left unchanged
    #tlen   : number of taper nodes
    
    #Verify equal spacing
    dx    = pos_arr[1] - pos_arr[0]
    eps   = 1e-8
    x_min = pos_arr[0]
    pos_no_offset_arr = pos_arr - x_min
    num_on_grid = np.sum(float_eq(0.0, (pos_no_offset_arr + 0.5*eps)%dx, epsilon = eps))
    npos  = len(pos_arr)
    if num_on_grid != npos:
        raise Exception("Not a regular grid provided")
    
    #now find the indices of ts1x and ts2x
    ind_e1 = np.where(float_eq(te1x, pos_arr) == True)[0][0]
    ind_s2 = np.where(float_eq(ts2x, pos_arr) == True)[0][0]

    ts1x   = te1x - tlen_l
    te2x   = ts2x + tlen_r

    ind_s1 = np.where(float_eq(ts1x, pos_arr) == True)[0][0]
    ind_e2 = np.where(float_eq(te2x, pos_arr) == True)[0][0]

    nwin_left  = int(np.round(tlen_l /dx))
    nwin_right = int(np.round(tlen_r/dx))

    ind_s1 = ind_e1 - nwin_left 
    ind_e2 = ind_s2 + nwin_right
     
    tap   = np.zeros(npos)
    #left side
    tap[ind_s1:ind_e1]   = np.exp(-(tlen_l + dx)/(pos_arr[ind_s1:ind_e1]-(ts1x-dx))*np.exp((tlen_l + dx)/(pos_arr[ind_s1:ind_e1]-te1x)))

    #middle
    tap[ind_e1:ind_s2+1] = 1.0
    
    #right
    tap[ind_s2+1:ind_e2+1] = np.exp(-(tlen_r+dx)/(-pos_arr[ind_s2+1:ind_e2+1]+(te2x+dx))*np.exp((tlen_r+dx)/(-pos_arr[ind_s2+1:ind_e2+1]+ ts2x)))
    
#  tap(ts2x+1:te2x-1)=exp(-(x(te2x)-x(ts2x))./(-x(ts2x+1:te2x-1)+x(te2x)).* ...
#            exp(-(x(ts2x)-x(te2x))./(-x(ts2x+1:te2x-1)+ ...
#                         x(ts2x))));    
    
    return tap

def exp_tap_2d(pos_x_arr, pos_z_arr, te1x, ts2x, te1z, ts2z, tlen_l, tlen_r, tlen_t, tlen_b):
    #logical extension of 1D input. See description above
    
    tap_x  = exp_tap_1d(pos_x_arr, te1x, ts2x, tlen_l, tlen_r)
    tap_z  = exp_tap_1d(pos_z_arr, te1z, ts2z, tlen_t, tlen_b)
    
    nx     = len(pos_x_arr)
    nz     = len(pos_z_arr)
    
    tap_2d = np.ones((nz,nx))
    tap_2d   *= tap_x
    
    tap_2d = (tap_2d.T * tap_z).T 
    return tap_2d

def exp_taper_arr_2d(arr_2d, pos_x_arr, pos_z_arr, te1x, ts2x, te1z, ts2z, tlen_l, tlen_r, tlen_t, tlen_b, bval=0):
    #apply a 2D exponential taper to array 'arr_2d'.
    #taper into background val 'bval'
    
    taper_2d   = exp_tap_2d(pos_x_arr, pos_z_arr, te1x, ts2x, te1z, ts2z, tlen_l, tlen_r, tlen_t, tlen_b)
    c_taper_2d = 1-taper_2d 
    
    ret        = arr_2d*taper_2d + bval*c_taper_2d
    return ret