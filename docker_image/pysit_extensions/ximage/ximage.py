import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pysit_extensions.ximage.get_clip import get_clip

def ximage(arr, perc=99.9, aspect=1.0):
    #Based on the matlab code that winston lewis sent to me
    clp = get_clip(arr, perc)
    
    nt,nr = arr.shape
    ratio = float(nr)/nt
    imshow_aspect = ratio/aspect
    plt.figure()
    im = plt.imshow(arr, interpolation='nearest', cmap=plt.get_cmap('Greys'), aspect=imshow_aspect)
    norm =  mpl.colors.Normalize(vmin=-clp, vmax=clp)
    im.set_norm(norm)    
    plt.colorbar()
    plt.show()
