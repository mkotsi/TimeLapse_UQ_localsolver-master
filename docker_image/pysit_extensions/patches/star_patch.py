from matplotlib.patches import Polygon

#from https://github.com/freephys/Beginning-Python-Visualization/blob/master/Chapter09/src/star_patch.py

# create a star patch object
from pylab import *

def star(R, x0, y0, color='w', N=5, thin = 0.5):
    """Returns an N-pointed star of size R at (x0, y0) (matplotlib patch)."""

    polystar = zeros((2*N, 2))
    for i in range(2*N):
        angle = i*pi/N
        r = R*(1-thin*(i%2))
        polystar[i] = [r*cos(angle)+x0, r*sin(angle)+y0]
    return Polygon(polystar, fc=color, ec=color)