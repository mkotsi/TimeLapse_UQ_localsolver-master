import copy

import numpy as np

from pysit.solvers.model_parameter import ModelParameterBase, ModelPerturbationBase

#from util import *

__all__ = ['ShotModelWindow', 'DepthModelWindow', 'IdentityModelWindow', 'TemporalDataWindow', 'IdentityDataWindow' ]

class WindowBase(object):

    def window(self, *args, **kwargs):
        raise NotImplementedError('')

    def adjoint_window(self, *args, **kwargs):
        return self.window(*args, **kwargs)

    def complement_window(self, *args, **kwargs):
        raise NotImplementedError('')

    def adjoint_complement_window(self, *args, **kwargs):
        return self.complement_window(*args, **kwargs)

class DataWindowBase(WindowBase):
    pass

class IdentityDataWindow(DataWindowBase):

    def window(self, shot, x, *args, **kwargs):
        return x

    def complement_window(self, shot, x, *args, **kwargs):
        return 0*x

class TemporalDataWindow(DataWindowBase):

    def __init__(self, start_shift=0, start_width=0.1, end_shift=None, end_width=None):

        self.start_shift = start_shift
        self.end_shift = end_shift
        self.start_width = start_width
        self.end_width = end_width

        if self.end_shift is not None and end_width is None:
            self.end_width = start_width

    def _build_window(self, shot, data):
        ts = shot.receivers.ts

        w = np.ones_like(ts)

        wc_left = _sine_window_profile(ts, self.start_shift, self.start_width, orientation='left', complement=True)

        if end_shift is not None:
            wc_right = _sine_window_profile(ts, self.end_shift, self.end_width, orientation='right', complement=True)
        else:
            wc_right = 0*wc_left

        w = 1 - np.maximum(wx_left, wc_right)

        w.shape=-1,1

        W = np.tile(w, shot.receivers.receiver_count)

        return W.copy()

    def _build_complement_window(self, shot, data):
        W = self._build_window(shot, data)
        return np.ones_like(W) - W

    def window(self, shot, data=None):

        W = self._build_window(shot, data)

        if data is None:
            data = shot.receivers.data

        # elementwise product
        return W*data

    def complement_window(self, shot, data=None):

        Wc = self._build_complement_window(shot, data)

        if data is None:
            data = shot.receivers.data

        # elementwise product
        return Wc*data

class ModelWindowBase(WindowBase):

    def window(self, m):

        if isinstance(m, ModelParameterBase):
            # Linearize, then window the array, return a perturbation
            windowed_m = m.without_padding().linearize(asperturbation=True)
            windowed_m.data *= self.W

        elif isinstance(m, ModelPerturbationBase):
            # window the array and return a perturbation
            windowed_m = copy.deepcopy(m)
            windowed_m = windowed_m.without_padding()
            windowed_m.data *= self.W

        elif type(m) is np.ndarray:
            # window the array and return the new array
            windowed_m = self.W*m
        else:
            raise TypeError('Window array has unknown type.')

        return windowed_m

    def complement_window(self, m):

        if isinstance(m, ModelParameterBase):
            # Linearize, then window the array, return a perturbation
            windowed_m = m.without_padding().linearize(asperturbation=True)
            windowed_m.data *= self.Wc

        elif isinstance(m, ModelPerturbationBase):
            # window the array and return a perturbation
            windowed_m = copy.deepcopy(m)
            windowed_m = windowed_m.without_padding()
            windowed_m.data *= self.Wc

        elif type(m) is np.ndarray:
            # window the array and return the new array
            windowed_m = self.Wc*m
        else:
            raise TypeError('Window array has unknown type.')

        return windowed_m



class IdentityModelWindow(ModelWindowBase):

    def window(self, m):

        if isinstance(m, ModelParameterBase):
            # Linearize, then window the array, return a perturbation
            windowed_m = m.without_padding().linearize(asperturbation=True)

        elif isinstance(m, ModelPerturbationBase):
            # window the array and return a perturbation
            windowed_m = copy.deepcopy(m)
            windowed_m = windowed_m.without_padding()

        elif type(m) is np.ndarray:
            # window the array and return the new array
            windowed_m = m
        else:
            raise TypeError('Window array has unknown type.')

        return windowed_m

    def complement_window(self, m):

        if isinstance(m, ModelParameterBase):
            # Linearize, then window the array, return a perturbation
            windowed_m = m.without_padding().linearize(asperturbation=True)
            windowed_m.data *= 0.0

        elif isinstance(m, ModelPerturbationBase):
            # window the array and return a perturbation
            windowed_m = copy.deepcopy(m)
            windowed_m = windowed_m.without_padding()
            windowed_m.data *= 0.0

        elif type(m) is np.ndarray:
            # window the array and return the new array
            windowed_m = 0.0*m
        else:
            raise TypeError('Window array has unknown type.')

        return windowed_m

class DepthModelWindow(ModelWindowBase):

    def __init__(self, mesh, depth, transition_width):

        self.mesh = mesh

        self.depth = depth
        self.transition_width = transition_width

        self.W = None
        self.Wc = None

        self._build_windows()

    def _build_windows(self):

        grid = self.mesh.mesh_coords()
        ZZ = grid[-1]

        W = _sine_window_profile(ZZ, self.depth, self.transition_width, orientation='left')

        self.W = W

        self.Wc = 1 - W

class ShotModelWindow(ModelWindowBase):

    def __init__(self, mesh, radius, transition_width, shots=[], positions=None):

        self.mesh = mesh

        self.radius = radius
        self.transition_width = transition_width

        if positions is None:
            positions = set()

            for shot in shots:
                # Currently doesn't handle multisourcing
                positions.add(shot.sources.position)

                for r in shot.receivers.receiver_list:
                    positions.add(r.position)
        else:
            positions = set(positions)

        self.positions = positions
        self._build_windows(positions)

    def _build_windows(self, positions):

        grid = self.mesh.mesh_coords()

        Wc = np.zeros_like(grid[0])

        for pos in self.positions:
            wc = _point_sine_window_profile(grid, pos, self.radius, self.transition_width, complement=True)

            Wc = np.maximum(Wc, wc, out=Wc)

        self.W = 1 - Wc
        self.Wc = Wc


# windowing function
# x is a numpy 1D array
def _sine_window_profile(x, shift, width, orientation='left', complement=False):

    w = np.ones_like(x)

    if orientation == 'left':

        shift_loc = np.where(x <= shift)
        if width > 0:
            transition_loc = np.where((x > shift) & (x <= shift+width))
            transition_value = np.sin( 0.5*np.pi*(x[transition_loc]-shift)/width)**2

    elif orientation == 'right':

        x_max = x.max()
        r_shift = x_max-shift
        shift_loc = np.where(x >= r_shift)
        if width > 0:
            transition_loc = np.where((x < r_shift) & (x >= r_shift-width))
            transition_value = np.sin(0.5*np.pi*(x_max - r_shift - x[transition_loc])/width)**2

    else:
        raise ValueError("Only 'left' and 'right' are valid orientations.")

    w[shift_loc] = 0.0
    if width > 0:
        w[transition_loc] = transition_value

    if complement:
        w = 1 - w

    return w

def _point_sine_window_profile(grid, center, radius, transition_width, complement=False):

    circle = np.sqrt(sum([(g-c)**2 for g,c in zip(grid, center)]))

    plateau_loc = np.where(circle <= radius)

    if not complement:
        w = np.ones_like(circle)
        w[plateau_loc] = 0.0
    else:
        w = np.zeros_like(circle)
        w[plateau_loc] = 1.0

    if transition_width > 0.0:
        trans_loc = np.where((circle > radius) & (circle <= (radius+transition_width)))
        trans_value = np.sin(0.5*np.pi*(circle[trans_loc]-radius)/transition_width)**2

        if not complement:
            w[trans_loc] = trans_value
        else:
            w[trans_loc] = 1-trans_value

    return w