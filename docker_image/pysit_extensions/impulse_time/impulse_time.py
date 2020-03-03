import numpy as np

from pysit.core.wave_source import *
from pysit.core.wave_source import _arrayify

#Right now only allow for impulsive source at t = 0. But can change easily later
class ImpulseTimeWavelet(SourceWaveletBase):
    def __init__(self, t0 = 0): #t0 is the time at which the impulse takes place
        if t0 != 0:
            raise Exception('For now only work with impulse at t=0. Would require time evaluation to change, and would also require phase shift in freq domain evaluation.') 
        self.t0 = t0


    def _evaluate_time(self, ts):

        # Vectorize the time list
        ts_was_not_array, ts = _arrayify(ts)
        v = np.ones(ts.shape)
        v[ts > 0] = 0.0

        return v[0] if ts_was_not_array else v

    def _evaluate_frequency(self, nus):
        
        # Vectorize the frequency list
        nus_was_not_array, nus = _arrayify(nus)
        v = np.ones(nus.size)
        
        return v[0] if nus_was_not_array else v
    
    @property
    def time_source(self):
        """bool, Indicates if wavelet is defined in time domain."""
        return True
    
    @property
    def frequency_source(self):
        """bool, Indicates if wavelet is defined in frequency domain."""
        return True