import numpy as np
import math
from pysit.core.wave_source import SourceWaveletBase, _arrayify

class UnifDistrPhaseSource(SourceWaveletBase):
	
	""" Complex exponential with random phase. In contrary to the WhiteNoiseSource the amplitude is always 1.0 and the phase has uniform (not normal) distribution
	    I try to test whether this implementation has better results, as is suggested by the 2009 Tang paper.	    

	Notes
	-----
	
	Do not use for both time and frequency simultaneously, as realizations are
	not coherent.
	
	"""
	
	@property
	def time_source(self): return True
	
	@property
	def frequency_source(self): return True
		
	def __init__(self, seed=None, **kwargs):
		
		# time domain storage, of dubious merit for implementing in this manner.
		self._f = dict()
		# frequency domain storage
		self._f_hat = dict()
		
		self.seed = seed
		if seed is not None:
			np.random.seed(seed)
			
	def _evaluate_time(self, ts):
		
		raise Exception('Time domain implementation not (yet?) implemented')
		
	def _evaluate_frequency(self, nus):
		
		# Vectorize the frequency list
		nus_was_not_array, nus = _arrayify(nus)
		
		v = list()
		for nu in nus:
			if nu not in self._f_hat:
				#self._f_hat[nu] = self.variance*(np.random.randn() + np.random.randn()*1j)
				self._f_hat[nu] = np.exp(1j*2*np.pi*np.random.rand())
			v.append(self._f_hat[nu])
		
		return v[0] if nus_was_not_array else np.array(v)
