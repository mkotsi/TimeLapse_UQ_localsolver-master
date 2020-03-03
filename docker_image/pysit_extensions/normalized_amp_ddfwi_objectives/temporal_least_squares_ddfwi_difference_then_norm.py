from __future__ import absolute_import

import numpy as np

from pysit.objective_functions.temporal_least_squares import TemporalLeastSquares
from pysit.util.parallel import ParallelWrapShotNull
from pysit.modeling.data_modeling import generate_seismic_data
from pysit.modeling.temporal_modeling import TemporalModeling
from pysit_extensions.normalize_amplitudes import *
from pysit_extensions.give_solver_specified_dt import *
import jpype 
import copy

__all__ = ['TemporalLeastSquaresDDFWIDiffThenNormUtoD']

__docformat__ = "restructuredtext en"

class TemporalLeastSquaresDDFWIDiffThenNormUtoD(TemporalLeastSquares):
    def __init__(self, solver, shots_d1, shots_d0, res_match_obj, parallel_wrap_shot=ParallelWrapShotNull(), imaging_period = 1):
        
        
        TemporalLeastSquares.__init__(self, solver, parallel_wrap_shot, imaging_period)
        self.res_match_obj = res_match_obj #The residual match object that is used for normalizing the data. 
        
        self.shots_u0_set = False #When this is set, the next two items will be generated. See _generate_data_on_shots_u0
        self.fixed_dt_synthetic = None
        self.fixed_ts = None
        self.shots_u0 = None
        
        self.shots_d0 = shots_d0
        self.shots_d1 = shots_d1
        self.shots_dict = dict()    #in _residual we will index this to find out which d0 and u0 shot we should use when forming the double difference.
                                    #d0 and d1 may have slightly different positions. I just need their order to be the same
                                    
        
        for shot in shots_d1:
            i = shots_d1.index(shot)
            pos = shot.sources.position
            self.shots_dict[pos] = i        
            
        #To reduce the probability of errors when matching d1 and d0 shots I am going to assume that all shots have increasing x positions.
        #Here I will just verify this
        self._verify_shots_x_position_increasing(shots_d1)
        self._verify_shots_x_position_increasing(shots_d0)

        if res_match_obj._synToFld:
            raise Exception("I'm assuming synToFld should be false. It determines the order of synthetic and true data in the call self.res_match_obj.match in the self._residual function. The covmatch option of the match routine will in this case pass fld as first argument and syn as second. The udmatch array contains the scaling factor for the first argument in self.res_match_obj.match to the second this way")

    def _multiply_data_shots_by_same_constant(self, shots_d1, shots_d0, shots_u0):
        #########################################################################################################################
        #compare energy in shots and then bring energy in d0 and d1 closer to u0. d0 and d1 are much smaller right now. 
        #Maybe this makes neglecting the B* in gradient worse? Pure speculation
        #########################################################################################################################
        
        #THIS IS NOT A VERY ACCURATE METHOD AT THIS POINT BECAUSE IT DOES NOT TAKE INTO ACCOUNT DIFFERENCES IN TRACE LENGTH BETWEEN TRUE DATA AND SIMULATED DATA.
        #WILL STILL GIVE ROUGH SCALING ESTIMATE THOUGH. 
        
        avg_energy_ratio = get_average_energy_ratio(shots_d0, shots_u0, self.parallel_wrap_shot)
        
        #Make sure I multiply with same constant. Otherwise I would cause 4D differences when subtracting data
        shots_d1[0].receivers.data *= np.sqrt(avg_energy_ratio)
        shots_d0[0].receivers.data *= np.sqrt(avg_energy_ratio)    
        
        #NEED TO USE AVERAGE RATIO. OTHERWISE EVERY SHOT WOULD USE DIFFERENT MULTIPLIER. AT THAT POINT WE WOULD WEIGH SHOTS DIFFERENTLY
        
        #########################################################################################################################
        #END OF ENERGY NORMALIZE
        #########################################################################################################################
        
    def _verify_shots_x_position_increasing(self, shots):
        prev_x = -np.inf
        for i in xrange(len(shots)):
            shot = shots[i]
            shot_x = shot.sources.position[0]
            
            if shot_x > prev_x:
                prev_x = shot_x
            else:
                raise Exception('Shot locations not increasing!')            

    def _generate_shots_u0_from_m0_and_d0(self, m0):

        CFL = self.solver.cfl_safety
        min_deltas = np.min(self.solver.mesh.deltas)
        C = m0.C
        max_C = max(abs(C.min()), C.max())
        max_CFL_dt = CFL*min_deltas / max_C
        self.fixed_dt_synthetic = max_CFL_dt / 1.15 #It is highly unlikely that the maximum velocity increases by more than 15% in DDFWI. 15% is even a relatively excessive quantity I think
        self._fix_solver_dt_to_suggested_dt()
        self.fixed_ts = self.solver.ts()
        
        shots_u0 = copy.deepcopy(self.shots_d0) #So the positions of u0 are exactly the same as for d0
        
        for shot in shots_u0:
            shot.reset_time_series(self.fixed_ts)
        
        #now generate synthetic data on u0 with the specified timestep
        print "Generating u0 with the fixed timestep"
        generate_seismic_data(shots_u0, self.solver, m0, verbose=False)
        
        self.shots_u0 = shots_u0
        self.shots_u0_set = True
        print "Finished generating u0"

    def _fix_solver_dt_to_suggested_dt(self):
        set_solver_dt(self.solver,self.fixed_dt_synthetic)
            #when creating u0 shot by copying d0 shot (ensures same position), reset time series of shot. This will clear data and create array of appropriate size
        
    def _residual(self, shot, m1, dWaveOp=None): #OVERRIDE. 
        """Computes residual in the usual sense.
        shot.receivers.data contains d1

        Parameters
        ----------
        shot : pysit.Shot
            Shot for which to compute the residual.
        dWaveOp : list of ndarray (optional)
            An empty list for returning the derivative term required for
            computing the imaging condition.

        """
        
        # If we will use the second derivative info later (and this is usually
        # the case in inversion), tell the solver to store that information, in
        # addition to the solution as it would be observed by the receivers in
        # this shot (aka, the simdata).
        
        if not self.shots_u0_set: #We start DDFWI apparently. m1 is initially the same as m0
            print "ASSUMING THAT THE MODEL AT THIS POINT IS THE BASELINE INVERTED MODEL."
            self._generate_shots_u0_from_m0_and_d0(m1) #m1 is m0 at this point according to assumption

            self._multiply_data_shots_by_same_constant(self.shots_d1, self.shots_d0, self.shots_u0)
        

        rp = ['simdata']
        if dWaveOp is not None:
            rp.append('dWaveOp')

        #Make sure that u1 will be computed with the same dt. But first set the model the solver model to m1, because each model change causes a timestep reset.
        self.solver.model_parameters = m1
        self._fix_solver_dt_to_suggested_dt() 

        # Run the forward modeling step. (This will do some extra work in first iter, because m1 = m0 and we already calculated data. But we did not calculate dWaveOp. This can be optimized somehow, but it is not worth it at this point
        retval = self.modeling_tools.forward_model(shot, m1, self.imaging_period, return_parameters=rp)
        # Compute the residual vector by interpolating the measured data to the
        # timesteps used in the previous forward modeling stage.
        # resid = map(lambda x,y: x.interpolate_data(self.solver.ts())-y, shot.gather(), retval['simdata'])
        
        source_index = self.shots_dict[shot.sources.position]
        
        assert(id(self.shots_d1[source_index]) == id(shot)) #make sure indexing is going right
        
        interpolated_d1 = self.shots_d1[source_index].receivers.interpolate_data(self.solver.ts()) #same sampling rate as simdata now.
        interpolated_d0 = self.shots_d0[source_index].receivers.interpolate_data(self.solver.ts())
        
        u0 = self.shots_u0[source_index].receivers.data
        u1 = retval['simdata']
        
        delta_d = interpolated_d1 - interpolated_d0
        delta_u = u1 - u0
        
        delta_u_normalized = match_amp_first_to_second(delta_u, delta_d, self.solver.dt, self.res_match_obj)
        resid = delta_d - delta_u_normalized        
        
        # If the second derivative info is needed, copy it out
        if dWaveOp is not None:
            dWaveOp[:]  = retval['dWaveOp'][:]

        return resid        
        
    def compute_gradient(self, shots, m1, aux_info={}, **kwargs):
        #WHEN USING FIRST PART ONLY, I ONLY NEED TO MODIFY RESIDUAL. THIS TAKES CARE OF OBJECTIVE FUNCTION AND GRADIENT
        
        #print "FOR NOW I AM ONLY USING THE FIRST PART OF THE GRADIENT. See email in GMAIL June 11th 2015. When taking derivative with respect to m, there should also be a term with the derivative of the normalization operator."
        #print "I am also ignoring a potential adjoint of the normalization. See notes Tuesday week 8.
        return TemporalLeastSquares.compute_gradient(self, shots, m1, aux_info=aux_info, **kwargs)  