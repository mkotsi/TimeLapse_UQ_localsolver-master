from __future__ import absolute_import

import copy

import numpy as np

from pysit.objective_functions.temporal_least_squares import TemporalLeastSquares
from pysit.util.parallel import ParallelWrapShotNull
from pysit.modeling.temporal_modeling import TemporalModeling
from pysit_extensions.normalize_amplitudes import *
from pysit_extensions.windowed_inversion import IdentityModelWindow, IdentityDataWindow
#from pysit_extensions.probing import IdentityModelWindow, IdentityDataWindow
import jpype 

__all__ = ['NormalizedAmplitudeWindowedTemporalLeastSquares']

__docformat__ = "restructuredtext en"



class NormalizedAmplitudeWindowedTemporalLeastSquares(TemporalLeastSquares):
    """ How to compute the parts of the objective you need to do optimization """

    def __init__(self, solver, res_match_obj, normalize_d_to_u = True , model_window=IdentityModelWindow(), data_window=IdentityDataWindow(), parallel_wrap_shot=ParallelWrapShotNull(), imaging_period = 1):

        self.solver = solver

        self.modeling_tools = TemporalModeling(solver)

        self.parallel_wrap_shot = parallel_wrap_shot

        self.W_model = model_window
        self.W_data = data_window

        TemporalLeastSquares.__init__(self, solver, parallel_wrap_shot, imaging_period)
        self.res_match_obj = res_match_obj #The residual match object that is used for normalizing the data. 
        self.normalize_d_to_u = normalize_d_to_u

        if res_match_obj._synToFld:
            raise Exception("I'm assuming synToFld should be false. It determines the order of synthetic and true data in the call self.res_match_obj.match in the self._residual function. The covmatch option of the match routine will in this case pass fld as first argument and syn as second. The udmatch array contains the scaling factor for the first argument in self.res_match_obj.match to the second this way")

    def set_fixed_model(self, m):
        """ If m is an array, it needs to be in linear form.  Better to pass a ModelParameter."""
        self.base_perturbation = self.W_model.complement_window(m)


    def _residual(self, shot, m0, dWaveOp=None): #OVERRIDE
        """Computes residual in the usual sense.

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
        rp = ['simdata']
        if dWaveOp is not None:
            rp.append('dWaveOp')

        # Run the forward modeling step
        retval = self.modeling_tools.forward_model(shot, m0, self.imaging_period, return_parameters=rp)



        # Compute the residual vector by interpolating the measured data to the
        # timesteps used in the previous forward modeling stage.
        # resid = map(lambda x,y: x.interpolate_data(self.solver.ts())-y, shot.gather(), retval['simdata'])
        
        interpolated_real_data = shot.receivers.interpolate_data(self.solver.ts()) #same sampling rate as simdata now.
 
        if self.normalize_d_to_u: #NORMALIZE DATA
            d_modified = match_amp_first_to_second(interpolated_real_data, retval['simdata'], self.solver.dt, self.res_match_obj)
            resid = d_modified - retval['simdata']            

            #print "??? really working as expected ? At one point I put array in there in a rotated fashion and the results were the same. Maybe bcecause in my tests the scaling was only in the t direction and the algorithm would be insensitive to which direction it would move ?"
            
            #print "??? will it matter that d_modified is derived from 32 bit data ? Should I write my own 64 bit function ?"
            
            #import matplotlib.pyplot as plt
            #receiver_plot_nr = np.round(2.0*n_receiver/3)
            #plt.plot(self.solver.ts(), interpolated_real_data[:,receiver_plot_nr], 'r', label='data true', lw=2)
            #plt.plot(self.solver.ts(), d_modified[:,receiver_plot_nr], 'b', label='data normalized', lw=2)
            #plt.plot(self.solver.ts(), retval['simdata'][:,receiver_plot_nr], 'g', label='simdata', lw=2)
            #plt.legend()
            #plt.show()
            
        else: #NORMALIZE SYNTHETIC
            sim_modified = match_amp_first_to_second(retval['simdata'], interpolated_real_data, self.solver.dt, self.res_match_obj)
            resid = interpolated_real_data - sim_modified
            
            #import matplotlib.pyplot as plt
            #receiver_plot_nr = np.round(2.0*n_receiver/3)
            #plt.plot(self.solver.ts(), interpolated_real_data[:,receiver_plot_nr], 'r', label='data true', lw=2)
            #plt.plot(self.solver.ts(), sim_modified[:,receiver_plot_nr], 'b', label='synthetic normalized', lw=2)
            #plt.plot(self.solver.ts(), retval['simdata'][:,receiver_plot_nr], 'g', label='simdata', lw=2)
            #plt.legend()
            #plt.show()

        # If the second derivative info is needed, copy it out
        if dWaveOp is not None:
            dWaveOp[:]  = retval['dWaveOp'][:]

        return resid

    def evaluate(self, shots, m0, **kwargs):
        """ Evaluate the least squares objective function over a list of shots."""

        m0_ = self._build_true_m0(m0)

        r_norm2 = 0
        for shot in shots:
            r = self._residual(shot, m0_)
            r_windowed = self.W_data.window(shot, r)
            r_norm2 += np.linalg.norm(r_windowed)**2

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:
            # Allreduce wants an array, so we give it a 0-D array
            new_r_norm2 = np.array(0.0)
            self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
            r_norm2 = new_r_norm2[()] # goofy way to access 0-D array element

        return 0.5*r_norm2*self.solver.dt

    def _gradient_helper(self, shot, m0, ignore_minus=False, ret_pseudo_hess_diag_comp = False, **kwargs):
        """Helper function for computing the component of the gradient due to a
        single shot.

        Computes F*_s(d - scriptF_s[u]), in our notation.

        Parameters
        ----------
        shot : pysit.Shot
            Shot for which to compute the residual.

        """

        # Compute the residual vector and its norm
        dWaveOp=[]
        r = self._residual(shot, m0, dWaveOp=dWaveOp, **kwargs)

        r_windowed = self.W_data.window(shot, r)

        # Perform the migration or F* operation to get the gradient component
        g = self.modeling_tools.migrate_shot(shot, m0, self.W_data.adjoint_window(shot, r_windowed), self.imaging_period, dWaveOp=dWaveOp)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if not ignore_minus:
            g = -1*g

        if ret_pseudo_hess_diag_comp:
            return g, r, r_windowed, self._pseudo_hessian_diagonal_component_shot(dWaveOp)
        else:
            return g, r, r_windowed

    def _pseudo_hessian_diagonal_component_shot(self, dWaveOp):
        #Shin 2001: "Improved amplitude preservation for prestack depth migration by inverse scattering theory". 
        #Basic illumination compensation. In here we compute the diagonal. It is not perfect, it does not include receiver coverage for instance.
        #Currently only implemented for temporal modeling. Although very easy for frequency modeling as well. -> np.real(omega^4*wavefield * np.conj(wavefield)) -> np.real(dWaveOp*np.conj(dWaveOp))
        
        mesh = self.solver.mesh
          
        import time
        tt = time.time()
        pseudo_hessian_diag_contrib = np.zeros(mesh.unpad_array(dWaveOp[0], copy=True).shape)
        for i in xrange(len(dWaveOp)):                          #Since dWaveOp is a list I cannot use a single numpy command but I need to loop over timesteps. May have been nicer if dWaveOp had been implemented as a single large ndarray I think
            unpadded_dWaveOp_i = mesh.unpad_array(dWaveOp[i])   #This will modify dWaveOp[i] ! But that should be okay as it will not be used anymore.
            pseudo_hessian_diag_contrib += unpadded_dWaveOp_i*unpadded_dWaveOp_i

        pseudo_hessian_diag_contrib *= self.imaging_period #Compensate for doing fewer summations at higher imaging_period

        print "Time elapsed when computing pseudo hessian diagonal contribution shot: %e"%(time.time() - tt)

        return pseudo_hessian_diag_contrib

    def compute_gradient(self, shots, m0, aux_info={}, **kwargs):
        """Compute the gradient for a set of shots.

        Computes the gradient as
            -F*(d - scriptF[m0]) = -sum(F*_s(d - scriptF_s[m0])) for s in shots

        Parameters
        ----------
        shots : list of pysit.Shot
            List of Shots for which to compute the gradient.
        i : int
            Current time index.

        """

        m0_ = self._build_true_m0(m0)

        # compute the portion of the gradient due to each shot
        grad = m0_.perturbation()
        r_norm2 = 0.0
        r_windowed_norm2 = 0.0
        pseudo_h_diag = np.zeros(m0_.asarray().shape)
        for shot in shots:
            if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
                g, r, r_windowed, h = self._gradient_helper(shot, m0_, ignore_minus=True, ret_pseudo_hess_diag_comp = True, **kwargs)
                pseudo_h_diag += h 
            else:
                g, r, r_windowed = self._gradient_helper(shot, m0_, ignore_minus=True, **kwargs)            
            
            grad -= g # handle the minus 1 in the definition of the gradient of this objective
            r_norm2 += np.linalg.norm(r)**2
            r_windowed_norm2 += np.linalg.norm(r_windowed)**2

        # Window in model space
        grad = m0_.perturbation(data=self.W_model.adjoint_window(grad.data))
        if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
            pseudo_h_diag = self.W_model.adjoint_window(pseudo_h_diag)

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:
            # Allreduce wants an array, so we give it a 0-D array
            new_r_norm2 = np.array(0.0)
            self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
            r_norm2 = new_r_norm2[()] # goofy way to access 0-D array element

            # Allreduce wants an array, so we give it a 0-D array
            new_r_windowed_norm2 = np.array(0.0)
            self.parallel_wrap_shot.comm.Allreduce(np.array(r_windowed_norm2), new_r_windowed_norm2)
            r_windowed_norm2 = new_r_windowed_norm2[()] # goofy way to access 0-D array element

            ngrad = np.zeros_like(grad.asarray())
            self.parallel_wrap_shot.comm.Allreduce(grad.asarray(), ngrad)
            grad=m0_.perturbation(data=ngrad)

            if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
                pseudo_h_diag_temp = np.zeros(pseudo_h_diag.shape)
                self.parallel_wrap_shot.comm.Allreduce(pseudo_h_diag, pseudo_h_diag_temp)
                pseudo_h_diag = pseudo_h_diag_temp 

        # account for the measure in the integral over time
        r_norm2 *= self.solver.dt
        r_windowed_norm2 *= self.solver.dt
        pseudo_h_diag *= self.solver.dt #The gradient is implemented as a time integral in TemporalModeling.adjoint_model(). I think the pseudo Hessian (F*F in notation Shin) also represents a time integral. So multiply with dt as well to be consistent.

        # store any auxiliary info that is requested
        if ('residual_norm' in aux_info) and aux_info['residual_norm'][0]:
            aux_info['residual_norm'] = (True, np.sqrt(r_norm2))
        if ('objective_value' in aux_info) and aux_info['objective_value'][0]:
            aux_info['objective_value'] = (True, 0.5*r_windowed_norm2)
        if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
            aux_info['pseudo_hess_diag'] = (True, pseudo_h_diag)

        return grad

    def apply_hessian(self, shots, m0, m1, hessian_mode='approximate', levenberg_mu=0.0, *args, **kwargs):

        modes = ['approximate', 'full', 'levenberg']
        if hessian_mode not in modes:
            raise ValueError("Invalid Hessian mode.  Valid options for applying hessian are {0}".format(modes))

        m0_ = self._build_true_m0(m0)

        result = m0_.perturbation()

        if hessian_mode in ['approximate', 'levenberg']:
            for shot in shots:
                # Run the forward modeling step
                retval = self.modeling_tools.forward_model(shot, m0_, return_parameters=['dWaveOp'])
                dWaveOp0 = retval['dWaveOp']

                linear_retval = self.modeling_tools.linear_forward_model(shot, m0_, self.W_model.window(m1), return_parameters=['simdata'], dWaveOp0=dWaveOp0)

                d1 = self.W_data.window(shot, linear_retval['simdata']) # data from F applied to m1
                d1 = self.W_data.adjoint_window(shot, d1)
                result += self.modeling_tools.migrate_shot(shot, m0_, d1, dWaveOp=dWaveOp0)

        elif hessian_mode == 'full':
            raise NotImplementedError('Full Hessian not implemented.')
#           for shot in shots:
#               # Run the forward modeling step
#               dWaveOp0 = list() # wave operator derivative wrt model for u_0
#               r0 = self._residual(shot, m0_, dWaveOp=dWaveOp0, **kwargs)
#
#               linear_retval = self.modeling_tools.linear_forward_model(shot, m0_, m1, return_parameters=['simdata', 'dWaveOp1'], dWaveOp0=dWaveOp0)
#               d1 = linear_retval['simdata']
#               dWaveOp1 = linear_retval['dWaveOp1']
#
#               # <q, u1tt>, first adjointy bit
#               dWaveOpAdj1=[]
#               res1 = self.modeling_tools.migrate_shot( shot, m0_, r0, dWaveOp=dWaveOp1, dWaveOpAdj=dWaveOpAdj1)
#               result += res1
#
#               # <p, u0tt>
#               res2 = self.modeling_tools.migrate_shot(shot, m0_, d1, operand_dWaveOpAdj=dWaveOpAdj1, operand_model=m1, dWaveOp=dWaveOp0)
#               result += res2

        result = m0_.perturbation(data=self.W_model.adjoint_window(result.data))

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:

            nresult = np.zeros_like(result.asarray())
            self.parallel_wrap_shot.comm.Allreduce(result.asarray(), nresult)
            result = m0_.perturbation(data=nresult)

        # Note, AFTER the application has been done in parallel do this.
        if hessian_mode == 'levenberg':
            result += levenberg_mu*m1

        return result

    def _build_true_m0(self, m):
        """ Constructs m0 = b + W*m """

        m0 = copy.deepcopy(m)
        m0.data *= np.infty
        m0 += self.base_perturbation
        m0 += self.W_model.window(m)

        if np.any(m0 == np.infty):
            raise ValueError('Prescribed window leaves infinite velocities.')

        return m0
