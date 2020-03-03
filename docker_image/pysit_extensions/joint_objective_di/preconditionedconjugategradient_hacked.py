from pysit.optimization.cg import ConjugateGradient
import numpy as np
import time
import copy

from pysit_extensions.joint_objective_di.optimization_hacked import *

__all__=['PreconditionedConjugateGradient']

__docformat__ = "restructuredtext en"

class PreconditionedConjugateGradient(OptimizationBase): #THE NORMAL NOT-JOINT PreconditionedConjugateGradient OBJECT INHERITS FROM CONJUGATEGRADIENT INSTEAD. BUT DONT HAVE HACKED VERSION OF THAT. SO I WILL JUST PUT NECESSARY PARTS INTO FILE HERE. 
    def __init__(self, objective_0, objective_1 = None, reset_length=None, beta_style='fletcher-reeves', max_vel_update = 80.0, geom_fac = 0.6, stability_factor = 0.01, *args, **kwargs):
        
        if objective_1 == None: #if we only provide one objective I assume model 0 and model 1 term are same objective. This will probably work for everything, even amplitude normalization (you just provide different shots to the objective. Either shots_baseline or shots_monitor
            objective_1 = objective_0
        
        OptimizationBase.__init__(self, objective_0, objective_1, geom_fac = geom_fac, *args, **kwargs)
        
        self.max_allowed_vel_update = max_vel_update
        self.stability_factor = stability_factor #prevent division by (nearly) zero
        print "not sure if ill compensating gradient and then using in nonlinear CG is actually preconditioned CG..."

        self.prev_alpha = None

        self.reset_length = reset_length

        self.prev_gradient = None
        self.prev_direction = None

        if beta_style not in ['fletcher-reeves', 'polak-ribiere']:
            raise ValueError('Invalid beta computation method.')
        self.beta_style = beta_style

    def inner_loop(self, shots_0, shots_1, beta, model_reg_term_scale, steps, objective_arguments={}, **kwargs): #Could not use the same one as in objective hacked
        """Inner loop the optimization iteration

        This is a separate method so that the workings of the inner loop can be
        overridden without duplicating the wrapper code in the call function.

        Parameters
        ----------
        shots : list of pysit.Shot
            List of Shots for which to compute the residual.
        steps : int
            Number of iterations to run.

        """

        for step in xrange(steps):
            # Zeroth step is always the initial condition.
            tt = time.time()
            i = self.iteration

            self.store_history('value', i, self.base_model)

            self._print('Iteration {0}'.format(i))

            # extra data to try to extract from gradient call
            aux_info = {'objective_value': (True, None),
                        'residual_norm': (True, None),
                        'pseudo_hess_diag': (True, None)
                        }



            # Compute the gradient. CHANGED FOR JOINT
            gradient, objective_value, h_0, h_1 = self.joint_gradient(shots_0, shots_1, beta, model_reg_term_scale, self.base_model, objective_arguments, aux_info=aux_info)
            divide_by_me_0 = h_0 + self.stability_factor*np.max(h_0)
            divide_by_me_1 = h_1 + self.stability_factor*np.max(h_1)
            
            prec_gradient = copy.deepcopy(gradient)
            
            prec_gradient_val_0 = gradient.p_0*(1./divide_by_me_0)
            prec_gradient_val_1 = gradient.p_1*(1./divide_by_me_1)
            
            prec_gradient.p_0.data = prec_gradient_val_0
            prec_gradient.p_1.data = prec_gradient_val_1
            
            # Compute step modifier
            step = self._select_step(shots_0, shots_1, beta, model_reg_term_scale, objective_value, prec_gradient, i, objective_arguments, **kwargs)

            # Process and store meta data about the step
            step_len = step.norm()
            self.store_history('step_length', i, step_len)
            self.store_history('step', i, step)

            # Apply new step
            self.base_model += step

            ttt = time.time()-tt
            self.store_history('run_time', i, ttt)

            self.iteration += 1

            self._print('  run time {0}s'.format(ttt))


    def _select_step(self, shots, current_objective_value, gradient, iteration, objective_arguments, **kwargs):
        """Compute the CG update for a set of shots.
        COPIED FROM NORMAL NOT-JOINT CG CLASS
            
        """

        reset = (self.reset_length is not None) and (not np.mod(iteration, self.reset_length))

        if (self.prev_gradient is None) or reset:
            direction = -1*gradient
            self.prev_direction = direction
            self.prev_gradient = gradient
        else: #compute new search direction
            gkm1 = self.prev_gradient
            gk = gradient

            if self.beta_style == 'fletcher-reeves':
                beta = gk.inner_product(gk) / gkm1.inner_product(gkm1)
            elif self.beta_style == 'polak-ribiere':
                beta = (gk-gkm1).inner_product(gk) / gkm1.inner_product(gkm1)

            direction = -1*gk + beta*self.prev_direction

            self.prev_direction = direction
            self.prev_gradient = gk

        alpha0_kwargs = {'reset' : False}
        if iteration == 0:
            alpha0_kwargs = {'reset' : True}

        alpha = self.select_alpha(shots, gradient, direction, objective_arguments,
                                  current_objective_value=current_objective_value,
                                  alpha0_kwargs=alpha0_kwargs, **kwargs)

        self._print('  alpha {0}'.format(alpha))
        self.store_history('alpha', iteration, alpha)

        step = alpha * direction

        return step
            
    def _compute_alpha0(self, phi0, prec_gradient, reset=False, upscale_factor=None, **kwargs):
        
        #do some rough scaling so the max velocity update value is self.max_vel_update
        poor_scaling = True
        tol = 0.1
        alpha = 1.0
        while poor_scaling: #since nonlinear i may need more than 1 iteration ? Or I may be tired...
            updated_model = self.base_model + alpha*prec_gradient
            valid = updated_model.validate()
            while not valid: #hopefully only entered once, when we first go into loop. Later it should converge. The reason for having this is that negative linear 1/c^2 values may be created. 
                alpha/=2 
                updated_model = self.base_model + alpha * prec_gradient
                valid = updated_model.validate()    
                   
            max_update_0 = np.max(np.abs(updated_model.m_0.C - self.base_model.m_0.C))
            max_update_1 = np.max(np.abs(updated_model.m_1.C - self.base_model.m_1.C))
            
            max_update = np.max(max_update_0, max_update_1)
              
            rel = np.abs(self.max_allowed_vel_update - max_update)/self.max_allowed_vel_update
                
            print max_update, self.max_allowed_vel_update
            print "dividing by source illum may be too much close to sources? See my plots about 4d experiments where i compute ill comp grads"
                
            if  rel < tol:
                poor_scaling = False
            else:
                alpha *= self.max_allowed_vel_update/max_update        
        
        return alpha #I do scaling after illumination compensation already
#         if reset or (self.prev_alpha is None):
#             return phi0 / (grad0.norm()*np.prod(self.solver.mesh.deltas))**2
#         else:
#             return self.prev_alpha / upscale_factor               