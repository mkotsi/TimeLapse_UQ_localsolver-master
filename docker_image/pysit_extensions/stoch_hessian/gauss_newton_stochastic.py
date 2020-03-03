from __future__ import absolute_import

import time
import copy

import numpy as np
from pysit.core import *
from scipy.sparse import lil_matrix
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import LinearOperator
from pyamg.krylov import cg, gmres

from pysit.optimization.optimization import OptimizationBase
from pysit.objective_functions.frequency_least_squares import FrequencyLeastSquares
from pysit_extensions.stoch_hessian.unif_distr_phase_source import UnifDistrPhaseSource

__all__=['GaussNewtonStochastic']

__docformat__ = "restructuredtext en"

#//TODO 2DO need to change the ensemble average functions perhaps, so that they take the receiver locations for the current shot. This would make it possible to compute the stochastic Hessian when the receivers are moving   

class GaussNewtonStochastic(OptimizationBase):

    def __init__(self, objective, shots, hessiantype, krylov_maxiter=50, n_realizations = 250, noise_type = 'white-noise',n_nodes = 0, sparse_mem_ensemble_avg = False, timing = False, use_diag_preconditioner = True, *args, **kwargs):
        OptimizationBase.__init__(self, objective, *args, **kwargs)


        
        #Certain Hessiantypes (Target Oriented) require some extra work if you want to use them for inversion, as the complete Hessian has many rows of 0.0? But then again, any Hessian has nullspace and we apply damping. Think about it more
        
        self.krylov_maxiter = krylov_maxiter
        self.n_realizations = n_realizations
        self.n_nodes = n_nodes
        self.use_diag_preconditioner = use_diag_preconditioner
        self.timing = timing
        self.used = False #THIS FLAG TRACKS WHETHER THE STOCHASTIC HESSIAN HAS BEEN CALCULATED BEFORE ALREADY. IF SO, WE NEED TO REGENERATE THE NOISE SUPERSHOTS BECAUSE OTHERWISE THE STOCHASTIC HESSIAN WILL BE EXACTLY THE SAME
        self.noise_type = noise_type
        self.sparse_mem_ensemble_avg = sparse_mem_ensemble_avg         #Using a sparse ensemble average matrix is slower because indexing takes more time. But it does not take (nx*nz)**2 elements in memory. For large problems the dense ensemble average size can become problematic. A dense matrix is normally used, but only the elements for which the Hessian will be computed are filled with non-zeros 
         
        
        
        if noise_type == 'white-noise':
            self.noisesource = WhiteNoiseSource
            self.autocorrelation = 2.0 #expected value is 2.0 (variance of real and variance of imaginary part)
        elif noise_type == 'const-amp-unif-distr-phase':
            self.noisesource = UnifDistrPhaseSource 
            self.autocorrelation = 1.0 #autocorrelation of the noise always gives amplitude 1.0 in this case
        else:
            raise Exception('The wrong information for noise source is supplied')
        
        if timing:
            self._timing_ensemble_source_list = [] #a list. On the top level there will be a dictionary for each nonlinear iteration. Each frequency will be an entry in such a dictionary with a list as key. This list will contain the time each realization of the noise correlation took for that specific frequency. Be careful, if LU decomposition is used the first realization of each frequency will cost more!!!
            self._timing_ensemble_receiver_list = []  
            self._timing_total_list = [] #a list, contains the total computation for each nonlinear iteration (noise realization, cross-correlation and Hessian element estimation, but not the shot generation at source and receiver locations or connectiondict since those are just abstractions)
        #CHECK SOURCES     
        for shot in shots:
            if type(shot.sources) != PointSource:
                raise TypeError('Stochastic Hessian only works with point sources so far')
            else:
                if shot.sources.approximation != "delta":
                    raise TypeError('Stochastic Hessian requires spatial delta functions right now')

        #USE SAME AMOUNT OF REALIZATIONS FOR SOURCES AND RECEIVERS
        n_realizations_sources = self.n_realizations
        n_realizations_receivers = self.n_realizations     

        #ASSUME THAT THE RECEIVERS DO NOT MOVE AND THAT THEY RECORD EACH SHOT. SO TO GET RECEIVER LOCATIONS WE JUST LOOP OVER RECEIVER LOCATIONS IN A SHOT
        #We only have to generate the random noise sources once, so might as well do it now. More efficient, since creating all the objects for this abstraction level is actually quite expensive
        print "Initializer: Creating supershots \n"
        self.shot_sourceloc = self._create_noise_supershots_sources(shots, n_realizations_sources)
        self.shot_receiverloc = self._create_noise_supershots_receivers(shots, n_realizations_receivers)        
        
        print "Initializer: Creating connectiondict \n"
        self._create_connectiondict(hessiantype)
        self.hessiantype = [hessiantype[0], hessiantype[1]]
        
        #Use this as a check in _get_stoch_hessian_elements. If a shots object is supplied to this routine, check if it is the same
        #as the one given to the constructor. If that is the case, don't redo the work.
        self.shots = shots
        print "Initializer: Done...\n"
        
    def _select_step(self, shots, current_objective_value, gradient, iteration, objective_arguments, **kwargs):
        """Compute the update for a set of shots.
        
        Gives the step s as a function of the gradient vector.  Implemented as in p178 of Nocedal and Wright.
        
        Parameters
        ----------
        shots : list of pysit.Shot
            List of Shots for which to compute.
        grad : ndarray
            Gradient vector.
        i : int
            Current time index.
        objective_arguments['frequencies'] gives a list with the frequencies
        
        nrealizations: should be passed on through kwargs 
            
        """
        if type(self.objective_function) != FrequencyLeastSquares:
            raise NotImplementedError("Stochastic Gauss Newton is only implemented for a frequency least squares objective function")
        
        rhs = -1*gradient.asarray()
        stochastic_hessian = self._get_stoch_hessian(objective_arguments['frequencies'], shots, **kwargs)
        
        #stochastic_hessian = stochastic_hessian / 900.0

        m = self.solver.mesh
        dx = m.x.delta
        dz = m.z.delta
        
        #Need to multiply stochastic hessian by (dx*dz) to get values at same order of magnitude as you would get using adjoint method to calculate H_appr
        #This will decrease entries in the result of 'd'
        #Difference probably due to difference in definition norm. volume integral vs sum etc...
        #If I don't multiply, the step I get a 'step' with entries that seem to be too large. After 10 reductions of alpha, norm is still decreasing each time alpha is reduced.
        #final result does not seem to change since value of alpha will adapt accordingly with different order of magnitude of 'd'  
        
        #stochastic_hessian = stochastic_hessian * float(dx*dz)
        
        resid = []
        if self.use_diag_preconditioner:
            diagonal = stochastic_hessian.diagonal()
            idiagonal = 1./diagonal
            Pinv = dia_matrix((idiagonal, 0), shape = stochastic_hessian.shape)
            d, info = gmres(stochastic_hessian, rhs, maxiter=self.krylov_maxiter, residuals=resid, M = Pinv)
        else:
            d, info = gmres(stochastic_hessian, rhs, maxiter=self.krylov_maxiter, residuals=resid)
        
        d.shape = rhs.shape    
        direction = self.solver.model_parameters.perturbation()
        direction = direction.without_padding()
        direction.data = d
        
#        d, info = cg(A, rhs, maxiter=self.krylov_maxiter, residuals=resid)
        
        
        #direction = mo.perturbation(data=d)
        
        if info < 0:
            print "CG Failure"
        if info == 0:
            print "CG Converge"
        if info > 0:
            print "CG ran {0} iterations".format(info)
        
        alpha0_kwargs = {'reset' : False} #COPIED FROM LBFGS THINK ABOUT THIS LATER
        if iteration == 0:
            alpha0_kwargs = {'reset' : True}
        
        alpha = self.select_alpha(shots, gradient, direction, objective_arguments, 
                                  current_objective_value=current_objective_value, 
                                  alpha0_kwargs=alpha0_kwargs, **kwargs)
        
        self._print('  alpha {0}'.format(alpha))
        self.store_history('alpha', iteration, alpha)
        
        step = alpha * direction        
        
        if kwargs.has_key('printfigs'):
            if kwargs['printfigs']: #if it is True
                import numpy as np
                import matplotlib.pyplot as plt
                nx = m.x.n
                nz = m.z.n
                
                gradientplot = np.reshape(gradient.data,(nz,nx),'F')
                directionplot = np.reshape(direction.data, (nz,nx), 'F')
                stepplot = np.reshape(step.data, (nz,nz), 'F')
                
                plt.ion()
                f1 = plt.figure(1)
                plt.imshow(gradientplot, interpolation = 'nearest')
                plt.colorbar()
                plt.title('gradient')
                
                f2 = plt.figure(2)
                plt.imshow(directionplot, interpolation = 'nearest')
                plt.colorbar()
                plt.title('Direction after applying Hessian')
                
                f3 = plt.figure(3)
                plt.imshow(stepplot, interpolation = 'nearest')
                plt.colorbar()
                plt.title('Step after applying Hessian and doing linesearch')
                
                f4 = plt.figure(4)
                plt.imshow(stochastic_hessian.todense(), interpolation = 'nearest')
                plt.colorbar()
                plt.title("incomplete stochastic hessian")
                
                wait = raw_input("PRESS ENTER TO CONTINUE.")
                plt.close(f1)
                plt.close(f2)
                plt.close(f3)
                plt.close(f4)
                
        return step
        
    def _compute_alpha0(self, phi0, grad0, reset=False, *args, **kwargs): #TAKEN FROM LBFGS EXAMPLE: returning a value of 1.0 is waaay to high and requires many refinements for low frequencies. For high frequencies an alpha of 1.0 was too low (probably because there is a large high frequency contribution in the layered test model I used?)
        if reset:
            return phi0 / (grad0.norm()*np.prod(self.solver.mesh.deltas))**2
        else:
            return 1.0 #a value of 1 is too small for larger frequencies sometimes?
    
    def _get_stoch_hessian(self, frequencies, shots, **kwargs):
        #nodecentered refers to getting a select set of off-diagonals corresponding to a region around each image point
        return self._get_stoch_hessian_elements(frequencies, shots, hessiantype = ['full','dummy'], **kwargs) #need to implement this better, without 'dummy'. At some parts hessiantype[0] and hessiantype[1] are directly assigned without checks right now (initializer and in get_elements
        
        
    def _get_stoch_hessian_elements(self, frequencies, hessiantype = None, shots = None, **kwargs):
        """
            frequencies: set of frequencies
            shots: The shots under consideration. Use to extract source and receiver locations
            **kwargs can contain n_realizations
            hessiantype:list. Element[0] contains flag
                        if "nodecenteredregion", then a the hessian elements are calculated for a certain region around
                        if "row", then a row of the hessian is calculated (which should be equal to a column). row, because elements in a row of a matrix are next to each other in memory in a numpy array. Still first initializing dense matrix and then turning it into sparse matrix. Not efficient. The column should be a dense list with nx * nz elements
                        if "zoomregion", then you want Hessian entries for node combinations within a certain region only
                        Element[1] contains further information. 
                        if Element[0] = "nodecenteredregion" -> Element[1] = n_nodes
                        if Element[0] = "column" -> Element[1] = columnnumber 
                        if Element[0] = "zoomregion" -> Element[1] = dictionary with geometric properties of the square. See _get_ensemble_average_for_elements function for details
                        
            If shots and hessiantype were already passed along to the constructor, they are not required anymore because all tasks for which they were required have already been done (Where to place supershots, and what nodes to calculate the Hessian for ->self.connectiondict())
        """
        
        if self.used:
            #USE SAME AMOUNT OF REALIZATIONS FOR SOURCES AND RECEIVERS
            n_realizations_sources = self.n_realizations
            n_realizations_receivers = self.n_realizations     
            print "Initializer: Creating supershots again!\n"
            self.shot_sourceloc = self._create_noise_supershots_sources(shots, n_realizations_sources)
            self.shot_receiverloc = self._create_noise_supershots_receivers(shots, n_realizations_receivers)
            self.used = False     
            
        
        if shots is not self.shots and shots is not None: #If different shots object is given 
            #ASSUME THAT THE RECEIVERS DO NOT MOVE AND THAT THEY RECORD EACH SHOT. SO TO GET RECEIVER LOCATIONS WE JUST LOOP OVER RECEIVER LOCATIONS IN A SHOT
            #We only have to generate the random noise sources once, so might as well do it now. More efficient, since creating all the objects for this abstraction level is actually quite expensive
            self.shot_sourceloc = self._create_noise_supershots_sources(shots, n_realizations_sources)
            self.shot_receiverloc = self._create_noise_supershots_receivers(shots, n_realizations_receivers)        
        
            #Use this as a check in _get_stoch_hessian_elements. If a shots object is supplied to this routine, check if it is the same
            #as the one given to the constructor. If that is the case, don't redo the work.
            self.shots = shots
            print "Just regenerated the supershots"
        
        #if frequencies is not in an iterable form, then make it! Will loop over frequencies is _get_ensemble_average_for_elements
        if not np.iterable(frequencies):
            frequencies = [frequencies] #not watertight, maybe frequencies is some strange type of object
        

        #USE SAME AMOUNT OF REALIZATIONS FOR SOURCES AND RECEIVERS
        n_realizations_sources = self.n_realizations
        n_realizations_receivers = self.n_realizations
        
            
        #make supershots at source locations. FOR NOW ASSUME THAT ALL THE SOURCE LOCATIONS ARE DISTINCT. NO SOURCE IS EXCITED TWICE AT THE SAME LOCATION. Think about implications of this later
        #Also work with delta sources right now. Sources are restricted to a single node
        solver = self.solver
        m = solver.mesh
        
        if m.dim == 2:
            nx = m.x.n
            nz = m.z.n
            
            #dx = m.x.delta
            #dz = m.z.delta

        if m.dim == 3:
            #nx = m.x.n
            #ny = m.y.n
            #nz = m.z.n
            
            #dx = m.x.delta
            #dy = m.y.delta
            #dz = m.z.delta
            raise Exception('Not yet implemented for 3D, but this code does not have to be changed a lot')

        if not hasattr(self, 'connectiondict'):
            self._create_connectiondict(hessiantype)
            self.hessiantype = [hessiantype[0], hessiantype[1]]
            print "Creating connectiondict in _get_stoch_hessian_elements function. Did you not pass hessiantype to the initializer?"
        elif not hasattr(self, 'hessiantype'):
            raise Exception('If the connectiondict is not none, but hessiantype is, then there is a problem')
        elif self.hessiantype[0] != hessiantype[0] or self.hessiantype[1] != hessiantype[1]: #new hessiantype supplied
            self._create_connectiondict(hessiantype)
            self.hessiantype = [hessiantype[0], hessiantype[1]]
            print "Creating connectiondict in _get_stoch_hessian_elements function. Did you not change hessiantype?"
            
        #determine nnodes (this in turn determines amount of diagonals)
        #first check kwargs, else default to 0, which will just give the diagonal

        if self.timing:
            self.time = time.time()
            self._timing_ensemble_source_list.append(dict())     #create dictionary for new nonlinear iteration
            self._timing_ensemble_receiver_list.append(dict())   #create dictionary for new nonlinear iteration
        
        source_ensemble_averages = self._get_ensemble_average_for_elements('sourceloc', frequencies, nx, nz)
        receiver_ensemble_averages = self._get_ensemble_average_for_elements('receiverloc', frequencies, nx, nz)
        
        #I AM ASSUMING THAT ALL THE ALL THE SOURCES HAVE THE SAME WAVELET.
        W = dict()
        for frequency in frequencies:
            W[frequency] = shots[0].sources.w(nu = frequency) #This computes the strength of the source at this frequency (Source used in seismic experiment as this appears in the definition of the Hessian, not the virtual white noise source)
        
        #the hessian contains a sum over the different frequencies involved
        
        constant = self.autocorrelation
        
        if hessiantype[0] == 'full':
            H_stochastic_incomplete = np.zeros((nx*nz,nx*nz))
            for frequency in frequencies:
                multiplier = (2*(np.pi)*frequency)**4*W[frequency]*np.conj(W[frequency])/(constant**2)/(self.n_realizations**2) #Divide by n_realizations^2 because the division has not been made in the ensemble_average subroutine. It would cost ~0.6 for each ensemble average on a 10000x10000 full matrix on my computer. More efficient to just do a single scalar matrix multiplication here because I need to multiply anyway. Same applies to sparse matrix ? don't know
                H_stochastic_incomplete += (np.multiply(source_ensemble_averages[frequency]*receiver_ensemble_averages[frequency],multiplier)).real #Using np.multiply seems to be more efficient than using the multiplication operator *
        else: #Use sparse matrix. NEED TO THINK ABOUT HOW TO DO THIS CONSISTENTLY IF THE RETURN OF THIS FUNCTION IS USED IN INVERSION. CANNOT SIMPLY GIVE BACK DIFFERENT DATATYPES? OR I WOULD HAVE TO HANDLE CASES IN INVERSION ALGORITHM
            H_stochastic_incomplete = lil_matrix((nx*nz, nx*nz))  
            for frequency in frequencies:
                multiplier = (2*(np.pi)*frequency)**4*W[frequency]*np.conj(W[frequency])/(constant**2)/(self.n_realizations**2) 
                H_stochastic_incomplete += (multiplier*source_ensemble_averages[frequency].multiply(receiver_ensemble_averages[frequency])).real 
    
            if hessiantype[0] == "row": #revise this? what if the inversion algorithm calls this routine and expects a matrix back, but now only gets a vector?
                H_stochastic_incomplete = H_stochastic_incomplete[hessiantype[1],:].todense() #The hessian only contains a row (which is equal to column). Instead of first making dense nx*nz, nx*nz matrix in source_ensemble_average it would be more efficient to directly only allocate space for a single column. But no time to implement more efficient right now
    
        if self.timing:
            self._timing_total_list.append(time.time()-self.time) #Not sure if I should include to 'todense()' step before
            print "The total time for the elements in this stochastic Hessian was %f seconds:\n"%(time.time()-self.time)
                
                
        self.used = True #For those rare occasions when you want multiple stochastic hessian realizations to be generated. This will result in the noise sources being regenerated (in order to avoid getting exactly the same stochastic hessian back)
        
        return H_stochastic_incomplete #incomplete because maybe needs damping?
             
    def _create_noise_supershots_sources(self, shots, n_realizations):
        
        #CHECK SOURCES     
        for shot in shots:
            if type(shot.sources) != PointSource:
                raise TypeError('Stochastic Hessian only works with point sources so far')
            else:
                if shot.sources.approximation != "delta":
                    raise TypeError('Stochastic Hessian requires spatial delta functions right now')
                
        m = self.solver.mesh
        shot_sourceloc = []
        for l in xrange(n_realizations):
            source_list = []
            for shot in shots:
                #at this point we know we are dealing with a delta pointsource already
                shotlocation = shot.sources.position #a tuple
                source_list.append(PointSource(m, shotlocation, self.noisesource(), approximation='delta'))
        
            junkreceiver = PointReceiver(m, shotlocation) #I believe a receiver is required to make a shot object, even though I will not use it
        
            # Create and store the shot
            shot_sourceloc.append(Shot(SourceSet(m,source_list), junkreceiver))
        
        return shot_sourceloc
        
    def _create_noise_supershots_receivers(self, shots, n_realizations):
        #ASSUME THAT THE RECEIVERS DO NOT MOVE AND THAT THEY RECORD EACH SHOT. SO TO GET RECEIVER LOCATIONS WE JUST LOOP OVER RECEIVER LOCATIONS IN A SHOT
        
        m = self.solver.mesh
        shot_receiverloc = []
        for l in xrange(n_realizations):
            source_list = []
            for receiver in shots[0].receivers.receiver_list: #ASSUMING ALL SOURCES HAVE SAME SET OF RECEIVERS WITH THE SAME LOCATIONS
                #at this point we know we are dealing with a delta pointsource already
                receiverlocation = receiver.position #a tuple
                source_list.append(PointSource(m, receiverlocation, self.noisesource(), approximation='delta'))
        
            junkreceiver = PointReceiver(m, receiverlocation) #I believe a receiver is required to make a shot object, even though I will not use it
        
            # Create and store the shot
            shot_receiverloc.append(Shot(SourceSet(m,source_list), junkreceiver))
        
        return shot_receiverloc
            
    def _get_ensemble_average_for_elements(self, indicator, frequencies, nx, nz):
        """
        Generate the ensemble average using the modeling tools.
        Do a forward model for each of the sourcesets in 'noise_sourcesets'.
    
        
        indicator -> string that is either 'sourceloc' or 'receiverloc'. It is a flag for determining which supershots should be generated.
        n_nodes -> determines how many diagonals are calculated (how many nodes you look around for approx hessian in your mesh)
            amount of diagonals is (2*n_nodes + 1)^2
        nx -> amount of nodes in horizontal direction
        nz -> amount of nodes in vertical direction
        """

        if self.connectiondict is None:
            raise Exception('The connectiondict should have been set when generating the ensemble averages.')
        
        #See if we fire the shots at the true source or true receiver locations
        if indicator == 'sourceloc':
            noise_sourcesets = self.shot_sourceloc
        elif indicator == 'receiverloc':
            noise_sourcesets = self.shot_receiverloc
        else:
            raise Exception('wrong argument supplied for string variable \'indicator\'')
        
        ensemble_averages = dict() #contains ensemble_averages for each frequency
        
        
        
        
        
        if self.timing:
            for frequency in frequencies: #Make sure all entries in the dictionary are available
                current_nonlinear_iter_index = len(self._timing_total_list) #gets incremented at the end of _get_stoch_hessian_elements
                if indicator == 'sourceloc':
                    self._timing_ensemble_source_list[current_nonlinear_iter_index][frequency] = np.zeros(self.n_realizations)
                elif indicator == 'receiverloc':
                    self._timing_ensemble_receiver_list[current_nonlinear_iter_index][frequency] = np.zeros(self.n_realizations)
                else:
                    raise Exception('wrong argument supplied for string variable \'indicator\'')
        
        
        for frequency in frequencies:
            modeling_tools = self.objective_function.modeling_tools
        
            #I am doing everything (combination of 2 nodes) double. Could cut work in half
    
            #USING NON-SPARSE MATRIX BECAUSE INDEXING
            #ensemble_average[k,connectiondict[k]] += wavefield[k]*np.ma.conjugate(wavefield[connectiondict[k]]) was extremely slow for some reason when using nodecenteredregion with 0 nodes to get the diagonal for instance. (Even with the dense hessian it is MUCH slower than computing a column of the stochastic hessian though. For a column, connectiondict only has a single ndarray entry that is very long. This is much easier. Slow loops....
            
            #ensemble_average = np.zeros((nx*nz,nx*nz), dtype='complex128')
            #if self.hessiantype[0] is 'full':
            #    ensemble_average = np.zeros((nx*nz,nx*nz), dtype='complex128')
            #else:
            #    print "Using sparse ensemble average"
            #    ensemble_average = lil_matrix((nx*nz, nx*nz), dtype='complex128')  
            #PUT ZEROS IN THE SPARSE MATRIX AT ALL LOCATIONS THAT WILL BE POPULATED LATER. OTHERWISE CANNOT DO += IF ELEMENT DOES NOT YET EXIST
    
            
            if not self.sparse_mem_ensemble_avg:
                #I REALLY NEED TO THINK OF A BETTER WAY THAN ALLOCATING A DENSE MATRIX IN WHICH ONLY A PART WILL BE FILLED WITH NON-ZEROS. ACCESSING MAY BE MUCH QUICKER, BUT SHOULD COME UP WITH SOME LISTS OF 1D ARRAYS, DEPENDING ON WHETHER WE WANT ROW, OR SOME OFFDIAGS, OR FULL
                #Make an ensemble_average object that treats all these different desired hessiantypes efficiently so this particular subroutine (_get_ensemble_average_for_elements) does not need as many hacks 
                ensemble_average = np.zeros((nx*nz,nx*nz), dtype='complex128')
            else:
                ensemble_average = lil_matrix((nx*nz, nx*nz), dtype='complex128') 
                
            nrealizations = len(noise_sourcesets)
            print "Now do a total of %i realizations"%nrealizations
    
            wavefield = np.zeros(nx*nz, dtype = 'complex128')
    
            for i in xrange(nrealizations):
                #start timer
                if self.timing:
                    self.time_inner = time.time()
                    
                r = modeling_tools.forward_model(noise_sourcesets[i], modeling_tools.solver.model_parameters.without_padding(), frequencies=frequency, return_parameters=['wavefield'])
        
                #find better way so that solver returns data in correct format. The [0] term is troublesome, cannot extract set of elements at the same time because of it
                #but only nx*nz things that are done extra. Relatively small to what happens in the loop down there

                
                wavefield = r['wavefield'][frequency].reshape(-1)

                if self.hessiantype[0] is 'full':
                    #for k in xrange(nx*nz): THIS WAS MUCH MORE EFFICIENT. See if np.outer will be even better
                        #ensemble_average[k, :] = wavefield[k]*np.ma.conjugate(wavefield) #Do one row at a time, since row-major order. 
                    ensemble_average += np.outer(wavefield,np.ma.conjugate(wavefield)) #THIS IS EVEN BETTER THAN THE COMMENTED OUT LINES ABOVE!!
                    #for a 240x40 model, using a full connectiondict and the 'else' code (not the {if self.hessiantype[0] is 'full'} code), the Hessian calculation took 400 seconds. When I loop over the rows of the Hessian and multiply the wavefield vector with a conjugate scalar entry of the wavefield vector the time was 90 seconds, but using the outer product it was only 38 seconds
 
                    #Should think about making this more memory efficient. Numbering is Row Major by default in numpy, so element [1,1] is next to [1,2] in memory, but not next to [1,7] or [2,1]. Connectiondict should be ordered I think for each row. Remember that node numbering goes down a column first though
                    #Currently, at the end of the generation of connectiondict I order the elements of each row in the dictionary. This should make memory access better. Test this out later by turning the ordering on and off and compare the speed.
                else:
                    for k in self.connectiondict.keys():
                        ensemble_average[k,self.connectiondict[k]] += wavefield[k]*np.ma.conjugate(wavefield[self.connectiondict[k]])
                
                #Write down time
                if self.timing:
                    if indicator == 'sourceloc':
                        self._timing_ensemble_source_list[current_nonlinear_iter_index][frequency][i] = time.time() - self.time_inner 
                    elif indicator == 'receiverloc':
                        self._timing_ensemble_receiver_list[current_nonlinear_iter_index][frequency][i] = time.time() - self.time_inner
                    else:
                        raise Exception('wrong argument supplied for string variable \'indicator\'')
            
            #ensemble_average = np.multiply(ensemble_average, 1.0/nrealizations) #This is faster than doing ensemble_average/nrealizations. But still takes a lot of time. 
            #Instead of doing this here for the source/receiver ensemble average, I can also do the division when computing the stochastic hessian (the place where the source and receiver ensemble averages are computed)
            if self.hessiantype[0] == 'full': #no need to sparsify now, would hog a lot of memory actually
                ensemble_averages[frequency] = ensemble_average
            else:
                ensemble_averages[frequency] = lil_matrix(ensemble_average) #return sparse version to save memory 
            
        return ensemble_averages

    def _create_connectiondict(self,hessiantype):
        #Working with 2D now
        if self.solver.mesh.dim == 3:
            raise Exception('Not yet implemented for 3D, but this code does not have to be changed a lot')        
        
        nx = self.solver.mesh.x.n 
        nz = self.solver.mesh.z.n
        
        self.connectiondict = dict()
        if hessiantype[0] == "column":
            raise Exception('column is deprecated. Due to fast memory direction is along rows of an NDarray, a row is calculated. This should be equal to a column though')
        if hessiantype[0] == "nodecenteredregion":
            try:
                hessiantype[1]
            except NameError:
                print "hessiantype[1] should contain n_nodes hessiantype[0] = nodecenteredregion"
                    
            n_nodes = hessiantype[1]
            for k in xrange(nx*nz):
                self.connectiondict[k] = self._determine_neighbours_2D(k,n_nodes, nx, nz)
        elif hessiantype[0] == "row":
            try:
                hessiantype[1]
            except NameError:
                print "hessiantype[1] should contain the element number you want the row for if hessiantype[0] = row"
                
            k = hessiantype[1]
            self.connectiondict[k] = range(0,nx*nz)
        elif hessiantype[0] == "zoomregion":
            try:
                hessiantype[1]
            except NameError:
                print "hessiantype[1] should contain a dictionary with keys zoomcolleft, zoomcolright, zoomrowtop, zoomrowbot. The corresponding values should contain the 2D array entry"
                    
            self._determine_neighbours_zoomregion_2D(self.connectiondict, hessiantype, nx, nz) #SHOULD REWRITE _determine_neighbours and make it more general! They share a lot of logic. But this was an easy fix. The variable connectiondict you pass along is manipulated. Also assuming 2D now
        elif hessiantype[0] == "full": #Don't really need connectiondict as we know we have connections between all the points. I won't actually use it when I get the ensemble averages 
            for k in xrange(nx*nz):
                self.connectiondict[k] = np.arange(0,nx*nz,dtype='int32') #arange will generate up to nx*nz-1 , which is the last node number if you start counting at 0.             
        else:
            raise Exception('hessiantype[0] contains unrecognized entry')
    
        #Now order the entries in the connectiondict, so that memory access will be more efficient later on (row-major layout)
    
        for k in self.connectiondict.keys(): #Temporary fix, only need to generate connectiondict once in initializer anyway. A better solution would be to order the entries in memory right from the start when the connectiondict is first generated.
            self.connectiondict[k].sort() #Sorts in place, does not reorder
    
    def _determine_neighbours_2D(self, k, n_nodes, nx, nz): #This gives the node numbers n_nodes around node with number k. This function is called through a loop over all nodes 'k' if hessiantype[0] == "nodecenteredregion"
   
        row = k%nz
        col = int(np.round((k-row)/float(nz)))
    
        #check how many returnelements there will be
    
        if (col - n_nodes) < 0:
            left = 0
        else:
            left = col - n_nodes
        
        if (col + n_nodes) > (nx - 1):
            right = nx - 1
        else:
            right = col + n_nodes
        
        if (row - n_nodes) < 0:
            top = 0
        else:
            top = row - n_nodes
        
        if (row + n_nodes) > (nz - 1):
            bot = nz - 1
        else:
            bot = row + n_nodes
                    
        nhor = (right - left + 1)
        nver = (bot - top + 1)
    
        n_entries = nhor * nver
    
        ret = np.zeros(n_entries, dtype='int32')        #This many node numbers will be returned
    
    
        for i in xrange(nver):
            for j in xrange(nhor):
                row_el = top + i
                col_el = left + j
                ret[i*nhor + j] = col_el*nz + row_el
            
        return ret

    def _determine_neighbours_zoomregion_2D(self, connectiondict, hessiantype, nx, nz):
        
        left = hessiantype[1]['zoomcolleft']
        right = hessiantype[1]['zoomcolright']
        top = hessiantype[1]['zoomrowtop']
        bot = hessiantype[1]['zoomrowbot']
        
        rows = np.arange(top,bot+1)         #Edited in +1 at 30 October 2013. Otherwise the bottom row is not included
        cols = np.arange(left,right+1)
        
        #node to make connections for
        for i in rows:
            for j in cols:
                nodenr_from = j*nz+i
                connectiondict[nodenr_from] = np.zeros((bot-top+1)*(right-left+1), dtype='int32')
                #node to which connection is made
                entry = 0
                for k in rows:
                    for l in cols:
                        nodenr_to = l*nz + k
                        connectiondict[nodenr_from][entry] = nodenr_to  
                        entry += 1
                        
