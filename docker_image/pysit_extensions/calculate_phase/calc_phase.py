#LATER MOVE THIS INTO PYSIT EXTENSION!
import numpy as np
import warnings
from scipy.signal import hilbert

__all__ = ['calc_phase',
           'envelope_trace_scipy',
           'envelope_trace_self',
           'calc_phase_trace',
           'correlation_coefficient',
           'rotate_phase'
           ]

def calc_phase(data, ret_corr_coeffs=False): #BASED ON XINFA ZHU SEG ABSTRACT WITH TITLE 'COMPARISON OF METHODS FOR MODELING PHASE VARIATION WITH ANGLE' FROM 2011
    sh = data.shape
    if len(sh) == 1: #single trace
        if ret_corr_coeffs:
            corr_coeffs = np.zeros(360)
        else:
            corr_coeffs = None
            
        best_phase = calc_phase_trace(data, corr_coeffs)
        
        if ret_corr_coeffs:
            return best_phase, corr_coeffs
        else:
            return best_phase
        
    if len(sh) == 2: #'data' is a gather
        #SHOULD USE MPI TO PARALLELIZE
        [_,nr] = sh
        best_phases = np.zeros(nr)
        
        if ret_corr_coeffs:
            corr_coeffs = np.zeros((nr,360))
        else:
            corr_coeffs = None        
        
        corr_coeffs_iter = None
        for ir in xrange(nr):
            print ir
            if ret_corr_coeffs:
                corr_coeffs_iter = corr_coeffs[ir,:]
                
            best_phases[ir] = calc_phase_trace(data[:,ir], corr_coeffs_iter)

        if ret_corr_coeffs:
            return best_phases, corr_coeffs
        else:
            return best_phases
            
    else:
        raise Exception('wrong input?')

def envelope_trace_scipy(trace):
    #(-imag part is hilbert. real part is original signal.
    #The hilbert is has zero freq zero (zero mean) in this implementation 
    #So abs gives amplitude complex valued analytic function)
    
    return np.abs(hilbert(trace))

def envelope_trace_self(trace):
    #(alternative to getting directly from the 'hilbert' function. I put zero frequency to zero as well. 
    #This gives an envelope that is much better for certain functions like the box-car. 
    #Witout zeroing the mean of the hilbert, the envelope will deviate much from zero far away from the box due to the mean shift
    hilbert_of_trace = rotate_phase(trace,90)
    hilbert_of_trace -= np.mean(hilbert_of_trace)
    return np.sqrt(trace**2 + hilbert_of_trace**2) 
    
def calc_phase_trace(trace, corr_coeffs = None):
    #WARNING SECTION
    warnings.warn('Be careful when the mean is not zero. Box-car example works correctly, but not sure if this will generally be the case.')
    
    #step 1, the locating (windowing?) of the trace is assumed to be done at this point
    trace_zero_mean = trace #Actually I'm not removing the mean right now anymore. I'm giving the freedom to the user 
    
    #Prepare correlation coefficient array.
    if type(corr_coeffs)==np.ndarray:
        if corr_coeffs.size == 360:
            corr_coeffs[:] = 0 #initialize
        else:
            raise Exception('wrong input')
    else:
        corr_coeffs = np.zeros(360)
    
    for shift in xrange(360): #Doing 1 degree at a time is somewhat arbitrary
        rotated_trace = rotate_phase(trace_zero_mean, shift) #step 2
        
        #Two equivalent ways of doing step 3. The hilbert transform is made zero mean. 
        #This is the default in scipy, and also seems to result in better envelopes in the boxcar example where 
        #the mean is significantly nonzero
        
        #step 3: way 1
        envelope_of_rotated_trace = envelope_trace_scipy(rotated_trace) 
        
        #step 3: way 2 
        #envelope_of_rotated_trace = envelope_trace_self(rotated_trace)
         
        corr_coeffs[shift] = correlation_coefficient(rotated_trace, envelope_of_rotated_trace) #step 4
        #step 5 is loop

        #print "Should I make both trace and its hilbert transform 0 mean? What is effect of deviation from mean on the phase according to algorithm?"
        #print '-imag(hilbert(rotated_trace)) ~ hilbert_of_rotated_trace -> but imag(hilbert_rotated_trace) has zero mean ! Only difference'
        #print 'real(hilbert(rotated_trace)) = rotated_trace'

        
    best_match_shift = np.argmax(corr_coeffs) #step 6
    return (360 - best_match_shift)%360 #step 7 (the mod is to get 0 instead of 360)
    
        
def correlation_coefficient(a, b): #'Local similarity with the envelope as a seismic phase detector', Fomel and van der Baan, 2010
    #a and b are two ndarrays of the same length
    corr_coeff = a.dot(b) / np.sqrt( a.dot(a) * b.dot(b))  
    return corr_coeff

def rotate_phase(trace,shift_deg):
    warnings.warn('Not touching zero frequency. Therefore rotating by 180 degrees does not gives minus the original signal.')
    #Should probably use np.fft.rfft instead since real function. More efficient ?
    #I AM NOT TOUCHING THE MEAN (0 FREQ) COMPONENT. THIS MAY GIVE UNEXPECTED RESULTS. 
    #ROTATING PHASE BY 180 DEGREES FOR INSTANCE WILL NOT GIVE MINUS THE ORIGINAL SIGNAL IF MEAN IS NOT ZERO.
     
    shift_rad = shift_deg/180.*np.pi
    
    #Do FFT
    trace_fft = np.fft.fft(trace)
    freq = np.fft.fftfreq(trace_fft.size)
    freq_shifted = np.fft.fftshift(freq)
    zero_index = np.where(freq_shifted == 0)[0][0]
    trace_fft_shifted = np.fft.fftshift(trace_fft)
    
    neg_shift = np.exp(-1j*shift_rad)
    pos_shift = np.exp( 1j*shift_rad)
    
    #Shift phase by 'shift'
    rotated_trace_fft_shifted = trace_fft_shifted #not necessary, but variable names easier to follow
    rotated_trace_fft_shifted[0:zero_index]    *= neg_shift
    rotated_trace_fft_shifted[(zero_index+1):] *= pos_shift 
    
    
    #do IFFT
    rotated_trace_fft = np.fft.ifftshift(rotated_trace_fft_shifted)
    rotated_trace = np.fft.ifft(rotated_trace_fft)
    return np.real(rotated_trace) #can get some 1e-16 imaginary values...
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pysit.core.wave_source import RickerWavelet
    peak_frequency = 25
    ts = np.linspace(0,1.0,1001)
    zero_phase_ricker = RickerWavelet(peak_frequency, t_shift = 0.5)
    zero_phase_ricker_trace = zero_phase_ricker._evaluate_time(ts)
    #zero_phase_ricker_trace = zero_phase_ricker_trace - np.mean(zero_phase_ricker_trace)
    
    thirty_phase_ricker_trace = rotate_phase(zero_phase_ricker_trace, 30)
    sixty_phase_ricker_trace  = rotate_phase(zero_phase_ricker_trace, 60)
    ninety_phase_ricker_trace = rotate_phase(zero_phase_ricker_trace, 90)
    plt.figure(1)
    plt.plot(ts, zero_phase_ricker_trace, 'r', label='0 degree')
    plt.plot(ts, thirty_phase_ricker_trace, 'b', label='30 degree')
    plt.plot(ts,  sixty_phase_ricker_trace, 'm', label='60 degree')
    plt.plot(ts, ninety_phase_ricker_trace, 'k', label='90 degree')
    plt.title('Rotated wavelets. Wrapping takes place in method') #Without the 't_shift' argument the zero phase ricker starts close to t = 0. After rotating it spreads out and wraps to the end 
    plt.legend()
    
    #investigate effect of mean. Does it change the peak angle for the correlation?
    arr_corr1 = np.zeros(360)
    arr_corr2 = np.zeros(360)
    arr_corr3 = np.zeros(360)
    print calc_phase_trace(sixty_phase_ricker_trace, corr_coeffs = arr_corr1)
    print calc_phase_trace(sixty_phase_ricker_trace + 0.1, corr_coeffs = arr_corr2)
    print calc_phase_trace(sixty_phase_ricker_trace +   1, corr_coeffs = arr_corr3)
    
    plt.figure(2)
    plt.plot(np.arange(360), arr_corr1/np.max(arr_corr1), 'r', label = '60 phase ricker')
    plt.plot(np.arange(360), arr_corr2/np.max(arr_corr2), 'b', label = '60 phase ricker changed mean')
    plt.plot(np.arange(360), arr_corr3/np.max(arr_corr3), 'g', label = '60 phase ricker changed mean more')
    plt.yticks(np.linspace(-1.5,1.5,31))
    plt.legend()
    
    #investigate estimation inaccuracy
    shift_angles = np.arange(360)
    error_ricker_estimate = np.zeros(360)
    for angle in shift_angles:
        print angle
        error_ricker_estimate[angle] = calc_phase_trace(rotate_phase(zero_phase_ricker_trace, angle)) - angle
        if error_ricker_estimate[angle] > 180 and error_ricker_estimate[angle]<360: #center around 0 this way
            error_ricker_estimate[angle] = 360 - error_ricker_estimate[angle]
         
    plt.figure(3)
    plt.plot(shift_angles, error_ricker_estimate)
    plt.title('estimation error Ricker at each angle')

    #Do the same with a zero phase wavelet with significant non-zero mean. Will estimation error still be zero?
    
    #Use box centered in middle
    nt = ts.size
    zero_phase_box_trace = np.zeros(ts.size); zero_phase_box_trace[np.round(nt/2)-50:np.round(nt/2)+50] = 1
    
    error_box_estimate = np.zeros(360)
    for angle in shift_angles:
        print angle
        error_box_estimate[angle] = calc_phase_trace(rotate_phase(zero_phase_box_trace, angle)) - angle
        if error_box_estimate[angle] > 180 and error_box_estimate[angle]<360: #center around 0 this way
            error_box_estimate[angle] = 360 - error_box_estimate[angle]    

    plt.figure(4)
    plt.plot(shift_angles, error_box_estimate)
    plt.title('estimation error Box at each angle')

    #Use box starting at zero, so there will be wrapping
    zero_phase_box_start_trace = np.zeros(ts.size); zero_phase_box_start_trace[0:100] = 1 
    
    error_box_start_estimate = np.zeros(360)
    for angle in shift_angles:
        print angle
        error_box_start_estimate[angle] = calc_phase_trace(rotate_phase(zero_phase_box_start_trace, angle)) - angle
        if error_box_start_estimate[angle] > 180 and error_box_start_estimate[angle]<360: #center around 0 this way
            error_box_start_estimate[angle] = 360 - error_box_start_estimate[angle]    

    plt.figure(5)
    plt.plot(shift_angles, error_box_start_estimate)
    plt.title('estimation error Box (at start) at each angle')

    plt.figure(6)
    plt.plot(ts, envelope_trace_scipy(zero_phase_box_trace), 'r', label="envelope 'box' by scipy implementation")
    plt.plot(ts, envelope_trace_scipy(zero_phase_box_start_trace), 'b', label="envelope 'box start' by scipy implementation")
    plt.plot(ts, envelope_trace_self( zero_phase_box_start_trace), 'g', label="envelope 'box start' by own implementation")
    plt.legend()
    plt.show()
