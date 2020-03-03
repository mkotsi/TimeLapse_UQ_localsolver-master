import jpype
import copy
import numpy as np

__all__ = ['match_amp_first_to_second']
__docformat__ = "restructuredtext en"

def match_amp_first_to_second(u1, u2, dt, res_match_obj):
    #MAKES ASSUMPTION THAT INTERPOLATION HAS BEEN DONE PRIOR TO ENTERING THIS ROUTINE, SO THAT THE SHAPES OF THE ARRAYS ARE EQUAL
    #By default will normalize u1 to u2. 
    #Enter u1 and u2 as 2D arrays in normal form: nt rows and nr receivers. All transposes and dtype changes will happen in this routine
    
    (nt, nr) = u1.shape
    assert((nt, nr) == u2.shape)
    
    u1_transposed_and_float32 = u1.astype('float32').transpose()
    u2_transposed_and_float32 = u2.astype('float32').transpose()
    udmatch_transposed_and_float32 = np.zeros((nr, nt), dtype='float32');
    
    u1_float_java = jpype.JArray(jpype.JFloat, u1_transposed_and_float32.ndim)(u1_transposed_and_float32.tolist())
    u2_float_java = jpype.JArray(jpype.JFloat, u2_transposed_and_float32.ndim)(u2_transposed_and_float32.tolist())
    udmatch_java = jpype.JArray(jpype.JFloat, udmatch_transposed_and_float32.ndim)(udmatch_transposed_and_float32.tolist())
    
    nrLiveSyn_java = jpype.JInt(nr) #each row is receiver
    nt_java = jpype.JInt(nt) #each column is a timestep
    dt_java = jpype.JFloat(dt)
    
    #This routine below normalizes its first entry to its second.       
    res_match_obj.match(u1_float_java, u2_float_java, udmatch_java, nt_java, dt_java, nrLiveSyn_java)

    #I'm not really sure how to efficiently transfer jpype array back. Can't find documentation for that.
    for i in xrange(nr): #not efficient, but dont know how to do else
        temp_arr = udmatch_java[i][:]
        nan_index = np.isnan(temp_arr) #when values in windows entirely 0 (before arrival), nan can be the result
        inf_index = np.isinf(temp_arr) #when values in windows entirely 0 (before arrival), inf can be the result
        temp_arr[nan_index] = 1.0 #convert nan values to 1.0
        temp_arr[inf_index] = 1.0 #convert inf values to 1.0
        udmatch_transposed_and_float32[i,:] = temp_arr    
        
    u1_modified_transposed_and_float32 =  copy.deepcopy(u1_transposed_and_float32) * udmatch_transposed_and_float32 #Quite sure I don't really need to copy, since the dtype change in line 12 probably already resulted in addressing a different piece of memory
    u1_modified = u1_modified_transposed_and_float32.transpose().astype('float64') #rest of program works with float64. Not sure how type differences would propagate to influence final result.
    
    return u1_modified    