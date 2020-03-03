from pysparse import spmatrix
import numpy as np

def csr_scp_lil_pysp(A_scp):
    [n_row, n_col] = A_scp.shape
    
    if n_row != n_col:
        raise Exception('give a square matrix please')
    
    n = n_row
    
    A_real_pysp = spmatrix.ll_mat(n,n,A_scp.nnz) #The third option is a sizehint, the number of nonzeros. Prevents new memory allocations.
    A_imag_pysp = spmatrix.ll_mat(n,n,A_scp.nnz)
    
    [row_nonzero, col_nonzero] = A_scp.nonzero()
    
    #pysparse assignments need python integers and not numpy.int32. Problems otherwise.  ->tolist() returns python ints.
    row_nonzero = row_nonzero.tolist()
    col_nonzero = col_nonzero.tolist()
    
    A_real_pysp.update_add_at(np.array(A_scp[row_nonzero,col_nonzero].real.tolist()[0]), row_nonzero, col_nonzero)
    A_imag_pysp.update_add_at(np.array(A_scp[row_nonzero,col_nonzero].imag.tolist()[0]), row_nonzero, col_nonzero)  

         
    return [A_real_pysp, A_imag_pysp]