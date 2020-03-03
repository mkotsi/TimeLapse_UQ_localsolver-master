from ctypes import *

NOUTVAR=12 #hardcoded in C code

class outparams(Structure): #Modeling the struct that is used in the c code
    _fields_ = [("l", c_int),
                ("ii", c_int),
                ("ie", c_int),
                ("di", c_int)
                ] 

class ssparams(Structure):
    _fields_ = [("var", c_int * NOUTVAR),
                ("m0", c_int),
                ("dm", c_int),
                ("mm", c_int),
                ("k0", c_int),
                ("dk", c_int),
                ("kk", c_int),
                ("M", c_int),
                ("K", c_int)
                ]

class boundary_params(Structure):
    _fields_ = [("rec_boundary", c_int),
                ("local_solve", c_int),
                ("rec_x_l", c_double),
                ("rec_x_r", c_double),
                ("rec_z_t", c_double),
                ("rec_z_b", c_double)
                ]

class info_struct(Structure):
    #Take pysit input and create a struct like object that can be passed to the C code.
    #This struct will basically replace the input script that is used in xindings default code.
    #Alternatively I could also just have generated input scripts each time I would call the C code.
    #But that would probably have been more cumbersome.
    
    #NOT INCLUDING BECAUSE WILL HARDCODE: 
    #input_file (will remove)
    #igrid (we always use given model)
    #model_file (will remove, directly pass model)
    #model_type, same as model_file
    #NDF -> 0
    #Iheter -> 0
    #iwavelet
    
    _fields_ = [('MM', c_int), #nx without nodes in pml
                ('KK', c_int), #nz without nodes in pml
                ("nabs", c_int),
                ("ifsbc", c_int),
                ("d0factor", c_double),
                ("PPW0", c_double),
                ("p_power", c_double),
                ("dx", c_double), #dx = dz
                ("dt", c_double),
                ("itimestep", c_int),
                ("gx0", c_double),
                ("gz0", c_double),
                ("iwavelet", c_int),
                ("amp0", c_double),
                ("freq0", c_double), 
                ("isourcecomp", c_int), 
                ("nsrc", c_int),
                ("srcx", POINTER(c_double)),
                ("srcz", POINTER(c_double)),                
                ("screen", outparams),
                ("snap", outparams),
                ("hist", outparams),
                ("histvar", c_int * NOUTVAR),
                ("bparams", boundary_params),
                ("ss", ssparams),
                ("nhist", c_int), #number of geophones
                ("rcvx", POINTER(c_double)),
                ("rcvz", POINTER(c_double)),
                ("traces_mem", c_int),
                ("snaps_mem", c_int),
                ('traces_output_dir', c_char_p),
                ('snaps_output_dir', c_char_p)
                ] 