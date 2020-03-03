from setuptools import setup, find_packages
from distutils.core import Extension
from distutils import sysconfig
import numpy as np
import platform
import os

#Hack to get rid of the excess compiler flags distutils adds by default...
#Right now I just strip the -g flag 
#http://stackoverflow.com/questions/13143294/distutils-how-to-disable-including-debug-symbols-when-building-an-extension

if platform.system() != 'Windows':  # When compilinig con visual no -g is added to params
    cflags = sysconfig.get_config_var('CFLAGS')
    opt = sysconfig.get_config_var('OPT')
    sysconfig._config_vars['CFLAGS'] = cflags.replace(' -g ', ' ')
    sysconfig._config_vars['OPT'] = opt.replace(' -g ', ' ')

if platform.system() == 'Linux':  # In macos there seems not to be -g in LDSHARED
    ldshared = sysconfig.get_config_var('LDSHARED')
    sysconfig._config_vars['LDSHARED'] = ldshared.replace(' -g ', ' ')

#os.environ is only visible to this process and children
#os.environ["CFLAGS"] = '' #I don't want the default -g flags etc which it grabs from 'python-config --cflags'
#os.environ["BASECFLAGS"]=""
#os.environ["OPT"]="" 



# An extension configuration has the following format:
# module_name' : { 'extension kwarg' : argument }
# This makes adding C from deep in the package trivial.  No more nested setup.py.

#I CAN'T GET THE WILDCARD FOR SOURCE TO WORK...
source_path = os.path.join(os.path.dirname(__file__), 'pysit_extensions','elastic_solver', 'elastic_c_code/')
extension_config = {'pysit_extensions.elastic_solver.elastic_c_code._elastic_solver' :
                          { 'sources' : [source_path +'fd2d_memsetup.c', source_path +'fd2d_model.c', source_path +'fd2d_output.c', source_path +'fd2d_pml.c', source_path +'fd2d_readin_struct.c', source_path +'fd2d_source.c', source_path +'fd2d_update_fsbc.c', source_path +'fd2d_update_RSG2.c', source_path +'fd2d_update_RSG4.c', source_path +'fd2d_update_SSG_Acoustic.c', source_path +'fd2d_update_SSG.c', source_path +'fd2dfracture.c', source_path + 'fd2d_rec_wavefields.c', source_path + 'fd2d_update_boundaries.c', source_path + 'get_physical_cpu_count.c'],
                            'extra_compile_args' :  ["-O3","-fopenmp","-ffast-math"],
                            'include_dirs' : [np.get_include(), os.path.join(os.path.dirname(__file__), 'pysit_extensions','elastic_solver', 'elastic_c_code')],
                            'libraries' :  ['gomp'],
                            'library_dirs' : ['/usr/lib64']
                          },
                   }

extensions = [Extension(key, **value) for key, value in extension_config.iteritems()]

setup( # Update these.  Replace 'template' with the name of your extension.
    version = '0.0', 
    name = 'pysit_extensions.example',
    description = 'A template for setting up pysit extension packages.',
    ext_modules = extensions,
    
    # Do not change any of these unless you know what you are doing.
    install_requires = ['setuptools'],
    packages=find_packages(),
    namespace_packages = ['pysit_extensions'],
    zip_safe = False
    )
