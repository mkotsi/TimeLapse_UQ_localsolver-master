from setuptools import setup, find_packages

setup( # Update these.  Replace 'template' with the name of your extension.
       version = '0.0', 
       name = 'pysit_extensions.petsc4py_complex_convenience',
       description = 'A petsc4py_complex_convenience for setting up pysit extension packages.',
       
       # Do not change any of these unless you know what you are doing.
       install_requires = ['setuptools'],
       packages=find_packages(),
       namespace_packages = ['pysit_extensions'],
       )