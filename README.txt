These codes perform time-lapse Metropolis-Hastings inversion using a local acoustic solver. This local acoustic solver was originally developed by Willemsen et. al (2016) 

-----------------

GETTING STARTED WITH DOCKER

The docker image was originally build by Gregory Ely (http://www.mit.edu/~elyg/). 

All python codes and extensions in this image are using python 2.7

0. Install docker desktop from https://www.docker.com/ and setup an account.
1. Build the image by opening a terminal and from within the docker_image directory run: "docker build -t uqtest ."
2. Move to a directory containing the scripts of interest i.e. metropolis_hastings
3. Run: docker run -v $(pwd):/home/shared -it uqtest
	   You will now be inside a docker instance with pysit installed!  The files that were contained in your
	   current directory are now mounted to  /home/shared in the docker image.
4. Move to the shared directory: cd /home/shared
5. Run a python script: python 1_generate_true_data.py
6. Now when you exit the docker image by typing exit() any files you save to /home/shared will appear in your local directory.

-----------------

FOLDER: METROPOLIS_HASTINGS

This folder contains all the main codes to reproduce the main results from "Kotsi M., A. Malcolm, and G. Ely, Uncertainty quantification in time-lapse seismic imaging: a full-wavefield approach : Geophysical Journal International, 2020". If you find these codes useful, we would appreciate a citation. 

Some of the following codes are written in Matlab. These cannot be run through the docker image. The user will need to run them locally on their machine. 

The codes should be run in the following order:

(1) generate_true_data.py : assuming that the true models are known and saved in the folder indata, the code generates the true data for baseline and monitor model for 64 shots and 6 frequencies. The data as well as their geometries will be saved in the folder indata.
                         
(2) generate_background_greens.py : sets up the boundaries of the local domain and compute the Green's functions of the background model. Here the background model is the true baseline model. The output information is saved in outdata/truncated_solver_components
								 
(3) generate_DCT_matrices.py : using the 2D DCT formula this code generates the total number of DCT matrices that will be used to generate the Phi matrix. The DCT matrices are saved in outdata/dct_components.
                              
(4) save_true_models_in_local_domain.py : loads the true models from indata, truncates them to the local domain size and saves then in outdata/dct_components.
                                        
(5) generate_alpha_coefficients_and_Phi_matrix.m : generates the Phi matrix, performs SVD to retrieve the alpha (DCT) coefficients, and chooses the subset of 20 coefficients. All information is saved outdata/dct_components.
                                               
(6) save_true_data_from_shot_32.py : The Metropolis Hastings algorithm at this stage is run using a single shot and a single frequency. This code saves in .mat format (so they can be easily used) the data for shot number 32 and frequency of 8 Hz. The data are saved in outdata/measuredField_at_shot_32_with_freq_8.
                                   
(7) create_noisy_measurements_for_shot_32.m : given the data saved from the previous code, this code generates uncorrelated Gaussian noise given a known covariance matrix matrix. Outputs are the covariance matrix, the covariance inverse, and the noisy time-lapse data, and are saved in outdata/measuredField_at_shot_32_with_freq_8.
                                            
(8) adaptive_metropolis_hastings_inversion.py : runs the Adaptive Metropolis Hastings inversion for a single chain and saves outputs in outdata/mcmc_results.

(9) histograms_of_alpha_coefficients.m : plots the histograms of the retrieved alpha coefficients

(10) quantities_of_interest.m : calculates the three quantities of interest (vertical extent, horizontal extent, average velocity) and plots their histograms. To calculate these quantities, I am using the function "findpeaks" which part of the signal processing toolbox. If this is not already installed in your matlab, you can easily do so through the "Adds-on" tab. 

-----------------

Please let me know if there are any issues!
