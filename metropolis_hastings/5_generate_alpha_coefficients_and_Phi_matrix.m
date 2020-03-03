
clc
clear

% Generate the full Phi matrix
load /outdata/dct_components/outputs.mat
N = size(outputs,1);
phimatrix_full = zeros(N,N);

for i = 1 : N
    outnew = squeeze(outputs(i,:,:));
    phimatrix_full(:,i) = outnew(:);
end


%% 

%load the true models

load true_models_in_local_domain.mat


%turn the time lapse model into a vector

tlchangevec = truncated_timelapse_velocity_2d(:);


% We assume that the time-lapse model can be decomposed such that 
% delta_m = Phi * alphas. Knowing the true time-lapse model and the phi 
% matrix, we first need to solve an inverse problem to recover all the DCT 
% coefficients (aka alphas).


% Singular value decomposition 

[u s v] = svd(phimatrix_full);


% singular values decay very fast after 300

figure(1)
subplot(1,2,1)
plot(log(diag(s)), 'linewidth',2)
subplot(1,2,2)
plot((diag(s)), 'linewidth',2)

%% 

% truncate after 300 values 

[u s v] = svds (phimatrix_full,300);

% pseudo inverse is.

myInv2 = v * s^-1 * u';


%% Get the alphas and the reconstructed 4d

alphas = myInv2 * tlchangevec;

reconstructed_tl_model = phimatrix_full * alphas; % do this for validation

reconstructed_tl_model_2d = reshape(reconstructed_tl_model, [25,44]);

figure(2);
imagesc(reconstructed_tl_model_2d)

%% Alphas as a function of m,n

alphas_2d = reshape(alphas, [44,25]);
alphas_2d = alphas_2d';

figure(3)
imagesc(alphas_2d)

%% Alpha subset

% Take a 5-by-5 block from the upper left corner

alphas_subset = alphas_2d(1:5,1:5);
figure; imagesc(alphas_subset)

% The 2nd column has very small values compared to the others, so I can
% exlude to reduce the number of parameters even more

alphas_20 = alphas_subset(:, [1 3:5]);

alphas_20_vector = alphas_20(:); % turn into a vector

% For each of those alphas, I will need the equivalent column of the
% phimatrix_full, so when they are multiplied to give me a meaniningful
% approximation of the time-lapse change

idx = [1,45,89,133,177,...
       3,47,91,135,179,...
       4,48,92,136,180,...
       5,49,93,137,181];
       
for ii=1:numel(idx)
    phi(:,ii) = phimatrix_full(:,idx(ii));
end

figure(4);
imagesc(phi)

%% Check the time-lapse model reconstruction using the 20 alphas

reconstruction_20    = phi * alphas_20_vector;
reconstruction_20_2d = reshape(reconstruction_20, 25,44);

% multiply with taper
reconstruction_20_2d_tapered = reconstruction_20_2d .* taper_2d;

figure(5);
imagesc(reconstruction_20_2d_tapered)

%% Save all information
save ('/outdata/dct_components/subset_of_alpha_coefficients.mat', 'idx', 'alphas_20', 'alphas_20_vector', 'alphas_2d', 'reconstruction_20', 'reconstruction_20_2d', 'reconstruction_20_2d_tapered');

save('/outdata/dct_components/phi_matrix.mat', 'phi')
