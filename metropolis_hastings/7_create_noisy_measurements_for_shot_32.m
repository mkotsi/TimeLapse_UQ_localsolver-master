
% load the baseline data
load /outdata/measuredField_at_shot_32_with_freq_8/baseline_data.mat

% load the monitor data
load /outdata/measuredField_at_shot_32_with_freq_8/monitor_data.mat

% load the time-lapse data
load /outdata/measuredField_at_shot_32_with_freq_8/residuals_recorded.mat %% d1-d0

% I do not want to worry about complex numbers, so I concatinate the
% imagianry part after the real part
data = [real(residuals_recorded(:));imag(residuals_recorded(:))];


stdev_res = sum(data.^2);
stdev = stdev_res/5000;


%%

% The covarinace matrix is the identity matrix multiplied by the energy in
% my data
covMatrix = eye(numel(residuals_recorded)*2) * (stdev);


% Generate Gaussian noise from that covariance matrix
myNoise = mvnrnd(zeros(numel(residuals_recorded)*2,1),covMatrix);
myNoise = myNoise(1:numel(residuals_recorded)) + 1i*myNoise([1:numel(residuals_recorded)] + numel(residuals_recorded));


% Add the noise to the time-lapse data
noisyField = residuals_recorded + reshape(myNoise,size(residuals_recorded));

%% Plot noisy vs. noiseless time-lapse data

figure(1)
subplot(2,1,1)
plot(real(noisyField), 'linewidth',3)
hold on
plot(real(residuals_recorded), 'linewidth',2)
legend('Noisy', 'Noiseless')
xlabel('Receiver Index')
ylabel('Amplitude')
title('Real part of true residuals')
set(gca, 'Fontsize',20)

subplot(2,1,2)
plot(imag(noisyField), 'linewidth',3)
hold on
plot(imag(residuals_recorded), 'linewidth',2)
legend('Noisy', 'Noiseless')
xlabel('Receiver Index')
ylabel('Amplitude')
title('Imaginary part of true residuals')
set(gca, 'Fontsize',20)

%%

% Concatinate the noisy time-lapse data so there is no complex numbers
measuredField = [real(noisyField(:));imag(noisyField(:))];


% The inverse of the covariance matrix
covInv = covMatrix^-1;


% Noise to signal ration
noise_signal_ratio = norm(residuals_recorded)/norm(myNoise);

%% Save

% covariance matrix
save('/outdata/measuredField_at_shot_32_with_freq_8/Covariance_matrix.mat', 'covMatrix')

% covariance inverse
save('/outdata/measuredField_at_shot_32_with_freq_8/Covariance_inverse.mat', 'covInv')

% complex noisy time-lapse data
save('/outdata/measuredField_at_shot_32_with_freq_8/Noisy_measured_field.mat', 'noisyField')

% concatinated noisy time-lapse data
save('/outdata/measuredField_at_shot_32_with_freq_8/Noisy_measured_field_2D.mat','measuredField')