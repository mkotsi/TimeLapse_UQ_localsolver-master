
% load mcmc results
load /outdata/mcmc_results/Chain1_freq8_initalphas1_truebaseback_iter_100000_initial_sigma_0.0056.mat

% load the true values of the alpha coefficients
load ('/outdata/dct_components/subset_of_alpha_coefficients.mat', 'alphas_20_vector')

% calculate the acceptance rate --> it should be between 0.1 and 0.6
% otherwise the step size will need to be adjusted

acceptance_rate = sum(acceptance_history==1)/numel(acceptance_history);

% plot the likelihood function -> the pstar that I have saved is without
% taking the exponential

figure(1)
plot(pStar_hist(1:100:end), 'linewidth',2)
xlabel('iterations')
ylabel('pStar')
title('Likehihood function')
set(gca, 'fontsize',20)

%% Plot histograms of alphas in 2D

num  = 20;
half = 50000; 

figure(2);

for ii = 1 : num 
    subplot(4,5,ii);
    a = alpha(:,ii);
    histogram(a(half+1:end),50);    
    ylim([0 4000]);
    xlim([-20500 20500]);    
    line([alphas_20_vector(ii) alphas_20_vector(ii)],[0 1000],'LineWidth',3, 'Color', 'g');    
    title("\alpha_{"+ num2str(ii)+"}", 'Interpreter', 'tex');    
    set(gca,'xtick',[]);    
    set(gca,'ytick',[]);   
    set(gca, 'fontsize',20);
end

%% Plot bivariate (3D) histograms

figure(3)

num2 = num - 1;

for ii = 1:num2 
    x = alpha(half:end, ii);
    y = alpha(half:end, ii+1);
    subplot(4,5, ii+1);
    hist3([x,y],'CdataMode','auto');
    xlabel("\alpha_{"+ num2str(ii)+"}", 'Interpreter', 'tex');
    ylabel("\alpha_{"+ num2str(ii+1)+"}", 'Interpreter', 'tex');
    %colorbar
    view(2);
    set(gca,'xtick',[]);
    set(gca,'ytick',[]);
    set(gca, 'fontsize',20);
end
