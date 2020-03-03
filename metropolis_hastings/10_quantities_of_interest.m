% load mcmc results
load /outdata/mcmc_results/Chain1_freq8_initalphas1_truebaseback_iter_100000_initial_sigma_0.0056.mat

% load the phi matrix
load /outdata/dct_components/phi_matrix.mat

% load the true time-lapse model
load('/outdata/dct_components/true_models_in_local_domain.mat', 'truncated_timelapse_velocity_2d')

% Generate the time-lapse models along the chain

n      = 100000;        % number of iterations
deltam = zeros(n,1100); % initialize the matrix for time-lapse models; each model is saved as a vector

for ii = 1:n
    al = alpha(ii,:);
    deltam(ii,:) = phi*al';
end

%% Calculate the average velocity

% initialize
vertical_line     = zeros(n,25);
vertical_extent   = zeros(1,n);
horizontal_line   = zeros(n,44);
horizontal_extent = zeros(1,n);
average_velocity  = zeros(n,1);

% calculate the average velocity for the true model
subdomain_true        = truncated_timelapse_velocity_2d(11:17,14:31);
sub_vec_true          = subdomain_true(:);
average_velocity_true = mean(sub_vec_true);

for jj = 1 : n
    dm                     = reshape(deltam(ii,:), 25,44); % reshape to a 2D model
    dm                     = dm.*taper_2d;                 % apply the taper
    Subdomain              = dm(11:17,14:31);              % get the subdomain for average velocity calculation
    Subdom_vec             = Subdom(:);                    % make it a vector
    average_velocity(ii)   = mean(velLine);                % average velocity calc
    
    
    vertical_line(ii,:)    = dm(:,22);                     % extract line for vertical extent calc
    horizontal_line(ii,:)  = dm(14,:);                     % extract line for horizontal extent calc
                      
end

%% Calculate the vertical extent

% extract a vertical line that passed through the middle of the anomaly; if
% you plot that line should look like a bell curve
vertical_line_true = truncated_timelapse_velocity_2d(:,22);

% use findpeaks to calculate the width of that bell curve
[~,~,vertical_extent_true,~] = findpeaks(vertical_line_true, 'SortStr','descend','Npeaks',1);


% do it for all models in the chain
for kk = 1:n
    [pks,locs,w,~] = findpeaks(vertical_line(kk,:), 'SortStr','descend','Npeaks',1);
    if isempty(w)
        w =0;
    end
    vertical_extent(kk) = w;
end

%% Calculate the horizontal extent

% extract a horizontal line that passed through the middle of the anomaly; if
% you plot that line should look like a bell curve
horizontal_line_true = truncated_timelapse_velocity_2d(14,:);

% use findpeaks to calculate the width of that bell curve
[~,~,horizontal_extent_true,~] = findpeaks(horizontal_line_true, 'SortStr','descend','Npeaks',1);

% do it for all models
for kk = 1:n
    [pks,locs,w,~] = findpeaks(horizontal_line(kk,:), 'SortStr','descend','Npeaks',1);
    if isempty(w)
        w =0;
    end
    horizontal_extent(kk) = w;
end

%% Plot histograms

half = n/2;

figure(1);

subplot(2,2,1);
imagesc(truncated_timelapse_velocity_2d);
hold on 
line([22,22],[1,25], 'Color', 'w','LineStyle',':','linewidth',4);
line([1,44] ,[14,14],'Color', 'w','LineStyle',':','linewidth',4);
rectangle('Position', [14 11 18 6], 'EdgeColor','w','linewidth',5);
xlabel('n \rightarrow');ylabel('\leftarrow m');
title('(a) True 4D change');
set(gca, 'fontsize',20);


subplot(2,2,2)
histogram(vertical_extent(half+1:end),50);
line([vertical_extent_true vertical_extent_true],[0 1500],'LineWidth',6, 'Color', 'g');
ylim([ 0 4500]);
ylabel('Counts');
xlabel('m');
title('(b) Vertical extent');
set(gca, 'fontsize',20);


subplot(2,2,3)
histogram(horizontal_extent(half+1:end),50);
line([horizontal_extent_true horizontal_extent_true],[0 1500],'LineWidth',6, 'Color', 'g');
ylim([ 0 4500]);
ylabel('Counts');
xlabel('n');
title('(c) Horizontal extent');
set(gca, 'fontsize',20);

subplot(2,2,4)
histogram(average_velocity(half+1:end),50);
line([average_velocity_true average_velocity_true],[0 1500],'LineWidth',6, 'Color', 'g');
ylim([ 0 4500]);
ylabel('Counts');
xlabel('\delta m (m/s)')
title('(d) Average velocity')
set(gca, 'fontsize',20);


