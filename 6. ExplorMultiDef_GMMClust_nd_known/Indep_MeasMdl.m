function [V_hat_i, Out_struct] = Indep_MeasMdl(V_Xe, X_e_spline, Par_MeasMdl, PrevUse_Data)

% a = Par_MeasMdl.a;
beta = Par_MeasMdl.beta;
thres_meas = Par_MeasMdl.thres_meas;
n_def = Par_MeasMdl.n_def;
iter = Par_MeasMdl.iter;

%% %%%%%%%%%%%%%%%% Preprocesing data %%%%%%%%%%%%%%%%%%%%%%%%%
idx_V = V_Xe > thres_meas; %Locate values above threshold
Preprocessed_V_Xe = V_Xe(idx_V);
Preprocessed_X_e = X_e_spline(idx_V,:);
Preprocessed_Data_current = [Preprocessed_X_e, Preprocessed_V_Xe];

% adding the previous useful data
Preprocessed_Data = [PrevUse_Data; Preprocessed_Data_current];

V_Xe_int = round(Preprocessed_Data(:,3));    % Accumulated V Conversion to int
Data_Xe_hist_V = repelem(Preprocessed_Data(:,1:2), V_Xe_int, 1); %Repeat elements on spatial domain (trajectory points)

%% %%%%%%%%%%%% Gaussian Mixture Model of Preprocessed data %%%%%%%%%%%%%%%

% If the number of clusters is not known, but a range given, evaluate
% cev = evalclusters(statsNorm,"gmdistribution","silhouette",...
%     KList=2:4);
% bestK = cev.OptimalK;

gmm_opt = statset('MaxIter', 2000);
GMModel = fitgmdist(Data_Xe_hist_V, n_def, "CovarianceType", "diagonal", ...
      'Options', gmm_opt, 'SharedCovariance', true, "Replicates", 10); %'SharedCovariance', false

% Solution 2 for unknown number of clusters, but a range given, computing
% AIC criterion
% AIC = zeros(1,4);
% GMModels = cell(1,4);
% gmm_opt = statset('MaxIter',2000);
% for k = 1:4
%     GMModels{k} = fitgmdist(X,k,'Options',gmm_opt,'CovarianceType','diagonal',...
%                   'SharedCovariance', true, "Replicates", 5);
%     AIC(k)= GMModels{k}.AIC;
% end
% 
% [minAIC,numComponents] = min(AIC);
% BestModel = GMModels{numComponents}

%% Re-ordering fitted distribution (defects) and clustering

if iter == 1
    GMM_ordered = GMModel; %Leave it as the GMM decides
    dist_mu_i = [0;0;0]; 
else
    GMM_ordered_last = Par_MeasMdl.GMM_ordered_last;
    %Given a new mu from GMM solution, Search the closest mu from last
    %ordered distribution
    [idx_ord, dist_mu_i] = dsearchn(GMModel.mu, GMM_ordered_last.mu);
    Mu_ordered = GMModel.mu(idx_ord,:); %Means
    p_ordered = GMModel.ComponentProportion(idx_ord); %Weights
    Sigma = GMModel.Sigma; %Variance (the same for every defect)
    GMM_ordered = gmdistribution(Mu_ordered, Sigma, p_ordered); %compute re-ordered distribution
end

grps_all = cluster(GMM_ordered, Preprocessed_Data(:,1:2)); %clustering position points of all time history
CurrentPosData_sz = size(Preprocessed_X_e, 1);
grps = grps_all( end-CurrentPosData_sz+1:end );

% grps = cluster(GMM_ordered, Preprocessed_X_e);

%% Definition of the measurement model V_hat and V_bar

% n_d = max(grps); %for unknown number of defects
n_d = n_def;

V_bar_i = NaN(size(V_Xe,1), n_d);

id_data = find(idx_V);

for i = 1:length(Preprocessed_V_Xe)
    V_bar_i( id_data(i),grps(i) ) = Preprocessed_V_Xe(i);
end

V_hat_i = V_bar_i;
V_hat_i(isnan(V_hat_i)) = beta;

%% Output variables

Out_struct.Preprocessed_V_Xe = Preprocessed_V_Xe;
Out_struct.Preprocessed_X_e = Preprocessed_X_e;
Out_struct.Preprocessed_Data = Preprocessed_Data;
Out_struct.Data_Xe_hist_V = Data_Xe_hist_V;
Out_struct.grps_all = grps_all;
Out_struct.grps = grps;
Out_struct.GMModel = GMModel;
Out_struct.GMM_ordered = GMM_ordered;
Out_struct.dist_mu_i = dist_mu_i;
Out_struct.V_hat_i = V_hat_i;
Out_struct.V_bar_i = V_bar_i;

end