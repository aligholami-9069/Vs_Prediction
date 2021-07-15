clear variables;
close all;
clc;

font_size = 12;
path = './Results/Optimization/';

%% Read Data

%********** Fast
nn_result_fast  = load('./Results/Neural Network/nn_result_fast.mat');
nn_y_pred_fast = nn_result_fast.y_pred_fast;

fl_y_pred_fast  = load('./Results/Fuzzy Logic/FuzzyResult.mat');
fl_y_pred_fast = fl_y_pred_fast.FuzzyResult;
[~,idx] = max([fl_y_pred_fast.rFast]);
fl_y_pred_fast = fl_y_pred_fast(1,idx).y_pred_fast;

nf_result_fast  = load('./Results/Neuro Fuzzy/nf_result_fast.mat');
nf_y_pred_fast = nf_result_fast.y_pred_fast;

tst_lbl_fast = table2array(readtable('./data/trn_tst/tst_lbl_fast.dat'));
tst_lbl_fast = tst_lbl_fast(~isnan(tst_lbl_fast));

%********** Slow
nn_result_slow  = load('./Results/Neural Network/nn_result_slow.mat');
nn_y_pred_slow = nn_result_slow.y_pred_slow;

fl_y_pred_slow  = load('./Results/Fuzzy Logic/FuzzyResult.mat');
fl_y_pred_slow = fl_y_pred_slow.FuzzyResult;
[~,idx] = max([fl_y_pred_slow.rSlow]);
fl_y_pred_slow = fl_y_pred_slow(1,idx).y_pred_slow;

nf_result_slow  = load('./Results/Neuro Fuzzy/nf_result_slow.mat');
nf_y_pred_slow = nf_result_slow.y_pred_slow;

tst_lbl_slow = table2array(readtable('./Data/trn_tst/tst_lbl_slow.dat'));
tst_lbl_slow = tst_lbl_slow(~isnan(tst_lbl_slow));

%% Simple Averaging

%********** Fast
sa_y_fast = (0.33333 * nn_y_pred_fast)+...
            (0.33333 * fl_y_pred_fast)+...
            (0.33333 * nf_y_pred_fast);
        
sa_x_fast = [0.33333, 0.33333, 0.33333];
        
sa_mse_fast = immse(tst_lbl_fast,sa_y_fast);

figure;
[~,~,sa_r_fast] = postreg(tst_lbl_fast',sa_y_fast');
xlabel(strcat('DTSM Fast (Avaraged with Simple Averaging)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DTSM Fast (Measured)'),'fontweight','bold','fontsize',font_size);
title(sprintf('R = %.5f',sa_r_fast),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'sa_Fast.png'))

%********** Slow
sa_y_slow = (0.33333 * nn_y_pred_slow)+...
            (0.33333 * fl_y_pred_slow)+...
            (0.33333 * nf_y_pred_slow);

sa_x_slow = [0.33333, 0.33333, 0.33333];

sa_mse_slow = immse(tst_lbl_slow,sa_y_slow);

figure;
[~,~,sa_r_slow] = postreg(tst_lbl_slow',sa_y_slow');
xlabel(strcat('DTSM Slow (Avaraged with Simple Averaging)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DTSM Slow (Measured)'),'fontweight','bold','fontsize',font_size);
title(sprintf('R = %.5f',sa_r_slow),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'sa_Slow.png'))

%% GA Algorithm

rng('default')
nvars = 3;
lb = [0,0,0];
ub = [1,1,1];
%options = optimoptions('ga','MaxGenerations',1e4,'PopulationSize',2000,'CrossoverFraction', 0.5);
options = optimoptions('ga',...
            'CrossoverFcn',str2func('crossoversinglepoint'),...
            'CrossoverFraction',0.7,...
            'PopulationSize',110,...
            'EliteCount',6,...
            'HybridFcn',str2func('patternsearch'),...
            'InitialPopulationRange',[0;1],...
            'MaxGenerations',300,...
            'MutationFcn',str2func('mutationgaussian'),...
            'SelectionFcn',str2func('selectionstochunif'));

%********** Fast
fun_fast = @(x_fast) MSE(x_fast(1),x_fast(2),x_fast(3),...
                         nn_y_pred_fast,fl_y_pred_fast,nf_y_pred_fast,...
                         tst_lbl_fast,size(tst_lbl_fast,1));

ga_x_fast = ga(fun_fast,nvars,[],[],[],[],lb,ub,[],options);

ga_y_fast = (ga_x_fast(1) * nn_y_pred_fast)+...
            (ga_x_fast(2) * fl_y_pred_fast)+...
            (ga_x_fast(3) * nf_y_pred_fast);

ga_mse_fast = immse(tst_lbl_fast,ga_y_fast);

figure;
[~,~,ga_r_fast] = postreg(tst_lbl_fast',ga_y_fast');
xlabel(strcat('DTSM Fast (Avaraged with GA)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DTSM Fast (Measured)'),'fontweight','bold','fontsize',font_size);
title(sprintf('R = %.5f',ga_r_fast),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'ga_Fast.png'))

%********** Slow
fun_slow = @(x_slow) MSE(x_slow(1),x_slow(2),x_slow(3),...
                         nn_y_pred_slow,fl_y_pred_slow,nf_y_pred_slow,...
                         tst_lbl_slow,size(tst_lbl_slow,1));

ga_x_slow = ga(fun_slow,nvars,[],[],[],[],lb,ub,[],options);

ga_y_slow = (ga_x_slow(1) * nn_y_pred_slow)+...
            (ga_x_slow(2) * fl_y_pred_slow)+...
            (ga_x_slow(3) * nf_y_pred_slow);

ga_mse_slow = immse(tst_lbl_slow,ga_y_slow);

figure;
[~,~,ga_r_slow] = postreg(tst_lbl_slow',ga_y_slow');
xlabel(strcat('DTSM Slow (Avaraged with GA)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DTSM Slow (Measured)'),'fontweight','bold','fontsize',font_size);
title(sprintf('R = %.5f',ga_r_slow),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'ga_Slow.png'))

%% Simulated Annealing

rng('default')
x0 = [0,0,0];
lb = [0,0,0];
ub = [1,1,1];
%options = optimoptions('simulannealbnd','PlotFcns',{@saplotbestx,@saplotbestf,@saplotx,@saplotf});
options = optimoptions('simulannealbnd',...
            'InitialTemperature',75,...
            'ReannealInterval',20,...           
            'FunctionTolerance',1e-6,...
            'HybridFcn',str2func('fminsearch'),...
            'HybridInterval','never',...
            'TemperatureFcn',str2func('temperatureexp'),...
            'AnnealingFcn',str2func('annealingfast'));

%********** Fast
fun_fast = @(x_fast) MSE(x_fast(1),x_fast(2),x_fast(3),...
                         nn_y_pred_fast,fl_y_pred_fast,nf_y_pred_fast,...
                         tst_lbl_fast,size(tst_lbl_fast,1));
                     
simann_x_fast = simulannealbnd(fun_fast,x0,lb,ub,options);

simann_y_fast = (simann_x_fast(1) * nn_y_pred_fast)+...
                (simann_x_fast(2) * fl_y_pred_fast)+...
                (simann_x_fast(3) * nf_y_pred_fast);

simann_mse_fast = immse(tst_lbl_fast,simann_y_fast);
            
figure;
[~,~,simann_r_fast] = postreg(tst_lbl_fast',simann_y_fast');
xlabel(strcat('DTSM Fast (Avaraged with Simulated Annealing)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DTSM Fast (Measured)'),'fontweight','bold','fontsize',font_size);
title(sprintf('R = %.5f',simann_r_fast),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'simann_Fast.png'))

%********** Slow
fun_slow = @(x_slow) MSE(x_slow(1),x_slow(2),x_slow(3),...
                         nn_y_pred_slow,fl_y_pred_slow,nf_y_pred_slow,...
                         tst_lbl_slow,size(tst_lbl_slow,1));
                     
simann_x_slow = simulannealbnd(fun_slow,x0,lb,ub,options);

simann_y_slow = (simann_x_slow(1) * nn_y_pred_slow)+...
                (simann_x_slow(2) * fl_y_pred_slow)+...
                (simann_x_slow(3) * nf_y_pred_slow);

simann_mse_slow = immse(tst_lbl_slow,simann_y_slow);

figure;
[~,~,simann_r_slow] = postreg(tst_lbl_slow',simann_y_slow');
xlabel(strcat('DTSM Slow (Avaraged with Simulated Annealing)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DTSM Slow (Measured)'),'fontweight','bold','fontsize',font_size);
title(sprintf('R = %.5f',simann_r_slow),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'simann_Slow.png'))

%% Ant Colony Algorithm

rng('default')
n_iter = 100; %number of iteration
NA = 100; % Number of Ants
alpha = 1; % alpha
beta = 0.6; % beta
roh = 0.3; % Evaporation rate
n_param = 3; % Number of paramters
LB = [0.1,0.1,0.1]; % lower bound
UB = [0.99,0.99,0.99]; % upper bound
n_node = 1000; % numbe

%********** Fast
aco_x_fast = ACO(nn_y_pred_fast,fl_y_pred_fast,nf_y_pred_fast,tst_lbl_fast,size(tst_lbl_fast,1),...
                 n_iter,NA,alpha,beta,roh,n_param,LB,UB,n_node);

aco_y_fast = (aco_x_fast(1) * nn_y_pred_fast)+...
             (aco_x_fast(2) * fl_y_pred_fast)+...
             (aco_x_fast(3) * nf_y_pred_fast);

aco_mse_fast = immse(tst_lbl_fast,aco_y_fast);

figure;
[~,~,aco_r_fast] = postreg(tst_lbl_fast',aco_y_fast');
xlabel(strcat('DTSM Fast (Avaraged with Ant Colony)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DTSM Fast (Measured)'),'fontweight','bold','fontsize',font_size);
title(sprintf('R = %.5f',aco_r_fast),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'aco_Fast.png'))

%********** Slow
aco_x_slow = ACO(nn_y_pred_slow,fl_y_pred_slow,nf_y_pred_slow,tst_lbl_slow,size(tst_lbl_slow,1),...
                 n_iter,NA,alpha,beta,roh,n_param,LB,UB,n_node);

aco_y_slow = (aco_x_slow(1) * nn_y_pred_slow)+...
             (aco_x_slow(2) * fl_y_pred_slow)+...
             (aco_x_slow(3) * nf_y_pred_slow);

aco_mse_slow = immse(tst_lbl_slow,aco_y_slow);
         
figure;
[~,~,aco_r_slow] = postreg(tst_lbl_slow',aco_y_slow');
xlabel(strcat('DTSM Slow (Avaraged with Ant Colony)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DTSM Slow (Measured)'),'fontweight','bold','fontsize',font_size);
title(sprintf('R = %.5f',aco_r_slow),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'aco_Slow.png'))

%% Total Averaging

%********** Fast
tavg_y_fast = (sa_y_fast + ga_y_fast + simann_y_fast + aco_y_fast) / 4;
tavg_x_fast = (sa_x_fast + ga_x_fast + simann_x_fast + aco_x_fast) ./ 4;

tavg_mse_fast = immse(tst_lbl_fast,tavg_y_fast);

figure;
[~,~,tavg_r_fast] = postreg(tst_lbl_fast',tavg_y_fast');
xlabel(strcat('DTSM Fast (Avarage of all Optimization Algorithms)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DTSM Fast (Measured)'),'fontweight','bold','fontsize',font_size);
title(sprintf('R = %.5f',tavg_r_fast),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'tavg_Fast.png'))

%********** Slow
tavg_y_slow = (sa_y_slow + ga_y_slow + simann_y_slow + aco_y_slow) / 4;
tavg_x_slow = (sa_x_slow + ga_x_slow + simann_x_slow + aco_x_slow) ./ 4;

tavg_mse_slow = immse(tst_lbl_slow,tavg_y_slow);

figure;
[~,~,tavg_r_slow] = postreg(tst_lbl_slow',tavg_y_slow');
xlabel(strcat('DTSM Slow (Avarage of all Optimization Algorithms)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DTSM Slow (Measured)'),'fontweight','bold','fontsize',font_size);
title(sprintf('R = %.5f',tavg_r_slow),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'tavg_Slow.png'))

%% Result Matrix

Results.R_Fast.Simple_Averaging = sa_r_fast;
Results.R_Fast.GA = ga_r_fast;
Results.R_Fast.Simulated_Annealing = simann_r_fast;
Results.R_Fast.Ant_Colony = aco_r_fast;
Results.R_Fast.Total_Averaging = tavg_r_fast;

Results.R_Slow.Simple_Averaging = sa_r_slow;
Results.R_Slow.GA = ga_r_slow;
Results.R_Slow.Simulated_Annealing = simann_r_slow;
Results.R_Slow.Ant_Colony = aco_r_slow;
Results.R_Slow.Total_Averaging = tavg_r_slow;

Results.MSE_Fast.Simple_Averaging = sa_mse_fast;
Results.MSE_Fast.GA = ga_mse_fast;
Results.MSE_Fast.Simulated_Annealing = simann_mse_fast;
Results.MSE_Fast.Ant_Colony = aco_mse_fast;
Results.MSE_Fast.Total_Averaging = tavg_mse_fast;

Results.MSE_Slow.Simple_Averaging = sa_mse_slow;
Results.MSE_Slow.GA = ga_mse_slow;
Results.MSE_Slow.Simulated_Annealing = simann_mse_slow;
Results.MSE_Slow.Ant_Colony = aco_mse_slow;
Results.MSE_Slow.Total_Averaging = tavg_mse_slow;

save(strcat(path,'result_all.mat'),'Results');

%% Result y Predict

y_Predict_Result.y_pred_fast.Simple_Averaging = sa_y_fast;
y_Predict_Result.y_pred_fast.GA = ga_y_fast;
y_Predict_Result.y_pred_fast.Simulated_Annealing = simann_y_fast;
y_Predict_Result.y_pred_fast.Ant_Colony = aco_y_fast;
y_Predict_Result.y_pred_fast.Total_Averaging = tavg_y_fast;

y_Predict_Result.y_pred_slow.Simple_Averaging = sa_y_slow;
y_Predict_Result.y_pred_slow.GA = ga_y_slow;
y_Predict_Result.y_pred_slow.Simulated_Annealing = simann_y_slow;
y_Predict_Result.y_pred_slow.Ant_Colony = aco_y_slow;
y_Predict_Result.y_pred_slow.Total_Averaging = tavg_y_slow;

save(strcat(path,'result_y_predict.mat'),'y_Predict_Result');

%% Functions

function f = MSE(w1,w2,w3,data_x1,data_x2,data_x3,data_real,N)
    squaredError = (((w1 * data_x1) + (w2 * data_x2) + (w3 * data_x3)) - data_real) .^2;
    f = sum(squaredError(:)) / N;
end

function f = ACO_MSE(w,data1,data2,data3,data_real,N)
    squaredError = (((w(1) * data1) + (w(2) * data2) + (w(3) * data3)) - data_real) .^2;
    f = sum(squaredError(:)) / N;
end

function opt_params = ACO(d1,d2,d3,d_real,d_count,n_iter,NA,alpha,beta,roh,n_param,LB,UB,n_node)
    % intializing some variables 
    cost_best_prev = inf;
    ant = zeros(NA,n_param);
    cost = zeros(NA,1);
    tour_selected_param = zeros(1,n_param);
    Nodes = zeros(n_node,n_param);
    prob = zeros(n_node, n_param);

    % Generating Nodes
    T = ones(n_node,n_param).*eps; % Phormone Matrix
    dT = zeros(n_node,n_param); % Change of Phormone
    for i = 1:n_param
        Nodes(:,i) = linspace(LB(i),UB(i),n_node); % Node generation at equal spaced points
    end

    % Iteration loop
    for iter = 1:n_iter
        for tour_i = 1:n_param
            prob(:,tour_i) = (T(:,tour_i).^alpha) .* ((1./Nodes(:,tour_i)).^beta);
            prob(:,tour_i) = prob(:,tour_i)./sum(prob(:,tour_i));
        end
        for A = 1:NA
            for tour_i = 1:n_param
                node_sel = rand;
                node_ind = 1;
                prob_sum = 0;
                for j = 1:n_node
                    prob_sum = prob_sum + prob(j,tour_i);
                    if prob_sum >= node_sel
                        node_ind = j;
                        break
                    end
                end
                ant(A,tour_i) = node_ind;
                tour_selected_param(tour_i) = Nodes(node_ind, tour_i);
            end
            cost(A) = ACO_MSE(tour_selected_param,d1,d2,d3,d_real,d_count);
            clc
            disp(['Ant number: ',num2str(A)])
            disp(['Ant Cost: ',num2str(cost(A))])
            disp(['Ant Paramters: ',num2str(tour_selected_param)])
            if iter ~= 1
                disp(['iteration: ',num2str(iter)])
                disp('_________________')
                disp(['Best cost: ',num2str(cost_best)])
                for i = 1:n_param
                    tour_selected_param(i) = Nodes(ant(cost_best_ind,i),i);
                end
                disp(['Best paramters: ',num2str(tour_selected_param)])
            end
        end
        [cost_best,cost_best_ind] = min(cost);

        % Elitsem
        if (cost_best>cost_best_prev) && (iter ~= 1)
            [~,cost_worst_ind] = max(cost);
            ant(cost_worst_ind,:) = best_prev_ant;
            cost_best = cost_best_prev;
            cost_best_ind = cost_worst_ind;
        else
            cost_best_prev = cost_best;
            best_prev_ant = ant(cost_best_ind,:);
        end
        dT = zeros(n_node,n_param); % Change of Phormone
        for tour_i = 1:n_param
            for A = 1:NA
                dT(ant(A,tour_i),tour_i) = dT(ant(A,tour_i),tour_i) + cost_best/cost(A);
            end
        end
        T = roh.*T + dT;        
    end   
    opt_params = tour_selected_param;
end