clear variables;
close all;
clc;

font_size = 12;
path = './Results/Optimization/ACO/';

%% Read Data

%********** Fast
nn_result_fast  = load('./Results/Neural Network/nn_result_fast.mat');
nn_y_pred_fast = nn_result_fast.y_pred_fast;

fl_y_pred_fast  = load('./Results/Fuzzy Logic/FuzzyResult.mat');
fl_y_pred_fast = fl_y_pred_fast.FuzzyResult;
[~,idx] = min([fl_y_pred_fast.mse_fast]);
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
[~,idx] = min([fl_y_pred_slow.mse_slow]);
fl_y_pred_slow = fl_y_pred_slow(1,idx).y_pred_slow;

nf_result_slow  = load('./Results/Neuro Fuzzy/nf_result_slow.mat');
nf_y_pred_slow = nf_result_slow.y_pred_slow;

tst_lbl_slow = table2array(readtable('./Data/trn_tst/tst_lbl_slow.dat'));
tst_lbl_slow = tst_lbl_slow(~isnan(tst_lbl_slow));

%% Ant Colony Algorithm

i = 0;
params = [];
col_num = [1,2,3,4,5,6,7,8,9,10,11];
csv_file_path = strcat(path,'aco_param_tuning_2.csv');
dlmwrite(csv_file_path,col_num,'delimiter',',');

for NA = 100:100:1000
    for alpha = 0.4:0.1:1
        for beta = 0.4:0.1:1
            for roh = 0.1:0.1:1
                for n_node = 500:100:1000

                    i = i + 1;

                    rng('default')
                    n_iter = 100;
                    n_param = 3; % Number of paramters
                    LB = [0.1,0.1,0.1]; % lower bound
                    UB = [0.99,0.99,0.99]; % upper bound

                    %********** Fast
                    aco_x_fast = ACO(nn_y_pred_fast,fl_y_pred_fast,nf_y_pred_fast,tst_lbl_fast,size(tst_lbl_fast,1),...
                                     n_iter,NA,alpha,beta,roh,n_param,LB,UB,n_node);

                    aco_y_fast = (aco_x_fast(1) * nn_y_pred_fast)+...
                                 (aco_x_fast(2) * fl_y_pred_fast)+...
                                 (aco_x_fast(3) * nf_y_pred_fast);

                    aco_mse_fast = immse(tst_lbl_fast,aco_y_fast);

                    %figure;
                    set(0,'DefaultFigureVisible','off');
                    [~,~,aco_r_fast] = postreg(tst_lbl_fast',aco_y_fast');

                    %********** Slow
                    aco_x_slow = ACO(nn_y_pred_slow,fl_y_pred_slow,nf_y_pred_slow,tst_lbl_slow,size(tst_lbl_slow,1),...
                                     n_iter,NA,alpha,beta,roh,n_param,LB,UB,n_node);

                    aco_y_slow = (aco_x_slow(1) * nn_y_pred_slow)+...
                                 (aco_x_slow(2) * fl_y_pred_slow)+...
                                 (aco_x_slow(3) * nf_y_pred_slow);

                    aco_mse_slow = immse(tst_lbl_slow,aco_y_slow);

                    %figure;
                    set(0,'DefaultFigureVisible','off');
                    [~,~,aco_r_slow] = postreg(tst_lbl_slow',aco_y_slow');

                    %********** Result
                    params = [i,n_iter,NA,alpha,beta,roh,n_node,aco_mse_fast,aco_r_fast,aco_mse_slow,aco_r_slow];
                    dlmwrite(csv_file_path,params,'delimiter',',','-append');
                end
            end
        end
    end
end

%% Functions

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