clear variables;
close all;
clc;

font_size = 12;
path = './Results/Optimization/GA/';

%% Read Data

%********** Fast
nn_y_pred_fast  = load('./Results/Neural Network/nn_result_fast.mat');
nn_y_pred_fast = nn_y_pred_fast.y_pred_fast;

fl_y_pred_fast  = load('./Results/Fuzzy Logic/FuzzyResult.mat');
fl_y_pred_fast = fl_y_pred_fast.FuzzyResult;
[~,idx] = min([fl_y_pred_fast.mse_fast]);
fl_y_pred_fast = fl_y_pred_fast(1,idx).y_pred_fast;

nf_y_pred_fast  = load('./Results/Neuro Fuzzy/nf_result_fast.mat');
nf_y_pred_fast = nf_y_pred_fast.y_pred_fast;

tst_lbl_fast = table2array(readtable('./data/trn_tst/tst_lbl_fast.dat'));
tst_lbl_fast = tst_lbl_fast(~isnan(tst_lbl_fast));

%********** Slow
nn_y_pred_slow  = load('./Results/Neural Network/nn_result_slow.mat');
nn_y_pred_slow = nn_y_pred_slow.y_pred_slow;

fl_y_pred_slow  = load('./Results/Fuzzy Logic/FuzzyResult.mat');
fl_y_pred_slow = fl_y_pred_slow.FuzzyResult;
[~,idx] = min([fl_y_pred_slow.mse_slow]);
fl_y_pred_slow = fl_y_pred_slow(1,idx).y_pred_slow;

nf_y_pred_slow  = load('./Results/Neuro Fuzzy/nf_result_slow.mat');
nf_y_pred_slow = nf_y_pred_slow.y_pred_slow;

tst_lbl_slow = table2array(readtable('./Data/trn_tst/tst_lbl_slow.dat'));
tst_lbl_slow = tst_lbl_slow(~isnan(tst_lbl_slow));

%% GA Algorithm

i = 0;
params = [];
col_num = [1,2,3,4,5];
csv_file_path = strcat(path,'ga_param_tuning.csv');
dlmwrite(csv_file_path,col_num,'delimiter',',');

for MaxGenerations = 150:100:850 %40:20:120 %1000:1000:1e4
for PopulationSize = 150:100:850 %50:10:100 %1000:100:2000
for EliteCount = ceil(0.05 * PopulationSize)
for CrossoverFraction = 0.5:0.1:1
for MutationFcn = {'mutationgaussian','mutationadaptfeasible','mutationuniform'}
for CrossoverFcn = {'crossoverscattered','crossoversinglepoint'}
for SelectionFcn = {'selectionstochunif','selectionroulette'}
for HybridFcn = {'patternsearch','fmincon'}
for MigrationFraction = 0.1:0.1:1
for InitialPopulationRange = [0;1]
    
    try
        i = i + 1;
        rng('default')
        nvars = 3;
        lb = [0.01,0.01,0.01];
        ub = [1,1,1];
        options = optimoptions('ga',...
            'CrossoverFcn',str2func(CrossoverFcn{1}),...
            'CrossoverFraction',CrossoverFraction,...
            'PopulationSize',PopulationSize,...
            'EliteCount',EliteCount,...
            'HybridFcn',str2func(HybridFcn{1}),...
            'InitialPopulationRange',InitialPopulationRange,...
            'MaxGenerations',MaxGenerations,...
            'MutationFcn',str2func(MutationFcn{1}),...
            'MigrationFraction',MigrationFraction,...
            'SelectionFcn',str2func(SelectionFcn{1}));                  
        
        %********** Fast
        fun_fast = @(x_fast) MSE(x_fast(1),x_fast(2),x_fast(3),...
                         nn_y_pred_fast,fl_y_pred_fast,nf_y_pred_fast,...
                         tst_lbl_fast,size(tst_lbl_fast,1));

        ga_x_fast = ga(fun_fast,nvars,[],[],[],[],lb,ub,[],options);

        ga_y_fast = (ga_x_fast(1) * nn_y_pred_fast)+...
                    (ga_x_fast(2) * fl_y_pred_fast)+...
                    (ga_x_fast(3) * nf_y_pred_fast);

        ga_mse_fast = immse(tst_lbl_fast,ga_y_fast);

        set(0,'DefaultFigureVisible','off');
        [~,~,ga_r_fast] = postreg(tst_lbl_fast',ga_y_fast');

        %********** Slow
        fun_slow = @(x_slow) MSE(x_slow(1),x_slow(2),x_slow(3),...
                                 nn_y_pred_slow,fl_y_pred_slow,nf_y_pred_slow,...
                                 tst_lbl_slow,size(tst_lbl_slow,1));

        ga_x_slow = ga(fun_slow,nvars,[],[],[],[],lb,ub,[],options);

        ga_y_slow = (ga_x_slow(1) * nn_y_pred_slow)+...
                    (ga_x_slow(2) * fl_y_pred_slow)+...
                    (ga_x_slow(3) * nf_y_pred_slow);

        ga_mse_slow = immse(tst_lbl_slow,ga_y_slow);

        set(0,'DefaultFigureVisible','off');
        [~,~,ga_r_slow] = postreg(tst_lbl_slow',ga_y_slow');
        
    catch ME
        warning(ME.message);
    end

        %********** Result
        params = [i,...
            CrossoverFcn,...
            CrossoverFraction,...
            PopulationSize,...
            EliteCount,...
            HybridFcn,...
            InitialPopulationRange,...
            MaxGenerations,...
            MutationFcn,...
            MigrationFraction,...
            SelectionFcn,... 
            ga_mse_fast,ga_r_fast,ga_mse_slow,ga_r_slow];         

        save(strcat(path,"Parameters/",int2str(i),".mat"),'params');
        
        params_csv = [i,ga_mse_fast,ga_r_fast,ga_mse_slow,ga_r_slow];
        dlmwrite(csv_file_path,params_csv,'delimiter',',','-append');
end
end
end
end
end
end
end
end
end
end
        
%% Functions

function f = MSE(w1,w2,w3,data_x1,data_x2,data_x3,data_real,N)
    squaredError = (((w1 * data_x1) + (w2 * data_x2) + (w3 * data_x3)) - data_real) .^2;
    f = sum(squaredError(:)) / N;
end