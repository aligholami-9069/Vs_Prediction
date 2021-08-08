clear variables;
close all;
clc;

font_size = 12;
path = './Results/Optimization/SA/';

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

%% Simulated Annealing

i = 0;
params = [];
col_num = [1,2,3,4,5];
csv_file_path = strcat(path,'simann_param_tuning.csv');
dlmwrite(csv_file_path,col_num,'delimiter',',');

for InitialTemperature = 1000:500:10000 %10:5:150 %50:50:1000 %100
for ReannealInterval = 1000:500:10000 %10:5:150 %50:50:1000 %100
for FunctionTolerance = 1e-6 %1e-8
for HybridFcn = {'fminsearch','patternsearch','fminunc','fmincon'}
for HybridInterval = {'never','end'} %Positive integer
for TemperatureFcn = {'temperatureboltz','temperaturefast','temperatureexp'}
for AnnealingFcn = {'annealingboltz','annealingfast'}

    try
        i = i + 1;

        rng('default')
        x0 = [0,0,0];
        lb = [0.01,0.01,0.01];
        ub = [1,1,1];
        options = optimoptions('simulannealbnd',...
            'InitialTemperature',InitialTemperature,...
            'ReannealInterval',ReannealInterval,...           
            'FunctionTolerance',FunctionTolerance,...
            'HybridFcn',str2func(HybridFcn{1}),...
            'HybridInterval',HybridInterval{1},...
            'TemperatureFcn',str2func(TemperatureFcn{1}),...
            'AnnealingFcn',str2func(AnnealingFcn{1}));    
        
        %********** Fast
        fun_fast = @(x_fast) MSE(x_fast(1),x_fast(2),x_fast(3),...
                                 nn_y_pred_fast,fl_y_pred_fast,nf_y_pred_fast,...
                                 tst_lbl_fast,size(tst_lbl_fast,1));

        simann_x_fast = simulannealbnd(fun_fast,x0,lb,ub,options);

        simann_y_fast = (simann_x_fast(1) * nn_y_pred_fast)+...
                        (simann_x_fast(2) * fl_y_pred_fast)+...
                        (simann_x_fast(3) * nf_y_pred_fast);

        simann_mse_fast = immse(tst_lbl_fast,simann_y_fast);

        set(0,'DefaultFigureVisible','off');
        [~,~,simann_r_fast] = postreg(tst_lbl_fast',simann_y_fast');

        %********** Slow
        fun_slow = @(x_slow) MSE(x_slow(1),x_slow(2),x_slow(3),...
                                 nn_y_pred_slow,fl_y_pred_slow,nf_y_pred_slow,...
                                 tst_lbl_slow,size(tst_lbl_slow,1));

        simann_x_slow = simulannealbnd(fun_slow,x0,lb,ub,options);

        simann_y_slow = (simann_x_slow(1) * nn_y_pred_slow)+...
                        (simann_x_slow(2) * fl_y_pred_slow)+...
                        (simann_x_slow(3) * nf_y_pred_slow);

        simann_mse_slow = immse(tst_lbl_slow,simann_y_slow);

        set(0,'DefaultFigureVisible','off');
        [~,~,simann_r_slow] = postreg(tst_lbl_slow',simann_y_slow');
        
    catch ME
        warning(ME.message);
    end

        %********** Result
        params = [i,...
            InitialTemperature,...
            ReannealInterval,...           
            FunctionTolerance,...
            HybridFcn,...
            HybridInterval,...
            TemperatureFcn,...
            AnnealingFcn,...
            simann_mse_fast,simann_r_fast,simann_mse_slow,simann_r_slow];
        
        save(strcat(path,"Parameters/",int2str(i),".mat"),'params');
        
        params_csv = [i,simann_mse_fast,simann_r_fast,simann_mse_slow,simann_r_slow];
        dlmwrite(csv_file_path,params_csv,'delimiter',',','-append');
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