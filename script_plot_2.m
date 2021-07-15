clear variables;
close all;
clc;

font_size = 12;
path = './Results/Plots/';

%% Load Data

tst_depth = table2array(readtable('./Data/trn_tst/tst_depth.dat'));
tst_depth = tst_depth(~isnan(tst_depth));

%Fast
y_measured_fast = table2array(readtable('./Data/trn_tst/tst_lbl_fast.dat'));
y_measured_fast = y_measured_fast(~isnan(y_measured_fast));

nn_result_fast = load('./Results/Neural Network/nn_result_fast.mat');
y_pred_fast_nn = nn_result_fast.y_pred_fast;

fl_y_pred_fast = load('./Results/Fuzzy Logic/FuzzyResult.mat');
fl_y_pred_fast = fl_y_pred_fast.FuzzyResult;
[~,idx] = max([fl_y_pred_fast.rFast]);
y_pred_fast_fl = fl_y_pred_fast(1,idx).y_pred_fast;

nf_result_fast = load('./Results/Neuro Fuzzy/nf_result_fast.mat');
y_pred_fast_nf = nf_result_fast.y_pred_fast;

y_pred_opt = load('./Results/Optimization/result_y_predict.mat');
y_pred_fast_opt_simple_averaging = y_pred_opt.y_Predict_Result.y_pred_fast.Simple_Averaging;
y_pred_fast_opt_ga = y_pred_opt.y_Predict_Result.y_pred_fast.GA;
y_pred_fast_opt_simulated_annealing = y_pred_opt.y_Predict_Result.y_pred_fast.Simulated_Annealing;
y_pred_fast_opt_ant_colony = y_pred_opt.y_Predict_Result.y_pred_fast.Ant_Colony;
y_pred_fast_opt_total_averaging = y_pred_opt.y_Predict_Result.y_pred_fast.Total_Averaging;

clear nn_result_fast fl_y_pred_fast nf_result_fast y_pred_opt

%Slow
y_measured_slow = table2array(readtable('./Data/trn_tst/tst_lbl_slow.dat'));
y_measured_slow = y_measured_slow(~isnan(y_measured_slow));

nn_result_slow = load('./Results/Neural Network/nn_result_slow.mat');
y_pred_slow_nn = nn_result_slow.y_pred_slow;

fl_y_pred_slow = load('./Results/Fuzzy Logic/FuzzyResult.mat');
fl_y_pred_slow = fl_y_pred_slow.FuzzyResult;
[~,idx] = max([fl_y_pred_slow.rSlow]);
y_pred_slow_fl = fl_y_pred_slow(1,idx).y_pred_slow;

nf_result_slow = load('./Results/Neuro Fuzzy/nf_result_slow.mat');
y_pred_slow_nf = nf_result_slow.y_pred_slow;

y_pred_opt = load('./Results/Optimization/result_y_predict.mat');
y_pred_slow_opt_simple_averaging = y_pred_opt.y_Predict_Result.y_pred_slow.Simple_Averaging;
y_pred_slow_opt_ga = y_pred_opt.y_Predict_Result.y_pred_slow.GA;
y_pred_slow_opt_simulated_annealing = y_pred_opt.y_Predict_Result.y_pred_slow.Simulated_Annealing;
y_pred_slow_opt_ant_colony = y_pred_opt.y_Predict_Result.y_pred_slow.Ant_Colony;
y_pred_slow_opt_total_averaging = y_pred_opt.y_Predict_Result.y_pred_slow.Total_Averaging;

clear idx nn_result_slow fl_y_pred_slow nf_result_slow y_pred_opt

%% Plot

%Fast
figure
plot(tst_depth,y_measured_fast,tst_depth,y_pred_fast_nn)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Fast (탎/ft)','fontweight','bold','fontsize',font_size);
title('Neural Network','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_fast_DTSM_depth_Neural Network.png'))

figure
plot(tst_depth,y_measured_fast,tst_depth,y_pred_fast_fl)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Fast (탎/ft)','fontweight','bold','fontsize',font_size);
title('Fuzzy Logic','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_fast_DTSM_depth_Fuzzy Logic.png'))

figure
plot(tst_depth,y_measured_fast,tst_depth,y_pred_fast_nf)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Fast (탎/ft)','fontweight','bold','fontsize',font_size);
title('Neuro Fuzzy','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_fast_DTSM_depth_Neuro Fuzzy.png'))

figure
plot(tst_depth,y_measured_fast,tst_depth,y_pred_fast_opt_simple_averaging)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Fast (탎/ft)','fontweight','bold','fontsize',font_size);
title('Simple Averaging','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_fast_DTSM_depth_Simple Averaging.png'))

figure
plot(tst_depth,y_measured_fast,tst_depth,y_pred_fast_opt_ga)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Fast (탎/ft)','fontweight','bold','fontsize',font_size);
title('GA','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_fast_DTSM_depth_GA.png'))

figure
plot(tst_depth,y_measured_fast,tst_depth,y_pred_fast_opt_simulated_annealing)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Fast (탎/ft)','fontweight','bold','fontsize',font_size);
title('Simulated Annealing','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_fast_DTSM_depth_Simulated Annealing.png'))

figure
plot(tst_depth,y_measured_fast,tst_depth,y_pred_fast_opt_ant_colony)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Fast (탎/ft)','fontweight','bold','fontsize',font_size);
title('Ant Colony','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_fast_DTSM_depth_Ant Colony.png'))

figure
plot(tst_depth,y_measured_fast,tst_depth,y_pred_fast_opt_total_averaging)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Fast (탎/ft)','fontweight','bold','fontsize',font_size);
title('Total Averaging','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_fast_DTSM_depth_Total Averaging.png'))

%Slow
figure
plot(tst_depth,y_measured_slow,tst_depth,y_pred_slow_nn)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Slow (탎/ft)','fontweight','bold','fontsize',font_size);
title('Neural Network','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_slow_DTSM_depth_Neural Network.png'))

figure
plot(tst_depth,y_measured_slow,tst_depth,y_pred_slow_fl)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Slow (탎/ft)','fontweight','bold','fontsize',font_size);
title('Fuzzy Logic','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_slow_DTSM_depth_Fuzzy Logic.png'))

figure
plot(tst_depth,y_measured_slow,tst_depth,y_pred_slow_nf)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Slow (탎/ft)','fontweight','bold','fontsize',font_size);
title('Neuro Fuzzy','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_slow_DTSM_depth_Neuro Fuzzy.png'))

figure
plot(tst_depth,y_measured_slow,tst_depth,y_pred_slow_opt_simple_averaging)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Slow (탎/ft)','fontweight','bold','fontsize',font_size);
title('Simple Averaging','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_slow_DTSM_depth_Simple Averaging.png'))

figure
plot(tst_depth,y_measured_slow,tst_depth,y_pred_slow_opt_ga)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Slow (탎/ft)','fontweight','bold','fontsize',font_size);
title('GA','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_slow_DTSM_depth_GA.png'))

figure
plot(tst_depth,y_measured_slow,tst_depth,y_pred_slow_opt_simulated_annealing)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Slow (탎/ft)','fontweight','bold','fontsize',font_size);
title('Simulated Annealing','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_slow_DTSM_depth_Simulated Annealing.png'))

figure
plot(tst_depth,y_measured_slow,tst_depth,y_pred_slow_opt_ant_colony)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Slow (탎/ft)','fontweight','bold','fontsize',font_size);
title('Ant Colony','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_slow_DTSM_depth_Ant Colony.png'))

figure
plot(tst_depth,y_measured_slow,tst_depth,y_pred_slow_opt_total_averaging)
legend('Measured','Predicted')
xlabel('Depth (m)','fontweight','bold','fontsize',font_size);
ylabel('DTSM\_Slow (탎/ft)','fontweight','bold','fontsize',font_size);
title('Total Averaging','fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'comparison_slow_DTSM_depth_Total Averaging.png'))