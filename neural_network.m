clear variables;
close all;
clc;

font_size = 12;
path = './Results/Neural Network/';

%% Read Net & Test Data

nn_net_fast  = load(strcat(path,'Net/Fast/Fast_1_n5/NeuralNetwork_Fast.mat'));
nn_fast = nn_net_fast.results_fast.net_fast;
nn_net_slow  = load(strcat(path,'Net/Slow/slow-6-n10/NeuralNetwork_Slow6.mat'));
nn_slow = nn_net_slow.results_slow6.net_slow6;

tst_dat = table2array(readtable('./Data/trn_tst/tst_dat.dat'));
tst_lbl_fast = table2array(readtable('./Data/trn_tst/tst_lbl_fast.dat'));
tst_lbl_fast = tst_lbl_fast(~isnan(tst_lbl_fast));
tst_lbl_slow = table2array(readtable('./Data/trn_tst/tst_lbl_slow.dat'));
tst_lbl_slow = tst_lbl_slow(~isnan(tst_lbl_slow));

%% Test the Model

y_pred_fast = sim(nn_fast,tst_dat');
y_pred_fast = y_pred_fast';
y_pred_slow = sim(nn_slow,tst_dat');
y_pred_slow = y_pred_slow';

%% Plot

figure;
[~,~,r_fast] = postreg(tst_lbl_fast',y_pred_fast');
xlabel(strcat('DTSM Fast (Predicted)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DTSM Fast (Measured)'),'fontweight','bold','fontsize',font_size);
%title_tmp = sprintf('Fast \n R = %.5f',r_fast);
%title(strcat('DTSM\_',title_tmp),'fontweight','bold','fontsize',font_size);
title(sprintf('R = %.5f',r_fast),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'nn_DTSM_Fast.png'))

figure;
[~,~,r_slow] = postreg(tst_lbl_slow',y_pred_slow');
xlabel(strcat('DTSM Slow (Predicted)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DTSM Slow (Measured)'),'fontweight','bold','fontsize',font_size);
%title_tmp = sprintf('Slow \n R = %.5f',r_slow);
%title(strcat('DTSM\_',title_tmp),'fontweight','bold','fontsize',font_size);
title(sprintf('R = %.5f',r_slow),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'nn_DTSM_Slow.png'))

%% Results

mse_fast = immse(tst_lbl_fast,y_pred_fast);
mse_slow = immse(tst_lbl_slow,y_pred_slow);

save(strcat(path,'nn_result_fast.mat'),'y_pred_fast','r_fast','mse_fast');
save(strcat(path,'nn_result_slow.mat'),'y_pred_slow','r_slow','mse_slow');
