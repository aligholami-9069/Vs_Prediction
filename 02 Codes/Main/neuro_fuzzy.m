clear variables;
close all;
clc;

font_size = 12;
path = './Results/Neuro Fuzzy/';

%% Read Net & Test Data

nn_net  = load(strcat(path,'Net/NF_fast.mat'));
NeuroFuzzy_Fast = nn_net.NF_fast;
nn_net  = load(strcat(path,'Net/NF_slow.mat'));
NeuroFuzzy_Slow = nn_net.NF_slow;

tst_dat = table2array(readtable('./Data/trn_tst/tst_dat.dat'));
tst_lbl_fast = table2array(readtable('./Data/trn_tst/tst_lbl_fast.dat'));
tst_lbl_fast = tst_lbl_fast(~isnan(tst_lbl_fast));
tst_lbl_slow = table2array(readtable('./Data/trn_tst/tst_lbl_slow.dat'));
tst_lbl_slow = tst_lbl_slow(~isnan(tst_lbl_slow));

%% Test the Model

y_pred_fast = evalfis(tst_dat, NeuroFuzzy_Fast);
y_pred_slow = evalfis(tst_dat, NeuroFuzzy_Slow);

%% Plot

mse_fast = immse(tst_lbl_fast,y_pred_fast);
mse_slow = immse(tst_lbl_slow,y_pred_slow);

figure;
[~,~,r_fast] = postreg(tst_lbl_fast',y_pred_fast');
xlabel(strcat('DTSM Fast (Predicted)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DTSM Fast (Measured)'),'fontweight','bold','fontsize',font_size);
title(sprintf('MSE = %.5f, R = %.5f',mse_fast,r_fast),'fontweight','bold','fontsize',font_size);
saveas(gcf,strcat(path,'nf_DTSM_Fast.png'))

figure;
[~,~,r_slow] = postreg(tst_lbl_slow',y_pred_slow');
xlabel(strcat('DTSM Slow (Predicted)'),'fontweight','bold','fontsize',font_size);
ylabel(strcat('DTSM Slow (Measured)'),'fontweight','bold','fontsize',font_size);
title(sprintf('MSE = %.5f, R = %.5f',mse_slow,r_slow),'fontweight','bold','fontsize',font_size);

saveas(gcf,strcat(path,'nf_DTSM_Slow.png'))

%% Results

save(strcat(path,'nf_result_fast.mat'),'y_pred_fast','r_fast','mse_fast');
save(strcat(path,'nf_result_slow.mat'),'y_pred_slow','r_slow','mse_slow');