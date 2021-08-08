clear variables;
close all;
clc;

font_size = 12;
path = './Results/Fuzzy Logic/';
data_path = './Data/trn_tst/';

%% Read Data

trn_dat = table2array(readtable(strcat(data_path,'trn_dat.dat')));
trn_lbl_fast = table2array(readtable(strcat(data_path,'trn_lbl_fast.dat')));
trn_lbl_fast = trn_lbl_fast(~isnan(trn_lbl_fast));
trn_lbl_slow = table2array(readtable(strcat(data_path,'trn_lbl_slow.dat')));
trn_lbl_slow = trn_lbl_slow(~isnan(trn_lbl_slow));

tst_dat = table2array(readtable(strcat(data_path,'tst_dat.dat')));
tst_lbl_fast = table2array(readtable(strcat(data_path,'tst_lbl_fast.dat')));
tst_lbl_fast = tst_lbl_fast(~isnan(tst_lbl_fast));
tst_lbl_slow = table2array(readtable(strcat(data_path,'tst_lbl_slow.dat')));
tst_lbl_slow = tst_lbl_slow(~isnan(tst_lbl_slow));

X_trn = trn_dat;
X_tst = tst_dat;
y_trn = horzcat(trn_lbl_fast,trn_lbl_slow);
y_tst = horzcat(tst_lbl_fast,tst_lbl_slow);

%% Fuzzy

i = 0;
for radii = 0.1:0.05:1
    
    i = i + 1;
    % Generate an FIS using subtractive clustering, and specify the cluster center range of influence.
    fis = genfis2(X_trn,y_trn,radii);
    % Evaluate generated FIS fuzzy inference
    y_pred = evalfis(X_tst,fis);
    
    mse_fast = immse(y_tst(:,1),y_pred(:,1));
    mse_slow = immse(y_tst(:,2), y_pred(:,2));
    
    figure;
    [~,~,r_fast] = postreg(y_tst(:,1)',y_pred(:,1)');
    xlabel(strcat('DTSM Fast (Predicted)'),'fontweight','bold','fontsize',font_size);
    ylabel(strcat('DTSM Fast (Measured)'),'fontweight','bold','fontsize',font_size);
    title(sprintf('Clustering Radius = %1.2f \n MSE = %.5f, R = %.5f',radii,mse_fast,r_fast),'fontweight','bold','fontsize',font_size);
    saveas(gcf,strcat(path,'fl_DTSM_Fast_ClustRadi_',num2str(radii,'%.2f'), '.png'))
    
    figure;
    [~,~,r_slow] = postreg(y_tst(:,2)',y_pred(:,2)');
    xlabel(strcat('DTSM Slow (Predicted)'),'fontweight','bold','fontsize',font_size);
    ylabel(strcat('DTSM Slow (Measured)'),'fontweight','bold','fontsize',font_size);
    title(sprintf('Clustering Radius = %1.2f \n MSE = %.5f, R = %.5f',radii,mse_slow,r_slow),'fontweight','bold','fontsize',font_size);
    saveas(gcf,strcat(path,'fl_DTSM_Slow_ClustRadi_',num2str(radii,'%.2f'), '.png'))
    
    % Put results in a structure
    FuzzyResult(i).Radius = radii;
    FuzzyResult(i).Fuzzy = fis;
    FuzzyResult(i).rFast = r_fast;
    FuzzyResult(i).y_pred_fast = y_pred(:,1);
    FuzzyResult(i).mse_fast = mse_fast;
    FuzzyResult(i).rSlow = r_slow;
    FuzzyResult(i).y_pred_slow = y_pred(:,2);
    FuzzyResult(i).mse_slow = mse_slow;

end

[mse_fast_min_y,idx] = min(cell2mat({FuzzyResult.mse_fast}));
mse_fast_min_x = FuzzyResult(1,idx).Radius;
mse_fast_min_model = FuzzyResult(1,idx).Fuzzy;
[mse_slow_min_y,idx] = min(cell2mat({FuzzyResult.mse_slow}));
mse_slow_min_x = FuzzyResult(1,idx).Radius;
mse_slow_min_model = FuzzyResult(1,idx).Fuzzy;

save(strcat(path,'FuzzyResult.mat'),'FuzzyResult');
save(strcat(path,'fl_model_mse_min_fast.mat'),'mse_fast_min_model');
save(strcat(path,'fl_model_mse_min_slow.mat'),'mse_slow_min_model');

figure;
hold on
plot(cell2mat({FuzzyResult.Radius}),cell2mat({FuzzyResult.mse_fast}),'LineWidth',2);
plot(cell2mat({FuzzyResult.Radius}),cell2mat({FuzzyResult.mse_slow}),'LineWidth',2);
plot(mse_fast_min_x,mse_fast_min_y,'o','LineWidth',2,'MarkerSize',8,'MarkerEdgeColor','b');
text(mse_fast_min_x,mse_fast_min_y+0.5,'Maximum value');
plot(mse_slow_min_x,mse_slow_min_y,'o','LineWidth',2,'MarkerSize',8,'MarkerEdgeColor',[0.91,0.41,0.17]);
text(mse_slow_min_x,mse_slow_min_y-0.5,'Maximum value');
legend('Fast','Slow','Location','east');
xlabel('Clustering Radius','fontweight','bold','fontsize',font_size);
ylabel('MSE','fontweight','bold','fontsize',font_size);
xticks(0.1:0.05:1)
xtickangle(45)
saveas(gcf,strcat(path,'fl_Plot_r.png'))