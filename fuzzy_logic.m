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
    
    figure;
    [~,~,r_fast] = postreg(y_tst(:,1)',y_pred(:,1)');
    xlabel(strcat('DTSM Fast (Predicted)'),'fontweight','bold','fontsize',font_size);
    ylabel(strcat('DTSM Fast (Measured)'),'fontweight','bold','fontsize',font_size);
    title(sprintf('Clustering Radius = %1.2f \n R = %.5f',radii,r_fast),'fontweight','bold','fontsize',font_size);
    saveas(gcf,strcat(path,'fl_DTSM_Fast_ClustRadi_',num2str(radii,'%.2f'), '.png'))
    
    figure;
    [~,~,r_slow] = postreg(y_tst(:,2)',y_pred(:,2)');
    xlabel(strcat('DTSM Slow (Predicted)'),'fontweight','bold','fontsize',font_size);
    ylabel(strcat('DTSM Slow (Measured)'),'fontweight','bold','fontsize',font_size);
    title(sprintf('Clustering Radius = %1.2f \n R = %.5f',radii,r_slow),'fontweight','bold','fontsize',font_size);
    saveas(gcf,strcat(path,'fl_DTSM_Slow_ClustRadi_',num2str(radii,'%.2f'), '.png'))
    
    % Put results in a structure
    FuzzyResult(i).Radius = radii;
    FuzzyResult(i).Fuzzy = fis;
    FuzzyResult(i).rFast = r_fast;
    FuzzyResult(i).y_pred_fast = y_pred(:,1);
    FuzzyResult(i).mse_fast = immse(y_tst(:,1),y_pred(:,1));
    FuzzyResult(i).rSlow = r_slow;
    FuzzyResult(i).y_pred_slow = y_pred(:,2);
    FuzzyResult(i).mse_slow = immse(y_tst(:,2), y_pred(:,2));

end

[rfast_max_y,idx] = max(cell2mat({FuzzyResult.rFast}));
rfast_max_x = FuzzyResult(1,idx).Radius;
[rslow_max_y,idx] = max(cell2mat({FuzzyResult.rSlow}));
rslow_max_x = FuzzyResult(1,idx).Radius;

figure;
hold on
plot(cell2mat({FuzzyResult.Radius}),cell2mat({FuzzyResult.rFast}),'LineWidth',2);
plot(cell2mat({FuzzyResult.Radius}),cell2mat({FuzzyResult.rSlow}),'LineWidth',2);
plot(rfast_max_x,rfast_max_y,'o','LineWidth',2,'MarkerSize',8,'MarkerEdgeColor','b');
text(rfast_max_x,rfast_max_y+0.01,'Maximum value');
plot(rslow_max_x,rslow_max_y,'o','LineWidth',2,'MarkerSize',8,'MarkerEdgeColor',[0.91,0.41,0.17]);
text(rslow_max_x,rslow_max_y+0.01,'Maximum value');
legend('Fast','Slow','Location','east');
xlabel('Clustering Radius','fontweight','bold','fontsize',font_size);
ylabel('Correlation Coefficient','fontweight','bold','fontsize',font_size);
xticks(0.1:0.05:1)
xtickangle(45)
saveas(gcf,strcat(path,'fl_Plot_r.png'))

save(strcat(path,'FuzzyResult.mat'),'FuzzyResult');