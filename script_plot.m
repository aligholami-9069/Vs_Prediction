clear variables;
close all;
clc;

font_size = 12;
path = './Results/Plots for Article/';

%% Load Data

data = readtable(strcat(path,'FL.xlsx'), 'Range','A:D');
data_fast = data(:, 1:3);
data_slow = data(:, [1:2 4]);

%% 2-side Plot

% Fast
figure
yyaxis left
plot(data_fast.ClusteringRadius,data_fast.No_OfRules,'LineWidth',2);
%title('Plots with Different y-Scales','fontweight','bold','fontsize',font_size)
xlabel('Clustering Radius','fontweight','bold','fontsize',font_size)
ylabel('No. of Rules','fontweight','bold','fontsize',font_size)
yyaxis right
plot(data_fast.ClusteringRadius,data_fast.MSEDTSMFast,'LineWidth',2);
ylabel('MSE Fast','fontweight','bold','fontsize',font_size)
xticks(0.1:0.05:1)
xtickangle(45)
saveas(gcf,strcat(path,'plot_fast.png'))

% Slow
figure
yyaxis left
plot(data_slow.ClusteringRadius,data_slow.No_OfRules,'LineWidth',2);
%title('Plots with Different y-Scales','fontweight','bold','fontsize',font_size)
xlabel('Clustering Radius','fontweight','bold','fontsize',font_size)
ylabel('No. of Rules','fontweight','bold','fontsize',font_size)
yyaxis right
plot(data_slow.ClusteringRadius,data_slow.MSEDTSMSlow,'LineWidth',2);
ylabel('MSE Slow','fontweight','bold','fontsize',font_size)
xticks(0.1:0.05:1)
xtickangle(45)
saveas(gcf,strcat(path,'plot_slow.png'))