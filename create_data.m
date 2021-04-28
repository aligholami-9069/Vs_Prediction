clear variables;
close all;
clc;

%% Read Data

data  = readtable('data\data_all.csv');
X  = table2array(data(:, 2:5));
y  = table2array(data(:, 6:7));
Depth = table2array(data(:, 1));

[data_rows_count, ~] = size(data);

path = './data/trn_tst/';

%% Training, and Test data

% Set size of Training sample
N_trn = fix(0.8 * data_rows_count);
% Set size of Test sample
N_tst = fix(0.2 * data_rows_count);

% Form Training data 
X_trn = X(1:N_trn, :);
y_trn = y(1:N_trn, :);
Depth_trn = Depth(1:N_trn, :);

% Form Test data
X_tst = X(N_trn + (1:N_tst), :);
y_tst = y(N_trn + (1:N_tst), :);
Depth_tst = Depth(N_trn + (1:N_tst), :);

trn_dat = X_trn;
trn_lbl_fast = y_trn(:,1);
trn_lbl_slow = y_trn(:,2);
trn_depth = Depth_trn;
save(strcat(path,'trn_dat.dat'),'trn_dat','-ascii');
save(strcat(path,'trn_lbl_fast.dat'),'trn_lbl_fast','-ascii');
save(strcat(path,'trn_lbl_slow.dat'),'trn_lbl_slow','-ascii');
save(strcat(path,'trn_depth.dat'),'trn_depth','-ascii');

tst_dat = X_tst;
tst_lbl_fast = y_tst(:,1);
tst_lbl_slow = y_tst(:,2);
tst_depth = Depth_tst;
save(strcat(path,'tst_dat.dat'),'tst_dat','-ascii');
save(strcat(path,'tst_lbl_fast.dat'),'tst_lbl_fast','-ascii');
save(strcat(path,'tst_lbl_slow.dat'),'tst_lbl_slow','-ascii');
save(strcat(path,'tst_depth.dat'),'tst_depth','-ascii');
