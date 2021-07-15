clear variables;
close all;
clc;

%% Load Data

col_names = {...
    'Depth'    ,'(m)';...
    'GR'       ,'(API)';...
    'DT'       ,'(µs/ft)';...
    'CGR'      ,'(API)';...
    'PEF'      ,'(B/E)';...
    'NPHI'     ,'(fr)';...
    'RHOB'     ,'(g/cm3)';...
    'DTSM_Fast','(µs/ft)';...
    'DTSM_Slow','(µs/ft)';...
    };

font_size = 12;

dataset = readtable('Data\main_dataset.xlsx', 'Range','A:I');
dataset.Properties.VariableNames = col_names(:,1);


%% Plot

for j = numel(col_names(:,1))-1:numel(col_names(:,1))
    
    ycolname_tmp = char(col_names(j,1));
    ycolname = strcat(ycolname_tmp(1:4),'\_',ycolname_tmp(6:end),{' '},col_names(j,2));
    y = dataset.(ycolname_tmp);
    
    figure;
    for i = 1:numel(col_names(:,1))-2 
        xcolname = char(col_names(i,1));
        x = dataset.(xcolname);
        subplot(2,4,i);
        scatter(x,y);
        xlabel(strcat(xcolname,{' '},col_names(i,2)),'fontweight','bold','fontsize',font_size);
        ylabel(ycolname,'fontweight','bold','fontsize',font_size);
        R = corrcoef(x,y);
        title(strcat('R =',{' '},num2str(R(2),'%.5f')),'fontweight','bold','fontsize',font_size);
        ls = lsline;
        set(ls,'color','r')
        
    end   
end
