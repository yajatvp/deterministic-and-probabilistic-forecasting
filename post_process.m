%%
clc
clear;

load('./run1_det.mat');

ts = string(Timestamp);
GHI = double(GHI);
%% Generate deterministic forecast plots
figure;set(gcf, 'WindowState', 'maximized');
plot(datetime(datenum(ts),'ConvertFrom','datenum'),GHI);
grid on;
xlabel('Timestamp');
ylabel('GHI value [W/m^2]')
set(gca,'fontsize',15);grid on;
legend('Actual GHI')

%%
figure;set(gcf, 'WindowState', 'maximized');
plot(datetime(datenum(ts),'ConvertFrom','datenum'),GHI_POC);
grid on; hold on;
plot(datetime(datenum(ts),'ConvertFrom','datenum'),GHI_RF);
plot(datetime(datenum(ts),'ConvertFrom','datenum'),GHI_RF_hr);
xlabel('Timestamp');ylim([0 1200])
ylabel('GHI value [W/m^2]')
set(gca,'fontsize',15);grid on;
legend('GHI-POC method','GHI-RF1 method','GHI-RF2 method')
%%
figure;set(gcf, 'WindowState', 'maximized');
plot(datetime(datenum(ts(8761:end)),'ConvertFrom','datenum'),movmedian(abs((GHI_POC(8761:end)) - GHI(8761:end)),6));
grid on; hold on;
plot(datetime(datenum(ts(8761:end)),'ConvertFrom','datenum'),movmedian(abs((GHI_RF(8761:end)) - GHI(8761:end)),6));
plot(datetime(datenum(ts(8761:end)),'ConvertFrom','datenum'),movmedian(abs((GHI_RF_hr(8761:end)) - GHI(8761:end)),6));
xlabel('Timestamp');
ylabel('6 hr moving meadian of error in GHI forecast [W/m^2]')
set(gca,'fontsize',15);grid on;
legend('GHI-POC method','GHI-RF1 method','GHI-RF2 method')
%%
figure;set(gcf, 'WindowState', 'maximized');
histogram(100.*((GHI_RF(8761:end)) - GHI(8761:end))./max(GHI(8761:end)),100,'Normalization','probability');grid on;
%histogram(GHI_RF_hr(8761:end),80,'Normalization','probability');grid on;
xlabel('% Error');ylabel('Normalized frequency');set(gca,'fontsize',14);

al = (hour(:)>6 & hour(:)<17); 
al(1:8760) = 0;
hold on;
histogram(100.*((GHI_RF(al)) - GHI(al))./max(GHI(al)),100,'Normalization','probability');grid on;
%histogram(GHI_RF_hr(8761:end),80,'Normalization','probability');grid on;
xlabel('% Error');ylabel('Normalized frequency');set(gca,'fontsize',14);
legend('All data','Day-time data')
%% Probabilistic forecasts: Optimal sigma at each training point of 2014 data
for i = 1:8760
    % use opt_sigma to get optimal sigma at each timestamp of 2014.
    sigma(i) = opt_sigma(GHI(i), GHI_RF(i));
    i
end
sigma(sigma(:)<0.00001) = 0;
save(['.\sigma_surrogate.mat'],'sigma')
%% Create sample plot of probabilistic plot of 2014
yup = zeros(1,17520); ydwn = zeros(1,17520);

for i = 1:8760
    [~, ydwn(i), yup(i)] = norminv(0.5,GHI_RF(i),sigma(i),[sigma(i),0;0,sigma(i)],0.00001);
end
yup(isnan(yup(:))) = 0; ydwn(isnan(ydwn(:))) = 0;

figure;set(gcf, 'WindowState', 'maximized');
%p = fill([(datenum(ts(1:30))) ;(datenum(ts(30:-1:1)))],...
p = fill(datetime([(datenum(ts(1:90))) ;(datenum(ts(90:-1:1)))],'ConvertFrom','datenum'),...
    [yup(1:90) ydwn(90:-1:1)],'red');
p.FaceColor = [1 0.8 0.8];  
xlabel('Time stamp');ylabel('Probabilistic GHI forecast [W/m^2]');set(gca,'fontsize',14);hold on;

plot(datetime(datenum(ts(1:90)),'ConvertFrom','datenum'),GHI_RF(1:90),'-')
%% Now go to forecast.py and run SVR surrogate model for getting sigma values at 2015 timestamps.
%% Load predicted values of sigma for 2015
load(['./run1_sig.mat']);
sig1(sig1(:)<0.01) = 0;

for i = 8761:17520
    [~, ydwn(i), yup(i)] = norminv(0.5,GHI_RF(i),sig1(i-8760),[sig1(i-8760),0;0,sig1(i-8760)],0.00001);
end
yup(isnan(yup(:))) = 0; ydwn(isnan(ydwn(:))) = 0;

s = 14445; e = 14550;
figure;set(gcf, 'WindowState', 'maximized');
%p = fill([(datenum(ts(1:30))) ;(datenum(ts(30:-1:1)))],...
p = fill(datetime([(datenum(ts(s:e))) ;(datenum(ts(e:-1:s)))],'ConvertFrom','datenum'),...
    [yup(s:e) ydwn(e:-1:s)],'red');
p.FaceColor = [1 0.8 0.8]; 
xlabel('Time stamp');ylabel('Probabilistic GHI forecast [W/m^2]');set(gca,'fontsize',14);hold on;

plot(datetime(datenum(ts(s:e)),'ConvertFrom','datenum'),GHI(s:e),'-');grid on
%% Calc. avg. pinball loss
for i = 8761:17520
    L(i-8760) = pinball(sig1(i-8760), GHI(i), GHI_RF(i));
end
L(isnan(L(:))) = 0;
%% Probabilistic forecast plot
figure;set(gcf, 'WindowState', 'maximized');
histogram(sigma(:),40,'Normalization','probability');grid on;hold on;
histogram(sigma(hour(1:8760)>6 & hour(1:8760)<17),40,'Normalization','probability');grid on;
xlabel('Standard deviation - \sigma [W/m^2]');ylabel('Normalized frequency');set(gca,'fontsize',14);
legend('All dataset','Day-time data')