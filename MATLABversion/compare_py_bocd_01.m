%compare if same as python 
%(bocd_only_bernoulli_checkMATLAB.ipynb)

%the following settings are copied from the ipynb
alpha0_true = 0.1;
beta0_true = 0.1;
cp_prob_true = 1/100;

%get data from the Python run
params = csvread('python_out2compare/params.csv');
xs = csvread('python_out2compare/xs.csv');
cps = csvread('python_out2compare/cps.csv');
est_p_py = csvread('python_out2compare/est_p.csv');
theR_py = csvread('python_out2compare/the_R.csv');

%run the ideal observer
out = bocd_01(xs,cp_prob_true,alpha0_true, beta0_true);

%the log_R seems identical:
mean(exp(out.log_R)-theR_py,'all')
max(exp(out.log_R)-theR_py,[],'all')
min(exp(out.log_R)-theR_py,[],'all')

%
plot(out.est_p,est_p_py,'x')
plot(out.est_p(1:end-1),est_p_py(2:end),'x')
plot(out.est_p(2:end),est_p_py(1:end-1),'x')

%there are some discrepancies between the Python & MATLAB predictions
%Python predictions seem to be one trial earlier than MATLAB?
figure;
plot(xs,'.','DisplayName','data')
hold on
plot(params,'DisplayName','true p')
plot(out.est_p,'k--','DisplayName','MATLAB version')
plot(est_p_py,'DisplayName','Python version')
legend('Location','Northwest')
set(gcf,'Position',[143 649 839 149])

