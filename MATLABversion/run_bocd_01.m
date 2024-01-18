trials = 1000;
alpha0_true = 0.1;
beta0_true = 0.1;
cp_prob_true = 1/100;

%simulate
rng(1010)
[params, xs, cps] = gen_data('bernoulli',...
    trials, alpha0_true, beta0_true,cp_prob_true);

%run the ideal observer
out = bocd_01(xs,cp_prob_true,alpha0_true, beta0_true);

%plot
figure;
plot(xs,'*','DisplayName','data')
hold on
plot(params,'DisplayName','true p')
plot(out.est_p,'k--','DisplayName','estimated')
hold off
legend('Location','Southwest')
set(gcf,'Position',[143 649 839 149])

figure;
imagesc(exp(out.log_R'))
%set(gca, 'YDir','reverse')
set(gcf,'Position',[440 577 877 221])
