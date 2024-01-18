function [params, xs, cps] = gen_data(set_dist, n_samples, alpha0, beta0,cp_prob)

%generate data; either Bernoulli or Gaussian
%translated from https://github.com/gwgundersen/bocd/blob/master/bocd.py

% switch set_dist
%     case 'bernoulli'
%         n_params = 1;
%     case 'normal'
%         n_params = 2;
%         %here: alpha0 is mean0, and beta0 is var0; sorry not sorry for misnomer
% end

curr_p = NaN;
params = nan(n_samples,1);
cps = [];
xs = nan(n_samples,1);

for t = 1:n_samples
    
    if isnan(curr_p) || rand < cp_prob
        %if we have a change point
        switch set_dist
            case 'bernoulli'
                curr_p = betarnd(alpha0,beta0,1,1); %generate new p for post CP
            case 'normal'
                mean_hyper = alpha0;%hyperprior for mean
                var_fixed = beta0;
                curr_p = normrnd(mean_hyper,var_fixed);
        end
        cps = [cps;t]; %add trial to list of known change points
    end
    
    %record current parameter (e.g., prob, or mean)
    params(t) = curr_p;
    switch set_dist
        case 'bernoulli'
            xs(t) = binornd(1,curr_p); %generate the data point
        case 'normal'
            xs(t) = normrnd(curr_p,var_fixed);
    end
    
end


end