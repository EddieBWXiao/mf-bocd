function out = bocd_01(data,hazard,alpha0,beta0)

%for Bernoulli

T           = length(data);
log_R       = -inf * ones(T+1, T+1);
log_R(1, 1) = 1;              %log 0 == 1
log_message = 1;   %log 0 == 1 %not sure what is going on here; initial condition issue?
est_p = nan(T,1);

for t = 1:T
    if t ==1 %initialise
        alpha = alpha0;
        beta = beta0;
    end
    
    %% make prediction? (bravely attempting to get an est of the Bernoulli p at each timepoint)
    weighted_alpha = sum(dot(exp(log_R(t,1:t)),alpha)); %each alpha est weighted by the plausibility of the corresponding run length
    weighted_beta = sum(dot(exp(log_R(t,1:t)),beta));
    est_p(t) = weighted_alpha/(weighted_alpha+weighted_beta);
    
    %% for new data
    x = data(t); %Observe new datum.
    
    %Evaluate predictive probabilities.
    log_pis = log_pred_prob(x,alpha,beta); %one number: x given current alpha & beta ests.
    
    %% for each possible run length, to continue or to change
        %p(r_t = l, data) and p(r_t = 0, data)
    %Calculate growth probabilities (size: each r_t considered) 
    log_growth_probs = log_pis + log_message + log(1-hazard);
        %if log_pis small, less likely growing
        %if hazard big, also less likely growing
        %message: this is p(r_t-1, data till t-1); is it like the prior??
    
    %Calculate changepoint probabilities: 
    log_cp_probs = logsumexp(log_pis+log_message+log(hazard));
    
    %% Determine run length distribution (posterior)
    %really getting the joint distribution:
    new_log_joint = [log_cp_probs,log_growth_probs];
        %IMPORTANT: here, everything has been shifted!!
        %the CP: r=0 becomes r=1
    
    %normalise / Calculate evidence?
    log_R(t+1,1:t+1) = new_log_joint - logsumexp(new_log_joint);
        
    %this posterior is ALSO the message we need to pass to next trial
    log_message = new_log_joint;
        %wait, so this is the prior for run length, right???
    
    %% Update sufficient statistics (the alpha and beta)
        %the alpha and beta will be used for the next trial
    new_alpha = alpha + x;
    alpha = [alpha0, new_alpha]; %keep it in a row vector
    new_beta = beta + 1-x;
    beta = [beta0, new_beta];
    
end

out.est_p=est_p;
out.log_R=log_R;
out.alpha = alpha;
out.beta = beta;


end
function out = log_pred_prob(x,a,b)
    %basically the beta function in log space?
    %a & b can probably be vectors
%     numer = x * log(a + eps) + (1 - x) * log(b + eps);
%     denom = log(a + b + eps);

    numer = x * log(a) + (1 - x) * log(b);
    denom = log(a + b);
    out = numer - denom;
end
function [lse,sm] = logsumexp(x)
%LOGSUMEXP  Log-sum-exp function.
%    lse = LOGSUMEXP(x) returns the log-sum-exp function evaluated at 
%    the vector x, defined by lse = log(sum(exp(x)).
%    [lse,sm] = LOGSUMEXP(x) also returns the softmax function evaluated
%    at x, defined by sm = exp(x)/sum(exp(x)).
%    The functions are computed in a way that avoids overflow and 
%    optimizes numerical stability.   

%    Reference:
%    P. Blanchard, D. J. Higham, and N. J. Higham.  
%    Accurately computing the log-sum-exp and softmax functions. 
%    IMA J. Numer. Anal., Advance access, 2020.

if ~isvector(x), error('Input x must be a vector.'), end

n = length(x);
s = 0; e = zeros(n,1);
[xmax,k] = max(x); a = xmax;
s = 0;
for i = 1:n
    e(i) = exp(x(i)-xmax);
    if i ~= k
       s = s + e(i);
    end   
end
lse = a + log1p(s);
if nargout > 1
   sm = e/(1+s);
end   
end
% function [log_cp_prob, log_growth_probs] = log_rl_joint(log_pis, log_message, hazard)
%     %Compute joint distribution p(r_{t} | x_{1:t}, s_{1:t}).
%     %that's the original comment; what is s_ ?????
%     %bad, abort mission
%     
% end