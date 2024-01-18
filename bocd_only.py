import numpy as np
from   scipy.special import logsumexp

def bocd(data, model, hazard):
    """Perform Bayesian online changepoint detection.
    """
    data, T, dim = _check_input(data)

    log_message = np.array([1])

    for t in range(1, T + 1):
        # 2. Observe new datum.
        x = data[t - 1]

        # 3. Evaluate predictive probabilities.
        log_pis = model.log_pred_prob(t, x)

        # 4. Calculate growth probabilities.
        # 5. Calculate changepoint probabilities.
        new_log_joint = log_rl_joint(log_pis, log_message, hazard)

        # 6. Calculate evidence
        # 7. Determine run length distribution.
        model.update_rl_posterior(t, new_log_joint)

        # 8. Update sufficient statistics.
        model.update_params(t, x)

        # Setup message passing.
        log_message = new_log_joint

        # Make model predictions.
        if t < T:
            model.predict(t)

    return model


# -----------------------------------------------------------------------------
# Utility functions.
# -----------------------------------------------------------------------------

def log_rl_joint(log_pis, log_message, hazard):
    """Compute joint distribution p(r_{t} | x_{1:t}, s_{1:t}).
    """
    log_growth_probs = log_pis + log_message + np.log(1 - hazard)
    log_cp_prob      = logsumexp(log_pis + log_message + np.log(hazard))
    return np.append(log_cp_prob, log_growth_probs)


def _check_input(data):
    """Check input?? Return data, number of samples T,
    and data dimension dim.
    """
    T    = len(data)
    data = np.array(data)
    dim  = data[0].size if not np.isscalar(data[0]) else 1
    return data, T, dim

def _init_prediction_vars(T, dim):
    """Initialize variables for prediction.
    """
    if dim > 1:
        pmean = np.zeros((T, dim))
        pvar = np.zeros((T, dim))
    else:
        pmean = np.zeros(T)
        pvar = np.zeros(T)
    return pmean, pvar