import numpy as np

from scipy.special import erfinv, erfcinv, erfc
from scipy.stats import chi2
from scipy.stats import poisson

def get_significance_from_p_value(p_value):
    sigma = np.sqrt(2.) * erfinv(1. - p_value)
    return sigma

def get_significance(chisquare, ndf):
    
    # probability of getting a more extreme result
    p_value = 1. - chi2.cdf(chisquare, ndf)
    
    sigma = np.sqrt(2.) * erfcinv(p_value)
    
    return p_value, sigma
    
def get_poisson_significance(measured, expected):
    
    # probability of getting an equal or more extreme result
    p_value = 1. - poisson.cdf(measured - 0.001, expected)
    
    sigma = np.sqrt(2.) * erfinv(1. - p_value)
    
    print(f"p value: {p_value}")
    print(f"significance: {sigma} sigma")


def get_p_value_from_significance(sigma):
    p_value = erfc(sigma / np.sqrt(2.))
    return p_value

def get_chi2(cov, data, pred, min_pred=0):

    data = np.array(data)
    pred = np.array(pred)
    cov = np.array(cov)

    valid_indices = np.where(pred >= min_pred)[0]
    
    data_cut = data[valid_indices]
    pred_cut = pred[valid_indices]
    cov_cut = cov[np.ix_(valid_indices, valid_indices)]
    
    diff = data_cut - pred_cut
    return np.linalg.multi_dot([diff, np.linalg.inv(cov_cut), diff])


def chi2_decomposition(cov, data, pred, min_pred=0):

    data = np.array(data)
    pred = np.array(pred)
    cov = np.array(cov)

    valid_indices = np.where(pred >= min_pred)[0]

    data_cut = data[valid_indices]
    pred_cut = pred[valid_indices]
    cov_cut = cov[np.ix_(valid_indices, valid_indices)]

    N = len(data_cut)

    eig_vals, eig_vecs = np.linalg.eig(cov_cut)
    deltas = data_cut - pred_cut

    decomp_deltas = np.matmul(deltas, eig_vecs)

    chi2s_decomp = []
    local_p_values = []
    look_elsewhere_corrected_p_values = []
    local_signed_sigmas = []
    for eigenval_i in range(len(eig_vals)):

        eig_val = eig_vals[eigenval_i]

        decomp_delta = decomp_deltas[eigenval_i]

        local_signed_sigma = decomp_delta / np.sqrt(eig_val)
        local_chi2 = decomp_delta**2 / eig_val
        local_p_value, local_sigma = get_significance(local_chi2, 1)
        look_elsewhere_corrected_p_value = 1 - (1 - local_p_value)**N

        chi2s_decomp.append(local_chi2)
        local_p_values.append(local_p_value)
        local_signed_sigmas.append(local_signed_sigma)
        look_elsewhere_corrected_p_values.append(look_elsewhere_corrected_p_value)
    
    chi2_tot = np.sum(chi2s_decomp)
    
    global_p_value = np.min(look_elsewhere_corrected_p_values)
    global_sigma = get_significance_from_p_value(global_p_value)

    return chi2s_decomp, chi2_tot, local_signed_sigmas, global_p_value, global_sigma

