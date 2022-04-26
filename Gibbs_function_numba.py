
# Import libraries!

import numpy as np
from scipy.special import erf, erfinv
from numba import float64, int32, njit, jit

# forceobj=True isn't ideal

@jit(nopython=False, forceobj=True)
def cdf_no_stats(x, mu, sig):
    arg = (x - mu)/(sig*np.sqrt(2))
    return (1 + erf(arg))/2

#@jit(nopython=False)
@jit(nopython=False, forceobj=True)
def ppf_no_stats(x, mu, sig):
    return mu + sig*np.sqrt(2)*erfinv(2*x - 1)

@jit(nopython=False, forceobj=True)
def conditional_samp_trunc(sampling_index, current_x,
                           mean, cov, p_low, p_high, sims):

    """
    Parameters
    ----------

    sampling_index : Integer, select variable that
                     we are finding a univariate draw for

    current_x : Numpy array with current draws,
                with dimensions of (Nobs X N Variables)

    mean      : Numpy array with means of the variables of interest,
                with dimensions of (N variables)

    cov       : Numpy array for variance-covariance matrix of variables,
                with dimensions (N variables X N variables)

    p_low     : Numpy array that determines that lower cutoff
              for each observation, with dimensions (Nobs)

    p_high     : Numpy array that determines that upper cutoff
          for each observation, with dimensions (Nobs)

    sims      : Numpy array that contains standard uniform
              draws for each var. These are then scaled based
              on model parameters, with dimensions (N variables)

    Returns
    ----------

    out       : Numpy array that updates the specified dimension
                with simulation draws holding fixed other dimensions,
                with dimensions (Nobs X N variables)

    See Also
    ----------
    Kenneth, Train (2003) Discrete Choice Methods with Simulation

    """

    N_var_axis = 1
    N_vars = current_x.shape[N_var_axis]

    var_inds = np.array(list(range(N_vars)))

    cond_index = np.delete(var_inds, sampling_index)
    cond_vals = current_x[:, cond_index, :]
    cond_vals = cond_vals.sum(axis=N_var_axis)

    cutoff_low = p_low - cond_vals[:, 0]
    cutoff_med1 = p_low - cond_vals[:, 1]
    cutoff_med2 = p_high - cond_vals[:, 1]
    cutoff_hig = p_high - cond_vals[:, 2]

    my_sim_i = sims[sampling_index]

#     ==================================
#             LOW TYPE
#     ==================================

# No lower bound restriction for low type

    ub_low = cdf_no_stats(cutoff_low,
             mean[sampling_index],
             cov[sampling_index, sampling_index])

    # Need to input a specific 3-triple of sim draws

    mu_bar_low =  np.array([my_sim_i])*ub_low


    draw_low = ppf_no_stats(mu_bar_low,
             mean[sampling_index],
             cov[sampling_index, sampling_index])


#     ==================================
#             MEDIUM TYPE
#     ==================================

    lb_med = cdf_no_stats(cutoff_med1,
                  mean[sampling_index],
                  cov[sampling_index, sampling_index])

    ub_med = cdf_no_stats(cutoff_med2,
              mean[sampling_index],
              cov[sampling_index, sampling_index])


    # Need to input a specific 3-triple of sim draws
    mu_bar_med =  np.array([my_sim_i])*ub_med + np.array([(1 - my_sim_i)])*lb_med

    # return mu_bar_med

    draw_med = ppf_no_stats(mu_bar_med,
             mean[sampling_index],
             cov[sampling_index, sampling_index])


#     ==================================
#             HIGH TYPE
#     ==================================

# No upper bound restriction for high type

    lb_hig = cdf_no_stats(cutoff_hig,
                  mean[sampling_index],
                  cov[sampling_index, sampling_index])

    # Need to input a specific 3-triple of sim draws
    mu_bar_hig =  np.array([1 - my_sim_i])*lb_hig + np.array([my_sim_i])*(1) # Times one means we are always below positive infinity

    draw_hig = ppf_no_stats(mu_bar_hig,
             mean[sampling_index],
             cov[sampling_index, sampling_index])

    # Deal with cases that are really extreme.
    # Only relevant for lowest case
    # -- Not sure if this will be problematic
    draw_low[draw_low == -np.inf] = cutoff_low[draw_low == -np.inf]
    draw_hig[draw_hig == np.inf] = cutoff_hig[draw_hig == np.inf]

    new_x = np.copy(current_x)

    low_ind = 0
    med_ind = 1
    hig_ind = 2

    new_x[:, sampling_index, low_ind] = draw_low
    new_x[:, sampling_index, med_ind] = draw_med
    new_x[:, sampling_index, hig_ind] = draw_hig

    return new_x

@jit(nopython=False, forceobj=True)
def trunc_gibbs_sampler(initial_point, mean, cov,
                        p_low, p_high, my_draws):

    num_samples = my_draws.shape[0]

    point = initial_point


    Nobs = len(p_low)
    N_vars = len(mean)
    N_types = 3

    samples = np.empty([Nobs, num_samples + 1, N_vars, N_types]) # sampled points
    samples[:, 0,:, :] = point
    tmp_points1 = np.empty([Nobs, num_samples, N_vars, N_types]) # inbetween points
    tmp_points2 = np.empty([Nobs, num_samples, N_vars, N_types]) # inbetween points

    for i in range(num_samples):

        # Sample from p(x_0|x_1, x_2)
        point = conditional_samp_trunc(0, point, mean, cov,
                                      p_low, p_high, my_draws[i, :])
        tmp_points1[:,i,:, :] = point

        # Sample from p(x_1|x_0, x_2)
        point = conditional_samp_trunc(1, point, mean, cov,
                                      p_low, p_high, my_draws[i, :])
        tmp_points2[:,i, :, :] = point

        # Sample from p(x_2|x_0, x_1)
        point = conditional_samp_trunc(2, point, mean, cov,
                                      p_low, p_high, my_draws[i, :])
        samples[:, i+1, :, :] = point

    return samples
