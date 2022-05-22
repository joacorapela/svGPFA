Other parameters
================

Section ``[other_params]`` should include:

* ``n_latents`` integer value indicating the number of latents of the model,

* ``trials_start_times`` list of length ``number_of_trials`` containing in the ith position the start time (in seconds) of the ith trial,

* ``trials_end_times`` list of length ``number_of_trials`` containing in the ith position the end time (in seconds) of the ith trial,

* ``n_quad`` integer value giving the number of points used for the approximation of the integral in the first term of Eq. 7 in :cite:t:`dunckerAndSahani18`,

* ``prior_cov_reg_param`` float value specifying the constant added to the diagonal of the inducing points prior covariance matrix , :math:`K_{zz}^{(k,r)}`, to mitigate numerical problems related to its poor conditioning.

