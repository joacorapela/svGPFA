Data structures
===============

1. ``spikes_times``: data structure containing spikes times
   (:math:`\mathbf{t}_n^{(r)}` in Eq. 6 of :cite:t:`dunckerAndSahani18`).

   .. Input ``measurements`` to method
      :meth:`svGPFA.stats.svLowerBound.SVLowerBound.setMeasurements`.

   Double list of length ``n_trials`` by ``n_neurons`` such that
   ``spikes_times[r][n]`` is a list-like collection of spikes times for trials
   ``r`` and  neuron ``n``.

2. ``ind_points_locs``: data structure containing inducing points locations
   (:math:`\mathbf{z}_k^{(r)}` in Eq. 3 of :cite:t:`dunckerAndSahani18`).

   .. Input ``locs`` to method
      :meth:`svGPFA.stats.svLowerBound.SVLowerBound.setIndPointsLocs`.

   List of length ``n_latents`` of PyTorch tensors of size (``n_trials``,
   ``n_ind_points``, 1), such that ``ind_points_locs[k][r, :, 0]`` gives the
   inducing points locations for trial ``r`` and latent ``k``.

3. ``leg_quad_points`` and ``leg_quad_weights``: data structures containing the
   Legendre quadrature points and weights, respectively, used for the
   calculation of the integral in the first term of the expected posterior
   log-likelihood in Eq. 7 in :cite:t:`dunckerAndSahani18`. 

   .. Input ``eLLCalculationParams`` to method
      :meth:`svGPFA.stats.svLowerBound.SVLowerBound.setInitialParamsAndData`.

   Both ``leg_quad_points`` and ``leg_quad_weights`` are tensors of size
   (``n_trials``, ``n_quad_elem``, 1), such that ``leg_quad_points[r, :, 0]``
   and ``leg_quad_weights[r, :, 0]`` give the quadrature points and weights,
   respectively, of trial ``r``.

4. ``var_mean``: mean of the variational distribution
   :math:`q(\mathbf{u}_k^{(r)})` (:math:`\mathbf{m}_k^{(r)}` in the paragraph
   above Eq. 4 of :cite:t:`dunckerAndSahani18`).

   List of length ``n_latents`` of PyTorch tensors of size (``n_trials``,
   ``n_ind_points``, 1), such that ``var_mean[k][r, :, 0]`` gives the
   variational mean for trial ``r`` and latent ``k``.

5. ``var_chol``: vectorized cholesky factor of the covariance of the
   variational distribution :math:`q(\mathbf{u}_k^{(r)})` (:math:`S_k^{(r)}` in the paragraph above Eq. 4 of
   :cite:t:`dunckerAndSahani18`).

   List of length ``n_latents`` of PyTorch tensors of size (``n_trials``, ``n_ind_points`` * (``n_ind_points`` + 1)/2, 1), such that ``var_chol[k][r, :, 0]`` gives the vectorized cholesky factor of the variational covariance for trial ``r`` and latent ``k``.

6. ``emb_post_mean_quad``: embedding posterior mean (:math:`\nu_n^{(r)}(t)` in Eq. 5
   of :cite:t:`dunckerAndSahani18`) evaluated at the Legendre quadrature
   points.

   List of length ``n_latents`` of PyTorch tensor of size (``n_trials``, n_ind_points, 1), such that ``emb_post_mean[k][r, :, 0]``  gives the embedding posterior mean for trial ``r`` and latent ``k``, evaluated at ``leg_quad_points[r, :, 0]``.

7. ``emb_post_var_quad``: embedding posterior variance (:math:`\sigma_n^{(r)}(t, t)` in Eq. 5
   of :cite:t:`dunckerAndSahani18`) evaluated at the Legendre quadrature
   points.

   List of length ``n_latents`` of PyTorch tensors of size (``n_trials``, n_ind_points, 1), such that ``emb_post_var[k][r, :, 0]``  gives the embedding posterior variance for trial ``r`` and latent ``k``, evaluated at ``leg_quad_points[r, :, 0]``.

8. ``Kzz``: kernel covariance function evaluated at inducing points locations (:math:`K_{zz}^{(kr)}` in Eq. 5 of :cite:t:`dunckerAndSahani18`).

   List of length ``n_latents`` of PyTorch tensors of PyTorch tensors of size (``n_trials``, ``n_ind_points``, ``n_ind_points``), such that ``Kzz[k][r, i, j]`` is the kernel covariance function for latent ``k`` evaluated at the ith and kth components of the inducing point locations for latent ``k`` and trial ``r`` (``Kzz[k][r, i, j]`` = :math:`\kappa_k(\mathbf{z}_k^{(r)}[i], \mathbf{z}_k^{(r)}[j])`).

9. ``Kzz_inv``: lower-triangular cholesky factors (:math:`L^{(kr)}`) of kernel covariance matrices evaluated at inducing points locations (:math:`K_{zz}^{(kr)}=L^{(kr)}\left(L^{(kr)}\right)^\intercal`).

   List of length ``n_latents`` of PyTorch tensors of PyTorch tensors of size (``n_trials``, ``n_ind_points``, ``n_ind_points``), such that ``Kzz_inv[k][r, :, :]`` is the lower-triangular cholesky factor :math:`L^{(kr)}`.

10. ``Ktz``: kernel covariance function evaluated at the quadrature points and at the inducing points locations (:math:`\kappa_k(t, \mathbf{z}_k^{(r)})` in Eq. 5 of :cite:t:`dunckerAndSahani18`).

   List of length ``n_latents`` of PyTorch tensors of size (``n_trials``, ``n_quad_elem``, ``n_ind_points``), such that ``Kzz[k][r, i, j]`` is the kernel covariance function for latent ``k`` evaluated at the ith quadrature time for trial ``r`` (``leg_quad_points[r, :, 0]``) and at the jth inducing points location for trial ``r`` and latent ``k`` (``Kzz[k][r, i, j]`` = :math:`\kappa_k(\text{leg_quad_points}[r, i, 0], \mathbf{z}_k^{(r)}[j]`).

11. ``Ktt``: kernel covariance function evaluated at quadrature points (:math:`\kappa_k(t, t)` in Eq. 5 of :cite:t:`dunckerAndSahani18`).
   
    Note: svGPFA does not need to evaluate :math:`\kappa_k(t, t')` for :math:`t\neq t'`. It only needs to evaluate :math:`\kappa_k(t, t)` to calculate the variance of the posterior embedding :math:`\sigma^2_n(t, t)`, which is used to compute :math:`\mathbb{E}_{q(h_n^{(r)})}\left[g(h_n^{(r)}(t))\right]`.

    List of length ``n_latents`` of PyTorch tensors of size (``n_trials``, ``n_quad_elem``, ``n_latents``), such that ``Ktt[k][r, i, k]`` is the kernel variance function for latent ``k`` evaluated at the ith quadrature time for trial ``r`` (``leg_quad_points[r, i, 0]``). That is ``Ktt[k][r, i, k]`` = :math:`\kappa_k(\text{leg_quad_points[r, i, 0]},  \text{leg_quad_points[r, i, 0]})`.

