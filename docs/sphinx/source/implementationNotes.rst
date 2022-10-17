Implementation notes
====================

The parameters optimized in svGPFA are:

1. variational parameters: means, :math:`\mathbf{m}_k^{(r)}\in\mathbb{R}^\text{n_ind_points(k)}`, and covariances, :math:`S_k^{(r)}\in\mathbb{R}^{\text{n_ind_points(k)}\times\text{n_ind_points(k)}}`, of the variational distributions :math:`q(\mathbf{u}_k^{(r)})=\mathcal{N}(\mathbf{u}_k^{(r)}|\mathbf{m}_k^{(r)}, S_k^{(r)})` (paragraph above Eq. 4 in :cite:t:`dunckerAndSahani18`),

2. embedding parameters: :math:`C\in\mathbb{R}^{\text{n_neurons}\times\text{n_latents}}` and :math:`d\in\mathbb{R}^\text{n_neurons}` (Eq. 1, middle row, in :cite:t:`dunckerAndSahani18`),

3. kernels parameters: parameters of :math:`\kappa_k(\cdot,\cdot)` in Eq. 1, top row, of :cite:t:`dunckerAndSahani18`,

4. inducing points locations: :math:`\mathbf{z}_k^{(r)}\in\mathbb{R}^\text{n_ind_points(k)}` in Eq. 2 of :cite:t:`dunckerAndSahani18`.

for :math:`k=1,\ldots,K` and :math:`r=1,\ldots,R`.

The estimation of svGPFA parameters is performed using the Expectation Conditional Maximization algorithm (:cite:t:`mcLachlanAndKrishnan08`, see :meth:`svGPFA.stats.svEM.SVEM.maximize`), which reduces to a sequence of numerical optimizations. Because we use PyTorch's autograd to compute derivatives, these optimizations only require the calculation of the svGPFA lower bound (left hand side of Eq. 4 in :cite:t:`dunckerAndSahani18`). Below we provide details about how the calculation of this lower bound is implemented.

