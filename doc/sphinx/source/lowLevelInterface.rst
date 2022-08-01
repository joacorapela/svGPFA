Low-level interface
===================

The class :class:`~svGPFA.stats.svGPFAModelFactory.SVGPFAModelFactory` creates
an svGPFA model and an instance of the class :class:`~svGPFA.stats.svEM.SVEM`
optimises its parameters.  Please refer to the svGPFA `class 
<https://github.com/joacorapela/svGPFA/blob/master/docs/design/Classes.pdf>`_ and `interaction <https://github.com/joacorapela/svGPFA/blob/master/docs/design/Interactions.pdf>`_ diagrams.

There is a one-to-one mapping between classes in the :mod:`svGPFA.stats` package and equations in :cite:t:`dunckerAndSahani18`.

* Class :class:`~svGPFA.stats.svLowerBound.SVLowerBound` corresponds to the right-hand-side of ``Eq.4``. This class uses the :class:`~svGPFA.stats.expectedLogLikelihood.ExpectedLogLikelihood` and :class:`~svGPFA.stats.klDivergence.KLDivergence` classes, described next.

* The abstract class :class:`~svGPFA.stats.expectedLogLikelihood.ExpectedLogLikelihood` corresponds to the first term of the right-hand-side of ``Eq.4``. 

  #. The abstract subclass :class:`~svGPFA.stats.expectedLogLikelihood.PointProcessELL` implements the functionality of :class:`~svGPFA.stats.expectedLogLikelihood.ExpectedLogLikelihood` for point-process observations, and corresponds to ``Eq.7``. If the link function (i.e., g in ``Eq.7``) is the exponential function, then the one-dimensional integral in the first term of ``Eq.7`` can be solved analytically (concrete subclass :class:`~svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink`). For other link functions we can solve this integral using Gaussian quadrature (concrete subclass :class:`~svGPFA.stats.expectedLogLikelihood.PointProcessELLQuad`).

  #. Similarly, the subclasses :class:`~svGPFA.stats.expectedLogLikelihood.PoissonELL`, :class:`~svGPFA.stats.expectedLogLikelihood.PoissonELLExpLink` and :class:`~svGPFA.stats.expectedLogLikelihood.PoissonELLQuad` implement the functionality of :class:`~svGPFA.stats.expectedLogLikelihood.ExpectedLogLikelihood` for Poisson observations.

* The concrete class :class:`~svGPFA.stats.klDivergence.KLDivergence` corresponds to the second term of the right-hand-side of ``Eq.4`` and implements the KL divergence between the prior on inducing points, :math:`p(\mathbf{u}_k^{(r)})`, and the posterior on inducing points, :math:`q(\mathbf{u}_k^{(r)})`.

* :class:`~svGPFA.stats.expectedLogLikelihood.ExpectedLogLikelihood` uses :class:`~svGPFA.stats.svEmbedding.SVEmbedding`, which calculates the mean and variance of the svGPFA embedding (:math:`h_n^{(r)}` in ``Eq.1``), given in ``Eq.5``. :class:`~svGPFA.stats.svEmbedding.SVEmbedding` is an abstract class, which has :class:`~svGPFA.stats.svEmbedding.LinearSVEmbedding` as abstract sublcass. Two concrete subclasses of :class:`~svGPFA.stats.svEmbedding.LinearSVEmbedding` are provided, which optimise the calculation of the embedding for two different uses in ``Eq.7``. 

  #. The first term in the right-hand-side of ``Eq.7`` requires the calculation of the embedding at sample times in a grid, which are the same for all neurons. This calculation is implemented in :class:`~svGPFA.stats.svEmbedding.LinearSVEmbeddingAllTimes`.  
  #. The second term in the right-hand-side of ``Eq.7`` requires the calculation of the embedding at spike times, which are different for each neuron. This calculation is implemented in :class:`~svGPFA.stats.svEmbedding.LinearSVEmbeddingAssocTimes`.

* :class:`~svGPFA.stats.svEmbedding.SVEmbedding` uses :class:`~svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatents`, which calculates the mean and variance of the latent variables, :math:`x_k^{(r)}` in ``Eq.1``. These means and variances are not described by their own equations in `Duncker and Sahani, 2018 <https://papers.nips.cc/paper/8245-temporal-alignment-and-latent-gaussian-process-factor-inference-in-population-spike-trains>`_, but are embedded in ``Eq.5``. They are 

  .. math::

     \nu_k^{(r)}(t) &= \kappa_k(t,z_k)K_{zz}^{(k)^{-1}}m_k^{(r)}

     \sigma_k^{(r)}(t) &= \kappa_k(t,t)+\mathbf{\kappa}_k(t,\mathbf{z}_k)\left(K_{zz}^{(k)^{-1}}S_k^{(r)}K_{zz}^{(k)^{-1}}-K_{zz}^{(k)^{  -1}}\right)\mathbf{\kappa}_k(\mathbf{z}_k,t)

  :class:`~svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatents` is an abstract class. As above, two concrete subclasses are provided. :class:`~svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes` computes the means and variances in a grid of time points and :class:`~svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatentsAssocTimes` calculates these statistics at spike times.

* :class:`~svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatents` uses :class:`~svGPFA.stats.kernelMatricesStore.KernelMatricesStore`, which stores kernel matrices between inducing points, :math:`K_{zz}`, between time points, :math:`K_{tt}`, and between time points and inducing points, :math:`K_{tz}`. :class:`~svGPFA.stats.kernelMatricesStore.KernelMatricesStore` is an abstract class with two subclasses. :class:`~svGPFA.stats.kernelMatricesStore.IndPointsLocsKMS` is a concrete subclass of :class:`~svGPFA.stats.kernelMatricesStore.KernelMatricesStore` that stores kernel matrices between inducing points, and their Cholesky decompositions. :class:`~svGPFA.stats.kernelMatricesStore.IndPointsLocsAndTimesKMS` is an abstract subclass of :class:`~svGPFA.stats.kernelMatricesStore.KernelMatricesStore` which stores covariance matrices between time points and between time points and inducing points. As above, :class:`~svGPFA.stats.kernelMatricesStore.IndPointsLocsAndAllTimes` and :class:`~svGPFA.stats.kernelMatricesStore.IndPointsLocsAndAssocTimes` are concrete subclasses of :class:`~svGPFA.stats.kernelMatricesStore.IndPointsLocsAndTimesKMS` for times points in a grid and for spike times, respectively.

* :class:`~svGPFA.stats.kernelMatricesStore.KernelMatricesStore` uses  :class:`~stats.kernels.Kernel`, which is an abstract class for constructing kernel matrices. Concrete subclasses construct kernel matrices for specific types of kernels (e.g., :class:`~stats.kernels.ExponentialQuadraticKernel` and :class:`~stats.kernels.PeriodicKernel`).

