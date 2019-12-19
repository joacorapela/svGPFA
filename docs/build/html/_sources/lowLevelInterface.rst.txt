Low-level interface
===================

The class :class:`~stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory` creates
an svGPFA model and an instance of the class :class:`~stats.svGPFA.svEM.SVEM`
optimizes its parameters.  Please refer to the `svGPFA class diagram
<https://github.com/joacorapela/svGPFA/blob/master/docs/design/Classes.pdf>`_.

There is a one-to-one mapping between classes in the stats.svGPFA package and
equations in `Duncker and Sahani, 2018 <https://papers.nips.cc/paper/8245-temporal-alignment-and-latent-gaussian-process-factor-inference-in-population-spike-trains>`_.

* Class :class:`~stats.svGPFA.svLowerBound.SVLowerBound` corresponds to the right-hand-side of ``Eq.4``. This class uses the :class:`~stats.svGPFA.expectedLogLikelihood.ExpectedLogLikelihood` and :class:`~stats.svGPFA.klDivergence.KLDivergence` classes, described next.

* The abstract class :class:`~stats.svGPFA.expectedLogLikelihood.ExpectedLogLikelihood` corresponds to the first term of the right-hand-side of ``Eq.4``. 

  #. The abstract subclass :class:`~stats.svGPFA.expectedLogLikelihood.PointProcessELL` implements the functionality of :class:`~stats.svGPFA.expectedLogLikelihood.ExpectedLogLikelihood` for point-process observations, and corresponds to ``Eq.7``. If the link function (i.e., g in ``Eq.7``) is the exponential function, then the one-dimensional integral in the first term of ``Eq.7`` can be solved analytically (concrete subclass :class:`~stats.svGPFA.expectedLogLikelihood.PointProcessELLExpLink`). For other link functions we can solve this integral using Gaussian quadrature (concrete subclass :class:`~stats.svGPFA.expectedLogLikelihood.PointProcessELLQuad`).

  #. Similarly, the subclasses :class:`~stats.svGPFA.expectedLogLikelihood.PoissonELL`, :class:`~stats.svGPFA.expectedLogLikelihood.PoissonELLExpLink` and :class:`~stats.svGPFA.expectedLogLikelihood.PoissonELLQuad` implement the functionality of :class:`~stats.svGPFA.expectedLogLikelihood.ExpectedLogLikelihood` for Poisson observations.

* The concrete class :class:`~stats.svGPFA.klDivergence.KLDivergence` corresponds to the second term of the right-hand-side of ``Eq.4`` and implements the KL divergence between the prior on inducing points, :math:`p(\mathbf{u}_k^{(r)})`, the posterior on inducing points, :math:`q(\mathbf{u}_k^{(r)})`.

* :class:`~stats.svGPFA.expectedLogLikelihood.ExpectedLogLikelihood` uses :class:`~stats.svGPFA.svEmbedding.SVEmbedding`, which calculates the mean and variance of the svGPFA embedding (:math:`h_n^{(r)}` in ``Eq.1``), given in ``Eq.5``. :class:`~stats.svGPFA.svEmbedding.SVEmbedding` is an abstract class, which has :class:`~stats.svGPFA.svEmbedding.LinearSVEmbedding` as abstract sublcass. Two concrete subclasses of :class:`~stats.svGPFA.svEmbedding.LinearSVEmbedding` are provided, which optimize the calculation of the embedding for two different uses in ``Eq.7``. 

  #. The first term in the righ-hand-side of ``Eq.7`` requires the calculation of the embedding at sample times in a grid, which are the same for all neurons. This calculation is implemented in :class:`~stats.svGPFA.svEmbedding.LinearSVEmbeddingAllTimes`.  
  #. The second term in the right-hand-side of ``Eq.7`` requires the calculation of the embedding at spike times, which are different for each neuron. This calculation is implemented in :class:`~stats.svGPFA.svEmbedding.LinearSVEmbeddingAssocTimes`.

* :class:`~stats.svGPFA.svEmbedding.SVEmbedding` uses :class:`~stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatents`, which calculates the mean and variance of the latent variables, :math:`x_k^{(r)}` in ``Eq.1``. These means and variances are not described by their own equations in `Duncker and Sahani, 2018 <https://papers.nips.cc/paper/8245-temporal-alignment-and-latent-gaussian-process-factor-inference-in-population-spike-trains>`_, but are embedded in ``Eq.5``. They are 

  .. math::

     \nu_k^{(r)}(t) &= \kappa_k(t,z_k)K_{zz}^{(k)^{-1}}m_k^{(r)}

     \sigma_k^{(r)}(t) &= \kappa_k(t,t)+\mathbf{\kappa}_k(t,\mathbf{z}_k)\left(K_{zz}^{(k)^{-1}}S_k^{(r)}K_{zz}^{(k)^{-1}}-K_{zz}^{(k)^{  -1}}\right)\mathbf{\kappa}_k(\mathbf{z}_k,t)

  :class:`~stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatents` is an abstract class. As above, two concrete subclasses are provided. :class:`~stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes` computes the means and variances in a grid of time points and :class:`~stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAssocTimes` calculates these statistics at spike times.

* :class:`~stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatents` uses :class:`~stats.svGPFA.kernelMatricesStore.KernelMatricesStore`, which stores kernel matrices between inducing points, :math:`K_{zz}`, between time points, :math:`K_{tt}`, and between time points and inducing points, :math:`K_{tz}`. :class:`~stats.svGPFA.kernelMatricesStore.KernelMatricesStore` is an abstract class with two subclasses. :class:`~stats.svGPFA.kernelMatricesStore.IndPointsLocsKMS` is a concrete subclass of :class:`~stats.svGPFA.kernelMatricesStore.KernelMatricesStore` that stores kernel matrices between inducing points, and their Cholesky decompositions. :class:`~stats.svGPFA.kernelMatricesStore.IndPointsLocsAndTimesKMS` is an abstract subclass of :class:`~stats.svGPFA.kernelMatricesStore.KernelMatricesStore` which stores covariance matrices between time points and between time points and inducing points. As above, :class:`~stats.svGPFA.kernelMatricesStore.IndPointsLocsAndAllTimes` and :class:`~stats.svGPFA.kernelMatricesStore.IndPointsLocsAndAssocTimes` are concrete subclasses of :class:`~stats.svGPFA.kernelMatricesStore.IndPointsLocsAndTimesKMS` for times points in a grid and for spike times, respectively.

* :class:`~stats.svGPFA.kernelMatricesStore.KernelMatricesStore` uses  :class:`~stats.kernels.Kernel`, which is an abstract class for constructing kernel matrices. Concrete subclasses contruct kernel matrices for specific types of kernels (e.g., :class:`~stats.kernels.ExponentialQuadraticKernel` and :class:`~stats.kernels.PeriodicKernel`).

