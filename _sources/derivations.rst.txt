Derivations
===========

GPFA model
----------

Equation :eq:`eq:gpfaModel` represents the GPFA model:

.. math::
   p(\{x_{kr}(\cdot)\}_{k=1,r=1}^{K,R}) &= \prod_{r=1}^R\prod_{k=1}^Kp(x_{kr}(\cdot))\\
   x_{kr}(\cdot) &\sim \mathcal{GP}(\mu_k(\cdot),\kappa_k(\cdot,\cdot))&&{\text{for}\; k=1, \ldots, K\;\text{and}\;r=1,\ldots, R}\\
   h_{nr}(\cdot) &= \sum_{k=1}^Kc_{nk}x_{kr}(\cdot) + d_n&&{\text{for}\; n=1, \ldots, N\;\text{and}\;r=1,\ldots, R}\\
   p(\{y_{nr}\}_{n=1,r=1}^{N,R}|\{h_{nr}(\cdot)\}_{n=1,r=1}^{N,R}) &= \prod_{r=1}^R\prod_{n=1}^Np(y_{nr}|h_{nr}(\cdot))
   :label: eq:gpfaModel

where :math:`x_{kr}(\cdot)` is the latent process :math:`k` in trial :math:`r`, :math:`h_{nr}(\cdot)` is the embedding process for neuron :math:`n` and trial :math:`r` and :math:`y_{nr}` is the activity of neuron :math:`n` in trial :math:`r`.

Notes:

    1. the first equation shows that the latent processes are independent,

    2. the second equation shows that the latent processes share mean and covariance functions across trials. That is, for any :math:`k`, the mean and covariance functions  of latents processes of different trials, :math:`x_{kr}(\cdot), r=1,\ldots, R`, are the same (:math:`\mu_k(\cdot)` and :math:`\kappa_k(\cdot,\cdot)`),

    3. the fourth equation shows that, given the embedding processes, the responses of different neurons are independent.

GPFA with inducing points model
-------------------------------

To use the sparse variational framework for Gaussian processes,
:cite:t:`dunckerAndSahani18` augmented the GPFA model by introducing inducing
points :math:`\mathbf{u}_{kr}` for each latent process :math:`k` and trial
:math:`r`. The inducing points :math:`\mathbf{u}_{kr}` represent evaluations of
the latent process :math:`x_{kr}(\cdot)` at locations
:math:`\mathbf{z}_{kr}=\left[z_{kr}[0],\ldots,z_{kr}[M_{kr}-1]\right]`. A
joint prior over the latent process :math:`x_{kr}(\cdot)` and its inducing
points :math:`\mathbf{u}_{kr}` is given in Eq. :eq:`eq:gpfaWithIndPointsPrior`.

.. math::
   p(\{x_{kr}(\cdot)\}_{k=1,r=1}^{K,R},\{\mathbf{u}_{kr}\}_{k=1,r=1}^{K,R}) &= p(\{x_{kr}(\cdot)\}_{k=1,r=1}^{K,R}|\{\mathbf{u}_{kr}\}_{k=1,r=1}^{K,R})p(\{\mathbf{u}_{kr}\}_  {k=1,r=1}^{K,R})\\
   p(\{x_{kr}(\cdot)\}_{k=1,r=1}^{K,R}|\{\mathbf{u}_{kr}\}_  {k=1,r=1}^{K,R}) &= \prod_{k=1}^k\prod_{r=1}^{R}p(x_{kr}(\cdot)|\mathbf{u}_{kr})\\
   p(\{\mathbf{u}_{kr}\}_{k=1,r=1}^{K,R})&=\prod_{k=1}^k\prod_{r=1}^{R}p(\mathbf{u}_{kr})\\
   p(\mathbf{u}_{kr})&=\mathcal{N}(\mathbf{0},K^{kr}_{zz})
   :label: eq:gpfaWithIndPointsPrior

where :math:`K_{zz}^{(kr)}[i,j]=\kappa_k(z_{kr}[i],z_{kr}[j])`.

We next derive the functional form of :math:`p(x_{kr}(\cdot)|\mathbf{u}_{kr})`.

Define the random vector :math:`\mathbf{x}_{kr}` as the random process
:math:`x_{kr}(\cdot)` evaluated at times
:math:`\mathbf{t}^{(r)}=\left\{t_1^{(r)},\ldots,t_M^{(r)}\right\}` (i.e.,
:math:`\mathbf{x}_{kr}=[x_{kr}(t_1^{(r)}),\ldots,x_{kr}(t_M^{(r)})]^\intercal`).
Because the inducing points :math:`\mathbf{u}_{kr}` are evaluations of the
latent process :math:`x_{kr}(\cdot)` at :math:`\mathbf{z}_{kr}`, then :math:`\mathbf{x}_{kr}` and
:math:`\mathbf{u}_{kr}` are jointly Gaussian:

.. math::
    p\left(\left[\begin{array}{c}
        \mathbf{u}_{kr}\\
        \mathbf{x}_{kr}
    \end{array}\right]\right)
    =\mathcal{N}\left(\left.\left[\begin{array}{c}
        \mathbf{u}_{kr}\\
        \mathbf{x}_{kr}
    \end{array}\right]\right|\left[\begin{array}{c}
        \mathbf{0}\\
        \mathbf{0}
    \end{array}\right],\left[\begin{array}{cc}
        K_\mathbf{zz}^{(kr)}&K_\mathbf{zt}^{(kr)}\\
        K_\mathbf{tz}^{(kr)}&K_\mathbf{tt}^{(r)}
    \end{array}\right]\right)
    :label: eq:prior

where
:math:`K_\mathbf{tz}^{(kr)}[i,j]=\kappa_k(t^{(r)}_i,z_{kr}[j])`,
:math:`K_\mathbf{zt}^{(kr)}[i,j]=\kappa_k(z_{kr}[i],t_j^{(r)})`
and
:math:`K_\mathbf{tt}^{(r)}[i,j]=\kappa_k(t_i^{(r)},t_j^{(r)})`.

Now, applying the formula for the conditional pdf for jointly Normal random
vectors :cite:p:`bishop06`, Eq. 2116, to Eq. :eq:`eq:prior`, we obtain

.. math::
   p(\mathbf{x}_{kr}|\mathbf{u}_{kr})=\mathcal{N}\left(\mathbf{x}_{kr}\left|K_\mathbf{tz}^{(kr)}\left(K_{zz}^{(kr)}\right)^{-1}\mathbf{u}_{kr},\;K_\mathbf{tt}^{(r)}-K_\mathbf{tz}^{(kr)}\left(K_{zz}^{(kr)}\right)^{-1}K_\mathbf{zt}^{(kr)}\right.\right)
   :label: eq:latentConditionalIndPointsVector

Because Eq. :eq:`eq:latentConditionalIndPointsVector` is valid for any
:math:`\mathbf{t}^{(r)}`, it follows that

.. math::
   p(x_{kr}(\cdot)|\mathbf{u}_{kr})=\mathcal{GP}\left(\tilde{\mu}_{kr}(\cdot), \tilde{\kappa}_{kr}(\cdot,\cdot\right))

with

.. math::
   \tilde{\mu}_{kr}(t)&=\kappa_k(t,\mathbf{z}_{kr})\left(K_{zz}^{(kr)}\right)^{-1}\mathbf{u}_{kr},\\
   \tilde{\kappa}_k(t,t')&=\kappa_k(t,t')-\kappa_k(t,\mathbf{z}_{kr})\left(K_{zz}^{(kr)}\right)^{-1}\kappa_k(\mathbf{z}_{kr},t')

which is Eq. 3 in :cite:t:`dunckerAndSahani18`.

svGPFA variational lower bound
------------------------------

:numref:`Theorem {number} <thmVariationalLowerBound>` proves Eq. 4 in
:cite:t:`dunckerAndSahani18`.

.. _thmVariationalLowerBound:
.. proof:theorem:: svGPFA Variational Lower Bound

   Let :math:`\mathcal{Y}=\{y_{nr}\}_{n=1,r=1}^{N,R}` then

   .. math::
      \log p(\mathcal{Y})\ge\sum_{n=1}^N\sum_{r=1}^R\mathbb{E}_{q\left(h_{nr}(\cdot)\right)}\left\{\log p(y_{nr}|h_{nr}(\cdot))\right\}-\sum_{r=1}^R\sum_{k=1}^KKL(q(\mathbf{u}_{kr})||p(\mathbf{u}_{kr}))
      :label: eq:variationalLowerBound

.. proof:proof::

   We begin with the joint-data likelihood of the full model, given in Eq.1 of
   the supplementary material in :cite:t:`dunckerAndSahani18`

   .. math::
      p\left(\mathcal{Y},\{x_{kr}(\cdot)\}_{k=1,r=1}^{K,R},\{\mathbf{u}_{kr}\}_{k=1,r=1}^{K,R}\right)=p\left(\mathcal{Y}|\{x_{kr}(\cdot)\}_{k=1,r=1}^{K,R}\right)\prod_{k=1}^K\prod_{r=1}^Rp(x_{kr}(\cdot)|\mathbf{u}_{kr})p(\mathbf{u}_{kr})
      :label: eq:jointDataLikelihood

   For notational clarity, from now on we omit the bounds of the :math:`k` and :math:`r` indices. From :numref:`Corollary {number} <corollaryVariationalInequality>` by taking :math:`x=\mathcal{Y}` and :math:`z=\left(\{x_{kr}(\cdot)\},\{\mathbf{u}_{kr}\}\right)`, we obtain

   .. math::
      \log p\left(\mathcal{Y}\right)\ge\int\int q\left(\{x_{kr}(\cdot)\},\{\mathbf{u}_{kr}\}\right)\log\frac{p\left(\mathcal{Y},\{x_{kr}(\cdot)\},\{\mathbf{u}_{kr}\}\right)}{q\left(\{x_{kr}(\cdot)\},\{\mathbf{u}_{kr}\}\right)}d\{x_{kr}(\cdot)\}d\{\mathbf{u}_{kr}\}
      :label: eq:vlbProof1

   Choosing

   .. math:: 
      q\left(\{x_{kr}(\cdot)\},\{\mathbf{u}_{kr}\}\right)=\prod_{r=1}^R\prod_{k=1}^Kp(x_{kr}(\cdot)|\mathbf{u}_{kr})q(\mathbf{u}_{kr})

   and using Eq. :eq:`eq:jointDataLikelihood` we can rewrite Eq. :eq:`eq:vlbProof1` as

    .. math::
       \log p\left(\mathcal{Y}\right)\ge&\int\int q\left(\{x_{kr}(\cdot)\},\{\mathbf{u}_{kr}\}\right)\left(\log p\left(\mathcal{Y}|\{x_{kr}(\cdot)\}\right)-\sum_{r=1}^R\sum_{k=1}^K\log\frac{q\left(\mathbf{u}_{kr}\right)}{p\left(\mathbf{u}_{kr}\right)}\right)d\{x_{kr}(\cdot)\}d\{\mathbf{u}_{kr}\}\nonumber\\
				        =&\int\int q\left(\{x_{kr}(\cdot)\},\{\mathbf{u}_{kr}\}\right)\log p\left(\mathcal{Y}|\{x_{kr}(\cdot)\}\right)d\{x_{kr}(\cdot)\}d\{\mathbf{u}_{kr}\}-\nonumber\\
				         &\int\int q\left(\{x_{kr}(\cdot)\},\{\mathbf{u}_{kr}\}\right)\sum_{r=1}^R\sum_{k=1}^K\log\frac{q\left(\mathbf{u}_{kr}\right)}{p\left(\mathbf{u}_{kr}\right)}d\{x_{kr}(\cdot)\}d\{\mathbf{u}_{kr}\}\nonumber\\
				        =&\int q\left(\{x_{kr}(\cdot)\}\right)\log p\left(\mathcal{Y}|\{x_{kr}(\cdot)\}\right)d\{x_{kr}(\cdot)\}-\sum_{r=1}^R\sum_{k=1}^K\int q\left(\mathbf{u}_{kr}\right)\log\frac{q\left(\mathbf{u}_{kr}\right)}{p\left(\mathbf{u}_{kr}\right)}d\mathbf{u}_{kr}\nonumber\\
				        =&\;\mathbb{E}_{q\left(\{x_{kr}(\cdot)\}\right)}\left\{\log p\left(\mathcal{Y}|\{x_{kr}(\cdot)\}\right)\right\}-\sum_{r=1}^R\sum_{k=1}^K\text{KL}\left(q\left(\mathbf{u}_{kr}\right)||p\left(\mathbf{u}_{kr}\right)\right)\\
				        =&\;\mathbb{E}_{q\left(\{h_{nr}(\cdot)\}\right)}\left\{\log p\left(\mathcal{Y}|\{h_{nr}(\cdot)\}\right)\right\}-\sum_{r=1}^R\sum_{k=1}^K\text{KL}\left(q\left(\mathbf{u}_{kr}\right)||p\left(\mathbf{u}_{kr}\right)\right)\\
				        =&\;\mathbb{E}_{q\left(\{h_{nr}(\cdot)\}\right)}\left\{\sum_{n=1}^N\sum_{r=1}^R\log p\left(y_{nr}|h_{nr}(\cdot)\right)\right\}-\sum_{r=1}^R\sum_{k=1}^K\text{KL}\left(q\left(\mathbf{u}_{kr}\right)||p\left(\mathbf{u}_{kr}\right)\right)\\
				        =&\;\sum_{n=1}^N\sum_{r=1}^R\mathbb{E}_{q\left(h_{nr}(\cdot)\right)}\left\{\log p\left(y_{nr}|h_{nr}(\cdot)\right)\right\}-\sum_{r=1}^R\sum_{k=1}^K\text{KL}\left(q\left(\mathbf{u}_{kr}\right)||p\left(\mathbf{u}_{kr}\right)\right)\nonumber

   Notes:
       1. the derivation of the equation in the sixth line from that in the fifth one is subtle. It assumes that there exists a measurable and injective change of variables function :math:`f(\{x_{kr}(\cdot)\})=\{h_{nr}(\cdot)\}`.
       2. the equation in the seventh line follows from that in the sixth one using the last line in Eq. :eq:`eq:gpfaModel`.

....


.. _lemmaVariationalEquality:
.. proof:lemma:: Variational Equality

   .. math::
      \log p(x) = \mathbb{E}_{q(z)}\left\{\log\frac{p(x,z)}{q(z)}\right\}+\text{KL}\left\{q(z)||p(z|x)\right\}

.. proof:proof::

   .. math::
      p(x)&=\frac{p(x,z)}{p(z|x)}=\frac{p(x,z)}{q(z)}\frac{q(z)}{p(z|x)}\\
      \log p(x)&=\log\frac{p(x,z)}{q(z)}+\log\frac{q(z)}{p(z|x)}\\
      \log p(x)&=\mathbb{E}_{q(z)}\left\{\log\frac{p(x,z)}{q(z)}\right\}+\mathbb{E}_{q(z)}\left\{\log\frac{q(z)}{p(z|x)}\right\}\\
      \log p(x)&=\mathbb{E}_{q(z)}\left\{\log\frac{p(x,z)}{q(z)}\right\}+\text{KL}\left\{q(z)||p(z|x)\right\}

   Notes:
       1. the first equation uses Bayes rule,
       2. the third equation applies the expected value to both sides of the second equation,
       3. the last equation uses the definition of the KL divergence.

....

.. _corollaryVariationalInequality:
.. proof:corollary:: Variational Inequality

   .. math::
      \log p(x) \ge \mathbb{E}_{q(z)}\left\{\log\frac{p(x,z)}{q(z)}\right\}
      :label: eq:variationalInequality

   with equality if and only if :math:`q(z)=p(z|x)`.

.. proof:proof::

   Equation :eq:`eq:variationalInequality` follows from
   :numref:`Lemma {number} <lemmaVariationalEquality>` by the
   fact that the KL divergence between two
   distributions is greater or equal than zero, with equality if and only if
   the distributions are equal (Information inequality,
   :cite:t:`coverAndThomas91`, Theorem 2.6.3).

....

Variational distribution of :math:`h_{nr}(\cdot)`
-------------------------------------------------

For the calculation of the lower bound in the right-hand side of
Eq. :eq:`eq:variationalLowerBound`, below we derive the distribution
:math:`q(h_{nr}(\cdot))`.

We first deduce the distribution :math:`q(x_{xr}(\cdot))`. Note, from
Eq. :eq:`eq:gpfaWithIndPointsPrior`, that for any :math:`P\in\mathbb{N}` and for any
:math:`\mathbf{t}=(t_1,\ldots,t_P)\in\mathbb{R}^P` the approximate variational
posterior of the random vectors
:math:`\mathbf{x}_{kr}=(x_{kr}(t_1),\ldots,x_{kr}(t_P))` and :math:`\mathbf{u}_{kr}` is
jointly Gaussian

.. math::
   q(\mathbf{x}_{kr},\mathbf{u}_{kr})&=p(\mathbf{x}_{kr}|\mathbf{u}_{kr})q(\mathbf{u}_{kr})\\
                                     &=\mathcal{N}\left(\mathbf{x}_{kr}|K_{tz}^{kr}(K_{zz}^{kr})^{-1}\mathbf{u}_{kr},\;K_{tt}^k-K_{tz}^{kr}(K_{zz}^{kr})^{-1}K_{zt}^{kr}\right)\mathcal{N}(\mathbf{u}_{kr}|\mathbf{m}_{kr},\;S_{kr})

where :math:`K_{tt}`, :math:`K_{tz}`, :math:`K_{zt}`, and :math:`K_{zz}` are covariance matrices obtained by evalating of :math:`\kappa_k(t,t')`, :math:`\kappa_k(t,z)`, :math:`\kappa_k(z,t)`, and :math:`\kappa_k(z,z')`, respectively, at :math:`t,t'\in \{t_1,\ldots t_P\}` and :math:`z,z'\in \{\mathbf{z}_{kr}[1],\ldots,\mathbf{z}_{kr}[M_{kr}]\}`. Next, using the expression for the marginal of a joint Gaussian distribution (e.g., Eq.~2.115 in :cite:t:`bishop06`) we obtain

.. math::
   q(\mathbf{x}_{kr})=\mathcal{N}\left(\mathbf{x}_{kr}|K_{tz}^{kr}(K_{zz}^{kr})^{-1}\mathbf{m}_{kr},\;K_{tt}^k+K_{tz}^{kr}\left((K_{zz}^{kr})^{-1}S_{kr}(K_{zz}^{kr})^{-1}-(K_{zz}^{kr})^{-1  }\right)K_{zt}^{kr}\right)
   :label: eq:qxRandomVec

Because Eq. :eq:`eq:qxRandomVec` holds for any :math:`P\in\mathbb{N}` and for any :math:`t_1,\ldots,t_P)\in\mathbb{R}^P` then

.. math::
   q(x_{kr}(\cdot))&=\mathcal{GP}\left(\breve\mu_{kr}(\cdot),\breve\kappa_{kr}(\cdot,\cdot)\right)\\
   \breve\mu_{kr}(t)&=\kappa_k(t,z_{kr})(K_{zz}^{kr})^{-1}\mathbf{m}_{kr},\\
   \breve\kappa_{kr}(t,t')&=\kappa_k(t,t')+\kappa_k(t,z_{kr})\left((K_{zz}^{kr})^{-1}S_{kr}(K_{zz}^{kr})^{-1}-(K_{zz}^{kr})^{-1}\right)\kappa_k(z_{kr},t')
   :label: eq:qx

Finally, because affine trasformations of Gaussians are Gaussians,
:math:`h_{nr}(\cdot)` is an affine transformation of :math:`\{x_{kr}(\cdot)\}` (which are
Gaussians, Eq. :eq:`eq:qx`), then the approximate posterior of :math:`h_{nr}(\cdot)`
is the Gaussian process in Eq. :eq:`eq:qh`.

.. math::
   q(h_{nr}(\cdot))&=\mathcal{GP}\left(\tilde\mu_{nr}(\cdot),\tilde\kappa_{nr}(\cdot,\cdot)\right)\\
   \tilde\mu_{nr}(t)&=\sum_{k=1}^Kc_{nk}\breve\mu_{kr}(t)+d_n\\
   \tilde\kappa_{nr}(t,t')&=\sum_{k=1}^Kc_{nk}^2\breve\kappa_{kr}(t,t')
   :label: eq:qh

which is Eq. 5 in :cite:t:`dunckerAndSahani18`.

