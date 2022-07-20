
Overview
========

This documentation describes the Python implementation of Sparse Variational
Gaussian Process Factor Analysis (``svGPFA``, :cite:t:`dunckerAndSahani18`)
destributed in this `respository <https://github.com/gatsby-sahani/svGPFA>`_.

``svGPFA`` identifies common latent structure in neural population
spike-trains, which allows for variability both in the trajectory and in the
rate of progression of the underlying computations. It uses shared latent
Gaussian processes, which are combined linearly as in Gaussian Process Factor
Analysis (GPFA, :cite:t:`yuEtAl09`).
``svGPFA`` extends GPFA to
handle unbinned spike-train data by using a continuous time point-process
likelihood model and achieving scalability using a sparse variational
approximation. Variability in the trajectory is decomposed in terms capturing
variability in individual trials, across subset of trials belonging to the same
experimental condition and across all trials. Variability in the timing of a
neural computation is modeled using a nested Gaussian process.
