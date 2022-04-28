
Overview
========

This documentation describes the Python implementation of Sparse Variational
Gaussian Process Factor Analysis (``svGPFA``, `Duncker and Sahani, 2018 <https://papers.nips.cc/paper/8245-temporal-alignment-and-latent-gaussian-process-factor-inference-in-population-spike-trains>`_) in `https://github.com/gatsby-sahani/svGPFA <https://github.com/gatsby-sahani/svGPFA>`_.

``svGPFA`` identifies common latent structure in neural population
spike-trains, which allows for variability both in the trajectory and in the
rate of progression of the underlying computations. It uses shared latent
Gaussian processes, which are combined linearly as in Gaussian Process Factor
Analysis (GPFA, `Yu et al., 2009
<https://www.ncbi.nlm.nih.gov/pubmed/19357332>`_).  ``svGPFA`` extends GPFA to
handle unbinned spike-train data by using a continuous time point-process
likelihood model and achieving scalability using a sparse variational
approximation. Variability in the trajectory is decomposed in terms capturing
variability in individual trials, across subset of trials belonging to the same
experimental condition and across all trials. Variability in the timing of a
neural computation is modeled using a nested Gaussian process.
