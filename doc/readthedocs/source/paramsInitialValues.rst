Initial values of model parameters
==================================

Initial values for four types of model parameters need to be specified:

* :ref:`variational_params`,
* :ref:`embedding_params`,
* :ref:`kernels_params`,
* :ref:`indPointsLocs_params`.

For most parameters types initial values can be specified in a binary format or
in a non-binary shorter or longer formats. In the binary format parameters are
given as Pytorch tensors. The shorter format provides the same initial value
for all latents and trials, whereas the longer format allows to specify
different initial values for each latent and trial. If both shorter nad longer
format are specified, the longer format take precedence.

.. _variational_params:

Variational parameters
----------------------

The variational parameters are the means (:math:`\mathbf{m}_k^{(r)}`,
:cite:t:`dunckerAndSahani18`, p.3) and covariances (:math:`S_k^{(r)}`,
:cite:t:`dunckerAndSahani18`, p.3) of the inducing points
(:math:`\mathbf{u}_k^{(r)}`, :cite:t:`dunckerAndSahani18`, p.3).

Binary format
^^^^^^^^^^^^^

Two items need to be specified:

* ``variatiopnal_mean`` should be a list of size ``n_latents``. The kth
  element of this list should be a ``torch.Tensor`` of type double and
  dimension (``n_trials``, ``n_indPoints[k]``, 1), where
  ``variatiopnal_mean[k][r, :, 0]`` gives the initial variational mean for
  latent ``k`` and trial ``r``.

* ``variatiopnal_cov`` should be a list of size ``n_latents``. The kth element
  of this list should be a ``torch.Tensor`` of type double and dimension
  (``n_trials``, ``n_indPoints[k]``, ``n_indPoints[k]``), where
  ``variatiopnal_cov[k][r, :, :]`` gives the initial variational covariance
  for latent ``k`` and trial ``r``.

Longer format
^^^^^^^^^^^^^

For every latent, k, and every trial, r, two items need to be specified:

* ``variational_mean_latent<k>_trial<r>_filename`` should provide the filename
  (csv format readable by pandas *read_csv* function) containing the initial
  values of the variational mean for latents k and trial r. This file should
  contain a vector of size *number_of_inducing_points*.

* ``variational_cov_latent<k>_trial<r>_filename`` should provide the filename (csv
  format readable by pandas *read_csv* function) containing the initial values
  of the variational covariance for all latents and trials. This file should
  contain a matrix of size *number_of_inducing_points* x
  *number_of_inducing_points*.

Example:

    .. code-block:: none
       :caption: example section [variational_params] of the configuration file in the longer format (2 latents, 2 trials)

        [variational_params]
        variational_mean_latent0_trial0_filename = ../data/uniform_0.00_1.00_len09.csv
        variational_cov_latent0_trial0_filename = ../data/identity_scaled1e-2_09x09.csv
        variational_mean_latent0_trial1_filename = ../data/gaussian_0.00_1.00_len09.csv
        variational_cov_latent0_trial1_filename = ../data/identity_scaled1e-4_09x09.csv
        variational_mean_latent1_trial0_filename = ../data/uniform_0.00_1.00_len09.csv
        variational_cov_latent1_trial0_filename = ../data/identity_scaled1e-2_09x09.csv
        variational_mean_latent1_trial1_filename = ../data/gaussian_0.00_1.00_len09.csv
        variational_cov_latent1_trial1_filename = ../data/identity_scaled1e-4_09x09.csv

Shorter format
^^^^^^^^^^^^^^
Two items need to be specified:

* ``variational_means_filename`` should provide the filename (csv format readable
  by pandas *read_csv* function) containing the initial values of the
  variational mean for all latents and trials. This file should contain a
  vector of size *number_of_inducing_points*.

* ``variational_covs_filename`` should provide the filename (csv format readable
  by pandas *read_csv* function) containing the initial values of the
  variational covariance for all latents and trials. This file should contain a
  matrix of size *number_of_inducing_points* x *number_of_inducing_points*.

Example:

    .. code-block:: none
       :caption: example section [variational_params] of the configuration file in the shorter format

       [variational_params]
       variational_means_filename = ../data/uniform_0.00_1.00_len09.csv 
       variational_covs_filename = ../data/identity_scaled1e-2_09x09.csv

.. _embedding_params:

Embedding parameters
----------------------

The embedding parameters are the loading matrix (:math:`C`, :cite:t:`dunckerAndSahani18`, Eq. 1, middle) and offset vector (:math:`\mathbf{d}`, :cite:t:`dunckerAndSahani18`, Eq. 1 middle). Their initial values should be provided in section ``[embedding_params]`` of the ``*.ini`` file.

* ``C_filename`` gives the filename (csv format readable by pandas *read_csv* function) containing the values of loading matrix ``C``,
* ``d_filename`` gives the filename (csv format readable by pandas *read_csv* function) containing the values of offset vector ``d``.

    .. code-block:: none
       :caption: example section [embedding_params] of the configuration file

       [embedding_params]
       C_filename = ../data/C_constant_1.00constant_100neurons_02latents.csv
       d_filename = ../data/d_constant_0.00constant_100neurons.csv

.. _kernels_params:

Kernel parameters
-----------------

The kernel parameters are the parameters of a Gaussian process covariance function (:math:`\kappa_k(\cdot,\cdot)`, Duncker and Sahani, p. 2). Their initial values should be  given in section ``[kernel_params]`` of the ``*.ini`` file.

Longer format
^^^^^^^^^^^^^

For each latent k, section ``[kernel_params]`` should contain item
``k_type_latent<k>`` giving the name of the kernel for latent k.  Other items
required in this section depend on the value of item ``k_type_latent<k>``. For
example, for ``k_type_latent<k>=exponentialQuadratiicKernel``, item
``k_lengthscale_latent<k>`` should specify the lengthscale parameter, and for
``k_type_latent<k>=periodicKernel`` items ``k_lengthscale_latent<k>`` and
``k_period_latent<k>`` should specify the lengthscale and period parameter of
the periodic kernel, respectively.

    .. code-block:: none
       :caption: example section [kernel_params] of the configuration file in the longer format (2 latents)

       [kernels_params]
        k_type_latent0 = exponentialQuadratic
        k_lengthscale_latent0 = 2.0

        k_type_latent1 = exponentialQuadratic
        k_lengthscale_latent1 = 1.0

Shorter format
^^^^^^^^^^^^^^

For all types of kernels section ``[kernel_params]`` should contain
item ``k_types``, giving the name name of the kernel for all latent variables.
Other items required in this section depend on the value of
item ``k_types``. For example, for ``k_types=exponentialQuadratiicKernel``,
item ``k_lengthscales`` should specify the lengthscale parameter, and for
``k_types=periodicKernel`` items ``k_lengthscales`` and ``k_periods`` should
specify the lengthscale and period parameter of the periodic kernel,
respectively.

    .. code-block:: none
       :caption: example section [kernel_params] of the configuration file in the shorter format

       [kernels_params]
       k_types = exponentialQuadratic
       k_lengthscales = 1.0

.. _indPointsLocs_params:

Inducing points locations parameters
------------------------------------

The inducing points locations, or input locations, are the points
(:math:`\mathbf{z}_k^{(r)}`, :cite:t:`dunckerAndSahani18`, p.3) where the Gaussian
process are evaluated to obtain the inducing points. Their initial values are
given in section ``[indPointsLocs_params]`` of the ``*.ini`` file.

Longer format
^^^^^^^^^^^^^

For each latent k and trial r, section ``[indPointsLocs_params]`` should
contain item
``indPointsLocs_latent<k>_trial<r>_filename=indPointsLocs_latentk_trialr.csv``
giving the name of the file (csv format readable by pandas *read_csv* function)
containing the initial inducing points locations for latent k and trial r.

    .. code-block:: none
       :caption: example section [indPointsLocs_params] of the configuration file in the longer format (2 latents, 2 trials)

       [indPointsLocs_params]
       indPointsLocs_latent0_trial0_filename = indPointsLocs_latent0_trial0.csv
       indPointsLocs_latent0_trial1_filename = indPointsLocs_latent0_trial1.csv
       indPointsLocs_latent1_trial0_filename = indPointsLocs_latent1_trial0.csv
       indPointsLocs_latent1_trial1_filename = indPointsLocs_latent1_trial1.csv

Shorter format 1
^^^^^^^^^^^^^^^^

The shorter format 1 requires the specification of the number of inducing points
in the item ``n_ind_points``. The layout of the initial inducing points
locations is given by the item ``ind_points_locs0_layout``. If
``ind_points_locs0_layout = equidistant`` the initial location of the inducing
points is equidistant between the trial start and trial end. If
``ind_points_locs0_layout = uniform`` the inducing points are uniformly
positioned between the start and end of the trial.

    .. code-block:: none
       :caption: example section [indPointsLocs_params] of the configuration file in the shorter format 1

       [indPointsLocs_params]
       n_ind_points = 9
       ind_points_locs0_layout = equidistant

Shorter format 2
^^^^^^^^^^^^^^^^

The shorter format 2 requires the specification, in section
``[indPointsLocs_params],``  of the item
``indPointsLocs_filename=indPointsLocs.csv`` giving the name of the file (csv
format readable by pandas *read_csv* function) containing the initial inducing points
locations for all latents and trials.

    .. code-block:: none
       :caption: example section [indPointsLocs_params] of the configuration file in the shorter format 2

       [indPointsLocs_params]
       indPointsLocs_filename=indPointsLocs.csv

