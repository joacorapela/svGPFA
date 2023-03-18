
.. _params_specs:

Parameters and their specification
##################################

svGPFA uses different groups of parameters. We provide a utility function
:func:`svGPFA.utils.initUtils.getParamsAndKernelsTypes` that builds them from
parameter specifications. These specifications are short descriptions on how
to build a parameter. For example, a parameter specification for the inducing
points locations can be ``equidistant``, indicating that the inducing points
locations should be set to equidistant values between the start and end of a
trial.

A parameter specification is a nested list (e.g.,
``param_spec[group_name][param_name]``) containing the specification of
parameter ``param_name`` in group ``group_name``. It can be built:

    1. automatically, using default values, with the utility function :func:`svGPFA.utils.initUtils.getDefaultParamsDict`,
           
    2. manually, by setting parameter specifications in the Python code,

    3. from the command line, with the utility function :func:`svGPFA.utils.initUtils.getParamsDictFromArgs`,

    4. from a configuration file, with the utility function :func:`svGPFA.utils.initUtils.getParamsDictFromStringsDict`.
          
    The `Colab <https://colab.research.google.com/github/joacorapela/svGPFA/blob/master/doc/ipynb/doEstimateAndPlot_collab.ipynb>`__ and `Jupyter <https://github.com/joacorapela/svGPFA/blob/master/examples/scripts/doEstimateSVGPFA.py>`__ notebooks automatically build this list (1). This `script <https://github.com/joacorapela/svGPFA/blob/master/examples/scripts/doEstimateSVGPFA_manualParamSpec.py>`__ demonstrates the manual construction of all parameters specifications (2). This `script <https://github.com/joacorapela/svGPFA/blob/master/examples/scripts/doEstimateSVGPFA.py>`__ builds the parameters specification list from the command line and from this `configuration file <https://github.com/joacorapela/svGPFA/blob/master/examples/params/00000545_estimation_metaData.ini>`__ (3, 4).

Below we describe all svGPFA parameters and their specifications. Refer to the
documentation of the above utility functions for details on how to use them.

.. _data_structure_params:

Data structure parameters
=========================

There are two data structure parameters ``trials_start_times`` and
``trials_end_times``, which are tensors of length ``n_trials`` giving the start
and end times of each trial; i.e., :math:`\tau_i` in Eq 7 in
:cite:t:`dunckerAndSahani18`.

These parameters can be specified in a trial-specific or trial-common format. If both are
specified, the longer format takes precedence.

Trial-specific format
---------------------

Two items need to be specified:

* ``trials_start_times`` should provide a list of length ``n_trials``, with float values indicating seconds, such that ``trials_start_times[i]`` gives the start time of the ith trial.

* ``trials_end_times`` should provide a list of length ``n_trials``, with float values indicating seconds, such that ``trials_end_times[i]`` gives the end time of the ith trial.

    .. code-block:: python
       :caption: adding ``data_structure_params`` in the trial-specific Python variable format to ``params_spec`` (3 trials)

        params_spec["data_structure_params"] = {
            "trials_start_times": [0.0, 0.4, 0.7],
            "trials_end_times":   [0.2, 0.5, 0.9],
        }

Trial-common format
-------------------

Two items need to be specified:

* ``trials_start_time`` should provide the start time (float value, secs) of all trials.

* ``trials_end_time`` should provide the end time (float value, secs) of all trials.

    .. code-block:: python
       :caption: adding ``data_structure_params`` in the trial-common Python variable format to ``params_spec``

        params_spec["data_structure_params"] = {
            "trials_start_time": 0.0,
            "trials_end_time":   1.0,
        }


Defaults
--------

All trials start at 0.0 sec and end at 1.0 sec.

    .. code-block:: python
       :caption: default ``data_structure_params`` 

        params_spec["data_structure_params"] = {
            "trials_start_time": 0.0,
            "trials_end_time":   1.0,
        }

.. _initial_value_params:

Initial values of model parameters
==================================

Initial values for four types of model parameters need to be specified:

* :ref:`variational_params0`,
* :ref:`embedding_params0`,
* :ref:`kernels_params0`,
* :ref:`ind_points_locs_params0`.

For most parameters types initial values can be specified in a binary format or
in a non-binary shorter or longer formats. In the binary format parameters are
given as PyTorch tensors. The shorter format provides the same initial value
for all latents and trials, whereas the longer format gives
different initial values for each latent and trial. If both shorter and longer
format are specified, the longer format take precedence.

.. _variational_params0:

Variational parameters
----------------------

The variational parameters are the means (:math:`\mathbf{m}_k^{(r)}`,
:cite:t:`dunckerAndSahani18`, p.3) and covariances (:math:`S_k^{(r)}`,
:cite:t:`dunckerAndSahani18`, p.3) of the inducing points
(:math:`\mathbf{u}_k^{(r)}`, :cite:t:`dunckerAndSahani18`, p.3). The data
structures for these parameters are described in the next section.

Python variable format
^^^^^^^^^^^^^^^^^^^^^^

Two items need to be specified:

* ``variational_mean0`` should be a list of size ``n_latents``. The kth
  element of this list should be a ``torch.DoubleTensor`` of
  dimension (``n_trials``, ``n_indPoints[k]``, 1), where
  ``variational_mean0[k][r, :, 0]`` gives the initial variational mean for
  latent ``k`` and trial ``r``.

* ``variational_cov0`` should be a list of size ``n_latents``. The kth element
  of this list should be a ``torch.DoubleTensor`` of dimension
  (``n_trials``, ``n_indPoints[k]``, ``n_indPoints[k]``), where
  ``variational_cov0[k][r, :, :]`` gives the initial variational covariance
  for latent ``k`` and trial ``r``.

    .. code-block:: python
       :caption: adding random ``variational_params0`` in the Python variable format to ``params_spec``

        n_latents = 3
        n_ind_points = [20, 10, 15]
        var_mean0 = [torch.normal(mean=0, std=1, size=(n_trials, n_ind_points[k], 1), dtype=torch.double) for k in range(n_latents)]
        diag_value = 1e-2 
        var_cov0 = [[] for r in range(n_latents)]
        for k in range(n_latents):
            var_cov0[k] = torch.empty((n_trials, n_ind_points[k], n_ind_points[k]), dtype=torch.double)
            for r in range(n_trials):
                var_cov0[k][r, :, :] = torch.eye(n_ind_points[k], dtype=torch.double)*diag_value
        params_spec["variational_params0"] = {
            "variational_mean0": var_mean0,
            "variational_cov0":  var_cov0,
        }

Latent-trial-specific filename format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For every latent, k, and every trial, r, two items need to be specified:

* ``variational_mean0_filename_latent<k>_trial<r>`` should provide the filename
  (csv format readable by pandas *read_csv* function) containing the initial
  values of the variational mean for latent k and trial r. This file should
  contain a vector of size *number_of_inducing_points*.

* ``variational_cov0_filename_latent<k>_trial<r>`` should provide the filename
  (csv format readable by pandas *read_csv* function) containing the initial
  values of the variational covariance for latent k and trial r. This file
  should contain a matrix of size *number_of_inducing_points* x
  *number_of_inducing_points*.

    .. code-block:: python
       :caption: adding ``variational_params0`` in the latent-trial-specific filename format to ``params_spec`` (2 trials and 2 latents)

        params_spec["variational_params0"] = {
            "variational_mean0_filename_latent0_trial0": "../data/uniform_0.00_1.00_len09.csv",
            "variational_cov0_filename_latent0_trial0": "../data/identity_scaled1e-2_09x09.csv",
            "variational_mean0_filename_latent0_trial1": "../data/gaussian_0.00_1.00_len09.csv",
            "variational_cov0_filename_latent0_trial1": "../data/identity_scaled1e-4_09x09.csv",
            "variational_mean0_filename_latent1_trial0": "../data/uniform_0.00_1.00_len09.csv",
            "variational_cov0_filename_latent1_trial0": "../data/identity_scaled1e-2_09x09.csv",
            "variational_mean0_filename_latent1_trial1": "../data/gaussian_0.00_1.00_len09.csv",
            "variational_cov0_filename_latent1_trial1": "../data/identity_scaled1e-4_09x09.csv",
        }

Latent-trial-common filename format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two items need to be specified:

* ``variational_means0_filename`` should provide the filename (csv format readable
  by pandas *read_csv* function) containing the initial values of the
  variational mean for all latents and trials. This file should contain a
  vector of size *number_of_inducing_points*.

* ``variational_covs0_filename`` should provide the filename (csv format readable
  by pandas *read_csv* function) containing the initial values of the
  variational covariance for all latents and trials. This file should contain a
  matrix of size *number_of_inducing_points* x *number_of_inducing_points*.

    .. code-block:: python
       :caption: adding ``variational_params0`` in the latent-trial-common filename format to ``params_spec``

        params_spec["variational_params0"] = {
            "variational_means0_filename": "../data/uniform_0.00_1.00_len09.csv",
            "variational_covs0_filename": "../data/identity_scaled1e-2_09x09.csv",
        }

Constant value format
^^^^^^^^^^^^^^^^^^^^^

This initialisation option sets the same variational mean and covariance across all latents and trials. The common variational mean has all elements equal to a constant value, and the common variational covariance is a scaled identity matrix.

Two items need to be specified:

* ``variational_mean0_constant_value`` should provide a float value giving the constant value of all elements of the common variational mean.
* ``variational_cov0_diag_value``  should provide a float value giving the diagonal value of the common variational covariance.

    .. code-block:: python
       :caption: adding ``variational_params0`` in the constant value format

        params_spec["variational_params0"] = {
            "variational_mean0_constant_value": 0.0,
            "variational_cov0_diag_value": 0.01,
        }

Defaults
^^^^^^^^

The default variational mean and covariance have constant values. For the variational mean the constant value is zero and for the variational covariance the constant diagonal value is 0.01.

    .. code-block:: python
       :caption: default ``variational_params0``

        params_spec["variational_params0"] = {
            "variational_mean0_constant_value": 0.0,
            "variational_cov0_diag_value": 0.01,
        }

.. _embedding_params0:

Embedding parameters
--------------------

The embedding parameters are the loading matrix (:math:`C`, :cite:t:`dunckerAndSahani18`, Eq. 1, middle) and offset vector (:math:`\mathbf{d}`, :cite:t:`dunckerAndSahani18`, Eq. 1 middle). The data structures for these parameters are described in the next section.

Python variable format
^^^^^^^^^^^^^^^^^^^^^^
Two items need to be specified:

* ``c0`` should be a ``torch.DoubleTensor`` of size (n_neurons, n_latents)

* ``d0`` should be a ``torch.DoubleTensor`` of size (n_neurons, 1)

    .. code-block:: python
       :caption: adding standard random ``embedding_params0`` in the Python variable format to ``params_spec``

        n_neurons = 100
        n_latents = 3

        params_spec["embedding_params0"] = {
            "c0": torch.normal(mean=0.0, std=1.0, size=(n_neurons, n_latents), dtype=torch.double),
            "d0":  torch.normal(mean=0.0, std=1.0, size=(n_neurons, 1), dtype=torch.double),
        }

Filename format
^^^^^^^^^^^^^^^

Two items need to be specified:

* ``c0_filename`` gives the filename (csv format readable by pandas *read_csv* function) containing the values of loading matrix ``C``,

* ``d0_filename`` gives the filename (csv format readable by pandas *read_csv* function) containing the values of offset vector ``d``.

    .. code-block:: python
        :caption: adding ``embedding_params0`` in the filename format to ``params_spec``

        params_spec["embedding_params0"] = {
            "c0_filename": "../data/C_constant_1.00constant_100neurons_02latents.csv",
            "d0_filename": "../data/d_constant_0.00constant_100neurons.csv",
        }

Random format
^^^^^^^^^^^^^

Eight items need to be specified:

* ``c0_distribution`` string value giving the name of the distribution of the loading matrix C (e.g., Normal).

* ``c0_loc`` float number giving the location of the distribution of the loading matrix C (e.g., 0.0).

* ``c0_scale`` float value giving the scale of the distribution of the loading matrix C (e.g., 1.0).

* ``c0_random_seed`` optional integer value giving the value of the random seed to be set prior to generating the random transition matrix ``C``. This value can be specified for replicability. If not given, the random seed is not changed prior to generating ``C``.

* ``d0_distribution`` string value giving the name of the distribution of the offset vector ``d`` (e.g., Normal).

* ``d0_loc`` float number giving the location of the distribution of the offset vector ``d`` (e.g., 0.3).

* ``d0_scale`` float value giving the scale of the distribution of the offset vector ``d`` (e.g., 1.0).

* ``d0_random_seed`` optional integer value giving the value of the random seed to be set prior to generating the random transition matrix ``d``. This value can be specified for replicability. If not given, the random seed is not changed prior to generating ``d``.

    .. code-block:: python
       :caption: adding ``embedding_params0`` in the random format to ``params_spec``

        params_spec["embedding_params0"] = {
            "c0_distribution": "Normal",
            "c0_loc": 0.0,
            "c0_scale": 1.0,
            "c0_random_seed": 102030,
            "d0_distribution": "Normal",
            "d0_loc": 0.0,
            "d0_scale": 1.0,
            "d0_random_seed": 203040,
        }

Defaults
^^^^^^^^
The default loading matrix C0/offset vector d0 is a zero mean standard normal random  matrix/vector.

    .. code-block:: python
       :caption: default ``embedding_params0``

        params_spec["embedding_params0"] = {
            "c0_distribution": "Normal",
            "c0_loc": 0.0,
            "c0_scale": 1.0,
            "d0_distribution": "Normal",
            "d0_loc": 0.0,
            "d0_scale": 1.0,
        }

.. _kernels_params0:

Kernel parameters
-----------------

The kernel parameters of latent k are those of the Gaussian process covariance
function (:math:`\kappa_k(\cdot,\cdot)`, :cite:t:`dunckerAndSahani18`, p. 2). The data
structures for these parameters are described in the next section.

Python variable format
^^^^^^^^^^^^^^^^^^^^^^

Two items need to be specified:

* ``k_types`` should be a list of size ``n_latents``. The kth element of this list should be a string with the type of kernel for the kth latent (e.g., ``k_types[k]=exponentialQuadratic``).

* ``k_params0`` should be a list of size ``n_latents``. The kth element of this list should be a ``torch.DoubleTensor`` containing the parameters of the kth kernel (e.g., ``k_params0[k]=torch.DoubleTensor([3.2])``).

    .. code-block:: python
       :caption: adding ``kernel_params`` in Python variable format (2 latents) to ``params_spec``

       expQuadK1_lengthscale = 2.9
       expQuadK2_lengthscale = 0.5
       periodK1_lengthscale = 3.1
       periodK1_period = 1.2
       params_spec["kernels_params0"] = {
            "k_types": ["exponentialQuadratic", "exponentialQuadratic", "periodic"],
            "k_params0": [torch.DoubleTensor([expQuadK1_lengthscale]),
                          torch.DoubleTensor([expQuadK2_lengthscale]),
                          torch.DoubleTensor([periodK1_lengthscale, periodK1_lengthscale]),
                         ],
       }

Latent-specific textual format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each latent k, item ``k_type_latent<k>`` needs to be specified, giving the
name of the kernel for latent k. Other items required depend on
the value of item ``k_type_latent<k>``. For example, for
``k_type_latent<k>=exponentialQuadratic``, item
``k_lengthscale0_latent<k>`` should specify the lengthscale parameter, and for
``k_type_latent<k>=periodic`` items ``k_lengthscale0_latent<k>`` and
``k_period0_latent<k>`` should specify the lengthscale and period parameter of
the periodic kernel, respectively.

    .. code-block:: python
       :caption: adding ``kernel_params`` in the latent-specific textual format (2 latents) to ``params_spec``

       params_spec["kernels_params0"] = {
            "k_type_latent0": "exponentialQuadratic",
            "k_lengthscale0_latent0": 2.0,
            "k_type_latent1": "periodic",
            "k_lengthscale0_latent1": 1.0,
            "k_period0_latent1": 0.75,
       }

Latent-common textual format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The shorter format requires
item ``k_types``, giving the name name of the kernel to be used for all latent variables.
Other required items depend on the value of
item ``k_types``. For example, for ``k_types=exponentialQuadratic``,
item ``k_lengthscales0`` should specify the lengthscale parameter, and for
``k_types=periodic`` items ``k_lengthscales0`` and ``k_periods0`` should
specify the lengthscale and period parameter of the periodic kernel,
respectively.

    .. code-block:: python
       :caption: adding ``kernel_params`` in the latent-common textual format to ``params_spec``

       params_spec["kernels_params0"] = {
           "k_types": "exponentialQuadratic",
           "k_lengthscales0": 1.0,
       }

Defaults
^^^^^^^^
For all latents, the default kernel is an exponential quadratic kernel with lengthscale 1.0.

    .. code-block:: python
       :caption: default ``kernel_params``

        params_spec["kernels_params0"] =  {
            "k_types": "exponentialQuadratic",
            "k_lengthscales0": 1.0,
        }

.. _ind_points_locs_params0:

Inducing points locations parameters
------------------------------------

The inducing points locations, or input locations, are the points
(:math:`\mathbf{z}_k^{(r)}`, :cite:t:`dunckerAndSahani18`, p.3) where the
Gaussian process are evaluated to obtain the inducing points. The data
structures for these parameters are described in the next section.

Python variable format
^^^^^^^^^^^^^^^^^^^^^^

One item needs to be specified:

* ``ind_points_locs0`` should be a list of size ``n_latents``. The kth element of
  this list should be a ``torch.DoubleTensor`` of size (``n_trials``,
  ``n_indPoints[k]``, 1), where ``ind_points_locs0[k][r, :, 0]`` gives the
  initial inducing points locations for latent k and trial r.

    .. code-block:: python
       :caption: adding ``ind_points_locs_params0`` in Python variable format with uniformly distributed inducing points locations to ``params_spec``

       n_latents = 3
       n_ind_points = (10, 20, 15)
       n_trials = 50
       trials_start_time = 0.0
       trials_end_time = 7.0
       params_spec["ind_points_locs_params0"] = {
            "ind_points_locs0": [trials_start_time + (trials_end_time-trials_start_time) * torch.rand(n_trials, n_ind_points[k], 1, dtype=torch.double) for k in range(n_latents)]
       }

Latent-trial-specific filename format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each latent k and trial r one item needs to be specified:

* ``ind_points_locs0_latent<k>_trial<r>_filename`` giving the name of the file
  (csv format readable by pandas *read_csv* function) containing the initial
  inducing points locations for latent k and trial r.

    .. code-block:: python
       :caption: adding ``ind_points_locs_params0`` in the latent-trial-specific filename format to ``params_spec`` (2 latents, 2 trials)

       params_spec["ind_points_locs_params0"] = {
           "ind_points_locs0_latent0_trial0_filename": "ind_points_locs0_latent0_trial0.csv",
           "ind_points_locs0_latent0_trial1_filename": "ind_points_locs0_latent0_trial1.csv",
           "ind_points_locs0_latent1_trial0_filename": "ind_points_locs0_latent1_trial0.csv",
           "ind_points_locs0_latent1_trial1_filename": "ind_points_locs0_latent1_trial1.csv",
       }

Latent-trial-common filename format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This shorter format requires the specification of the item
``ind_points_locs0_filename`` giving the name of the file (csv format readable by
pandas *read_csv* function) containing the initial inducing points locations
for all latents and trials.

    .. code-block:: python
       :caption: adding ``ind_points_locs_params0`` in the latent-trial-common filename format to ``params_spec``

       params_spec["ind_points_locs_params0"] = {
           "ind_points_locs0_filename": "ind_points_locs0.csv",
       }

Layout format
^^^^^^^^^^^^^

The layout format requires the specification of the number of inducing points
in the item ``n_ind_points`` or in the item ``common_n_ind_points``. Item
``n_ind_points`` is a list of length ``n_trials``, such that
``n_ind_points[r]`` gives the number of inducing points in trial ``r``. Item
``common_n_ind_points`` is an integer, that gives the number of inducing points
of all trials.

The layout of the initial inducing points locations is given by the item
``ind_points_locs0_layout``. If ``ind_points_locs0_layout = equidistant`` the
initial locations of the inducing points are equidistant between the trial
start and trial end. If ``ind_points_locs0_layout = uniform`` the initial
inducing points are distributed randomly between the start and end of the
trial.

    .. code-block:: python
       :caption: adding ``ind_points_locs_params0`` in the layout format to ``params_spec``

       n_ind_points = (10, 20, 15)
       params_spec["ind_points_locs_params0"] = {
           "n_ind_points": n_ind_points,
           "ind_points_locs0_layout": "equidistant",
       }

Defaults
^^^^^^^^

The default inducing points locations for trial r and latent k are equidistant n_ind_points[k] between the start and end of trial r.

    .. code-block:: python
       :caption: default ``ind_points_locs_params0``

       n_ind_points = (10, 10, 10)
        params_spec["ind_points_locs_params0"] = {
            "n_ind_points": n_ind_points,
            "ind_points_locs0_layout": "equidistant",
        }

.. _optim_params:

Optimisation parameters
=======================

Parameters values that control the optimisation should be specified
in section ``[optim_params]``.

* ``optim_method`` specifies the method used for for parameter optimisation. 
  
  If ``optim_method = ECM`` then the Expectation Conditional Maximisation
  method is used (:cite:t:`mcLachlanAndKrishnan08`, section 5.2).  Here the
  M-step is broken into three conditional maximisation steps: maximisation of
  the lower bound wrt the embedding parameters (mstep-embedding), wrt the
  kernels parameters (mstep-kernels) and wrt the inducing points locations
  (mstep-indPointsLocs). Thus, one ECM iteration comprises one E-step (i.e.,
  maximisation of the lower bound wrt the embedding parameters) followed by
  the three previous M-step conditional maximisation's.

  If ``optim_method = mECM`` then the Multicycle ECM is used
  (:cite:t:`mcLachlanAndKrishnan08`, section 5.3). Here
  one E-step maximisation is performed before each of the M-step conditional
  maximisation's. Thus, one mECM iteration comprises estep, mstep-embedding,
  estep,  mstep-kernels, estep, mstep-indPointsLocs.

* ``em_max_iter`` integer value specifying the maximum number of EM iterations.

* ``verbose`` boolean value indicating whether the optimisation should be
  verbose or silent.

For each ``<step> in {estep,mstep_embedding,mstep_kernels,mstep_indPointsLocs}``
section ``[optim_params]`` should contain items:

* ``<step>_estimate`` boolean value indicating whether ``<step>`` should be
  estimated or not.

* ``<step>_max_iter`` integer value indicating the maximum number of iterations
  used by ``torch.optim.LBFGS`` for the optimisation of the ``<step>`` within
  one EM iteration.

* ``<step>_lr`` float value indicating the learning rate used by
  ``torch.optim.LBFGS`` for the optimisation of the ``<step>`` within one EM
  iteration.
  
* ``<step>_tolerance_grad`` float value indicating the termination tolerance on
  first-order optimality used by ``torch.optim.LBFGS`` for the optimisation of
  the ``<step>`` within one EM iteration.
  
* ``<step>_tolerance_change`` float value indicating the termination tolerance
  on function value per parameter changes used by ``torch.optim.LBFGS`` for the
  optimisation of the ``<step>`` within one EM iteration.
  
* ``<step>_line_search_fn`` string value indicating the line search method used
  by ``torch.optim.LBFGS``. If ``<step>_line_search_fn=strong_wolfe`` line
  search is performed using the strong_wolfe method. If
  `<step>_line_search_fn=None`` line search is not used.

    .. code-block:: python
       :caption: adding ``optimisation_params`` to ``params_spec``

        params_spec["optim_params"] = {
            "n_quad": 200,
            "prior_cov_reg_param": 1e-5,
            #
            "optim_method": "ECM",
            "em_max_iter": 200,
            #
            "estep_estimate": True,
            "estep_max_iter": 20,
            "estep_lr": 1.0,
            "estep_tolerance_grad": 1e-7,
            "estep_tolerance_change": 1e-9,
            "estep_line_search_fn": "strong_wolfe",
            #
            "mstep_embedding_estimate": True,
            "mstep_embedding_max_iter": 20,
            "mstep_embedding_lr": 1.0,
            "mstep_embedding_tolerance_grad": 1e-7,
            "mstep_embedding_tolerance_change": 1e-9,
            "mstep_embedding_line_search_fn": "strong_wolfe",
            #
            "mstep_kernels_estimate": True,
            "mstep_kernels_max_iter": 20,
            "mstep_kernels_lr": 1.0,
            "mstep_kernels_tolerance_grad": 1e-7,
            "mstep_kernels_tolerance_change": 1e-9,
            "mstep_kernels_line_search_fn": "strong_wolfe",
            #
            "mstep_indpointslocs_estimate": True,
            "mstep_indpointslocs_max_iter": 20,
            "mstep_indpointslocs_lr": 1.0,
            "mstep_indpointslocs_tolerance_grad": 1e-7,
            "mstep_indpointslocs_tolerance_change": 1e-9,
            "mstep_indpointslocs_line_search_fn": "strong_wolfe",
            #
            "verbose": True,
       }


Defaults
--------

The default optimisation parameters are shown below.

    .. code-block:: python
       :caption: default ``optimisation_params``

        n_quad = 200
        prior_cov_reg_param = 1e-3
        em_max_iter = 50

        params_spec["optim_params"] = {
            "n_quad": n_quad,
            "prior_cov_reg_param": prior_cov_reg_param,
            "optim_method": "ecm",
            "em_max_iter": em_max_iter,
            "verbose": True,
            #
            "estep_estimate": True,
            "estep_max_iter": 20,
            "estep_lr": 1.0,
            "estep_tolerance_grad": 1e-7,
            "estep_tolerance_change": 1e-9,
            "estep_line_search_fn": "strong_wolfe",
            #
            "mstep_embedding_estimate": True,
            "mstep_embedding_max_iter": 20,
            "mstep_embedding_lr": 1.0,
            "mstep_embedding_tolerance_grad": 1e-7,
            "mstep_embedding_tolerance_change": 1e-9,
            "mstep_embedding_line_search_fn": "strong_wolfe",
            #
            "mstep_kernels_estimate": True,
            "mstep_kernels_max_iter": 20,
            "mstep_kernels_lr": 1.0,
            "mstep_kernels_tolerance_grad": 1e-7,
            "mstep_kernels_tolerance_change": 1e-9,
            "mstep_kernels_line_search_fn": "strong_wolfe",
            #
            "mstep_indpointslocs_estimate": True,
            "mstep_indpointslocs_max_iter": 20,
            "mstep_indpointslocs_lr": 1.0,
            "mstep_indpointslocs_tolerance_grad": 1e-7,
            "mstep_indpointslocs_tolerance_change": 1e-9,
            "mstep_indpointslocs_line_search_fn": "strong_wolfe"
        }

