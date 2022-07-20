Optimization parameters
=======================

Parameters values that control the optimization should be specified
in section ``[optim_params]``.

* ``optim_method`` specifies the method used for for parameter optimization. 
  
  If ``optim_method = ECM`` then the Expectation Conditional Maximization
  method is used (:cite:t:`mcLachlanAndKrishnan08`, section 5.2).  Here the
  M-step is broken into three conditional maximization steps: maximization of
  the lower bound wrt the embedding parameters (mstep-embedding), wrt the
  kernels parameters (mstep-kernels) and wrt the inducing points locations
  (mstep-indPointsLocs). Thus, one ECM iteration comprises one E-step (i.e.,
  maximiziation of the lower bound wrt the embedding parameters) followed by
  the three previous M-step conditional maximizations.

  If ``optim_method = mECM`` then the Multicycle ECM is used
  (:cite:t:`mcLachlanAndKrishnan08`, section 5.3). Here
  one E-step maximization is performed before each of the M-step conditional
  maximizations. Thus, one mECM iteration comprises estep, mstep-embedding,
  estep,  mstep-kernels, estep, mstep-indPointsLocs.

* ``em_max_iter`` boolean value specifying the maximum number of EM iterations.

* ``verbose`` boolean value indicating whether the optimization should be
  verbose or silent.

For each ``<step> in {estep,mstep_embedding,mstep_kernels,mstep_indPointsLocs}``
section ``[optim_params]`` should contain items:

* ``<step>_estimate`` boolean value indicating whether ``<step>`` should be
  estimated or not.

* ``<step>_max_iter`` integer value indicating the maximum number of iterations
  used by ``torch.optim.LBFGS`` for the optimization of the ``<step>`` within
  one EM iteration.

* ``<step>_lr`` float value indicating the learning rate used by
  ``torch.optim.LBFGS`` for the optimization of the ``<step>`` within one EM
  iteration.
  
* ``<step>_tolerance_grad`` float value indicating the termination tolerance on
  first-order optimality used by ``torch.optim.LBFGS`` for the optimization of
  the ``<step>`` within one EM iteration.
  
* ``<step>_tolerance_change`` float value indicating the termination tolerance
  on function value per parameter changes used by ``torch.optim.LBFGS`` for the
  optimization of the ``<step>`` within one EM iteration.
  
* ``<step>_line_search_fn`` string value indicating the line search method used
  by ``torch.optim.LBFGS``. If ``<step>_line_search_fn=strong_wolfe`` line
  search is performed using the strong_wolfe method. If
  `<step>_line_search_fn=None`` line search is not used.

    .. code-block:: none
       :caption: example section [optim_params] of the configuration file

        [optim_params]
        n_quad = 200
        prior_cov_reg_param = 1e-5
        #
        optim_method = ECM
        em_max_iter = 200
        #
        estep_estimate = True
        estep_max_iter = 20
        estep_lr = 1.0
        estep_tolerance_grad = 1e-7
        estep_tolerance_change = 1e-9
        estep_line_search_fn = strong_wolfe
        #
        mstep_embedding_estimate = True
        mstep_embedding_max_iter = 20
        mstep_embedding_lr = 1.0
        mstep_embedding_tolerance_grad = 1e-7
        mstep_embedding_tolerance_change = 1e-9
        mstep_embedding_line_search_fn = strong_wolfe
        #
        mstep_kernels_estimate = True
        mstep_kernels_max_iter = 20
        mstep_kernels_lr = 1.0
        mstep_kernels_tolerance_grad = 1e-7
        mstep_kernels_tolerance_change = 1e-9
        mstep_kernels_line_search_fn = strong_wolfe
        #
        mstep_indpointslocs_estimate = True
        mstep_indpointslocs_max_iter = 20
        mstep_indpointslocs_lr = 1.0
        mstep_indpointslocs_tolerance_grad = 1e-7
        mstep_indpointslocs_tolerance_change = 1e-9
        mstep_indpointslocs_line_search_fn = strong_wolfe
        #
        verbose = True
        
