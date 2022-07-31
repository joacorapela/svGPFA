Model structure parameters
==========================

The only model parameter is ``n_latents`` that should contain the number of latents variables in the svGPFA model; i.e., :math:`K` in Eq. 1 in :cite:t:`dunckerAndSahani18`.

Example:

    .. code-block:: none
       :caption: example section [model_structure_params] of the configuration file

        [model_structure_params]
        n_latents = 7

