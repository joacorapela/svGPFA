High-level interface
====================

Characterizing neural populations with svGPFA models involves the following steps:

1. Building an empty model

    .. code-block:: python

        model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
            conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
            linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
            embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
            kernels=kernels)

    by specifying a conditional distribution (e.g., point-process, :py:const:`~svGPFA.stats.svGPFAModelFactory.PointProcess`), an embedding type (e.g., linear, :py:const:`~svGPFA.stats.svGPFAModelFactory.LinearEmbedding`), a link function (e.g., :py:func:`~torch.exp`) and providing a set of kernels (:py:class:`~svGPFA.stats.kernels.Kernel`).

2. Setting initial parameters

    .. code-block:: python

       model.setParamsAndData(
           measurements=spikes_times,
           initial_params=params["initial_params"],
           eLLCalculationParams=params["ell_calculation_params"],
           priorCovRegParam=params["optim_params"]["prior_cov_reg_param"])

3. Estimating parameters

    .. code-block:: python

       svEM = stats.svGPFA.svEM.SVEM()
       lowerBoundHist = svEM.maximize(model=model, measurements=spikeTimes,
                                      initialParams=initialParams,
                                      quadParams=quadParams,
                                      optimParams=optimParams)

    by providing a set of measurements, ``spikeTimes``, initial parameters, ``initialParams``, quadrature parameters, ``quadParams`` and optimisation parameters, ``optimParams``.

4. Assessing goodness-of-fit
  
    .. code-block:: python

       diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = \
           gcnu_common.stats.pointProcesses.tests.\
               KSTestTimeRescalingNumericalCorrection(spikes_times=spikes_times_GOF,
                                                      cif_times=trial_times_GOF,
                                                      cif_values=cif_values_GOF,
                                                      gamma=ksTestGamma)
       fig = svGPFA.plot.plotUtilsPlotly.\
           getPlotResKSTestTimeRescalingNumericalCorrection(
               diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx,
               estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb,
               title=title)
       fig.show()

    .. image:: images/25994470_ksTestTimeRescaling_numericalCorrection_trial000_neuron000.png
       :alt: time rescaling KS test

5. Plotting model's parameters

    .. code-block:: python

       fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(times=trials_times.numpy(), latentsMeans=testMuK, latentsSTDs=torch.sqrt(testVarK), indPointsLocs=indPointsLocs, latentToPlot=latentToPlot, trials_colors=trials_colors, xlabel="Time (msec)")
       fig.show()

    .. image:: images/latent0AcrossTrials.png
       :alt: Latent 0 across all trials

