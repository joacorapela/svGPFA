High-level interface
====================

To estimate a sparse variational Gaussian process factor analysis model we:

1. Construct an empty model

    .. code-block:: python

        model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
            conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
            linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
            embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
            kernels=kernels)

    by specifying a conditional distribution (e.g., point-process, :const:`~stats.svGPFA.svGPFAModelFactory.PointProcess`), an embedding type (e.g., linear, :const:`~stats.svGPFA.svGPFAModelFactory.LinearEmbedding`), a link function (e.g., :func:`~torch.exp`) and providing a set of kernels (:class:`~stats.kernels.Kernel`).

2. Estimate the parameters of the model

    .. code-block:: python

       svEM = stats.svGPFA.svEM.SVEM()
       lowerBoundHist = svEM.maximize(model=model, measurements=spikeTimes,
                                      initialParams=initialParams,
                                      quadParams=quadParams,
                                      optimParams=optimParams)

    by providing a set of measurements, ``spikeTimes``, initial parameters, ``initialParams``, quadrature parameters, ``quadParams`` and optimisation parameters, ``optimParams``.

3. Plot the lower bound history of the estimated model

    .. code-block:: python

       plot.svGPFA.plotUtils.plotLowerBoundHist(lowerBoundHist=lowerBoundHist)

    .. image:: images/77594376_lowerBoundHist.png

and model parameters (e.g., latents).

    .. code-block:: python

       plot.svGPFA.plotUtils.plotTrueAndEstimatedLatents(times=testTimes, muK=testMuK, varK=testVarK, indPointsLocs=indPointsLocs, trueLatents=trueLatentsSamples, trialToPlot=trialToPlot)

   .. image:: images/77594376_trial0_estimatedLatents.png


Please refer to the following `notebook
<https://github.com/joacorapela/svGPFA/blob/master/ipynb/doEstimateAndPlot_jupyter.ipynb>`_
for a full piece of code running svGPFA and plotting its estimates.

