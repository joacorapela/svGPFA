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

    by specifying a conditional distribution (e.g., point-process, :py:const:`~svGPFA.stats.svGPFAModelFactory.PointProcess`), an embedding type (e.g., linear, :py:const:`~svGPFA.stats.svGPFAModelFactory.LinearEmbedding`), a link function (e.g., :py:func:`~torch.exp`) and providing a set of kernels (:py:class:`~svGPFA.stats.kernels.Kernel`).

2. Estimate the parameters of the model

    .. code-block:: python

       svEM = stats.svGPFA.svEM.SVEM()
       lowerBoundHist = svEM.maximize(model=model, measurements=spikeTimes,
                                      initialParams=initialParams,
                                      quadParams=quadParams,
                                      optimParams=optimParams)

    by providing a set of measurements, ``spikeTimes``, initial parameters, ``initialParams``, quadrature parameters, ``quadParams`` and optimisation parameters, ``optimParams``.

3. Plot estimated model parameters and perform goodness-of-fit tests.

Please refer to the following
`Google Colab notebook
<https://colab.research.google.com/github/joacorapela/svGPFA/blob/master/doc/ipynb/doEstimateAndPlot_collab.ipynb>`_,
`Jupyter notebook
<https://github.com/joacorapela/svGPFA/blob/master/doc/ipynb/doEstimateAndPlot.ipynb>`_, or `Python script <https://github.com/joacorapela/svGPFA/blob/master/examples/scripts/doEstimateSVGPFA.py>`_
for examples of using this interface.
