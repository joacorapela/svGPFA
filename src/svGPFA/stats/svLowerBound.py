
import pdb
import torch
import warnings

class SVLowerBound:

    def __init__(self, eLL, klDiv):
        super(SVLowerBound, self).__init__()
        self._eLL = eLL
        self._klDiv = klDiv

    def setParamsAndData(self, measurements, initial_params,
                         eLLCalculationParams, priorCovRegParam):
        """Sets model parameters and data.

        :param measurements: ``measurements[r][n]`` are the measurements for trial ``r`` and neuron ``n``.

            For a point-process SVLowerBound (i.e., for a SVLowerBound constructed with an expected log-likelihood of class :class:`svGPFA.stats.expectedLogLikelihood.PointProcessELL`) ``measurements[r][n]`` should be a list of spikes times for trial ``r`` and neuron ``n``.

            For a Poisson SVLowerBound (i.e., for a SVLowerBound constructed with an expected log-likelihood of class :class:`svGPFA.stats.expectedLogLikelihood.PoissonELL`) ``measurements[r][n]`` should be a tuple of length ``n_bins`` containing the spike counts in bins.

        :type  measurements: nested list

        :param initial_params: initial parameters as returned by :func:`svGPFA.utils.initUtils.getParamsAndKernelsTypes`.

        :type  initial_params: dictionary

        :param eLLCalculationParams: parameters used to calculate the expected log likelighood.

            For a point-process SVLowerBound ``eLLCalculationParams`` should be a dictionary with keys ``leg_quad_points`` and ``leg_quad_points``, containing the Legendre quadrature points and weights, respectivey, used to calculate the integral of the expected log likelihood (Eq. 7 in :cite:t:`dunckerAndSahani18`).

            For a Poisson SVLowerBound ``eLLCalculationParams`` should be a dictionary with key ``binTimes`` containing the mean time of every bin.

        :type eLLCalculationParams: dictionary

        :param priorCovRegParam: regularization parameter for the prior covariance matrix (:math:`K_{zz}` in Eq. 2 of :cite:t:`dunckerAndSahani18`).

        :type priorCovRegParam: float
        """
        self.setMeasurements(measurements=measurements)
        self.setInitialParams(initial_params=initial_params)
        self.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)
        self.setPriorCovRegParam(priorCovRegParam=priorCovRegParam)
        self.buildKernelsMatrices()

    def eval(self):
        eLLEval = self._eLL.evalSumAcrossTrialsAndNeurons()
        klDivEval = self._klDiv.evalSumAcrossLatentsAndTrials()
        theEval = eLLEval-klDivEval
        if torch.isinf(theEval):
            # raise RuntimeError("infinity lower bound detected")
            warnings.warn("infinity lower bound detected")
        return theEval

    def sampleCIFs(self, times, nudget=1e-3):
        answer = self._eLL.sampleCIFs(times=times, nudget=nudget)
        return answer

#     def computeCIFsMeans(self, times):
#         answer = self._eLL.computeCIFsMeans(times=times)
#         return answer
# 
    def computeExpectedPosteriorCIFs(self, times):
        answer = self._eLL.computeExpectedPosteriorCIFs(times=times)
        return answer

    def computeEmbeddingMeansAndVarsAtTimes(self, times):
        answer = self._eLL.computeEmbeddingsMeansAndVarsAtTimes(times=times)
        return answer

    def evalELLSumAcrossTrialsAndNeurons(self, svPosteriorOnLatentsStats):
        answer = self._eLL.evalSumAcrossTrialsAndNeurons(
            svPosteriorOnLatentsStats=svPosteriorOnLatentsStats)
        return answer

    def buildKernelsMatrices(self):
        self._eLL.buildKernelsMatrices()

    def computeSVPosteriorOnLatentsStats(self):
        return self._eLL.computeSVPosteriorOnLatentsStats()

    def setInitialParams(self, initial_params):
        self._eLL.setInitialParams(initial_params=initial_params)

    def setKernels(self, kernels):
        self._eLL.setKernels(kernels=kernels)

    def setMeasurements(self, measurements):
        self._eLL.setMeasurements(measurements=measurements)

    def setIndPointsLocs(self, locs):
        self._eLL.setIndPointsLocs(locs=locs)

    def setPriorCovRegParam(self, priorCovRegParam):
        self._eLL.setPriorCovRegParam(priorCovRegParam=priorCovRegParam)

    def setELLCalculationParams(self, eLLCalculationParams):
        self._eLL.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)

    def getSVPosteriorOnIndPointsParams(self):
        return self._eLL.getSVPosteriorOnIndPointsParams()

    def getSVEmbeddingParams(self):
        return self._eLL.getSVEmbeddingParams()

    def getKernels(self):
        return self._eLL.getKernels()

    def getKernelsParams(self):
        return self._eLL.getKernelsParams()

    def getIndPointsLocs(self):
        return self._eLL.getIndPointsLocs()

    def predictLatents(self, times):
        return self._eLL.predictLatents(times=times)

    def predictEmbedding(self, times):
        return self._eLL.predictEmbedding(times=times)


class SVLowerBoundWithParamsGettersAndSetters(SVLowerBound):

    def __init__(self, eLL, klDiv):
        super(SVLowerBoundWithParamsGettersAndSetters, self).__init__(
            eLL, klDiv)
        # shortcuts
        self._svPosteriorOnIndPoints = klDiv.get_svPosteriorOnIndPoints()
        self._svEmbedding = eLL.get_svEmbeddingAllTimes()
        self._indPointsLocsKMS = klDiv.get_indPointsLocsKMS()

    # svPosterioOnIndPoints
    def get_flattened_svPosteriorOnIndPoints_params(self):
        flattened_params = self._svPosteriorOnIndPoints.get_flattened_params()
        return flattened_params

    def get_flattened_svPosteriorOnIndPoints_params_grad(self):
        flattened_params_grad = self._svPosteriorOnIndPoints.get_flattened_params_grad()
        return flattened_params_grad

    def set_svPosteriorOnIndPoints_params_from_flattened(self, flattened_params):
        self._svPosteriorOnIndPoints.set_params_from_flattened(flattened_params=flattened_params)

    def set_svPosteriorOnIndPoints_params_requires_grad(self, requires_grad):
        self._svPosteriorOnIndPoints.set_params_requires_grad(requires_grad=requires_grad)

    # svEmbedding
    def get_flattened_svEmbedding_params(self):
        flattened_params = self._svEmbedding.get_flattened_params()
        return flattened_params

    def get_flattened_svEmbedding_params_grad(self):
        flattened_params_grad = self._svEmbedding.get_flattened_params_grad()
        return flattened_params_grad

    def set_svEmbedding_params_from_flattened(self, flattened_params):
        self._svEmbedding.set_params_from_flattened(flattened_params=flattened_params)

    def set_svEmbedding_params_requires_grad(self, requires_grad):
        self._svEmbedding.set_params_requires_grad(requires_grad=requires_grad)

    # kernels_params
    def get_flattened_kernels_params(self):
        flattened_params = self._indPointsLocsKMS.get_flattened_kernels_params()
        return flattened_params

    def get_flattened_kernels_params_grad(self):
        flattened_params_grad = self._indPointsLocsKMS.get_flattened_kernels_params_grad()
        return flattened_params_grad

    def set_kernels_params_from_flattened(self, flattened_params):
        self._indPointsLocsKMS.set_kernels_params_from_flattened(flattened_params=flattened_params)

    def set_kernels_params_requires_grad(self, requires_grad):
        self._indPointsLocsKMS.set_kernels_params_requires_grad(requires_grad=requires_grad)

    # indPointsLocs
    def get_flattened_indPointsLocs(self):
        flattened_params = self._indPointsLocsKMS.get_flattened_indPointsLocs()
        return flattened_params

    def get_flattened_indPointsLocs_grad(self):
        flattened_params_grad = self._indPointsLocsKMS.get_flattened_indPointsLocs_grad()
        return flattened_params_grad

    def set_indPointsLocs_from_flattened(self, flattened_params):
        self._indPointsLocsKMS.set_indPointsLocs_from_flattened(flattened_params=flattened_params)

    def set_indPointsLocs_requires_grad(self, requires_grad):
        self._indPointsLocsKMS.set_indPointsLocs_requires_grad(requires_grad=requires_grad)

