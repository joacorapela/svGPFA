
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

    def evalELLSumAcrossTrialsAndNeurons(self, posteriorOnLatentsStats):
        answer = self._eLL.evalSumAcrossTrialsAndNeurons(
            posteriorOnLatentsStats=posteriorOnLatentsStats)
        return answer

    def buildKernelsMatrices(self):
        self._eLL.buildKernelsMatrices()

    def buildVariationalCov(self):
        self._eLL.buildVariationalCov()

    def computePosteriorOnLatentsStats(self):
        return self._eLL.computePosteriorOnLatentsStats()

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

    def getVariationalDistParams(self):
        return self._eLL.getVariationalDistParams()

    def getPreIntensityParams(self):
        return self._eLL.getPreIntensityParams()

    def getKernels(self):
        return self._eLL.getKernels()

    def getKernelsParams(self):
        return self._eLL.getKernelsParams()

    def getIndPointsLocs(self):
        return self._eLL.getIndPointsLocs()

    def predictLatents(self, times):
        return self._eLL.predictLatents(times=times)

    def predictPreIntensity(self, times):
        return self._eLL.predictPreIntensity(times=times)


