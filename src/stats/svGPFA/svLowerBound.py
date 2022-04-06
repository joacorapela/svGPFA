
import pdb
import torch
import warnings

class SVLowerBound:

    def __init__(self, eLL, klDiv):
        super(SVLowerBound, self).__init__()
        self._eLL = eLL
        self._klDiv = klDiv

    def setInitialParamsAndData(self, measurements, initialParams,
                                eLLCalculationParams, indPointsLocsKMSRegEpsilon):
        self.setMeasurements(measurements=measurements)
        self.setInitialParams(initialParams=initialParams)
        self.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)
        self.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)
        self.buildKernelsMatrices()

    def eval(self):
        eLLEval = self._eLL.evalSumAcrossTrialsAndNeurons()
        klDivEval = self._klDiv.evalSumAcrossLatentsAndTrials()
        theEval = eLLEval-klDivEval
#         if torch.abs(theEval)>1e30:
#             import pdb; pdb.set_trace()
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

    def setInitialParams(self, initialParams):
        self._eLL.setInitialParams(initialParams=initialParams)

    def setKernels(self, kernels):
        self._eLL.setKernels(kernels=kernels)

    def setMeasurements(self, measurements):
        self._eLL.setMeasurements(measurements=measurements)

    def setIndPointsLocs(self, locs):
        self._eLL.setIndPointsLocs(locs=locs)

    def setIndPointsLocsKMSRegEpsilon(self, indPointsLocsKMSRegEpsilon):
        self._eLL.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)

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
        super(SVLowerBoundWithParamsGettersAndSetters, self).__init__(eLL, klDiv)
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

