
import pdb
import torch

class SVLowerBound:

    def __init__(self, eLL, klDiv):
        super(SVLowerBound, self).__init__()
        self._eLL = eLL
        self._klDiv = klDiv
        # shortcuts
        self._svPosteriorOnIndPoints = klDiv.get_svPosteriorOnIndPoints()

    def eval(self):
        eLLEval = self._eLL.evalSumAcrossTrialsAndNeurons()
        klDivEval = self._klDiv.evalSumAcrossLatentsAndTrials()
        theEval = eLLEval-klDivEval
        if torch.isinf(theEval):
            raise RuntimeError("infinity lower bound detected")
        return theEval

    def sampleCIFs(self, times):
        answer = self._eLL.sampleCIFs(times=times)
        return answer

    def computeMeanCIFs(self, times):
        answer = self._eLL.computeMeanCIFs(times=times)
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

    def setQuadParams(self, quadParams):
        self._eLL.setQuadParams(quadParams=quadParams)

    def getSVPosteriorOnIndPointsParams(self):
        return self._eLL.getSVPosteriorOnIndPointsParams()

    def getSVEmbeddingParams(self):
        return self._eLL.getSVEmbeddingParams()

    def getIndPointsLocs(self):
        return self._eLL.getIndPointsLocs()

    def getKernelsParams(self):
        return self._eLL.getKernelsParams()

    def predictLatents(self, newTimes):
        return self._eLL.predictLatents(newTimes=newTimes)

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

