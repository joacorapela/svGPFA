
import pdb
import torch

class SVLowerBound:

    def __init__(self, eLL, klDiv, paramsLogPriors):
        super(SVLowerBound, self).__init__()
        self._eLL = eLL
        self._klDiv = klDiv
        self._paramsLogPriors = paramsLogPriors

    def setInitialParamsAndData(self, measurements, initialParams, quadParams, indPointsLocsKMSRegEpsilon):
        self.setMeasurements(measurements=measurements)
        self.setInitialParams(initialParams=initialParams)
        self.setQuadParams(quadParams=quadParams)
        self.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)
        self.buildKernelsMatrices()

    def eval(self):
        eLLEval = self._eLL.evalSumAcrossTrialsAndNeurons()
        klDivEval = self._klDiv.evalSumAcrossLatentsAndTrials()
        theEval = eLLEval-klDivEval
        paramsLogPriorEval = self._evalParamsLogPrior()
        theEval = eLLEval-klDivEval+paramsLogPriorEval
#         if torch.isinf(theEval):
#             raise RuntimeError("infinity lower bound detected")
#         pdb.set_trace()
        return theEval

    def _evalParamsLogPrior(self):
        embeddingLogPriorValue = self._paramsLogPriors["embedding"](self.getSVEmbeddingParams())
        kernelsLogPriorValue = self._paramsLogPriors["kernels"](self.getKernelsParams())
        indPointsLogPriorValue = self._paramsLogPriors["indPointsLocs"](self.getIndPointsLocs())
        answer = embeddingLogPriorValue + kernelsLogPriorValue + indPointsLogPriorValue
        return answer

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

