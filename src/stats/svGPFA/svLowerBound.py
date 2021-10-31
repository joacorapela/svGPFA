
import pdb
import torch

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
        if torch.isinf(theEval):
            raise RuntimeError("infinity lower bound detected")
        return theEval

    def sampleCIFs(self, times):
        answer = self._eLL.sampleCIFs(times=times)
        return answer

    def computeCIFsMeans(self, times):
        answer = self._eLL.computeCIFsMeans(times=times)
        return answer

    def computeExpectedCIFs(self, times):
        answer = self._eLL.computeExpectedCIFs(times=times)
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

    def getIndPointsLocs(self):
        return self._eLL.getIndPointsLocs()

    def getKernelsParams(self):
        return self._eLL.getKernelsParams()

    def predictLatents(self, newTimes):
        return self._eLL.predictLatents(newTimes=newTimes)

