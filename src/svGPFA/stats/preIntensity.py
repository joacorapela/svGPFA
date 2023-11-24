
import pdb
import abc
import torch

class PreIntensity(abc.ABC):

    def __init__(self, posteriorOnLatents):
        self._posteriorOnLatents = posteriorOnLatents

    def computeMeansAndVars(self, posteriorOnLatentsStats=None):
        if posteriorOnLatentsStats is None:
            posteriorOnLatentsStats = \
                self._posteriorOnLatents.computeMeansAndVars()
        means, vars = self._computeMeansAndVarsGivenSVPosteriorOnLatentsStats(
            means=posteriorOnLatentsStats[0],
            vars=posteriorOnLatentsStats[1])
        return means, vars

    def computeSVPosteriorOnLatentsStats(self):
        return self._posteriorOnLatents.computeMeansAndVars()

    def buildKernelsMatrices(self):
        self._posteriorOnLatents.buildKernelsMatrices()

    def buildVariationalCov(self):
        self._posteriorOnLatents.buildVariationalCov()

    @abc.abstractmethod
    def setInitialParams(self, initial_params):
        pass

    def setIndPointsLocs(self, indPointsLocs):
        self._posteriorOnLatents.setIndPointsLocs(indPointsLocs=indPointsLocs)

    def setKernels(self, kernels):
        self._posteriorOnLatents.setKernels(kernels=kernels)

    def setTimes(self, times):
        self._posteriorOnLatents.setTimes(times=times)

    @abc.abstractmethod
    def getParams(self):
        pass

    def getSVPosteriorOnIndPointsParams(self):
        return self._posteriorOnLatents.getSVPosteriorOnIndPointsParams()

    def getIndPointsLocs(self):
        return self._posteriorOnLatents.getIndPointsLocs()

    def getKernels(self):
        return self._posteriorOnLatents.getKernels()

    def getKernelsParams(self):
        return self._posteriorOnLatents.getKernelsParams()

    @abc.abstractmethod
    def _computeMeansAndVarsGivenSVPosteriorOnLatentsStats(self, means, vars):
        pass

class LinearPreIntensity(PreIntensity):

    def setInitialParams(self, initial_params):
        svEmbeddingInitialParams = initial_params["embedding"]
        self._C = svEmbeddingInitialParams["C0"]
        self._d = svEmbeddingInitialParams["d0"]
        posteriorOnLatentsInitialParams = initial_params["posterior_on_latents"]
        self._posteriorOnLatents.setInitialParams(
            initial_params=posteriorOnLatentsInitialParams)

    def sample(self, times, nudget=1e-3):
        latentsSamples, _, _ = self._posteriorOnLatents.sample(times=times,
                                                                 nudget=nudget)
        answer = [self._C.matmul(latentsSamples[r])+self._d for r in range(len(latentsSamples))]
        return answer

    def getParams(self):
        return [self._C, self._d]

class LinearPreIntensityQuadTimes(LinearPreIntensity):

    def _computeMeansAndVarsGivenSVPosteriorOnLatentsStats(self, means, vars):
        # means[r], vars[r] \in nQuad[r] x nLatents 
        # emb_post_mean[r], emb_post_var[r] \in nQuad[r] x nNeurons
        nTrials = len(means)
        emb_post_mean = [[] for r in nTrials]
        emb_post_var = [[] for r in nTrials]
        for r in range(nTrials):
            emb_post_mean[r] = torch.matmul(means[r], torch.t(self._C)) + self._d.T # using broadcasting
            emb_post_var[r] = torch.matmul(vars[r], self._C.T**2)
        return emb_post_mean, emb_post_var

    def predictLatents(self, times):
        return self._posteriorOnLatents.predict(times=times)

    def predict(self, times):
        qKMu, qKVar = self._posteriorOnLatents.predict(times=times)
        answer = self._computeMeansAndVarsGivenSVPosteriorOnLatentsStats(means=qKMu, vars=qKVar)
        return answer

    def computeMeansAndVarsAtTimes(self, times):
        qKMu, qKVar = self._posteriorOnLatents.computeMeansAndVarsAtTimes(times=times)
        answer = self._computeMeansAndVarsGivenSVPosteriorOnLatentsStats(means=qKMu, vars=qKVar)
        return answer

    def setPriorCovRegParam(self, priorCovRegParam):
        self._posteriorOnLatents.setPriorCovRegParam(priorCovRegParam=priorCovRegParam)

class LinearPreIntensitySpikesTimes(LinearPreIntensity):

    def setNeuronForSpikeIndex(self, neuronForSpikeIndex):
        self._neuronForSpikeIndex = neuronForSpikeIndex

    def _computeMeansAndVarsGivenSVPosteriorOnLatentsStats(self, means, vars):
        # means[r], vars[r] \in nQuad[r] x nLatents 
        # emb_post_mean[r], emb_post_var[r] \in nSpikesFromAllNeuronsInTrial[r]
        nTrials = len(self._neuronForSpikeIndex)
        emb_post_mean = [[None] for tr in range(nTrials)]
        emb_post_var = [[None] for tr in range(nTrials)]
        for r in range(nTrials):
            emb_post_mean[r] = torch.sum(means[r]*self._C[(self._neuronForSpikeIndex[r]).tolist(),:], dim=1)+self._d[(self._neuronForSpikeIndex[r]).tolist()].squeeze()
            emb_post_var[r] = torch.sum(vars[r]*(self._C[(self._neuronForSpikeIndex[r]).tolist(),:])**2, dim=1)
        return emb_post_mean, emb_post_var

