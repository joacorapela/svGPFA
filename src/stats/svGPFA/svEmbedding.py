
import pdb
from abc import ABC, abstractmethod
import torch

class SVEmbedding(ABC):

    def __init__(self, svPosteriorOnLatents):
        self._svPosteriorOnLatents = svPosteriorOnLatents

    def computeMeansAndVars(self, svPosteriorOnLatentsStats=None):
        if svPosteriorOnLatentsStats is None:
            svPosteriorOnLatentsStats = \
                self._svPosteriorOnLatents.computeMeansAndVars()
        means, vars = self._computeMeansAndVarsGivenSVPosteriorOnLatentsStats(
            means=svPosteriorOnLatentsStats[0],
            vars=svPosteriorOnLatentsStats[1])
        return means, vars

    def computeSVPosteriorOnLatentsStats(self):
        return self._svPosteriorOnLatents.computeMeansAndVars()

    def buildKernelsMatrices(self):
        self._svPosteriorOnLatents.buildKernelsMatrices()

    @abstractmethod
    def setInitialParams(self, initialParams):
        pass

    def setIndPointsLocs(self, indPointsLocs):
        self._svPosteriorOnLatents.setIndPointsLocs(indPointsLocs=indPointsLocs)

    def setKernels(self, kernels):
        self._svPosteriorOnLatents.setKernels(kernels=kernels)

    def setTimes(self, times):
        self._svPosteriorOnLatents.setTimes(times=times)

    @abstractmethod
    def getParams(self):
        pass

    def getSVPosteriorOnIndPointsParams(self):
        return self._svPosteriorOnLatents.getSVPosteriorOnIndPointsParams()

    def getIndPointsLocs(self):
        return self._svPosteriorOnLatents.getIndPointsLocs()

    def getKernelsParams(self):
        return self._svPosteriorOnLatents.getKernelsParams()

    @abstractmethod
    def _computeMeansAndVarsGivenSVPosteriorOnLatentsStats(self, means, vars):
        pass

class LinearSVEmbedding(SVEmbedding):

    def setInitialParams(self, initialParams):
        svEmbeddingInitialParams = initialParams["svEmbedding"]
        self._C = svEmbeddingInitialParams["C0"]
        self._d = svEmbeddingInitialParams["d0"]
        svPosteriorOnLatentsInitialParams = initialParams["svPosteriorOnLatents"]
        self._svPosteriorOnLatents.setInitialParams(initialParams=svPosteriorOnLatentsInitialParams)

    def sample(self, times):
        latentsSamples = self._svPosteriorOnLatents.sample(times=times)
        answer = [self._C.matmul(latentsSamples[r])+self._d for r in range(len(latentsSamples))]
        return answer

    def getParams(self):
        return [self._C, self._d]

    def get_flattened_params(self):
        flattened_params = []
        flattened_params.extend(self._C.flatten().tolist())
        flattened_params.extend(self._d.flatten().tolist())
        return flattened_params

    def get_flattened_params_grad(self):
        flattened_params_grad = []
        flattened_params_grad.extend(self._C.grad.flatten().tolist())
        flattened_params_grad.extend(self._d.grad.flatten().tolist())
        return flattened_params_grad

    def set_params_from_flattened(self, flattened_params):
        flattened_param = flattened_params[:self._C.numel()]
        self._C = torch.tensor(flattened_param, dtype=torch.double).reshape(self._C.shape)
        flattened_params = flattened_params[self._C.numel():]
        flattened_param = flattened_params[:self._d.numel()]
        self._d = torch.tensor(flattened_param, dtype=torch.double).reshape(self._d.shape)
        flattened_param = flattened_params[self._d.numel():]

    def set_params_requires_grad(self, requires_grad):
        self._C.requires_grad = requires_grad
        self._d.requires_grad = requires_grad

class LinearSVEmbeddingAllTimes(LinearSVEmbedding):

    def _computeMeansAndVarsGivenSVPosteriorOnLatentsStats(self, means, vars):
        qHMu = torch.matmul(means, torch.t(self._C)) + torch.reshape(input=self._d, shape=(1, 1, len(self._d))) # using broadcasting
        qHVar = torch.matmul(vars, (torch.t(self._C))**2)
        return qHMu, qHVar

    def predictLatents(self, newTimes):
        return self._svPosteriorOnLatents.predict(newTimes=newTimes)

    def computeMeans(self, times):
        qKMu = self._svPosteriorOnLatents.computeMeans(times=times)
        qHMu = torch.matmul(qKMu, torch.t(self._C)) + torch.reshape(input=self._d, shape=(1, 1, len(self._d))) # using broadcasting
        return qHMu

    def setIndPointsLocsKMSRegEpsilon(self, indPointsLocsKMSRegEpsilon):
        self._svPosteriorOnLatents.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)

class LinearSVEmbeddingAssocTimes(LinearSVEmbedding):

    def setNeuronForSpikeIndex(self, neuronForSpikeIndex):
        self._neuronForSpikeIndex = neuronForSpikeIndex

    def _computeMeansAndVarsGivenSVPosteriorOnLatentsStats(self, means, vars):
        nTrials = len(self._neuronForSpikeIndex)
        qHMu = [[None] for tr in range(nTrials)]
        qHVar = [[None] for tr in range(nTrials)]
        for trialIndex in range(nTrials):
            qHMu[trialIndex] = torch.sum(means[trialIndex]*self._C[(self._neuronForSpikeIndex[trialIndex]).tolist(),:], dim=1)+self._d[(self._neuronForSpikeIndex[trialIndex]).tolist()].squeeze()
            qHVar[trialIndex] = torch.sum(vars[trialIndex]*(self._C[(self._neuronForSpikeIndex[trialIndex]).tolist(),:])**2, dim=1)
        return qHMu, qHVar

