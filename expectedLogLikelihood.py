
import pdb
from abc import ABC, abstractmethod
import torch
# import warnings

class ExpectedLogLikelihood(ABC):
    '''

    Abstract base class for expected log-likelihood subclasses 
    (e.g., PointProcessExpectedLogLikelihood).


    '''
    def __init__(self, approxPosteriorForH, hermQuadPoints, hermQuadWeights, linkFunction):
        self._approxPosteriorForH = approxPosteriorForH
        self._hermQuadPoints=hermQuadPoints
        self._hermQuadWeights=hermQuadWeights
        self._linkFunction=linkFunction

    def getApproxPosteriorForHParams(self):
        return self._approxPosteriorForH.getApproxPosteriorForHParams()

    @abstractmethod
    def evalSumAcrossTrialsAndNeurons(self):
        pass


class PointProcessExpectedLogLikelihood(ExpectedLogLikelihood):
    def __init__(self, approxPosteriorForH, hermQuadPoints, hermQuadWeights, legQuadPoints, legQuadWeights, linkFunction):
        super().__init__(approxPosteriorForH=approxPosteriorForH, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, linkFunction=linkFunction)
        self.__legQuadPoints = legQuadPoints
        self.__legQuadWeights = legQuadWeights

    def evalSumAcrossTrialsAndNeurons(self):
        qHMeanAtQuad, qHVarAtQuad = self._approxPosteriorForH.getMeanAndVarianceAtQuadPoints()
        qHMeanAtSpike, qHVarAtSpike = self._approxPosteriorForH.getMeanAndVarianceAtSpikeTimes()
        # warnings.warn("Use of analytical calculation has been disabled for testing")
        # if False:
        if self._linkFunction==torch.exp:
            # intval \in nTrials x nQuadLeg x nNeurons
            intval = torch.exp(input=qHMeanAtQuad+0.5*qHVarAtQuad)
            # logLink \in 
            logLink = torch.cat([torch.squeeze(input=qHMeanAtSpike[trial]) for trial in range(len(qHMeanAtSpike))])
        else:
            # intval = permute(mtimesx(m.wwHerm',permute(m.link(qHmeanAtQuad + sqrt(2*qHMVarAtQuad).*permute(m.xxHerm,[2 3 4 1])),[4 1 2 3])),[2 3 4 1]);

            # aux2 \in  nTrials x nQuadLeg x nNeurons
            aux2 = torch.sqrt(2*qHVarAtQuad)
            # aux3 \in nTrials x nQuadLeg x nNeurons x nQuadLeg
            aux3 = torch.einsum('ijk,l->ijkl', aux2, torch.squeeze(self._hermQuadPoints))
            # aux4 \in nQuad x nQuadLeg x nTrials x nQuadLeg
            aux4 = torch.add(input=aux3, other=qHMeanAtQuad.unsqueeze(dim=3))
            # aux5 \in nQuad x nQuadLeg x nTrials x nQuadLeg
            aux5 = self._linkFunction(input=aux4)
            # intval \in  nTrials x nQuadHerm x nNeurons
            intval = torch.einsum('ijkl,l->ijk', aux5, self._hermQuadWeights.squeeze())

            # log_link = cellvec(cellfun(@(x,y) log(m.link(x + sqrt(2*y).* m.xxHerm'))*m.wwHerm,mu_h_Spikes,var_h_Spikes,'uni',0));
            # aux1[trial] \in nSpikes[trial]
            aux1 = [2*qHVarAtSpike[trial] for trial in range(len(qHVarAtSpike))]
            # aux2[trial] \in nSpikes[trial] x nQuadLeg
            aux2 = [torch.einsum('i,j->ij', aux1[trial].squeeze(), self._hermQuadPoints.squeeze()) for trial in range(len(aux1))]
            # aux3[trial] \in nSpikes[trial] x nQuadLeg
            aux3 = [torch.add(input=aux2[trial], other=qHMeanAtSpike[trial]) for trial in range(len(aux2))]
            # aux4[trial] \in nSpikes[trial] x nQuadLeg
            aux4 = [torch.log(input=self._linkFunction(x=aux3[trial])) for trial in range(len(aux3))]
            # aux5[trial] \in nSpikes[trial] x 1
            aux5 = [torch.tensordot(a=aux4[trial], b=self._hermQuadWeights, dims=([1], [0])) for trial in range(len(aux4))]
            logLink = torch.cat(tensors=aux5)

        # self.__legQuadWeights \in nTrials x nQuadHerm x 1
        # aux0 \in nTrials x 1 x nQuadHerm
        aux0 = torch.transpose(input=self.__legQuadWeights, dim0=1, dim1=2)
        # intval \in  nTrials x nQuadHerm x nNeurons
        # aux1 \in  nTrials x 1 x nNeurons
        aux1 = torch.matmul(aux0, intval)
        sELLTerm1 = torch.sum(aux1)
        sELLTerm2 = torch.sum(logLink)
        return -sELLTerm1+sELLTerm2

class PoissonExpectedLogLikelihood(ExpectedLogLikelihood):
    def __init__(self, approxPosteriorForH, hermQuadPoints, hermQuadWeights, linkFunction, Y, binWidth):
        super().__init__(approxPosteriorForH=approxPosteriorForH, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, linkFunction=linkFunction)
        # Y \in nTrials x nNeurons x maxNBins
        self.__Y = Y
        self.__binWidth = binWidth

    def evalSumAcrossTrialsAndNeurons(self):
        qHMeanAtQuad, qHVarAtQuad = self._approxPosteriorForH.getMeanAndVarianceAtQuadPoints()
        # warnings.warn("Use of analytical calculation has been disabled for testing")
        # if False:
        if self._linkFunction==torch.exp:
            # intval \in nTrials x nQuadLeg x nNeurons
            intval = torch.exp(input=qHMeanAtQuad+0.5*qHVarAtQuad)
            # logLink \in nTrials x maxNBins x nNeurons
            logLink = qHMeanAtQuad
        else:
            # intval = permute(mtimesx(m.wwHerm',permute(m.link(qHmeanAtQuad + sqrt(2*qHMVarAtQuad).*permute(m.xxHerm,[2 3 4 1])),[4 1 2 3])),[2 3 4 1]);

            # aux2 \in  nTrials x maxNBins x nNeurons
            aux2 = torch.sqrt(2*qHVarAtQuad)
            # aux3 \in nTrials x maxNBins x nNeurons x nQuadLeg
            aux3 = torch.einsum('ijk,l->ijkl', aux2, torch.squeeze(self._hermQuadPoints))
            # aux4 \in maxNTrials x maxNBins x nNeurons x nQuadLeg
            aux4 = torch.add(input=aux3, other=qHMeanAtQuad.unsqueeze(dim=3))
            # aux5a \in maxNTrials x maxNBins x nNeurons x nQuadLeg
            aux5a = self._linkFunction(input=aux4)
            aux5b = aux5a.log()
            # intval \in  maxNTrials x maxNBins x nQuadLeg
            loglink = torch.einsum('ijkl,l->ijk', aux5a, self._hermQuadWeights.squeeze())
            intval = torch.einsum('ijkl,l->ijk', aux5b, self._hermQuadWeights.squeeze())

        aux1 = torch.matmul(aux0, intval)
        sELLTerm1 = self.__binWidth*intval.sum()
        # Y \in nTrials x nNeurons x maxNBins
        sELLTerm2 = (Y*logLink.permute(0, 2, 1)).sum()
        return -sELLTerm1+sELLTerm2
