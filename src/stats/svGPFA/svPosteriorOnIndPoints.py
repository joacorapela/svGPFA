
import pdb
import torch

class SVPosteriorOnIndPoints:

    def __init__(self):
        super(SVPosteriorOnIndPoints, self).__init__()

    def setInitialParams(self, initialParams):
        nLatents = len(initialParams["qMu0"])
        self._qMu = [initialParams["qMu0"][k] for k in range(nLatents)]
        self._qSRSigma = [initialParams["qSRSigma0"][k] for k in range(nLatents)]

    def getParams(self):
        listOfTensors = []
        listOfTensors.extend([self._qMu[k] for k in range(len(self._qMu))])
        listOfTensors.extend([self._qSRSigma[k] for k in range(len(self._qSRSigma))])
        return listOfTensors

    def getQMu(self):
        return self._qMu

    def buildQSigma(self):
        K = len(self._qSRSigma)
        R = self._qSRSigma[0].shape[0]
        qSigma = [[None] for k in range(K)]
        for k in range(K):
            nIndPointsK = self._qSRSigma[k].shape[1]
            qSigma[k] = torch.empty((R, nIndPointsK, nIndPointsK), dtype=torch.double)
            for r in range(R):
                qSigma[k][r,:,:] = torch.matmul(self._qSRSigma[k][r,:,:], torch.transpose(self._qSRSigma[k][r,:,:], 0, 1))
        return(qSigma)

