
import pdb
import math
import torch
import utils.svGPFA.miscUtils

class SVPosteriorOnIndPoints:

    def __init__(self):
        super(SVPosteriorOnIndPoints, self).__init__()

    def setInitialParams(self, initialParams):
        nLatents = len(initialParams["qMu0"])
        self._qMu = [initialParams["qMu0"][k] for k in range(nLatents)]
        self._srQSigmaVecs = [initialParams["srQSigma0Vecs"][k] for k in range(nLatents)]

    def getParams(self):
        listOfTensors = []
        listOfTensors.extend([self._qMu[k] for k in range(len(self._qMu))])
        listOfTensors.extend([self._srQSigmaVecs[k] for k in range(len(self._srQSigmaVecs))])
        return listOfTensors

    def getQMu(self):
        return self._qMu

    def buildQSigma(self):
        # begin patch for older version of the code
        if hasattr(self, "_qSRSigmaVec"):
            self._srQSigmaVecs = [self._qSRSigmaVec[k].unsqueeze(-1) for k in range(len(self._qSRSigmaVec))]
        elif self._srQSigmaVecs[0].dim()==2:
            self._srQSigmaVecs = [self._srQSigmaVecs[k].unsqueeze(-1) for k in range(len(self._srQSigmaVecs))]
        # end patch for older version of the code
        qSigma = utils.svGPFA.miscUtils.buildQSigmasFromSRQSigmaVecs(srQSigmaVecs=self._srQSigmaVecs)
        return qSigma


