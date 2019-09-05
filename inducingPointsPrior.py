import torch
from utils import build3DdiagFromDiagVector, flattenListsOfArrays

class InducingPointsPrior:

    def __init__(self, qMu, qSVec, qSDiag, varRnk):
        self.__qMu = qMu
        self.__qSVec = qSVec
        self.__qSDiag = qSDiag
        self.__varRnk = varRnk

    def getQMu(self):
        return self.__qMu

    def getParams(self):
        listOfTensors = []
        listOfTensors.extend([self.__qMu[k] for k in range(len(self.__qMu))])
        listOfTensors.extend([self.__qSVec[k] for k in range(len(self.__qSVec))])
        listOfTensors.extend([self.__qSDiag[k] for k in range(len(self.__qSDiag))])
        return listOfTensors

    def buildQSigma(self):
        # qSVec[k]  \in nTrials x (nInd[k]*varRnk[k]) x 1
        # qSDiag[k] \in nTrials x nInd[k] x 1

        R = self.__qSVec[0].shape[0]
        K = len(self.__qSVec)
        qSigma = [None] * K
        for k in range(K):
            nIndK = self.__qSDiag[k].shape[1]
            # qq \in nTrials x nInd[k] x varRnk[k]
            qq = self.__qSVec[k].reshape(shape=(R, nIndK, self.__varRnk[k]))
            # dd \in nTrials x nInd[k] x varRnk[k]
            nIndKVarRnkK = self.__qSVec[k].shape[1]
            dd = build3DdiagFromDiagVector(v=(self.__qSDiag[k].flatten())**2, M=R, N=nIndKVarRnkK)
            qSigma[k] = torch.matmul(qq, torch.transpose(a=qq, dim0=1, dim1=2)) + dd
        return(qSigma)

