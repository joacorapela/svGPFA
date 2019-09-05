
import pdb
import torch

class SparseVariationalLowerBound:

    def __init__(self, eLL, klDiv):
        self.__eLL = eLL
        self.__klDiv = klDiv

    def eval(self):
        eLLEval = self.__eLL.evalSumAcrossTrialsAndNeurons()
        klDivEval = self.__klDiv.evalSumAcrossLatentsAndTrials()
        theEval = eLLEval-klDivEval
        return theEval

    def getApproxPosteriorForHParams(self):
        return self.__eLL.getApproxPosteriorForHParams()
