
import pdb
import torch

class CovarianceMatricesStore:

    def __init__(self, Kzz, Kzzi, quadKtz, quadKtt):
        self.__Kzz = Kzz
        self.__Kzzi = Kzzi
        self.__quadKtz = quadKtz
        self.__quadKtt = quadKtt

    def getKzz(self):
        return self.__Kzz

    def getKzzi(self):
        return self.__Kzzi

    def getQuadKtz(self):
        return self.__quadKtz

    def getQuadKtt(self):
        return self.__quadKtt

class PointProcessCovarianceMatricesStore(CovarianceMatricesStore):
    def __init__(self, Kzz, Kzzi, quadKtz, quadKtt, spikeKtz, spikeKtt):
        super().__init__(Kzz=Kzz, Kzzi=Kzzi, quadKtz=quadKtz, quadKtt=quadKtt)
        self.__spikeKtz = spikeKtz
        self.__spikeKtt = spikeKtt

    def getSpikeKtz(self):
        return self.__spikeKtz

    def getSpikeKtt(self):
        return self.__spikeKtt
