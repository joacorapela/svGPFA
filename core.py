
import pdb
from abc import ABC
import numpy as np
from scipy.optimize import minimize
from utils import build3DdiagFromDiagVector

class ExpectedLogLikelihood(ABC):
    '''

    Abstract base class for expected log-likelihood subclasses 
    (e.g., PointProcessExpectedLogLikelihood).


    '''

    def __init__(self, legQuadPoints, legQuadWeights, linkFunction):
        '''

        Parameters
        ----------
        data : array
               observations

        quadPoints : array
                     points x_i's used to compute the expectation integral
                     by Gauss-Hermite quadrature.

        quadWeights : array
                      weights w_i's used to compute the expectation
                      integral by Gauss-Hermite quadrature.

        spikeTimes : array
                     array of spike times

        '''
        self._legQuadPoints=legQuadPoints
        self._legQuadWeights=legQuadWeights
        self._linkFunction=linkFunction

    def evalSumAcrossTrialsAndNeuronsWithGradientOnQ(self, qHMeanAtQuad, qHVarAtQuad, **kwargs):
        '''Evaluates the expected log likelihood of a svGPFA

        Parameters
        ----------
        qMean : array
                mean of q(h_n), as computed by
                SparseVariationalLowerBound::computeQ()

        qVar : array
               variance of q(h_n), as computed by
               SparseVariationalLowerBound::computeQ()

        Returns
        -------
        value : double
                value of expectation

        gradient: array
                  gradient of expectation wrt mean and covariance of q

        '''
        pass


# class PointProcessExpectedLogLikelihood():
class PointProcessExpectedLogLikelihood(ExpectedLogLikelihood):

    def __init__(self, legQuadPoints, legQuadWeights, hermQuadPoints, hermQuadWeights, linkFunction):
        '''

        Parameters
        ----------
        data : array
               observations

        quadPoints : array
                     points x_i's used to compute the expectation integral
                     by Gauss-Hermite quadrature.

        quadWeights : array
                      weights w_i's used to compute the expectation
                      integral by Gauss-Hermite quadrature.

        spikeTimes : array
                     array of spike times

        '''
        super().__init__(legQuadPoints=legQuadPoints, 
                          legQuadWeights=legQuadWeights,
                          linkFunction=linkFunction)
        self.__hermQuadPoints = hermQuadPoints
        self.__hermQuadWeights = hermQuadWeights

    def evalSumAcrossTrialsAndNeuronsWithGradientOnQ(self, qHMeanAtQuad, qHVarAtQuad, **kwargs):
        ''' Evaluates the expected log likelihood of a svGPFA

        Parameters
        ----------
        qMean : array
                mean of q(h_n), as computed by
                SparseVariationalLowerBound::computeQ()

        qVar : array
               variance of q(h_n), as computed by
               SparseVariationalLowerBound::computeQ()

        Returns
        -------
        value : double
                value of expectation

        gradient: array
                  gradient of expectation wrt mean and covariance of q

        '''

        try:
            qHMeanAtSpike = kwargs.get('qHMeanAtSpike')
        except KeyError:
           raise KeyError('Missing named argument qhMeanAtSpike')
        try:
            qHVarAtSpike = kwargs.get('qHVarAtSpike')
        except KeyError:
           raise KeyError('Missing named argument qhVarAtSpike')

        if self._linkFunction==np.exp:
            # intval \in nTrials x nQuadLeg x nNeurons
            intval = np.exp(qHMeanAtQuad+0.5*qHVarAtQuad)
            # logLink \in 
            logLink = np.concatenate(np.squeeze(qHMeanAtSpike))
        else:
            # intval = permute(mtimesx(m.wwHerm',permute(m.link(qHmeanAtQuad + sqrt(2*qHMVarAtQuad).*permute(m.xxHerm,[2 3 4 1])),[4 1 2 3])),[2 3 4 1]);

            # aux2 \in  nTrials x nQuadLeg x nNeuros
            aux2 = np.sqrt(2*qHVarAtQuad)
            # aux3 \in nTrials x nQuadLeg x nNeuros x nQuadLeg
            aux3 = np.multiply.outer(aux2, super()._hermQuadPoints)
            # aux4 \in nQuad x nQuadLeg x nTrials x nQuadLeg
            aux4 = np.add(aux3, qHMeanAtQuad)
            # aux5 \in nQuad x nQuadLeg x nTrials x nQuadLeg
            aux5 = self._linkFunction(x=aux4)
            # intval \in  nTrials x nQuadHerm x nNeurons
            intval = np.tensordot(a=aux5, b=super()._hermQuadWeights, axes=([4], [0]))
            # log_link = cellvec(cellfun(@(x,y) log(m.link(x + sqrt(2*y).* m.xxHerm'))*m.wwHerm,mu_h_Spikes,var_h_Spikes,'uni',0));
            # aux1[trial] \in nSpikes[trial]
            aux1 = [2*qHVarAtSpike[trial] for trial in range(len(qHVarAtSpike))]
            # aux2[trial] \in nSpikes[trial] x nQuadLeg
            aux2 = [np.multiply.outer(aux1[trial], super()._hermQuadPoints) for trial in range(len(aux1))]
            # aux3[trial] \in nSpikes[trial] x nQuadLeg
            aux3 = [np.add(aux2[trial], qHMeanAtSpike[trial]) for trial in range(len(aux2))]
            # aux4[trial] \in nSpikes[trial] x nQuadLeg
            aux4 = [np.log(self._linkFunction(x=aux3[trial])) for trial in range(len(aux3))]
            # aux5[trial] \in nSpikes[trial] x 1
            aux5 = [np.tensordot(a=aux4[trial], b=super()._hermQuadWeights, axes=([1], [0])) for trial in range(len(aux4))]
            logLink = np.concatentate(aux5)

        # self._legQuadWeights \in nTrials x nQuadHerm x 1
        # aux0 \in nTrials x 1 x nQuadHerm
        aux0 = np.transpose(self._legQuadWeights, (0, 2, 1)) 
        # intval \in  nTrials x nQuadHerm x nNeurons
        # aux1 \in  nTrials x 1 x nNeurons
        aux1 = np.matmul(aux0, intval)
        sELLTerm1 = np.sum(aux1)
        sELLTerm2 = np.sum(logLink)
        return -sELLTerm1+sELLTerm2

class KLDivergence:

    def evalSumAcrossLatentsAndTrials(self, Kzzi, qMu, qSigma):
        klDiv = 0
        for k in range(len(Kzzi)):
            klDivK = self.evalSumAcrossTrials(Kzzi=Kzzi[k], qMu=qMu[k], qSigma=qSigma[k])
            klDiv += klDivK
        return klDiv

    def evalSumAcrossTrials(self, Kzzi, qMu, qSigma):
        '''
        Answers the sum of KL divergences across trials.
        (e.g., answer= \sum_k KL(N(0, s0Inv[tr,:]), N(qMu[tr,:], qSigma[tr,:])))

        Parameters
        ----------
        Kzzi : array \in nTrials x nInd x nInd
        qMu : array \in nTrials x nInd x 1
        qSigma : array \in nTrials x nInd x nInd

        Returns
        -------
        value : double
                value of KL divergence
        '''

        # ESS \in nTrials x nInd x nInd
        ESS = qSigma + np.matmul(qMu, np.transpose(a=qMu, axes=(0,2,1)))
        nTrials = qMu.shape[0]
        answer = 0
        for trial in range(nTrials):
            _, logdetKzzi = np.linalg.slogdet(a=Kzzi[trial,:,:])
            _, logdetQSigma = np.linalg.slogdet(a=qSigma[trial,:,:])
            # aux3 = np.dot(np.ndarray.flatten(Kzzi[trial,:,:]), np.ndarray.flatten(ESS[trial,:,:]))
            # aux4 = ESS.shape[1]
            trialKL = .5*(-logdetKzzi-logdetQSigma
                           +np.dot(np.ndarray.flatten(Kzzi[trial,:,:]), np.ndarray.flatten(ESS[trial,:,:]))
                           -ESS.shape[1])
            answer += trialKL
            # print("aux1=%f\naux2=%f\naux3=%f\naux4=%f\nkldiv_nn=%f,kldiv=%f"%(logdetKzzi, logdetQSigma, aux3, aux4, trialKL, answer))
            # pdb.set_trace()
        return answer

class CovarianceMatricesStore:

    def __init__(self, Kzz, Kzzi, quadKtz, quadKtt, spikeKtz, spikeKtt):
        self.__Kzz = Kzz
        self.__Kzzi = Kzzi
        self.__quadKtz = quadKtz
        self.__quadKtt = quadKtt
        self.__spikeKtz = spikeKtz
        self.__spikeKtt = spikeKtt

    def getKzz(self):
        return self.__Kzz

    def getKzzi(self):
        return self.__Kzzi

    def getQuadKtz(self):
        return self.__quadKtz

    def getQuadKtt(self):
        return self.__quadKtt

    def getSpikeKtz(self):
        return self.__spikeKtz

    def getSpikeKtt(self):
        return self.__spikeKtt

class SparseVariationalEM:

    def __init__(self, eLL, covMatricesStore):
        self.__eLL = eLL
        self.__covMatricesStore = covMatricesStore

    def maximize(self, y, qMu0, SVec0, SDiag0, C0, d0, kernelParms0, z0, 
                       maxEMIter, maxEStepIter, maxMStepModelParamsIter,
                       maxMStepKernelParmsIter, maxMStepIndIter, 
                       tol):
        '''
        m0, SVec0, SDiag0 \in nInd x nNeruons x nTrials
        '''
        iter = 0
        svlb = SparseVariationalLowerBound(y=y, eLL=self.__eLL, 
                                               covMatricesStore=
                                                self.__covMatricesStore,
                                               qMu=qMu0, qSVec0=qSVec0,
                                               qSDiag0=qSDiag0, 
                                               C0=C0, d0=d0,
                                               kernelParms0=kernelParms0,
                                               z0=z0)
        qMu = qMu0
        SVec = SVec0
        SDiag = SDiag0
        while iter<maxEMIter:
            flattenedQParams = self.__eStep(lowerBound=svL, 
                                             qMu=qMu, 
                                             SVec=SVec, 
                                             sDiag=sDiag, 
                                             maxIter=maxEStepIter, 
                                             tol=tol)
            svlb.setVariationalProposalParams(flatteneQdParams=flattenedQParams)
            '''
            modelParams = self.__mStepModelParams(nInd=nInd, 
                                        maxMStepModelParamsIter=
                                         maxMStepModelParamsIter, 
                                        maxMStepKernelParmsIter=
                                         maxMStepModelParamsIter, 
                                        maxMStepModelParamsIter=
                                         maxMStepModelParamsIter, 
                                         tol=tol)
            vL.setModelParmams(modelParams=othterParmsa$modelParams,
                                kernelParams=otherParams$kernelParams,
                                zParms=otherParams$z,
                                inducingPointsParams=
                                 otherParams$inducingPointsParam)
            
            self.__mStep()
            '''

    def __eStep(self, lowerBound, qMu, qSVec, qSDiag, maxIter, tol):
        x0 = lowerBound.flattenVariationalProposalParams(qMu=qMu, qSVec=SVec, 
                                                                  qSDiag=SDiag)
        res = minimize(lowerBound.evalWithGradientOnQ, x0=x0, 
                        method='BFGS', options={'xtol': tol, 'disp': True},
                        args=(len(m)))
        return res


class SparseVariationalLowerBound:

    def __init__(self, eLL, covMatricesStore, qMu, qSVec, qSDiag, C, d, 
                  kernelParams, varRnk, neuronForSpikeIndex):
        self.__eLL = eLL
        self.__covMatricesStore = covMatricesStore
        self.__qMu = qMu
        self.__qSVec = qSVec
        self.__qSDiag = qSDiag
        self.__C = C
        self.__d = d
        self.__kernelParams = kernelParams
        self.__varRnk = varRnk
        self.__neuronForSpikeIndex = neuronForSpikeIndex

    def evalWithGradOnQ(self, x):
        '''
        Answers the value and gradient of the lower bound (Eq. (4) of Duncker and Sahani, 2018).

        Parameters
        ----------
        x : vector containing the concatenation of flattened qMu, qSVec and qSDiag.
        '''

        qMu, qSVec, qSDiag = self.__unflattenVariationalProposalParams(flattenedQParams=x)
        qSigma = self.__buildQSigma(qSVec=qSVec, qSDiag=qSDiag)
        qH = PointProcessSparseVariationalProposal()
        qHMeanAtQuad, qHVarAtQuad = qH.getMeanAndVarianceAtQuadPoints(qMu=qMu, qSigma=qSigma, C=self.__C, d=self.__d, Kzzi=self.__covMatricesStore.getKzzi(), Kzz=self.__covMatricesStore.getKzz(), Ktz=self.__covMatricesStore.getQuadKtz(), Ktt=self.__covMatricesStore.getQuadKtt())
        qHMeanAtSpike, qHVarAtSpike = qH.getMeanAndVarianceAtSpikeTimes(qMu=qMu, qSigma=qSigma, C=self.__C, d=self.__d, Kzzi=self.__covMatricesStore.getKzzi(), Kzz=self.__covMatricesStore.getKzz(), Ktz=self.__covMatricesStore.getSpikeKtz(), Ktt=self.__covMatricesStore.getSpikeKtt(), neuronForSpikeIndex=self.__neuronForSpikeIndex)
        eLLEval = self.__eLL.evalSumAcrossTrialsAndNeuronsWithGradientOnQ(qHMeanAtQuad=qHMeanAtQuad, qHVarAtQuad=qHVarAtQuad, qHMeanAtSpike=qHMeanAtSpike, qHVarAtSpike=qHVarAtSpike)
        klDiv = KLDivergence()
        klDivEval = klDiv.evalSumAcrossLatentsAndTrials(Kzzi=self.__covMatricesStore.getKzzi(), qMu=qMu, qSigma=qSigma)

        answer = -eLLEval+klDivEval
        return answer

    def __buildQSigma(self, qSVec, qSDiag):
        # qSVec[k]  \in nTrials x (nInd[k]*varRnk[k]) x 1
        # qSDiag[k] \in nTrials x nInd[k] x 1

        R = qSVec[0].shape[0]
        K = len(qSVec)
        qSigma = [None] * K
        for k in range(K):
            nIndK = qSDiag[k].shape[1]
            # qq \in nTrials x nInd[k] x varRnk[k]
            qq = np.reshape(a=qSVec[k], newshape=(R, nIndK, self.__varRnk[k]))
            # dd \in nTrials x nInd[k] x varRnk[k]
            nIndKVarRnkK = qSVec[k].shape[1]
            dd = build3DdiagFromDiagVector(v=(qSDiag[k].flatten())**2, M=R, N=nIndKVarRnkK)
            qSigma[k] = np.matmul(qq, np.transpose(a=qq, axes=(0,2,1))) + dd
        return(qSigma)

    def setVariationalProposalParams(self, flattenedQParams):
        self.__qMu, self.__qSVec, self.__qSDiag = \
         self.__unflattenVariationalProposalParams(flattenedQParams=
                                                    flattenedQParams)

    def flattenVariationalProposalParams(self, qMu, qSVec, qSDiag):
        return self.__flattenListOfArrays(qMu, qSVec, qSDiag)

    def __flattenListOfArrays(self, *lists):
        aListOfArrays = []
        for arraysList in lists:
            for array in arraysList:
                aListOfArrays.append(array.flatten())
        return np.concatenate(aListOfArrays)

    def __unflattenVariationalProposalParams(self, flattenedQParams):
        unflattenedLists =  \
         self.__unflattenListsOfArrays(flattenedParams=flattenedQParams,
                                        referenceLists=(self.__qMu,
                                                         self.__qSVec,
                                                         self.__qSDiag))
        return unflattenedLists
        
    def __unflattenListsOfArrays(self, flattenedParams, referenceLists):
        unflattenedLists = [None] * len(referenceLists)
        fromIndex = 0
        for i in range(len(referenceLists)):
            unflattenedLists[i], fromIndex = self.__unflattenListOfArrays(flattenedParams=flattenedParams, fromIndex=fromIndex, referenceList=referenceLists[i])
        return unflattenedLists

    def __unflattenListOfArrays(self, flattenedParams, fromIndex, referenceList):
        unflattenedList = [None] * len(referenceList)
        for k in range(len(referenceList)):
            unflattenedList[k] = np.reshape(a=flattenedParams[fromIndex+np.arange(referenceList[k].size)], newshape=referenceList[k].shape)
            fromIndex += referenceList[k].size
        return unflattenedList, fromIndex

    def __evalKLDivergenceTerm(Kzzi, qMu, qSigma):
        term = 0
        nLatents = Kzzi.dim[2]
        klDivergence = KLDivergence()
        for k in range(nLatents):
            term += klDivergence.evalWithGradientOnQ(S0Inv=Kzzi[:,:,k],
                                                      mu1=qMu[:,k],
                                                      S1=qSigma[:,:,k])
        return term

class SparseVariationalProposal:

    def getMeanAndVarianceAtQuadPoints(self, qMu, qSigma, C, d, Kzzi, Kzz, Ktz, Ktt):
        '''
        Answers the mean and variance in Eq. (5) of Dunker and Sahani, 2018,
        at quadrature times.

        Parameters
        ----------
        qMu : list of length nLatent. 
              qMu[k] array \in  nTrials x nInd[k] x 1
        qSigma : list of length nLatent. 
                 qSigma[k] array \in nTrials x nInd[k] x nInd[k]
        C : array \in nNeurons x nLatent
        d : array \in nNeurons
        Kzzi : list of length nLatent. 
               Kzzi[k] array \in nTrials x nInd[k] x nInd[k]
        Kzz : list of length nLatent. 
              Kzz[k] array \in nTrials x nInd[k] x nInd[k]
        Ktz : list of length nLatent. 
              Ktz[k] \in nTrials x nQuad x nInd[k]
        Ktt: array \in nTrials x nQuad x nLatent

        Returns
        -------
        qHMu : array \in nTrials x nQuad x nLatent
        qHVar : array \in nTrials x nQuad x nLatent
        '''

        nTrials = Ktt.shape[0]
        nQuad = Ktt.shape[1]
        nLatent = Ktt.shape[2]

        muK = np.empty((nTrials, nQuad, nLatent))
        varK = np.empty((nTrials, nQuad, nLatent))

        for k in range(len(qMu)):
            # Ak \in nTrials x nInd[k] x 1 
            Ak = np.matmul(Kzzi[k], qMu[k]) 
            muK[:,:,k] = np.squeeze(np.matmul(Ktz[k], Ak))
            # Bkf \in nTrials x nInd[k] x nQuad
            Bkf = np.matmul(Kzzi[k], np.transpose(a=Ktz[k], axes=(0, 2, 1)))
            # mm1f \in nTrials x nInd[k] x nQuad
            mm1f = np.matmul(qSigma[k]-Kzz[k], Bkf)

            # aux1 \in nTrials x nInd[k] x nQuad
            aux1 = Bkf*mm1f
            # aux2 \in nTrials x nQuad
            aux2 = np.sum(a=aux1, axis=1)
            # aux3 \in nTrials x nQuad
            aux3 = Ktt[:,:,k]+aux2
            # varK \in nTrials x nQuad x nLatent
            varK[:,:,k] = aux3
            # varK[:,:,k] = Ktt[:,:,k]+np.tensordot(a=Bkf, b=mm1f, axes=([1], [1]))

        qHMu = np.matmul(muK, C.T) + np.reshape(a=d, newshape=(1, 1, len(d))) # using broadcasting
        qHVar = np.matmul(varK, (C.T)**2)
        return (qHMu, qHVar)

class PointProcessSparseVariationalProposal(SparseVariationalProposal):

    def getMeanAndVarianceAtSpikeTimes(self, qMu, qSigma, C, d, Kzzi, Kzz, Ktz,
                                             Ktt, neuronForSpikeIndex):
        '''
        Answers the mean and variance in Eq. (5) of Dunker and Sahani, 2018,
        at spike times.

        Parameters
        ----------
        qMu : qMu[k] array \in  nTrials x nInd[k] x 1
        qSigma : qSigma[k] array \in nTrials x nInd[k] x nInd[k]
        C : array \in nNeurons x nLatent
        d : array \in nNeurons
        Kzzi : list of length nLatent. 
               Kzzi[k] array \in nTrials x nInd[k] x nInd[k]
        Ktz : list[k][trialIndex] \in nSpikesForTrial[trialIndex] x nInd[k]
        Ktt: list[k][trialIndex] \in nSpikesForTrial[trialIndex] x 1
        neuronForSpikeIndex: neuronForSpikeIndex[trialIndex][i]==j if neuron j
                             produced spike i in trialIndex.
        Returns
        -------
        qHMu :  list[trialIndex] \in nSpikesForTrial[trialIndex] x 1
        qHVar : list[trialIndex] \in nSpikesForTrial[trialIndex] x 1
        '''

        nTrials = Ktt.shape[1]
        nLatent = len(qMu)
        # Ak[k] \in nTrial x nInd[k] x 1
        Ak = [np.matmul(Kzzi[k], qMu[k]) for k in range(nLatent)]
        qKMu = [None] * nTrials
        qKVar = [None] * nTrials
        for trialIndex in range(nTrials):
            nSpikesForTrial = Ktt[0][trialIndex].shape[0]
            # qKMu[trialIndex] \in nSpikesForTrial[trialIndex] x nLatent
            qKMu[trialIndex] = np.empty((nSpikesForTrial, nLatent))
            qKVar[trialIndex] = np.empty((nSpikesForTrial, nLatent))
            for k in range(nLatent):
                qKMu[trialIndex][:,k] = \
                 np.squeeze(Ktz[k][trialIndex].dot(Ak[k][trialIndex,:,:]))
                # Bfk \in nInd[k] x nSpikesForTrial[trialIndex]
                Bfk = np.matmul(Kzzi[k][trialIndex,:,:], 
                                 np.transpose(a=Ktz[k][trialIndex]))
                # mm1f \in nInd[k] x nSpikesForTrial[trialIndex]
                mm1f = np.matmul(qSigma[k][trialIndex,:,:]-
                                  Kzz[k][trialIndex,:,:], Bfk)
                # qKVar[trialIndex] \in nSpikesForTrial[trialIndex] x nLatent
                qKVar[trialIndex][:,k] = np.squeeze(Ktt[k][trialIndex])+np.sum(a=Bfk*mm1f, axis=0)
        qHMu = [None] * nTrials
        qHVar = [None] * nTrials
        for trialIndex in range(nTrials):
            qHMu[trialIndex] = np.sum(qKMu[trialIndex]*C[neuronForSpikeIndex[trialIndex]-1,:],axis=1)+d[neuronForSpikeIndex[trialIndex]-1]
            qHVar[trialIndex] = np.sum(qKVar[trialIndex]*(C[neuronForSpikeIndex[trialIndex]-1,:])**2,axis=1)
        return qHMu, qHVar
