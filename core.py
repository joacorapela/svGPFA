
import pdb
from abc import ABCMeta
import numpy as np
from scipy.optimize import minimize

class ExpectedLogLikelihood(ABCMeta):
    '''

    Abstract base class for expected log-likelihood subclasses 
    (e.g., PointProcessExpectedLogLikelihood).


    '''

    def __init__(data, quadPoints, quadWeights, spikeTimes):
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
        pass

    def evalWithGradientOnQ(qMean, qVar):
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


class KLDivergenceCalculator:
    def evalSumAcrossTrials(S0Inv, mu1, S1):
        '''
        Answers the sum of KL divergences across trials.
        (e.g., answer= \sum_k KL(N(0, s0Inv[k,:]), N(mu1[k,:], S1[k,:]))).

        Parameters
        ----------
        S0Inv : array \in nTrials x nInd x nInd
        mu1 : array \in nTrials x nInd x 1
        S1 : array \in nTrials x nInd x nInd

        Returns
        -------
        value : double
                value of KL divergence
        '''

        ESS = S1 + np.matmul(mu, np.transpose(a=mu, axes=(0,2,1)))
        answer = 0
        for trial in range(mu1.shape[0]):
            trialKL = .5*(-np.logdet(a=S0Inv[trial,:,:])
                           -logdet(a=S1[trial,:,:])
                           +np.dot(np.flatten(S0inv[:,:trial]),
                                    np.flatten(ESS[:,:,trial]))
                           -ESS.shape[0])
            answer += trialKL
        return answer

class PointProcessExpectedLogLikelihood(ExpectedLogLikelihood):

    # def __init__(eLL, covMatricesStore, legQuadPoints, legQuadWeights, hermQuadPoints, hermQuadWeights, linkFunction):
    def __init__(legQuadPoints, legQuadWeights, hermQuadPoints, hermQuadWeights, linkFunction):
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
        self.__legQuadPoints = legQuadPoints
        self.__legQuadWeights = legQuadWeights
        self.__hermQuadPoints = hermQuadPoints
        self.__hermQuadWeights = hermQuadWeights
        self.__linkFunction = linkFunction

    def evalSumAcrossTrialsAndNeuronsWithGradientOnQ(qHMeanAtQuad, 
                                                      qHVarAtQuad,
                                                      qHMeanAtSpike, 
                                                      qHVarAtSpike,
            C, d, Kzzi, 
                                                      Kzz, KtzAtQuad, KttAtQuad,
                                                      KtzAtSpike, KttAtSpike):
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

        qH = PointProcessSparseVariationalProposal()
        qHMeanAtQuad, qHVarAtQuad = \
         qH.getMeanAndVarianceAtQuadPoints(qMu=qMu, qSigma=qSigma, 
                                                    C=C, d=d,
                                                    Kzzi=Kzzi,
                                                    Kzz=Kzz,
                                                    Ktz=KtzAtQuad,
                                                    Ktt=KttAtQuad)
        qHMeanAtSpike, qHVarAtSpike = \
             qH.getMeanAndVarianceAtSpikeTimes(qMu=qMu, qSigma=qSigma, 
                                                    C=C, d=d,
                                                    Kzzi=Kzzi,
                                                    Kzz=Kzz,
                                                    Ktz=KtzAtSpike,
                                                    Ktt=KttAtSpike)
        if self.__linkFunction == self.exp:
            intval = np.exp(qHMeanAtQuad + 0.5*qHVarAtQuad)
            logLink = qHMeanAtSpike
        else:
            # intval = permute(mtimesx(m.wwHerm',permute(m.link(qHmeanAtQuad + sqrt(2*qHMVarAtQuad).*permute(m.xxHerm,[2 3 4 1])),[4 1 2 3])),[2 3 4 1]);

            # aux2 \in nQuad x nNeuros x nTrials
            aux2 = np.sqrt(2*qHVarAtQuad)
            # aux3 \in nQuad x nNeuros x nTrials x trLen
            aux3 = np.multiply.outer(aux2, self.__hermQuadPoints)
            # aux4 \in nQuad x nNeuros x nTrials x trLen
            aux4 = np.add(aux3, qHMeanAtQuad)
            # aux5 \in nQuad x nNeuros x nTrials x trLen
            aux5 = self.__linkFunction(x=aux4)
            # intval \in nQuad x nNeuros x nTrials
            intval = np.tensordot(a=aux5, b=self.__hermQuadWeights, axes=([4], [0]))

            # log_link = cellvec(cellfun(@(x,y) log(m.link(x + sqrt(2*y).* m.xxHerm'))*m.wwHerm,mu_h_Spikes,var_h_Spikes,'uni',0));
            # aux1[trial] \in nSpikes[trial]
            aux1 = [2*qHVarAtSpike[trial] for trial in range(len(qHVarAtSpike))]
            # aux2[trial] \in nSpikes[trial] x nQuadLeg
            aux2 = [np.multiply.outer(aux1[trial], self.__hermQuadPoints) for trial in range(len(aux2))]
            # aux3[trial] \in nSpikes[trial] x nQuadLeg
            aux3 = [np.add(aux2[trial], qHMeanAtSpike[trial]) for trial in range(len(aux3))]
            # aux4[trial] \in nSpikes[trial] x nQuadLeg
            aux4 = [np.log(self.__linkFunction(x=aux3[trial])) for trial in range(len(aux4))]
            # logLink \in nSpikes[trial] x 1
            logLink = np.tensordot(a=aux4, b=self.__hermQuadWeights, axes=([1], [0]))

        # aux1 \in 1 x nNeurons x nTrials
        aux1 = np.matmul(np.transpose(a=self.__hermQuadWeights), intval)
        sELLTerm1 = np.sum(aux1)
        sELLTerm2 = np.sum(logLink)
        return -sELLTerm1+sELLTerm2

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
        svl = SparseVariationalLowerBound(y=y, eLL=self.__eLL, 
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
            svl.setVariationalProposalParams(flatteneQdParams=flattenedQParams)
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

    def __eStep(self, lowerBound, qMu, SVec, SDiag, maxIter, tol):
        x0 = lowerBound.flattenVariationalProposalParms(qMu=qMu, SVec=SVec, 
                                                                 SDiag=SDiag)
        res = minimize(lowerBound.evalWithGradientOnQ, x0=x0, 
                        method='BFGS', options={'xtol': tol, 'disp': True},
                        args=(len(m)))
        return res


class SparseVariationalLowerBound:

    def __init__(self, eLL, covMatricesStore, qMu, qSVec, qSDiag, 
                       C, d, kernelParms):
        self.__y = y
        self.__eLL = eLL
        self.__covMatricesStore = __covMaatricesStore
        self.__qMu = qMu
        self.__qSVec = qSVec
        self.__qSDiag = qSDiag
        self.__C = C
        self.__d = d
        self.__kernelParams = kernelParams

    def evalWithGradOnQ(self, x):
        '''
        Answers the value and gradient of the lower bound (Eq. (4) of Duncker and Sahani, 2018).

        Parameters
        ----------
        x : vector containing the concatenation of flattened qMu, qSVec and qSDiag.
        '''

        qMu, qSVec, qSDiag = \
         self.__unflattenVariationalProposalParams(flattenedQParams=x0)
        qSigma = self.__buildQSigma(qSVec=qSVec, qSDiag=qSDiag)
        qHMeanAtSpike, qHMVarAtSpike = qH.getMeanAndVarianceAtSpikeTimes()
    
        eLLEval = self.__eLL.evalWithGradOnQ(eLL=eLL, 
                                              covMatricesStore=
                                               covMatricesStore, 
                                              qMu=qMu, qSigma=qSigma,
                                              C=self.__C, d=self.__d,
                                              trials=trials)

        klDivEval = self.__evalKLDivergenceTerm(Kzzi=Kzzi, qMu=qMu, 
                                                           qSigma=qSigma)

        answer = eLLEval - klDivEval
        return answer

    def setVariationalProposalParams(self, flattenedQParams):
        self.__qMu, self.__qSVec, self.__qSDiag = \
         self.__unflattenVariationalProposalParams(flattenedQParams=
                                                    flattenedQParams)

    def flattenVariationalProposalParams(qMu, qSVec, qSDiag):
        return self.__flattenArrays(qMu, qSVec, qSDiag)
    '''
    def __unflattenVariationalProposalParams(flattenedQParams):
        fromIndex = 0
        qMu = np.reshape(a=flattenedQParmas[fromIndex+(0:self.__qMu.size)],
                          newshape=self.__qMu.shape)
        fromIndex += self.__qMu.size
        qSVec = np.reshape(a=flattenedQParmas[fromIndex+(0:self.__qSVec.size)],
                            newshape=self.__qSVec.shape)
        fromIndex += self.__qSVec.size
        qSVec = np.reshape(a=flattenedQParmas[fromIndex+(0:self.__qSDiag.size)],
                            newshape=self.__qSDiag.shape)
        return qMu, qSVec, qSDiag
    '''
    def __flattenArrays(self, *args):
        answer = []
        for arg in args:
            answer.append(arg.flatten)
        return answer

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
        return qHMu, qHVar, qKMu, qKVar
