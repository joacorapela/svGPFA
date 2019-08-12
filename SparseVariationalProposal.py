
class SparseVariationalProposal:

    def getMeanAndVarianceAtQuadPoints(self, qMu, qSigma, C, d, Kzzi, Ktz, Ktt):
        '''
        Answers the mean and variance in Eq. (5) of Dunker and Sahani, 2018.

        Parameters
        ----------
        qMu : list of length nLatent. 
              qMu[k] \in  nTrials x nInducingPoints[k] x 1
        qSigma : list of length nLatent. 
                 qSigma[k] \in nTrials x nInducingPoints[k] x nInducingPoints[k]
        C : matrix \in nNeurons x nLatent
        d : vector \in nNeurons
        Kzzi : list of length nLatent. 
               Kzzi[k] \in nTrials x nInducingPoints[k] x nInducingPoints[k]
        Ktz : list of length nLatent. 
              Ktz[k] \in nTrials x nQuad x nInducingPoints[k]
        Ktt: \in nTrials x nQuad x nLatent

        Returns
        -------
        qHMu : array \in nTrials x nQuad x nLatent
        qHVar : array \in nTrials x nQuad x nLatent
        '''

        nTrials = qMu.shape[0]
        nQuad = Ktz.shape[1]
        nLatent = C.shape[1]

        muK = np.empty((nTrials, nQuad, nLatent))
        sigmaK = np.empty((nTrials, nQuad, nLatent))

        for k = 1:len(qMu):
            # Ak \in nTrials x nInducingPoints[k] x 1 
            Ak = np.matmul(x=Kzzi[k], x=qMu[k]) 
            muK[:,:,k] = np.matmul(x1=Ktz[k], x2=Ak)
            # Bkf \in nTrials x nInducingPoints[k] x nQuad
            Bkf = np.matmul(x1=Kzzi[k], x2=np.transpose(a=Ktz, axes=(0, 2, 1)))
            # mm1f \in nTrials x nInducingPoints[k] x nQuad
            mm1f = np.matmul(x1=qSigma[k]-Kzz[k], x2=Bkf)

            # aux1 \in nTrials x nInducingPoints[k] x nQuad
            aux1 = Bkf*mm1f
            # aux2 \in nTrials x nQuad
            aux2 = np.sum(a=aux1, axis=1)
            # aux3 \in nTrials x nQuad
            aux3 = Ktt[:,:,k]+aux2
            # varK \in nTrials x nQuad x nLatent
            varK[:,:,k] = aux3
            # varK[:,:,k] = Ktt[:,:,k]+np.tensordot(a=Bkf, b=mm1f, axes=([1], [1]))

        qHMu = np.matmul(x1=muK, x2=C.T) + \
                np.reshape(a=b, newshape=(1, len(b), 1)) # using broadcasting
        qHVar = np.matmul(x1=varK, x2=(C.T)**2)
        return (qHMu, qHVar)
