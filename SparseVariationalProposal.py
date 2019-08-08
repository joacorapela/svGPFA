
class SparseVariationalProposal:

    def getMeanAndVarianceAtQuadPoints(self, qMu, qSigma, C, d, Kzzi, Ktz, Ktt):
        '''
        Answers the mean and variance in Eq. (5) of Dunker and Sahani, 20018.

        Parameters
        ----------
        qMu : list of length nLatent. 
              qMu[k] \in  nTrials x nInducingPoints[k] x 1
        qSigma : list of length nLatent. 
                 qSigma[k] \in nTrials x nInducingPoints[k] x nInducingPoints[k]
        C : matrix \in nNeurons x nLatent
        d : vector \in nLatent
        Kzzi : list of length nLatent. 
               Kzzi[k] \in nTrials x nInducingPoints[k] x nInducingPoints[k]
        Ktz : list of length nLatent. 
              Ktz[k] \in nTrials x nQuad x nInducingPoints[k]

        Returns
        -------
        qHMu : array \in nTrials x nQuad x nNeurons
        qHVar : array \in nTrials x nQuad x nNeurons
        '''

        nTrials = qMu.shape[0]
        nInducingPoints = qMu.shape[1]
        nQuad = Ktz.shape[1]
        nNeurons = C.shape[0]
        nLatent = C.shape[1]

        muK = np.empty((nTrials, nQuad, nLatent))
        sigmaK = np.empty((nTrials, nQuad, nLatent))

        for k = 1:len(qMu):
            Ak = np.matmul(Kzzi[k], qMu[k]) # \in nTrials x nInducingPoints[k] x 1
            muK[:,:,k] = np.matmul(Ktz[k], Ak) # \in nTrials x nQuad x 1
    def evalWithGradient(self, c, z, m, d):
