
def class SparseVariationalLowerBound:

    def __init__(y, eLL, m0, SVec0=SVec0, SDiag0=SDiag0, c0=c0, d0=d0,
                                               kernelParms0=kernelParms0,
                                               z0=z0)
    def evalWithGradOnQ(self, x):
        m = x[
        qH = SparseVariationalProposal()
        qHMeanAtQuad, qHMVarAtQuad = qH.getMeanAndVarianceAtQuadPoints()
        qHMeanAtSpikes, qHMVarAtSpikes = qH.getMeanAndVarianceAtSpikeTimes()
    
        eLLEval = self.__eLL.evalWithGradOnQ(qHMeanAtQuad=qHMeanAtQuad, 
                                              qHVarAtQuad=qHVarAtQuad, 
                                              qHMeanAtSpikes=qHMeanAtSpikes, 
                                              qHVarAtSpikes=qHVarAtSpikes,
                                              trials=trials)

        klDivEval = self.__evalKLDivergenceTerm(Kzzi, qMu, qSigma)

        answer = eLLEval - klDivEval
        return answer

    def __evalKLDivergenceTerm(Kzzi, qMu, qSigma):
        term = 0
        nLatents = Kzzi.dim[2]
        for k in range(nLatents):
            term += klDivergence.evalWithGradientOnQ(S0Inv=Kzzi[:,:,k],
                                                      mu1=qMu[:,k],
                                                      S1=qSigma[:,:,k])
        return term
