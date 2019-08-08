
from scipy.optimize import minimize

def class SparseVariationalEM:

    def maximize(self, y, eLL, g, m0, SVec0, SDiag0, c0, d0, kernelParms0, z0, 
                       maxEMIter, maxEStepIter, maxMStepModelParamsIter,
                       maxMStepKernelParmsIter, maxMStepInducingPointsIter, 
                       tol):
        '''
        m0, SVec0, SDiag0 \in nInducingPoints x nNeruons x nTrials
        '''
        iter = 0
        q = VariationalProposal()
        svl = SparseVariationalLowerBound(y=y, eLL=eLL,
                                               m0=m0, SVec0=SVec0,
                                               SDiag0=SDiag0, 
                                               c0=c0, d0=d0,
                                               kernelParms0=kernelParms0,
                                               z0=z0)
        m = m0
        SVec = SVec0
        SDiag = SDiag0
        while iter<maxEMIter:
            qParams = self.__eStep(lowerBound=svL, m=m, SVec=SVec, sDiag=sDiag, 
                                     maxIter=maxEStepIter, tol=tol)
            vL.setQParams(qParams=qParams)
            
            modelParams = self.__mStepModelParams(nInducingPoints=nInducingPoints, 
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

    def __eStep(self, lowerBound, m, SVec, SDiag, maxIter, tol):
        x0 = m.append(SVec, SDiag)
        res = minimize(lowerBound.evalWithGradientOnQ, x0=x0, 
                        method='BFGS', options={'xtol': tol, 'disp': True})
