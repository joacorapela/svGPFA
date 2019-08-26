
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
from core import SparseVariationalLowerBound, CovarianceMatricesStore, PointProcessExpectedLogLikelihood

def test_buildQSigma():
    tol = 1e-5
    dataFilename = os.path.expanduser('~/dev/research/gatsby/svGPFA/code/test/data/get_full_from_lowplusdiag.mat')

    mat = loadmat(dataFilename)
    q_sqrt = [mat['q_sqrt'][(0,i)].transpose((2,0,1)) for i in range(mat['q_sqrt'].shape[1])]
    q_diag = [mat['q_diag'][(0,i)].transpose((2,0,1)) for i in range(mat['q_diag'].shape[1])]
    q_sigma = [mat['q_sigma'][(0,i)].transpose((2,0,1)) for i in range(mat['q_sigma'].shape[1])]

    svLowerBound = SparseVariationalLowerBound(eLL=None, covMatricesStore=None, qMu=None, qSVec=None, qSDiag=None, C=None, d=None, kernelParams=None, varRnk=np.ones(shape=3, dtype=np.int8))
    qSigma = svLowerBound._SparseVariationalLowerBound__buildQSigma(qSVec=q_sqrt, qSDiag=q_diag)

    error = np.array([np.linalg.norm(x=qSigma[k]-q_sigma[k]) for k in range(len(qSigma))]).sum()

    assert(error<tol)

def test_flattenUnFlattenVariationalProposalParams():
    tol = 1e-5
    dataFilename = os.path.expanduser('~/dev/research/gatsby/svGPFA/code/test/data/Estep_Objective_PointProcess_svGPFA.mat')

    mat = loadmat(dataFilename)
    qMu0 = [mat['q_mu'][(0,i)].transpose((2,0,1)) for i in range(mat['q_mu'].shape[1])]
    qSVec0 = [mat['q_sqrt'][(0,i)].transpose((2,0,1)) for i in range(mat['q_sqrt'].shape[1])]
    qSDiag0 = [mat['q_diag'][(0,i)].transpose((2,0,1)) for i in range(mat['q_diag'].shape[1])]

    legQuadPoints = np.transpose(mat['ttQuad'], (2, 0, 1))
    legQuadWeights = np.transpose(mat['wwQuad'], (2, 0, 1))
    hermQuadPoints = mat['xxHerm']
    hermQuadWeights = mat['wwHerm']

    q_sqrt = [mat['q_sqrt'][(0,i)].transpose((2,0,1)) for i in range(mat['q_sqrt'].shape[1])]
    q_sqrt = [mat['q_sqrt'][(0,i)].transpose((2,0,1)) for i in range(mat['q_sqrt'].shape[1])]
    q_diag = [mat['q_diag'][(0,i)].transpose((2,0,1)) for i in range(mat['q_diag'].shape[1])]

    Kzz = [mat['Kzz'][(i,0)].transpose((2,0,1)) for i in range(mat['Kzz'].shape[0])]
    Kzzi = [mat['Kzzi'][(i,0)].transpose((2,0,1)) for i in range(mat['Kzzi'].shape[0])]
    quadKtz= mat['quadKtz']
    quadKtt= mat['quadKtt']
    spikeKtz= mat['spikeKtz']
    spikeKtt= mat['spikeKtt']

    C = mat['C']
    b = np.squeeze(mat['b'])

    varRnk = mat['varRnk']

    funEvalPrs0 = mat['funEvalPrs0']

    linkFunction = np.exp

    covMatricesStore = CovarianceMatricesStore(Kzz=Kzz, 
                                                Kzzi=Kzzi,
                                                quadKtz=quadKtz, 
                                                quadKtt=quadKtt, 
                                                spikeKtz=spikeKtz,
                                                spikeKtt=spikeKtt)
    ppELL = PointProcessExpectedLogLikelihood(legQuadPoints=legQuadPoints, 
                                               legQuadWeights=legQuadWeights, 
                                               hermQuadPoints=hermQuadPoints, 
                                               hermQuadWeights=hermQuadWeights, 
                                               linkFunction=linkFunction)
    svlb = SparseVariationalLowerBound(eLL=ppELL, 
                                        covMatricesStore=covMatricesStore,
                                        qMu=qMu0, 
                                        qSVec=qSVec0,
                                        qSDiag=qSDiag0, 
                                        C=C, d=b,
                                        kernelParams=None,
                                        varRnk=varRnk)
    x = svlb.flattenVariationalProposalParams(qMu=qMu0, qSVec=qSVec0, 
                                                        qSDiag=qSDiag0)
    uQMu, uQSVec, uQSDiag = svlb._SparseVariationalLowerBound__unflattenVariationalProposalParams(flattenedQParams=x)

    for k in range(len(qMu0)):
        assert(np.array_equal(qMu0[k], uQMu[k]))

    for k in range(len(qSVec0)):
        assert(np.array_equal(qSVec0[k], uQSVec[k]))

    for k in range(len(qSDiag0)):
        assert(np.array_equal(qSDiag0[k], uQSDiag[k]))

def test_evalWithGradOnQ():
    tol = 1e-5
    dataFilename = os.path.expanduser('~/dev/research/gatsby/svGPFA/code/test/data/Estep_Objective_PointProcess_svGPFA.mat')

    mat = loadmat(dataFilename)
    qMu0 = [mat['q_mu'][(0,i)].transpose((2,0,1)) for i in range(mat['q_mu'].shape[1])]
    qSVec0 = [mat['q_sqrt'][(0,i)].transpose((2,0,1)) for i in range(mat['q_sqrt'].shape[1])]
    qSDiag0 = [mat['q_diag'][(0,i)].transpose((2,0,1)) for i in range(mat['q_diag'].shape[1])]
    legQuadPoints = np.transpose(mat['ttQuad'], (2, 0, 1))
    legQuadWeights = np.transpose(mat['wwQuad'], (2, 0, 1))
    hermQuadPoints = mat['xxHerm']
    hermQuadWeights = mat['wwHerm']

    q_sqrt = [mat['q_sqrt'][(0,i)].transpose((2,0,1)) for i in range(mat['q_sqrt'].shape[1])]
    q_sqrt = [mat['q_sqrt'][(0,i)].transpose((2,0,1)) for i in range(mat['q_sqrt'].shape[1])]
    q_diag = [mat['q_diag'][(0,i)].transpose((2,0,1)) for i in range(mat['q_diag'].shape[1])]

    Kzz = [mat['Kzz'][(i,0)].transpose((2,0,1)) for i in range(mat['Kzz'].shape[0])]
    Kzzi = [mat['Kzzi'][(i,0)].transpose((2,0,1)) for i in range(mat['Kzzi'].shape[0])]
    quadKtz= [mat['quadKtz'][(i,0)].transpose((2,0,1)) for i in range(mat['quadKtz'].shape[0])]
    quadKtt= mat['quadKtt'].transpose((2,0,1))
    spikeKtz= mat['spikeKtz']
    spikeKtt= mat['spikeKtt']

    C = mat['C']
    b = np.squeeze(mat['b'])

    varRnk = mat['varRnk'][0]
    index = [mat['index'][i,0][:,0] for i in range(mat['index'].shape[0])]

    funEvalPrs0 = mat['funEvalPrs0'][0,0]

    linkFunction = np.exp

    covMatricesStore = CovarianceMatricesStore(Kzz=Kzz, 
                                                Kzzi=Kzzi,
                                                quadKtz=quadKtz, 
                                                quadKtt=quadKtt, 
                                                spikeKtz=spikeKtz,
                                                spikeKtt=spikeKtt)
    ppELL = PointProcessExpectedLogLikelihood(legQuadPoints=legQuadPoints, 
                                               legQuadWeights=legQuadWeights, 
                                               hermQuadPoints=hermQuadPoints, 
                                               hermQuadWeights=hermQuadWeights, 
                                               linkFunction=linkFunction)
    svlb = SparseVariationalLowerBound(eLL=ppELL, 
                                        covMatricesStore=covMatricesStore,
                                        qMu=qMu0, 
                                        qSVec=qSVec0,
                                        qSDiag=qSDiag0, 
                                        C=C, d=b,
                                        kernelParams=None,
                                        varRnk=varRnk,
                                        neuronForSpikeIndex=index)
    x = svlb.flattenVariationalProposalParams(qMu=qMu0, qSVec=qSVec0, 
                                                        qSDiag=qSDiag0)
    evalRes = svlb.evalWithGradOnQ(x=x)

    assert(abs(evalRes-funEvalPrs0)<tol)
    
    pdb.set_trace()

if __name__=='__main__':
    # test_buildQSigma()
    # test_flattenUnFlattenVariationalProposalParams()
    test_evalWithGradOnQ()
