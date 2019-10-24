
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
import torch
from kernels import PeriodicKernel, ExponentialQuadraticKernel
from kernelMatricesStore import IndPointsLocsKMS, IndPointsLocsAndAllTimesKMS,\
                                IndPointsLocsAndAssocTimesKMS
from svPosteriorOnIndPoints import SVPosteriorOnIndPoints
from svPosteriorOnLatents import SVPosteriorOnLatentsAllTimes,\
                                 SVPosteriorOnLatentsAssocTimes
from svEmbedding import LinearSVEmbeddingAllTimes, LinearSVEmbeddingAssocTimes
from expectedLogLikelihood import PointProcessELLExpLink, PointProcessELLQuad

def test_evalSumAcrossTrialsAndNeurons_pointProcessExpLink():
    tol = 3e-4
    yNonStackedFilename = os.path.expanduser("data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(yNonStackedFilename)
    YNonStacked = mat['YNonStacked']

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    t = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
    hermQuadWeights = torch.from_numpy(mat['wwHerm']).type(torch.DoubleTensor)
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    Elik = torch.from_numpy(mat['Elik'])

    linkFunction = torch.exp

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
    kernels = [[None] for k in range(nLatents)]
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = PeriodicKernel(scale=1.0)
            kernelsParams0[k] = {"lengthScale": float(hprs[k,0][0]),
                                    "period": float(hprs[k,0][1])}
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0)
            kernelsParams0[k] = {"lengthScale": float(hprs[k,0][0])}
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))
    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    qHParams0 = {"C0": C0, "d0": b0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    initialParams = {"svPosteriorOnIndPoints": qUParams0,
                     "kernelsMatricesStore": kmsParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints, 
                  "legQuadWeights": legQuadWeights}

    qU = SVPosteriorOnIndPoints()
    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsAndAllTimesKMS = IndPointsLocsAndAllTimesKMS()
    indPointsLocsAndAssocTimesKMS = IndPointsLocsAndAssocTimesKMS()
    qKAllTimes = SVPosteriorOnLatentsAllTimes(
        svPosteriorOnIndPoints=qU, 
        indPointsLocsKMS=indPointsLocsKMS, 
        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
    qKAssocTimes = SVPosteriorOnLatentsAssocTimes(
        svPosteriorOnIndPoints=qU, 
        indPointsLocsKMS=indPointsLocsKMS, 
        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
    qHAllTimes = LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qKAllTimes)
    qHAssocTimes = LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=
                                               qKAssocTimes)
    eLL = PointProcessELLExpLink(svEmbeddingAllTimes=qHAllTimes,
                                 svEmbeddingAssocTimes=qHAssocTimes)

    eLL.setKernels(kernels=kernels)
    eLL.setInitialParams(initialParams=initialParams)
    eLL.setMeasurements(measurements=YNonStacked)
    eLL.setQuadParams(quadParams=quadParams)
    eLL.buildKernelsMatrices()
    sELL = eLL.evalSumAcrossTrialsAndNeurons()

    sELLerror = abs(sELL-Elik)

    assert(sELLerror<tol)

def test_evalSumAcrossTrialsAndNeurons_pointProcessQuad():
    tol = 3e-4
    yNonStackedFilename = os.path.expanduser("data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(yNonStackedFilename)
    YNonStacked = mat['YNonStacked']

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    t = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
    hermQuadWeights = torch.from_numpy(mat['wwHerm']).type(torch.DoubleTensor)
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    Elik = torch.from_numpy(mat['Elik'])

    linkFunction = torch.exp

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
    kernels = [[None] for k in range(nLatents)]
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = PeriodicKernel(scale=1.0)
            kernelsParams0[k] = {"lengthScale": float(hprs[k,0][0]),
                                    "period": float(hprs[k,0][1])}
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0)
            kernelsParams0[k] = {"lengthScale": float(hprs[k,0][0])}
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))
    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    qHParams0 = {"C0": C0, "d0": b0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    initialParams = {"svPosteriorOnIndPoints": qUParams0,
                     "kernelsMatricesStore": kmsParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"hermQuadPoints": hermQuadPoints, 
                  "hermQuadWeights": hermQuadWeights,
                  "legQuadPoints": legQuadPoints, 
                  "legQuadWeights": legQuadWeights}


    qU = SVPosteriorOnIndPoints()
    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsAndAllTimesKMS = IndPointsLocsAndAllTimesKMS()
    indPointsLocsAndAssocTimesKMS = IndPointsLocsAndAssocTimesKMS()
    qKAllTimes = SVPosteriorOnLatentsAllTimes(
        svPosteriorOnIndPoints=qU, 
        indPointsLocsKMS=indPointsLocsKMS, 
        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
    qKAssocTimes = SVPosteriorOnLatentsAssocTimes(
        svPosteriorOnIndPoints=qU, 
        indPointsLocsKMS=indPointsLocsKMS, 
        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
    qHAllTimes = LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qKAllTimes)
    qHAssocTimes = LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=
                                               qKAssocTimes)
    eLL = PointProcessELLQuad(svEmbeddingAllTimes=qHAllTimes,
                              svEmbeddingAssocTimes=qHAssocTimes,
                              linkFunction=torch.exp)

    eLL.setKernels(kernels=kernels)
    eLL.setInitialParams(initialParams=initialParams)
    eLL.setMeasurements(measurements=YNonStacked)
    eLL.setQuadParams(quadParams=quadParams)
    eLL.buildKernelsMatrices()
    sELL = eLL.evalSumAcrossTrialsAndNeurons()

    sELLerror = abs(sELL-Elik)

    assert(sELLerror<tol)

"""

def test_evalSumAcrossTrialsAndNeurons_poisson():
    tol = 3e-4
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatent = len(mat['q_mu'])
    nTrials = mat['q_mu'][0,0].shape[2]
    nNeurons = mat['Y'].shape[0]
    qMu = [torch.from_numpy(mat['q_mu'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qSVec = [torch.from_numpy(mat['q_sqrt'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qSDiag = [torch.from_numpy(mat['q_diag'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
    hermQuadWeights = torch.from_numpy(mat['wwHerm']).type(torch.DoubleTensor)
    t_tmp = torch.from_numpy(mat['tt']).type(torch.DoubleTensor).squeeze()
    Z = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    Y = torch.from_numpy(mat['Y']).type(torch.DoubleTensor).permute(2,0,1) 
    binWidth = torch.from_numpy(mat['BinWidth'])
    Elik = torch.from_numpy(mat['Elik'])
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    # t_tmp \in nQuad and we want t \in nTrials x nQuad x 1
    t = torch.ger(input=torch.ones(nTrials, dtype=torch.double), vec2=t_tmp).unsqueeze(dim=2)

    linkFunction = torch.exp

    kernels = [[None] for k in range(nLatent)]
    for k in range(nLatent):
        if np.char.equal(kernelNames[0,k][0], 'PeriodicKernel'):
            kernels[k] = PeriodicKernel(scale=1.0, lengthScale=float(hprs[k,0][0]), period=float(hprs[k,0][1]))
        elif np.char.equal(kernelNames[0,k][0], 'rbfKernel'):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0, lengthScale=float(hprs[k,0][0]))
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
    kernelMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)
    qH = ApproxPosteriorForHForAllNeuronsAllTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore)

    eLL = PoissonExpectedLogLikelihood(approxPosteriorForHForAllNeuronsAllTimes=qH, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, linkFunction=linkFunction, Y=Y, binWidth=binWidth)
    sELL = eLL.evalSumAcrossTrialsAndNeurons()

    sELLerror = abs(sELL-Elik)

    assert(sELLerror<tol)

def test_evalSumAcrossTrialsAndNeurons_withKFactors_pointProcess():
    tol = 3e-4
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    t = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    Z = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Y = [torch.from_numpy(mat['Y'][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    index = [torch.from_numpy(mat['index'][i,0][:,0]).type(torch.ByteTensor) for i in range(nTrials)]
    hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
    hermQuadWeights = torch.from_numpy(mat['wwHerm']).type(torch.DoubleTensor)
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    Elik = torch.from_numpy(mat['Elik'])

    linkFunction = torch.exp

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
    kernels = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], 'PeriodicKernel'):
            kernels[k] = PeriodicKernel(scale=1.0, lengthScale=float(hprs[k,0][0]), period=float(hprs[k,0][1]))
        elif np.char.equal(kernelNames[0,k][0], 'rbfKernel'):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0, lengthScale=float(hprs[k,0][0]))
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
    kernelMatricesStore= KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)

    qH_allNeuronsAllTimes = ApproxPosteriorForHForAllNeuronsAllTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore)
    qH_allNeuronsAssociatedTimes = ApproxPosteriorForHForAllNeuronsAssociatedTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore, neuronForSpikeIndex=index)

    eLL = PointProcessExpectedLogLikelihood(approxPosteriorForHForAllNeuronsAllTimes=qH_allNeuronsAllTimes, approxPosteriorForHForAllNeuronsAssociatedTimes=qH_allNeuronsAssociatedTimes, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, legQuadPoints=legQuadPoints, legQuadWeights=legQuadWeights, linkFunction=linkFunction)
    kFactors = eLL.buildKFactors()
    sELL_withoutKFactors = eLL.evalSumAcrossTrialsAndNeurons()
    sELL_withKFactors = eLL.evalSumAcrossTrialsAndNeurons(kFactors=kFactors)

    error = abs(sELL_withoutKFactors-sELL_withKFactors)

    assert(error<tol)

"""

if __name__=="__main__":
    test_evalSumAcrossTrialsAndNeurons_pointProcessExpLink()
    test_evalSumAcrossTrialsAndNeurons_pointProcessQuad()
    # test_evalSumAcrossTrialsAndNeurons_pointProcess()
    # test_evalSumAcrossTrialsAndNeurons_poisson()
    # test_evalSumAcrossTrialsAndNeurons_withKFactors_pointProcess()
