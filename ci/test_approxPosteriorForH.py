
import sys
import pdb
import os
import math
from scipy.io import loadmat
import numpy as np
import torch
from approxPosteriorForH import ApproxPosteriorForHForAllNeuronsAllTimes, ApproxPosteriorForHForAllNeuronsAssociatedTimes
from inducingPointsPrior import InducingPointsPrior
from kernelMatricesStore import KernelMatricesStore
from kernels import PeriodicKernel, ExponentialQuadraticKernel

def test_getMeansAndVariances_allNeuronsAllTimes():
    tol = 5e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat["Z"].shape[0]
    nTrials = mat["Z"][0,0].shape[2]
    qMu = [torch.from_numpy(mat["q_mu"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec = [torch.from_numpy(mat["q_sqrt"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag = [torch.from_numpy(mat["q_diag"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    t = torch.from_numpy(mat["ttQuad"]).type(torch.DoubleTensor).permute(2, 0, 1)
    Z = [torch.from_numpy(mat["Z"][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Y = [torch.from_numpy(mat["Y"][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor)
    mu_h = torch.from_numpy(mat["mu_h_Quad"]).type(torch.DoubleTensor).permute(2,0,1)
    var_h = torch.from_numpy(mat["var_h_Quad"]).type(torch.DoubleTensor).permute(2,0,1)

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
    kernels = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = PeriodicKernel(scale=1.0, lengthScale=float(hprs[k,0][0]), period=float(hprs[k,0][1]))
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0, lengthScale=float(hprs[k,0][0]))
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))


    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))

    kernelMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)
    qH = ApproxPosteriorForHForAllNeuronsAllTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore)
    qHMu, qHVar = qH.getMeansAndVariances()

    qHMuError = math.sqrt(((mu_h-qHMu)**2).mean())
    qHVarError = math.sqrt(((var_h-qHVar)**2).mean())

    assert(qHMuError<tol)
    assert(qHVarError<tol)

def test_buildKFactors_allNeuronsAllTimes():
    tol = 5e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat["Z"].shape[0]
    nTrials = mat["Z"][0,0].shape[2]
    qMu = [torch.from_numpy(mat["q_mu"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec = [torch.from_numpy(mat["q_sqrt"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag = [torch.from_numpy(mat["q_diag"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    t = torch.from_numpy(mat["ttQuad"]).type(torch.DoubleTensor).permute(2, 0, 1)
    Z = [torch.from_numpy(mat["Z"][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Y = [torch.from_numpy(mat["Y"][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor)
    mu_k = torch.from_numpy(mat["mu_k_Quad"]).type(torch.DoubleTensor).permute(2,0,1)
    var_k = torch.from_numpy(mat["var_k_Quad"]).type(torch.DoubleTensor).permute(2,0,1)

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
    kernels = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = PeriodicKernel(scale=1.0, lengthScale=float(hprs[k,0][0]), period=float(hprs[k,0][1]))
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0, lengthScale=float(hprs[k,0][0]))
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))


    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))

    kernelMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)
    qH = ApproxPosteriorForHForAllNeuronsAllTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore)
    kFactors = qH.buildKFactors()
    qKMu = kFactors["qKMu"]
    qKVar = kFactors["qKVar"]

    qKMuError = math.sqrt(((mu_k-qKMu)**2).mean())
    assert(qKMuError<tol)
    qKVarError = math.sqrt(((var_k-qKVar)**2).mean())
    assert(qKVarError<tol)

def test_getMeansAndVariances_allNeuronsAssociatedTimes():
    tol = 1e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat["Z"].shape[0]
    nTrials = mat["Z"][0,0].shape[2]
    qMu = [torch.from_numpy(mat["q_mu"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec = [torch.from_numpy(mat["q_sqrt"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag = [torch.from_numpy(mat["q_diag"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    t = torch.from_numpy(mat["ttQuad"]).type(torch.DoubleTensor).permute(2, 0, 1)
    Y = [torch.from_numpy(mat["Y"][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    Z = [torch.from_numpy(mat["Z"][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor)
    mu_k = [torch.from_numpy(mat["mu_k_Spikes"][0,tr]).type(torch.DoubleTensor) for tr in range(nTrials)]
    var_k = [torch.from_numpy(mat["var_k_Spikes"][0,tr]).type(torch.DoubleTensor) for tr in range(nTrials)]
    index = [torch.from_numpy(mat["index"][i,0][:,0]).type(torch.ByteTensor) for i in range(nTrials)]

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
    kernels = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = PeriodicKernel(scale=1.0, lengthScale=float(hprs[k,0][0]), period=float(hprs[k,0][1]))
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0, lengthScale=float(hprs[k,0][0]))
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))


    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))

    kernelMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)
    qH = ApproxPosteriorForHForAllNeuronsAssociatedTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore, neuronForSpikeIndex=index)
    kFactors = qH.buildKFactors()
    qKMu = kFactors["qKMu"]
    qKVar = kFactors["qKVar"]

    for tr in range(nTrials):
        qKMuError = math.sqrt(((mu_k[tr]-qKMu[tr])**2).mean())
        assert(qKMuError<tol)
        qKVarError = math.sqrt(((var_k[tr]-qKVar[tr])**2).mean())
        assert(qKVarError<tol)

def test_predict_allNeuronsAllTimes():
    tol = 5e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/predictNew_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat["Z"].shape[0]
    nTrials = mat["Z"][0,0].shape[2]
    qMu = [torch.from_numpy(mat["q_mu"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec = [torch.from_numpy(mat["q_sqrt"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag = [torch.from_numpy(mat["q_diag"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    t = torch.from_numpy(mat["ttQuad"]).type(torch.DoubleTensor).permute(2, 0, 1)
    Z = [torch.from_numpy(mat["Z"][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Y = [torch.from_numpy(mat["Y"][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor)
    mu_k = torch.from_numpy(mat["muK"]).type(torch.DoubleTensor).permute(2,0,1)
    var_k = torch.from_numpy(mat["varK"]).type(torch.DoubleTensor).permute(2,0,1)
    mu_h = torch.from_numpy(mat["muH"]).type(torch.DoubleTensor).permute(2,0,1)
    var_h = torch.from_numpy(mat["varH"]).type(torch.DoubleTensor).permute(2,0,1)
    testTimes = torch.from_numpy(mat["testTimes"]).type(torch.DoubleTensor)

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
    kernels = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = PeriodicKernel(scale=1.0, lengthScale=float(hprs[k,0][0]), period=float(hprs[k,0][1]))
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0, lengthScale=float(hprs[k,0][0]))
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))


    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
    kernelMatricesStore= KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)
    qH_allNeuronsAllTimes = ApproxPosteriorForHForAllNeuronsAllTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore)

    qHMu, qHVar, qKMu, qKVar = qH_allNeuronsAllTimes.predict(testTimes=testTimes)

    qKMuError = math.sqrt(((mu_k-qKMu)**2).mean())
    assert(qKMuError<tol)
    qKVarError = math.sqrt(((var_k-qKVar)**2).mean())
    assert(qKVarError<tol)
    qHMuError = math.sqrt(((mu_h-qHMu)**2).mean())
    assert(qHMuError<tol)
    qHVarError = math.sqrt(((var_h-qHVar)**2).mean())
    assert(qHVarError<tol)

    # pdb.set_trace()

if __name__=="__main__":
    # test_getMeansAndVariances_allNeuronsAllTimes()
    # test_buildKFactors_allNeuronsAllTimes()
    # test_getMeansAndVariances_allNeuronsAssociatedTimes()
    # test_buildKFactors_allNeuronsAssociatedTimes()
    test_predict_allNeuronsAllTimes()
