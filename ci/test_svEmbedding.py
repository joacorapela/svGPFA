
import sys
import pdb
import os
import math
from scipy.io import loadmat
import numpy as np
import scipy.optimize
import torch
sys.path.append("../src")
import utils.svGPFA.initUtils
import stats.kernels
import stats.svGPFA.kernelsMatricesStore
import stats.svGPFA.svPosteriorOnIndPoints
import stats.svGPFA.svPosteriorOnLatents
import stats.svGPFA.svEmbedding
import stats.svGPFA.expectedLogLikelihood
import stats.svGPFA.klDivergence
import stats.svGPFA.svLowerBound

def test_get_flattened_params():
    svEmbedding = stats.svGPFA.svEmbedding.LinearSVEmbeddingAllTimes(svPosteriorOnLatents=None)
    C0 = torch.tensor([[1.,2.,3.],[4.,5.,6.]], dtype=torch.double)
    d0 = torch.tensor([[1.,1.,1.]], dtype=torch.double)
    svEmbedding._C = C0
    svEmbedding._d = d0
    true_flattened_params = torch.cat((C0.flatten(), d0.flatten()))
    flattened_params = svEmbedding.get_flattened_params()
    assert(torch.all(torch.eq(true_flattened_params, flattened_params)))

def test_set_params_from_flattened():
    svEmbedding = stats.svGPFA.svEmbedding.LinearSVEmbeddingAllTimes(svPosteriorOnLatents=None)
    C0 = torch.tensor([[1.,2.,3.],[4.,5.,6.]], dtype=torch.double)
    d0 = torch.tensor([[1.,1.,1.]], dtype=torch.double)
    svEmbedding._C = C0
    svEmbedding._d = d0
    C1 = torch.tensor([[10.,20.,30.],[40.,50.,60.]], dtype=torch.double)
    d1 = torch.tensor([[10.,10.,10.]], dtype=torch.double)
    to_set_flattened_params = torch.cat((C1.flatten(), d1.flatten()))
    svEmbedding.set_params_from_flattened(flattened_params=to_set_flattened_params)
    flattened_params = svEmbedding.get_flattened_params()
    torch.all(torch.eq(to_set_flattened_params, flattened_params))

def test_set_params_requires_grad():
    svEmbedding = stats.svGPFA.svEmbedding.LinearSVEmbeddingAllTimes(svPosteriorOnLatents=None)
    C0 = torch.tensor([[1.,2.,3.],[4.,5.,6.]], dtype=torch.double)
    d0 = torch.tensor([[1.,1.,1.]], dtype=torch.double)
    svEmbedding._C = C0
    svEmbedding._d = d0
    svEmbedding.set_params_requires_grad(requires_grad=True)
    assert(C0.requires_grad and d0.requires_grad)
    svEmbedding.set_params_requires_grad(requires_grad=False)
    assert(not C0.requires_grad and not d0.requires_grad)

def test_computeMeansAndVars_allTimes():
    tol = 5e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat["Z"].shape[0]
    nTrials = mat["Z"][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat["q_mu"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat["q_sqrt"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat["q_diag"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    t = torch.from_numpy(mat["ttQuad"]).type(torch.DoubleTensor).permute(2, 0, 1)
    Z0 = [torch.from_numpy(mat["Z"][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Y = [torch.from_numpy(mat["Y"][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor)
    mu_h = torch.from_numpy(mat["mu_h_Quad"]).type(torch.DoubleTensor).permute(2,0,1)
    var_h = torch.from_numpy(mat["var_h_Quad"]).type(torch.DoubleTensor).permute(2,0,1)
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    kernels = [[None] for k in range(nLatents)]
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = stats.kernels.PeriodicKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]), 
                                              float(hprs[k,0][1])], 
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = stats.kernels.ExponentialQuadraticKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
                                             dtype=torch.double)
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
    indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS()
    indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
    qK = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(svPosteriorOnIndPoints=qU, 
                                      indPointsLocsKMS=indPointsLocsKMS, 
                                      indPointsLocsAndTimesKMS=
                                       indPointsLocsAndAllTimesKMS)
    qH = stats.svGPFA.svEmbedding.LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qK)
    qH.setKernels(kernels=kernels)

    qUParams0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    qH.setInitialParams(initialParams=initialParams)
    qH.setTimes(times=t)
    qH.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=1e-5) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
    qH.buildKernelsMatrices()
    qHMu, qHVar = qH.computeMeansAndVars()

    qHMuError = math.sqrt(((mu_h-qHMu)**2).mean())
    assert(qHMuError<tol)
    qHVarError = math.sqrt(((var_h-qHVar)**2).mean())
    assert(qHVarError<tol)

def test_computeMeansAndVars_assocTimes():
    tol = 5e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat["Z"].shape[0]
    nTrials = mat["Z"][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat["q_mu"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat["q_sqrt"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat["q_diag"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    t = torch.from_numpy(mat["ttQuad"]).type(torch.DoubleTensor).permute(2, 0, 1)
    Z0 = [torch.from_numpy(mat["Z"][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Y = [torch.from_numpy(mat["Y"][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor)
    mu_h = [torch.from_numpy(mat["mu_h_Spikes"][0,i]).type(torch.DoubleTensor).squeeze() for i in range(nTrials)]
    var_h = [torch.from_numpy(mat["var_h_Spikes"][0,i]).type(torch.DoubleTensor).squeeze() for i in range(nTrials)]
    index = [torch.from_numpy(mat["index"][i,0][:,0]).type(torch.ByteTensor)-1 for i in range(nTrials)]

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
    kernels = [[None] for k in range(nLatents)]
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = stats.kernels.PeriodicKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]), 
                                              float(hprs[k,0][1])], 
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = stats.kernels.ExponentialQuadraticKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
                                             dtype=torch.double)
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
    indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS()
    indPointsLocsAndAssocTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAssocTimesKMS()
    qK = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAssocTimes(svPosteriorOnIndPoints=qU, 
                                        indPointsLocsKMS=indPointsLocsKMS, 
                                        indPointsLocsAndTimesKMS=
                                         indPointsLocsAndAssocTimesKMS)
    qH = stats.svGPFA.svEmbedding.LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=qK)
    qH.setKernels(kernels=kernels)

    qUParams0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    qH.setInitialParams(initialParams=initialParams)
    qH.setKernels(kernels=kernels)
    qH.setTimes(times=Y)
    qH.setNeuronForSpikeIndex(neuronForSpikeIndex=index)

    # begin patches because we are not using SVPosteriorOnLatentsAssocTimes in 
    # conjunction with SVPosteriorOnLatentsAllTimes
    qU.setInitialParams(initialParams=qUParams0)
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setInitialParams(initialParams=kmsParams0)
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setEpsilon(epsilon=1e-5) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
    indPointsLocsKMS.buildKernelsMatrices()
    # end patches because we are not using SVPosteriorOnLatentsAssocTimes in 
    # conjunction with SVPosteriorOnLatentsAllTimes

    qH.buildKernelsMatrices()
    qHMu, qHVar = qH.computeMeansAndVars()

    for i in range(len(mu_h)):
        qHMuError = math.sqrt(torch.sum((mu_h[i]-qHMu[i])**2))/mu_h[i].shape[0]
        assert(qHMuError<tol)
        qHVarError = math.sqrt(torch.sum((var_h[i]-qHVar[i])**2))/\
                     var_h[i].shape[0]
        assert(qHVarError<tol)

def test_svEmbedding_grads():
    tol = 2e-3
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/variationalEM.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor).contiguous()
    b0 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze().contiguous()
    indPointsLocsKMSRegEpsilon = 1e-2
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).contiguous().type(torch.DoubleTensor)

    linkFunction = torch.exp

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs0"]
    leasLowerBound = mat['lowerBound'][0,0]
    kernels = [[None] for k in range(nLatents)]
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = stats.kernels.PeriodicKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]),
                                              float(hprs[k,0][1])],
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = stats.kernels.ExponentialQuadraticKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
                                             dtype=torch.double)
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
    indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS()
    indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
    indPointsLocsAndAssocTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAssocTimesKMS()
    qKAllTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
    qKAssocTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAssocTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
    qHAllTimes = stats.svGPFA.svEmbedding.LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qKAllTimes)
    qHAssocTimes = stats.svGPFA.svEmbedding.LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=
                                               qKAssocTimes)
    eLL = stats.svGPFA.expectedLogLikelihood.PointProcessELLExpLink(svEmbeddingAllTimes=qHAllTimes,
                                 svEmbeddingAssocTimes=qHAssocTimes)
    klDiv = stats.svGPFA.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                         svPosteriorOnIndPoints=qU)
    svlb = stats.svGPFA.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
    svlb.setKernels(kernels=kernels)

    qUParams0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}
    optimParams = {"emMaxIter":3,
                   #
                   "eStepEstimate": True,
                   "eStepMaxIter":20,
                   "eStepTol":1e-2,
                   "eStepLR":1e-2,
                   "eStepLineSearchFn": "strong_wolfe",
                   "eStepNIterDisplay":1,
                   #
                   "mStepEmbeddingEstimate": True,
                   "mStepEmbeddingMaxIter":20,
                   "mStepEmbeddingTol":1e-2,
                   "mStepEmbeddingLR":1e-3,
                   "mStepEmbeddingLineSearchFn": "strong_wolfe",
                   "mStepEmbeddingNIterDisplay":1,
                   #
                   "mStepKernelsEstimate": True,
                   "mStepKernelsMaxIter":20,
                   "mStepKernelsTol":1e-2,
                   "mStepKernelsLR":1e-4,
                   "mStepKernelsLineSearchFn": "strong_wolfe",
                   "mStepKernelsNIterDisplay":1,
                   #
                   "mStepIndPointsEstimate": True,
                   "mStepIndPointsMaxIter":20,
                   "mStepIndPointsTol":1e-2,
                   "mStepIndPointsLR":1e-3,
                   "mStepIndPointsLineSearchFn": "strong_wolfe",
                   "mStepIndPointsNIterDisplay":1,
                   #
                   "verbose": True}
    svlb.setInitialParamsAndData( measurements=YNonStacked, initialParams=initialParams, quadParams=quadParams, indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)

    def eval_func(z):
        # pdb.set_trace()
        svlb.set_svEmbedding_params_from_flattened(flattened_params=z)
        svlb.set_svEmbedding_params_requires_grad(requires_grad=True)
        value = -svlb.eval()
        return value

    def value_func(z):
        # pdb.set_trace()
        value = eval_func(z=torch.from_numpy(z))
        return value.item()

    def grad_func(z):
        # pdb.set_trace()
        value = eval_func(z=torch.from_numpy(z))
        value.backward(retain_graph=True)
        grad_list = svlb.get_flattened_svEmbedding_params_grad()
        grad = np.array(grad_list)
        return grad

    x0 = svlb.get_flattened_svEmbedding_params().numpy()
    err = scipy.optimize.check_grad(func=value_func, grad=grad_func, x0=x0)
    assert(err<tol)

if __name__=="__main__":
    # test_get_flattened_params()
    # test_set_params_from_flattened()
    # test_set_params_requires_grad ()
    # test_computeMeansAndVars_allTimes()
    # test_computeMeansAndVars_assocTimes()
    test_svEmbedding_grads()
