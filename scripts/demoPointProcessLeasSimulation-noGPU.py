
import sys
import os
import time
import pdb
import argparse
import cProfile, pstats
from scipy.io import loadmat
import pickle
import torch
import numpy as np
sys.path.append("../src")
import stats.kernels
import stats.svGPFA.svGPFAModelFactory
import stats.svGPFA.svEM
import plot.svGPFA.plotUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", help="perform profiling", action="store_true")
    args = parser.parse_args()
    if args.profile:
        profile = True
    else:
        profile = False

    tol = 1e-3
    ppSimulationFilename = os.path.join(os.path.dirname(__file__), "data/pointProcessSimulation.mat")
    initDataFilename = os.path.join(os.path.dirname(__file__), "data/pointProcessInitialConditions.mat")
    lowerBoundHistFigFilename = "figures/leasLowerBoundHist_{:s}.png".format("cpu")
    modelSaveFilename = "results/estimationResLeasSimulation_{:s}.pickle".format("cpu")
    profilerFilenamePattern = "results/demoPointProcessLeasSimulation_{:d}Iter.pstats"

    mat = loadmat(initDataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Z0 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze()
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)

    yMat = loadmat(ppSimulationFilename)
    YNonStacked_tmp = yMat['Y']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor)

    kernelNames = mat["kernelNames"]
    hprs0 = mat["hprs0"]
    indPointsLocsKMSEpsilon = 1e-4

    # create kernels
    kernels = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = stats.kernels.PeriodicKernel()
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = stats.kernels.ExponentialQuadraticKernel()
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    # create initial parameters
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernelsParams0[k] = torch.tensor([1.0,
                                              float(hprs0[k,0][0]),
                                              float(hprs0[k,0][1])],
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernelsParams0[k] = torch.tensor([1.0,
                                              float(hprs0[k,0][0])],
                                             dtype=torch.double)
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
    optimParams = {"emMaxNIter":5, "eStepMaxNIter":100, "mStepModelParamsMaxNIter":100, "mStepKernelParamsMaxNIter":20, "mStepIndPointsMaxNIter":10, "mStepIndPointsLR": 1e-2}

    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels,
        indPointsLocsKMSEpsilon=indPointsLocsKMSEpsilon)

    # start debug code
    # parametersList = []
    # i = 0
    # for parameter in model.parameters():
    #     print("Inside for loop")
    #     print(i, parameter)
    #     parametersList.append(parameter)
    # print("Outside for loop")
    # pdb.set_trace()
    # ned debug code

    # maximize lower bound
    svEM = stats.svGPFA.svEM.SVEM()
    if profile:
        pr = cProfile.Profile()
        pr.enable()
    tStart = time.time()
    lowerBoundHist, elapsedTimeHist = \
        svEM.maximize(model=model,
                      measurements=YNonStacked,
                      initialParams=initialParams,
                      quadParams=quadParams,
                      optimParams=optimParams)
    tElapsed = time.time()-tStart
    print("Completed maximize in {:.2f} seconds".format(tElapsed))

    # start debug code
    # parametersList = []
    # i = 0
    # for parameter in model.parameters():
    #     print("Inside for loop")
    #     print(i, parameter)
    #     parametersList.append(parameter)
    #     i += 1
    # print("Outside for loop")
    # pdb.set_trace()
    # end debug code

    if profile:
        pr.disable()
        profilerFilename = profilerFilenamePattern.format(optimParams["emMaxNIter"])
        s = open(profilerFilename, "w")
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s)
        ps.strip_dirs().sort_stats(sortby).print_stats()
        s.close()

    resultsToSave = {"lowerBoundHist": lowerBoundHist, "elapsedTimeHist": elapsedTimeHist, "model": model}
    with open(modelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)

    # plot lower bound history
    plot.svGPFA.plotUtils.plotLowerBoundHist(lowerBoundHist=lowerBoundHist, elapsedTimeHist=elapsedTimeHist, figFilename=lowerBoundHistFigFilename)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

