
import sys
import os
import time
import pdb
import random
import argparse
import cProfile, pstats
from scipy.io import loadmat
import pickle
import configparser
import torch
import numpy as np
sys.path.append("../src")
import stats.kernels
import stats.svGPFA.svGPFAModelFactory
import stats.svGPFA.svEM
import plot.svGPFA.plotUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mEstNumber", help="Matlab's estimation number", type=int)
    parser.add_argument("--deviceName", help="name of device (cpu or cuda)", default="cpu")
    parser.add_argument("--profile", help="perform profiling", action="store_true")
    args = parser.parse_args()
    if args.profile:
        profile = True
    else:
        profile = False

    mEstNumber = args.mEstNumber
    deviceName = args.deviceName
    if not torch.cuda.is_available():
        deviceName = "cpu"
    device = torch.device(deviceName)
    print("Using {:s}".format(deviceName))

    mEstConfig = configparser.ConfigParser()
    mEstConfig.read("../../matlabCode/scripts/results/{:08d}-pointProcessEstimationParams.ini".format(mEstNumber))
    mSimNumber = int(mEstConfig["data"]["simulationNumber"])
    indPointsLocsKMSEpsilon = float(mEstConfig["control_variables"]["epsilon"])
    ppSimulationFilename = os.path.join(os.path.dirname(__file__), "../../matlabCode/scripts/results/{:08d}-pointProcessSimulation.mat".format(mSimNumber))
    initDataFilename = os.path.join(os.path.dirname(__file__), "../../matlabCode/scripts/results/{:08d}-pointProcessInitialConditions.mat".format(mEstNumber))

    # save estimated values
    estimationPrefixUsed = True
    while estimationPrefixUsed:
        pEstNumber = random.randint(0, 10**8)
        estimMetaDataFilename = \
                "results/{:08d}_leasSimulation_estimationPinv_metaData_{:s}.ini".format(pEstNumber, deviceName)
        if not os.path.exists(estimMetaDataFilename):
           estimationPrefixUsed = False
    modelSaveFilename = \
        "results/{:08d}_leasSimulation_estimatedModelPinv_{:s}.pickle".format(pEstNumber, deviceName)
    profilerFilenamePattern = \
        "results/{:08d}_leaseSimulation_estimatedModelPinv_{:s}.pstats".format(pEstNumber, deviceName)
    lowerBoundHistFigFilename = \
        "figures/{:08d}_leasSimulation_lowerBoundHistPinv_{:s}.png".format(pEstNumber, deviceName)

    mat = loadmat(initDataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).to(device) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).to(device) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).to(device) for i in range(nLatents)]
    Z0 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1).to(device) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor).to(device)
    b0 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze().to(device)
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1).to(device)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1).to(device)

    yMat = loadmat(ppSimulationFilename)
    YNonStacked_tmp = yMat['Y']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor).to(device)

    kernelNames = mat["kernelNames"]
    hprs0 = mat["hprs0"]

    # create kernels
    kernels = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = stats.kernels.PeriodicKernel(scale=1.0)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = stats.kernels.ExponentialQuadraticKernel(scale=1.0)
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    # create initial parameters
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernelsParams0[k] = torch.tensor([float(hprs0[k,0][0]),
                                              float(hprs0[k,0][1])],
                                             dtype=torch.double).to(device)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernelsParams0[k] = torch.tensor([float(hprs0[k,0][0])],
                                              dtype=torch.double).to(device)
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
    optimParams = {"emMaxIter":50,
                   #
                   "eStepEstimate":True,
                   "eStepMaxIter":100,
                   "eStepTol":1e-3,
                   "eStepLR":1e-3,
                   "eStepLineSearchFn":"strong_wolfe",
                   # "eStepLineSearchFn":"None",
                   "eStepNIterDisplay":1,
                   #
                   "mStepEmbeddingEstimate":True,
                   "mStepEmbeddingMaxIter":100,
                   "mStepEmbeddingTol":1e-3,
                   "mStepEmbeddingLR":1e-3,
                   "mStepEmbeddingLineSearchFn":"strong_wolfe",
                   # "mStepEmbeddingLineSearchFn":"None",
                   "mStepEmbeddingNIterDisplay":1,
                   #
                   "mStepKernelsEstimate":True,
                   "mStepKernelsMaxIter":10,
                   "mStepKernelsTol":1e-3,
                   "mStepKernelsLR":1e-3,
                   "mStepKernelsLineSearchFn":"strong_wolfe",
                   # "mStepKernelsLineSearchFn":"None",
                   "mStepKernelsNIterDisplay":1,
                   "mStepKernelsNIterDisplay":1,
                   #
                   "mStepIndPointsEstimate":True,
                   "mStepIndPointsMaxIter":20,
                   "mStepIndPointsTol":1e-3,
                   "mStepIndPointsLR":1e-8,
                   "mStepIndPointsLineSearchFn":"strong_wolfe",
                   # "mStepIndPointsLineSearchFn":"None",
                   "mStepIndPointsNIterDisplay":1,
                   #
                   "verbose":True
                  }
    estimConfig = configparser.ConfigParser()
    estimConfig["data"] = {"mEstNumber": mEstNumber}
    estimConfig["optim_params"] = optimParams
    estimConfig["control_params"] = {"indPointsLocsKMSEpsilon": indPointsLocsKMSEpsilon}
    with open(estimMetaDataFilename, "w") as f: estimConfig.write(f)

    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels,
    )

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

    # model.to(device)

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
                      optimParams=optimParams,
                      indPointsLocsKMSEpsilon=indPointsLocsKMSEpsilon,
                     )
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
        profilerFilename = profilerFilenamePattern.format(optimParams["emMaxIter"])
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

