
import sys
import os
import pdb
import math
from scipy.io import loadmat
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from svFrontEnd import SVGPFAFrontEnd

from kernelMatricesStore import KernelMatricesStore
from kernels import PeriodicKernel, ExponentialQuadraticKernel
from approxPosteriorForH import ApproxPosteriorForHForAllNeuronsAllTimes, ApproxPosteriorForHForAllNeuronsAssociatedTimes
from inducingPointsPrior import InducingPointsPrior
from expectedLogLikelihood import PointProcessExpectedLogLikelihood, PoissonExpectedLogLikelihood
from sparseVariationalLowerBound import SparseVariationalLowerBound
from klDivergence import KLDivergence
from sparseVariationalEM import SparseVariationalEM

def main(argv):
    dataFilename = "data/demo_PointProcess.mat"
    approxPosteriorForH_allNeuronsAllTimes_pickleFilename = "data/approxPosteriorForH_allNeuronsAllTimes.pickle"

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]
    qMu = [torch.from_numpy(mat['q_mu0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec = [torch.from_numpy(mat['q_sqrt0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag = [torch.from_numpy(mat['q_diag0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    t = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    Z0 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Y = [torch.from_numpy(mat['Y'][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    index = [torch.from_numpy(mat["index"][i,0][:,0]).type(torch.ByteTensor) for i in range(nTrials)]
    C0 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze()
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
    kernelNames = mat["kernelNames"]
    hprs0 = mat["hprs0"]
    testTimes = torch.from_numpy(mat['testTimes']).type(torch.DoubleTensor).squeeze()
    trueLatents = [[torch.from_numpy(mat['trueLatents'][tr,k]).type(torch.DoubleTensor) for tr in range(nTrials)] for k in range(nLatents)]

    linkFunction = torch.exp
    spikeTimes = unstackSpikeTimes(stackedSpikeTimes=Y, indices=indices)
    svPosteriorOnInducingPointsParams0 = {"mean":qMu, "covVec":qSVec, "covDiag":qSDiag}
    nLegendreQuadPoints = legQuadPoints.shape[0]
    nHermiteQuadPoints = hermQuadPoints.shape[0]

    frontEnd = SVGPFAFrontEndFactory.getSVGPFAFrontEnd(
                conditionalDist=\
                 SVGPFAFrontEnd.PointProcessWithLinearLatents)

    indPointLocs, kernelsHyperParams, svPosteriorOnIndPointsParams, \
        sufficientStatsParams = frontEnd.estimate(
                spikeTimes=spikeTimes, 
                testTimes=testTimes, 
                inducingPointLocs0=Z0, 
                svPosteriorOnInducingPointsParams0=\
                    svPosteriorOnInducingPointsParams0, 
                latentsToSufficientStatsMatrix0=C0, 
                sufficientStatsConstantVector0=b0, 
                nLegendreQuadPoints=nLegendreQuadPoints, 
                nHermiteQuadPoints=nHermiteQuadPoints, 
                kernelNames=kernelNames, 
                kernelHyperParams0=hprs0, 
                linkFunction=linkFunction)

    indPointLocs, kernelsHyperParams, svPosteriorOnIndPointsParams, \
        sufficientStatsParams = frontEnd.predict(testTimes=testTimes)

    kernels = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], 'PeriodicKernel'):
            kernels[k] = PeriodicKernel(scale=1.0,
                                        lengthScale=float(hprs[k,0][0]),
                                        period=float(hprs[k,0][1]))
        elif np.char.equal(kernelNames[0,k][0], 'rbfKernel'):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0,
                                                    lengthScale=float(hprs[k,0][0]))
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
    kernelMatricesStore= KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)
    qH_allNeuronsAllTimes = ApproxPosteriorForHForAllNeuronsAllTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore)
    qH_allNeuronsAssociatedTimes = ApproxPosteriorForHForAllNeuronsAssociatedTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore, neuronForSpikeIndex=index)

    eLL = PointProcessExpectedLogLikelihood(approxPosteriorForHForAllNeuronsAllTimes=qH_allNeuronsAllTimes, approxPosteriorForHForAllNeuronsAssociatedTimes=qH_allNeuronsAssociatedTimes, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, legQuadPoints=legQuadPoints, legQuadWeights=legQuadWeights, linkFunction=linkFunction)
    klDiv = KLDivergence(kernelMatricesStore=kernelMatricesStore, inducingPointsPrior=qU)
    svLB = SparseVariationalLowerBound(eLL=eLL, klDiv=klDiv)
    svEM = SparseVariationalEM(lowerBound=svLB, eLL=eLL, kernelMatricesStore=kernelMatricesStore)
    maxRes = svEM.maximize(emMaxNIter=50, eStepMaxNIter=50, mStepModelParamsMaxNIter=50, mStepKernelParamsMaxNIter=50, mStepKernelParamsLR=1e-5, mStepInducingPointsMaxNIter=50)

    file = open(approxPosteriorForH_allNeuronsAllTimes_pickleFilename, 'wb')
    pickle.dump(obj=qH_allNeuronsAllTimes, file=file)
    file.close()

    qHMu, qHVar, qKMu, qKVar = qH_allNeuronsAllTimes.predict(testTimes=testTimes)

    Z = qH_allNeuronsAllTimes._kernelMatricesStore.getZ()
    testTimesToPlot = testTimes.numpy()
    trialToPlot = 0
    f, axes = plt.subplots(nLatents, 1, sharex=True)
    for k in range(nLatents):
        trueLatentToPlot = trueLatents[k][trialToPlot].numpy().squeeze()
        qKMuToPlot = qKMu[trialToPlot,:,k].numpy()
        errorToPlot = qKVar[trialToPlot,:,k].sqrt().numpy()
        axes[k].plot(testTimesToPlot, trueLatentToPlot, label="true", color="black")
        axes[k].plot(testTimesToPlot, qKMuToPlot, label="estimated", color="blue")
        axes[k].fill_between(testTimesToPlot, qKMuToPlot-errorToPlot, qKMuToPlot+errorToPlot, color="lightblue")
        for i in range(Z[k].shape[1]):
            axes[k].axvline(x=Z[k][trialToPlot,i, 0], color="red")
        axes[k].set_ylabel("Latent %d"%(k))
    axes[-1].set_xlabel("Sample")
    axes[-1].legend()
    plt.xlim(left=np.min(testTimesToPlot)-1, right=np.max(testTimesToPlot)+1)
    plt.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
