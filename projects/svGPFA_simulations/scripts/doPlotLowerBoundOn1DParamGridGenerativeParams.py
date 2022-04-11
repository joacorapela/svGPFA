
import sys
import pdb
import math
import argparse
import pickle
import configparser
import torch
import numpy as np
import plotly.io as pio
sys.path.append("../src")
import utils.svGPFA.initUtils
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import stats.svGPFA.svGPFAModelFactory
import plot.svGPFA.plotUtilsPlotly
import lowerBoundVsOneParamUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simResNumber", help="simulation result number", type=int)
    parser.add_argument("paramType", help="Parameter type: indPointsPosteriorMean, indPointsPosteriorCov, indPointsLocs, kernel, embeddingC, embeddingD")
    parser.add_argument("indPointsLocsKMSRegEpsilon", help="regularization epsilon for the inducing points locations covariance", type=float)
    parser.add_argument("--trial", help="Parameter trial number", type=int, default=0)
    parser.add_argument("--latent", help="Parameter latent number", type=int, default=0)
    parser.add_argument("--neuron", help="Parameter neuron number", type=int, default=0)
    parser.add_argument("--kernelParamIndex", help="Kernel parameter index", type=int, default=0)
    parser.add_argument("--indPointIndex", help="Parameter inducing point index", type=int, default=0)
    parser.add_argument("--indPointIndex2", help="Parameter inducing point index2 (used for inducing points covariance parameters)", type=int, default=0)
    parser.add_argument("--paramValueStart", help="Start parameter value", type=float, default=1.0)
    parser.add_argument("--paramValueEnd", help="End parameters value", type=float, default=20.0)
    parser.add_argument("--paramValueStep", help="Step for parameter values", type=float, default=0.01)
    parser.add_argument("--yMin", help="Minimum y value", type=float, default=-math.inf)
    parser.add_argument("--yMax", help="Minimum y value", type=float, default=+math.inf)
    parser.add_argument("--nQuad", help="Number of quadrature points", type=int, default=200)

    args = parser.parse_args()
    simResNumber = args.simResNumber
    paramType = args.paramType
    indPointsLocsKMSRegEpsilon = args.indPointsLocsKMSRegEpsilon
    trial = args.trial
    latent = args.latent
    neuron = args.neuron
    kernelParamIndex = args.kernelParamIndex
    indPointIndex = args.indPointIndex
    indPointIndex2 = args.indPointIndex2
    paramValueStart = args.paramValueStart
    paramValueEnd = args.paramValueEnd
    paramValueStep = args.paramValueStep
    yMin = args.yMin
    yMax = args.yMax
    nQuad = args.nQuad

    # load data and initial values
    simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]
    simResFilename = simResConfig["simulation_results"]["simResFilename"]

    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    nLatents = int(simInitConfig["control_variables"]["nLatents"])
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)
    trials_start_times = [0.0 for i in range(nTrials)]
    trials_end_times = trialsLengths
    # firstIndPointLoc = float(simInitConfig["control_variables"]["firstIndPointLoc"])
    # indPointsLocsKMSRegEpsilon = float(simInitConfig["control_variables"]["indPointsLocsKMSRegEpsilon"])

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]
    # KzzChol = simRes["KzzChol"]
    Kzz = simRes["Kzz"]
    indPointsMeans = simRes["indPointsMeans"]
    C, d = utils.svGPFA.configUtils.getLinearEmbeddingParams(CFilename=simInitConfig["embedding_params"]["C_filename"], dFilename=simInitConfig["embedding_params"]["d_filename"])

    legQuadPoints, legQuadWeights = \
            utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(
                nQuad=nQuad, trials_start_times=trials_start_times,
                trials_end_times=trials_end_times)


    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig, forceUnitScale=True)
    kernelsParams0 = utils.svGPFA.initUtils.getKernelsParams0(kernels=kernels, noiseSTD=0.0)

    # Z0 = utils.svGPFA.initUtils.getIndPointLocs0(nIndPointsPerLatent=nIndPointsPerLatent, trialsLengths=trialsLengths, firstIndPointLoc=firstIndPointLoc)
    Z0 = utils.svGPFA.configUtils.getIndPointsLocs0(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)
    nIndPointsPerLatent = [Z0[k].shape[1] for k in range(nLatents)]

    # patch to acommodate Lea's equal number of inducing points across trials
    qMu0 = [[] for k in range(nLatents)]
    for k in range(nLatents):
        qMu0[k] = torch.empty((nTrials, nIndPointsPerLatent[k], 1), dtype=torch.double)
        for r in range(nTrials):
            qMu0[k][r,:,:] = indPointsMeans[r][k]
    # end patch

    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVecsFromKzz(Kzz=Kzz)

    qUParams0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C, "d0": d}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    # create model
    kernelMatrixInvMethod = stats.svGPFA.svGPFAModelFactory.kernelMatrixInvChol
    indPointsCovRep = stats.svGPFA.svGPFAModelFactory.indPointsCovChol
    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModelPyTorch(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels, kernelMatrixInvMethod=kernelMatrixInvMethod,
        indPointsCovRep=indPointsCovRep)

    model.setMeasurements(measurements=spikesTimes)
    model.setInitialParams(initialParams=initialParams)
    model.setELLCalculationParams(eLLCalculationParams=quadParams)
    model.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)
    model.buildKernelsMatrices()

    refParam = lowerBoundVsOneParamUtils.getReferenceParam(paramType=paramType, model=model, trial=trial, latent=latent, neuron=neuron, kernelParamIndex=kernelParamIndex, indPointIndex=indPointIndex, indPointIndex2=indPointIndex2)
    paramUpdateFun = lowerBoundVsOneParamUtils.getParamUpdateFun(paramType=paramType)
    paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
    lowerBoundValues = np.empty(paramValues.shape)
    for i in range(len(paramValues)):
        paramUpdateFun(model=model, paramValue=paramValues[i], trial=trial, latent=latent, neuron=neuron, kernelParamIndex=kernelParamIndex, indPointIndex=indPointIndex, indPointIndex2=indPointIndex2)
        Kzz = model._eLL._svEmbeddingAllTimes._svPosteriorOnLatents._indPointsLocsKMS._Kzz
        # begin debug code
#         for k in range(nLatents):
#             nTrial = Kzz[k].shape[0]
#             for r in range(nTrial):
#                 # torch.sum(torch.abs(torch.eig(Kzz[k][0,:,:]).eigenvalues[:,1]))==0
#                 imEigenval = torch.eig(Kzz[k][0,:,:]).eigenvalues[:,1]
#                 if torch.any(imEigenval!=0.0):
#                     print("{:f}".format(paramValues[i]))
#         if paramValues[i]>=16.48:
#             pdb.set_trace()
        # end debug code
#         if paramValues[i]>=4.0:
#             pdb.set_trace()
        lowerBoundValues[i] = model.eval()
    # begin fix plotly problem with nInf values
    # nInfIndices = np.where(lowerBoundValues==-np.inf)[0]
    # minNoNInfLowerBound = lowerBoundValues[nInfIndices.max()+1]
    # lowerBoundNoNInfValues = lowerBoundValues
    # lowerBoundNoNInfValues[nInfIndices] = minNoNInfLowerBound
    # end fix plotly problem with nInf values

    # begin fix plotly problem with nInf values
    boundedLowerBound = lowerBoundValues
    smallIndices = np.where(lowerBoundValues<yMin)[0]
    boundedLowerBound[smallIndices] = None
    largeIndices = np.where(lowerBoundValues>yMax)[0]
    boundedLowerBound[largeIndices] = None
    # end fix plotly problem with nInf values
    title = lowerBoundVsOneParamUtils.getParamTitle(paramType=paramType, trial=trial, latent=latent, neuron=neuron, kernelParamIndex=kernelParamIndex, indPointIndex=indPointIndex, indPointIndex2=indPointIndex2, indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)
    figFilenamePattern = lowerBoundVsOneParamUtils.getFigFilenamePattern(prefixNumber=simResNumber, descriptor="lowerBoundVs1DParam_generativeParams", paramType=paramType, trial=trial, latent=latent, neuron=neuron, indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon, kernelParamIndex=kernelParamIndex, indPointIndex=indPointIndex, indPointIndex2=indPointIndex2)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotLowerBoundVsOneParam(paramValues=paramValues, lowerBoundValues=lowerBoundValues, refParams=[refParam], title=title, yMin=yMin, yMax=yMax, lowerBoundLineColor="blue", refParamsLineColors=["blue"])
    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))
    pio.renderers.default = "browser"
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

