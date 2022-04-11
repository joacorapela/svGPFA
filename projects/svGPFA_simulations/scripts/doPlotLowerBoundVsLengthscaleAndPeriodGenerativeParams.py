
import sys
import pdb
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
import stats.kernels
import stats.svGPFA.svGPFAModelFactory
import plot.svGPFA.plotUtilsPlotly

def getReferenceParams(model, latent):
    kernelsParams = model.getKernelsParams()
    refParams = kernelsParams[latent]
    refParams = [refParams[i].clone() for i in range(len(refParams))]
    return refParams

def updateKernelParams(model, period, lengthscale, latent):
    kernelsParams = model.getKernelsParams()
    kernelsParams[latent][0] = lengthscale
    kernelsParams[latent][1] = period
    model.buildKernelsMatrices()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simResNumber", help="simulation result number", type=int)
    parser.add_argument("indPointsLocsKMSRegEpsilon", help="regularization epsilong for the inducing points locations covariance", type=float)
    parser.add_argument("--latent", help="Parameter latent number", type=int, default=0)
    parser.add_argument("--lowerBoundQuantile", help="Quantile of the smallest lower bount to plot", type=float, default=0.5)
    parser.add_argument("--lengthscaleStartValue", help="Center value to plot for lengthscale parameter", type=float)
    parser.add_argument("--lengthscaleScale", help="Scale for lengthscale parameter", type=float)
    parser.add_argument("--lengthscaleScaledDT", help="Scaled half width for the lengthscale parameter", type=float, default=1.0)
    parser.add_argument("--lengthscaleNSamples", help="Number of samples for lengthscale parameter", type=float, default=100)
    parser.add_argument("--periodStartValue", help="Center value for period parameter", type=float)
    parser.add_argument("--periodScale", help="Scale for period parameter", type=float)
    parser.add_argument("--periodScaledDT", help="Scaled half width for period parameter", type=float)
    parser.add_argument("--periodNSamples", help="Number of samples for period parameter", type=float, default=6.5)
    parser.add_argument("--zMin", help="Minimum z value", type=float, default=None)
    parser.add_argument("--zMax", help="Minimum z value", type=float, default=None)
    parser.add_argument("--nQuad", help="Number of quadrature points", type=int, default=200)

    args = parser.parse_args()
    simResNumber = args.simResNumber
    indPointsLocsKMSRegEpsilon = args.indPointsLocsKMSRegEpsilon
    latent = args.latent
    lowerBoundQuantile = args.lowerBoundQuantile
    lengthscaleStartValue = args.lengthscaleStartValue
    lengthscaleScale = args.lengthscaleScale
    lengthscaleScaledDT = args.lengthscaleScaledDT
    lengthscaleNSamples = args.lengthscaleNSamples
    periodStartValue = args.periodStartValue
    periodScale = args.periodScale
    periodScaledDT = args.periodScaledDT
    periodNSamples = args.periodNSamples
    zMin = args.zMin
    zMax = args.zMax

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
    firstIndPointLoc = float(simInitConfig["control_variables"]["firstIndPointLoc"])
    nQuad = args.nQuad

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]
    KzzChol = simRes["KzzChol"]
    indPointsMeans = simRes["indPointsMeans"]
    C, d = utils.svGPFA.configUtils.getLinearEmbeddingParams(CFilename=simInitConfig["embedding_params"]["C_filename"], dFilename=simInitConfig["embedding_params"]["d_filename"])

    legQuadPoints, legQuadWeights = utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(nQuad=nQuad, trialsLengths=trialsLengths)

    baseKernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig, forceUnitScale=True)
    baseParams = baseKernels[0].getParams()
    kernel = stats.kernels.PeriodicKernel(scale=1.0, lengthscaleScale=lengthscaleScale, periodScale=periodScale)
    kernel.setParams(params=torch.tensor([baseParams[0]*lengthscaleScale, baseParams[1]*periodScale]))
    kernels = [kernel]

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

    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVecsFromSRMatrices(srMatrices=KzzChol)
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
    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels)

    model.setMeasurements(measurements=spikesTimes)
    model.setInitialParams(initialParams=initialParams)
    model.setQuadParams(quadParams=quadParams)
    model.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)
    model.buildKernelsMatrices()

    refParams = getReferenceParams(model=model, latent=latent)
    refParamsLowerBound = model.eval()

    lengthscaleScaledStartValue = lengthscaleStartValue*lengthscaleScale
    lengthscaleScaledEndValue = lengthscaleScaledStartValue+lengthscaleScaledDT*periodNSamples
    lengthscaleScaledValues = np.arange(lengthscaleScaledStartValue, lengthscaleScaledEndValue, lengthscaleScaledDT)

    periodScaledStartValue = periodStartValue*periodScale
    periodScaledEndValue = periodScaledStartValue+periodScaledDT*periodNSamples
    periodScaledValues = np.arange(periodScaledStartValue, periodScaledEndValue, periodScaledDT)

    allLowerBoundValues = []
    allUnlengthscaleScaledValues = []
    allUnperiodScaledValues = []
    for i in range(len(periodScaledValues)):
        print("Processing period {:f} ({:d}/{:d})".format(periodScaledValues[i]/periodScale, i, len(periodScaledValues)))
        for j in range(len(lengthscaleScaledValues)):
            updateKernelParams(model=model, lengthscale=lengthscaleScaledValues[j], period=periodScaledValues[i], latent=latent)
            lowerBound = model.eval()
            if(torch.isinf(lowerBound).item()):
                pdb.set_trace()
            allLowerBoundValues.append(lowerBound.item())
            allUnperiodScaledValues.append(periodScaledValues[i]/periodScale)
            allUnlengthscaleScaledValues.append(lengthscaleScaledValues[j]/lengthscaleScale)
    title = "Kernel Periodic, Latent {:d}, Epsilon {:f}".format(latent, indPointsLocsKMSRegEpsilon)
    figFilenamePattern = "figures/{:08d}_generativeParams_epsilon{:f}_kernel_periodic_latent{:d}.{{:s}}".format(simResNumber, indPointsLocsKMSRegEpsilon, latent)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotLowerBoundVsTwoParamsParam(param1Values=allUnperiodScaledValues, param2Values=allUnlengthscaleScaledValues, lowerBoundValues=allLowerBoundValues, refParam1=refParams[0], refParam2=refParams[1], refParamText="Generative Value", refParamsLowerBound=refParamsLowerBound, title=title, lowerBoundQuantile=lowerBoundQuantile, param1Label="Period", param2Label="Lengthscale", lowerBoundLabel="Lower Bound", zMin=zMin, zMax=zMax)
    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))
    pio.renderers.default = "browser"
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

