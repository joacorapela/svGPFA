
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
import stats.svGPFA.svGPFAModelFactory
import plot.svGPFA.plotUtilsPlotly

def getReferenceParams(model, latent):
    kernelsParams = model.getKernelsParams()
    refParams = kernelsParams[latent]
    refParams = [refParams[i].clone() for i in range(len(refParams))]
    return refParams

def updateKernelParams(model, period, lengthscale, latent):
        kernelsParams = model.getKernelsParams()
        kernelsParams[latent][0] = period
        kernelsParams[latent][1] = lengthscale
        model.buildKernelsMatrices()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simResNumber", help="simulation result number", type=int)
    parser.add_argument("indPointsLocsKMSRegEpsilon", help="regularization epsilong for the inducing points locations covariance", type=float)
    parser.add_argument("--latent", help="Parameter latent number", type=int, default=0)
    parser.add_argument("--periodValueStart", help="Start value for period parameter", type=float, default=1.0)
    parser.add_argument("--periodValueEnd", help="End value for period parameter", type=float, default=10.0)
    parser.add_argument("--periodValueStep", help="Step size for period parameter", type=float, default=0.02)
    parser.add_argument("--lengthscaleValueStart", help="Start value for lengthscale parameter", type=float, default=1.0)
    parser.add_argument("--lengthscaleValueEnd", help="End value for lengthscale parameter", type=float, default=10.0)
    parser.add_argument("--lengthscaleValueStep", help="Step size for lengthscale parameter", type=float, default=0.02)
    parser.add_argument("--zMin", help="Minimum z value", type=float, default=None)
    parser.add_argument("--zMax", help="Minimum z value", type=float, default=None)
    parser.add_argument("--nQuad", help="Number of quadrature points", type=int, default=200)

    args = parser.parse_args()
    simResNumber = args.simResNumber
    indPointsLocsKMSRegEpsilon = args.indPointsLocsKMSRegEpsilon
    latent = args.latent
    periodValueStart = args.periodValueStart
    periodValueEnd = args.periodValueEnd
    periodValueStep = args.periodValueStep
    lengthscaleValueStart = args.lengthscaleValueStart
    lengthscaleValueEnd = args.lengthscaleValueEnd
    lengthscaleValueStep = args.lengthscaleValueStep
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
    periodValues = np.arange(periodValueStart, periodValueEnd, periodValueStep)
    lengthscaleValues = np.arange(lengthscaleValueStart, lengthscaleValueEnd, lengthscaleValueStep)
    allLowerBoundValues = []
    allLengthscaleValues = []
    allPeriodValues = []
    for i in range(len(periodValues)):
        print("Processing period {:f} ({:d}/{:d})".format(periodValues[i], i, len(periodValues)))
        for j in range(len(lengthscaleValues)):
            updateKernelParams(model=model, period=periodValues[i], lengthscale=lengthscaleValues[j], latent=latent)
            lowerBound = model.eval()
            allLowerBoundValues.append(lowerBound.item())
            allPeriodValues.append(periodValues[i])
            allLengthscaleValues.append(lengthscaleValues[j])
    title = "Kernel Periodic, Latent {:d}, Epsilon {:f}".format(latent, indPointsLocsKMSRegEpsilon)
    figFilenamePattern = "figures/{:08d}_generativeParams_epsilon{:f}_kernel_periodic_latent{:d}.{{:s}}".format(simResNumber, indPointsLocsKMSRegEpsilon, latent)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotLowerBoundVsTwoParamsParam(param1Values=allPeriodValues, param2Values=allLengthscaleValues, lowerBoundValues=allLowerBoundValues, refParam1=refParams[0], refParam2=refParams[1], refParamText="Generative Value", refParamsLowerBound=refParamsLowerBound, title=title, xlabel="Lengthscale", ylabel="Period", zlabel="Lower Bound", zMin=zMin, zMax=zMax)
    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))
    pio.renderers.default = "browser"
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

