
import sys
import os
import pdb
import argparse
import scipy.io
import pickle
import configparser
import torch
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
sys.path.append("../src")
import stats.svGPFA.svGPFAModelFactory
import plot.svGPFA.plotUtils
import utils.svGPFA.initUtils
import utils.svGPFA.configUtils
# import utils.svGPFA.miscUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simResNumber", help="Simulation result number", type=int)
    parser.add_argument("--periodStart", help="Start period value", type=float, default=0.1)
    parser.add_argument("--periodEnd", help="End period value", type=float, default=20.0)
    parser.add_argument("--periodBy", help="Interval between period values (sec)", type=float, default=0.1)
    parser.add_argument("--indPointsLocsKMSEpsilon", help="Inducing points locations kernel matrix store epsilon", type=int, default=1e-2)
    parser.add_argument("--nQuad", help="Number of quadrature points", type=int, default=200)
    parser.add_argument("--firstIndPointLoc", help="First inducing point location", type=float, default=1e-3)
    args = parser.parse_args()
    simResNumber = args.simResNumber
    periodStart = args.periodStart
    periodEnd = args.periodEnd
    periodBy = args.periodBy
    indPointsLocsKMSEpsilon = args.indPointsLocsKMSEpsilon
    nQuad = args.nQuad
    firstIndPointLoc = args.firstIndPointLoc

    simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]

    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    nLatents = int(simInitConfig["control_variables"]["nLatents"])
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    dtCIF = float(simInitConfig["control_variables"]["dtCIF"])
    nTrials = len(trialsLengths)

    simResFilename = "results/{:08d}_simRes.pickle".format(simResNumber)
    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikes = simRes["spikes"]

    C, d = utils.svGPFA.configUtils.getLinearEmbeddingParams(nNeurons=nNeurons, nLatents=nLatents, CFilename=simInitConfig["embedding_params"]["C_filename"], dFilename=simInitConfig["embedding_params"]["d_filename"])
    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig)
    kernelsParams0 = utils.svGPFA.initUtils.getKernelsParams0(kernels=kernels, noiseSTD=0.0)
    qMu0 = utils.svGPFA.configUtils.getQMu0(nTrials=nTrials, nLatents=nLatents, config=simInitConfig)
    qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
    indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS()
    indPointsLocsKMS.setEpsilon(epsilon=indPointsLocsKMSEpsilon)
    indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
    qKAllTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS
    )
    legQuadPoints, legQuadWeights = utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(nQuad=nQuad, trialsLengths=trialsLengths)
    qKAllTimes.setKernels(kernels=kernels)
    nIndPointsPerLatent = [qMu0[r].shape[1] for r in range(len(qMu0))]
    Z0 = utils.svGPFA.initUtils.getIndPointLocs0(nIndPointsPerLatent=nIndPointsPerLatent, trialsLengths=trialsLengths, firstIndPointLoc=firstIndPointLoc)
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    indPointsLocsKMS.setInitialParams(initialParams=kmsParams0)
    indPointsLocsKMS.buildKernelsMatrices()
    qSRSigma0 = indPointsLocsKMS.getKzzChol()
    qUParams0 = {"qMu0": qMu0, "qSRSigma0": qSRSigma0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C, "d0": d}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels,
    )

    model.setMeasurements(measurements=spikes)
    model.setInitialParams(initialParams=initialParams)
    model.setQuadParams(quadParams=quadParams)
    model.setIndPointsLocsKMSEpsilon(indPointsLocsKMSEpsilon=indPointsLocsKMSEpsilon)
    model.buildKernelsMatrices()

#     periodValues = np.arange(periodStart, periodEnd, periodBy)
#     lowerBoundValues = np.empty(periodValues.shape)
#     for i in range(len(periodValues)):
#         periodValue = periodValues[i]
#         model._eLL._svEmbeddingAllTimes._svPosteriorOnLatents._indPointsLocsKMS._kernels[0]._params[1]=periodValue
#         model.buildKernelsMatrices()
#         lowerBoundValues[i] = model.eval()
#     xlabel = "Period Value"

    paramValues = np.arange(0, 10, .1)
    lowerBoundValues = np.empty(paramValues.shape)
    for i in range(len(paramValues)):
        paramValue = paramValues[i]
        pdb.set_trace()
        model._eLL._svEmbeddingAllTimes._svPosteriorOnLatents._svPosteriorOnIndPoints._qMu[0][0,0,0]=paramValue
        lowerBoundValues[i] = model.eval()
    xlabel = "qMu0[0] Value"

    layout = {
        "xaxis": {"title": xlabel},
        "yaxis": {"title": "Lower Bound"},
    }
    data = []
    data.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": paramValues,
                "y": lowerBoundValues,
            },
    )
    fig = go.Figure(
        data=data,
        layout=layout,
    )
    pio.renderers.default = "browser"
    fig.show()
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

