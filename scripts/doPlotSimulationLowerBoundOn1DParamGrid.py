
import sys
import pdb
import math
import argparse
import scipy.io
import pickle
import configparser
import torch
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
sys.path.append("../src")
import utils.svGPFA.initUtils
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import stats.svGPFA.svGPFAModelFactory

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simResNumber", help="simulation result number", type=int)
    parser.add_argument("--paramValueStart", help="Start parameter value", type=float, default=2.7)
    parser.add_argument("--paramValueEnd", help="End parameters value", type=float, default=20.0)
    parser.add_argument("--paramValueStep", help="Step for parameter values", type=float, default=0.01)
    parser.add_argument("--yMin", help="Minimum y value", type=float, default=-math.inf)
    parser.add_argument("--yMax", help="Minimum y value", type=float, default=+math.inf)
    parser.add_argument("--nQuad", help="Number of quadrature points", type=int, default=200)
    args = parser.parse_args()
    simResNumber = args.simResNumber
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
    nIndPointsPerLatent = [int(str) for str in simInitConfig["control_variables"]["nIndPointsPerLatent"][1:-1].split(",")]
    nLatents = len(nIndPointsPerLatent)
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)
    firstIndPointLoc = float(simInitConfig["control_variables"]["firstIndPointLoc"])
    indPointsLocsKMSEpsilon = float(simInitConfig["control_variables"]["indPointsLocsKMSEpsilon"])

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]
    KzzChol = simRes["KzzChol"]
    indPointsMeans = simRes["indPointsMeans"]
    C, d = utils.svGPFA.configUtils.getLinearEmbeddingParams(nNeurons=nNeurons, nLatents=nLatents, CFilename=simInitConfig["embedding_params"]["C_filename"], dFilename=simInitConfig["embedding_params"]["d_filename"])

    # patch to acommodate Lea's equal number of inducing points across trials
    qMu0 = [[] for k in range(nLatents)]
    for k in range(nLatents):
        qMu0[k] = torch.empty((nTrials, nIndPointsPerLatent[k], 1), dtype=torch.double)
        for r in range(nTrials):
            qMu0[k][r,:,0] = indPointsMeans[r][k]
    # end patch

    legQuadPoints, legQuadWeights = utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(nQuad=nQuad, trialsLengths=trialsLengths)

    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig, forceUnitScale=False)
    kernelsParams0 = utils.svGPFA.initUtils.getKernelsParams0(kernels=kernels, noiseSTD=0.0)

    Z0 = utils.svGPFA.initUtils.getIndPointLocs0(nIndPointsPerLatent=nIndPointsPerLatent, trialsLengths=trialsLengths, firstIndPointLoc=firstIndPointLoc)
    qUParams0 = {"qMu0": qMu0, "qSRSigma0": KzzChol}
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

    pdb.set_trace()
    paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
    lowerBoundValues = np.empty(paramValues.shape)
    for i in range(len(paramValues)):
        kernelsParams = model.getKernelsParams()
        kernelsParams[0][1] = paramValues[i]
        model.buildKernelsMatrices()
        lowerBoundValues[i] = model.eval()
    xlabel = "Period Value"

    layout = {
        "xaxis": {"title": xlabel},
        # "yaxis": {"title": "Lower Bound"},
        "yaxis": {"title": "Lower Bound", "range": [yMin, yMax]},
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

