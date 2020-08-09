
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
import plot.svGPFA.plotUtils
import utils.svGPFA.initUtils
import utils.svGPFA.configUtils
# import utils.svGPFA.miscUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simResNumber", help="simulation result number", type=int)
    parser.add_argument("--paramValueStart", help="Start parameter value", type=float, default=0.1)
    parser.add_argument("--paramValueEnd", help="End parameters value", type=float, default=20.0)
    parser.add_argument("--paramValueStep", help="Step for parameter values", type=float, default=0.1)
    args = parser.parse_args()
    estResNumber = args.estResNumber
    paramValueStart = args.paramValueStart
    paramValueEnd = args.paramValueEnd
    paramValueStep = args.paramValueStep

    # load data and initial values
    simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]
    simResFilename = simResConfig["simulation_results"]["simResFilename"]

    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    nIndPointsPerLatent = [float(str) for str in simInitConfig["control_variables"]["nIndPointsPerLatent"][1:-1].split(",")]
    nLatents = len(nIndPointsPerLatent)
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]
    Kzz = simRes["Kzz"]
    indPointsMeans = simRes["indPointsMeans"]

    legQuadPoints, legQuadWeights = utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(nQuad=nQuad, trialsLengths=trialsLengths)

    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=estInitConfig)
    kernelsParams0 = utils.svGPFA.initUtils.getKernelsParams0(kernels=kernels, noiseSTD=0.0)

    qMu0, qSVec0, qSDiag0 = utils.svGPFA.initUtils.getSVPosteriorOnIndPointsParams0(nIndPointsPerLatent=nIndPointsPerLatent, nLatents=nLatents, nTrials=nTrials, scale=initCondIndPointsScale)

    Z0 = utils.svGPFA.initUtils.getIndPointLocs0(nIndPointsPerLatent=nIndPointsPerLatent, trialsLengths=trialsLengths, firstIndPointLoc=firstIndPointLoc)
    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": d0}
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

    paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
    lowerBoundValues = np.empty(paramValues.shape)
    for i in range(len(paramValues)):
        kernelsParams = model.getKernelsParams()
        pdb.set_trace()
        kernelsParams[0][1] = paramValues[i]
        model.buildKernelsMatrices()
        lowerBoundValues[i] = model.eval()
    xlabel = "Period Value"

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

