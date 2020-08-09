
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
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--paramValueStart", help="Start parameter value", type=float, default=0.1)
    parser.add_argument("--paramValueEnd", help="End parameters value", type=float, default=20.0)
    parser.add_argument("--paramValueStep", help="Step for parameter values", type=float, default=0.1)
    args = parser.parse_args()
    estResNumber = args.estResNumber
    paramValueStart = args.paramValueStart
    paramValueEnd = args.paramValueEnd
    paramValueStep = args.paramValueStep

    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)

    with open(modelSaveFilename, "rb") as f: estResults = pickle.load(f)
    model = estResults["model"]


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

