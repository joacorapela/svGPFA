
import sys
import os
import torch
import pdb
import pickle
import argparse
import configparser
from scipy.io import loadmat
import plotly.graph_objs as go
import plotly.offline
import plotly.io as pio
sys.path.append("../src")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mEstNumber", help="Matlab's estimation number", type=int)
    args = parser.parse_args()
    mEstNumber = args.mEstNumber

    mEstParamsFilename = "../../matlabCode/working/scripts/results/{:08d}-pointProcessEstimationParams.ini".format(mEstNumber)
    mEstConfig = configparser.ConfigParser()
    mEstConfig.read(mEstParamsFilename)
    pEstNumber = int(mEstConfig["data"]["pEstNumber"])

    pEstimMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(pEstNumber)
    pEstConfig = configparser.ConfigParser()
    pEstConfig.read(pEstimMetaDataFilename)
    pSimNumber = int(pEstConfig["simulation_params"]["simResNumber"])

    mModelSaveFilename = "../../matlabCode/working/scripts/results/{:08d}-pointProcessEstimationRes.mat".format(mEstNumber)
    pModelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(pEstNumber)

    lowerBoundVsIterNoFigFilenamePattern = "figures/{:08d}_{:08d}_lowerBoundVsIterNo.{{:s}}".format(pEstNumber, mEstNumber)
    lowerBoundVsElapsedTimeFigFilenamePattern = "figures/{:08d}_{:08d}_lowerBoundVsRuntime.{{:s}}".format(pEstNumber, mEstNumber)

    with open(pModelSaveFilename, "rb") as f: res = pickle.load(f)
    # pLowerBound = -torch.stack(res["lowerBoundHist"]).detach().numpy()
    pLowerBound = res["lowerBoundHist"]
    pElapsedTime = res["elapsedTimeHist"]

    loadRes = loadmat(mModelSaveFilename)
    mLowerBound = torch.cat(tuple(torch.from_numpy(loadRes["lowerBound"])))
    mElapsedTime = loadRes["m"]["elapsedTime"][0][0][:, 0]

    trace1 = go.Scatter(
        y=pLowerBound,
        # line=dict(color='rgb(0,100,80)'),
        line=dict(color='red'),
        mode='lines+markers',
        name='Python',
        showlegend=True,
    )
    trace2 = go.Scatter(
        y=mLowerBound,
        # line=dict(color='rgb(0,100,80)'),
        line=dict(color='blue'),
        mode='lines+markers',
        name='Matlab',
        showlegend=True,
    )
    trace3 = go.Scatter(
        x=pElapsedTime,
        y=pLowerBound,
        # line=dict(color='rgb(0,100,80)'),
        line=dict(color='red'),
        mode='lines+markers',
        name='Python',
        showlegend=True,
    )
    trace4 = go.Scatter(
        x=mElapsedTime,
        y=mLowerBound,
        # line=dict(color='rgb(0,100,80)'),
        line=dict(color='blue'),
        mode='lines+markers',
        name='Matlab',
        showlegend=True,
    )
    pio.renderers.default = "browser"

    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.update_xaxes(title_text="Iteration Number")
    fig.update_yaxes(title_text="Lower Bound")
    fig.write_image(lowerBoundVsIterNoFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundVsIterNoFigFilenamePattern.format("html"))
    fig.show()

    fig = go.Figure()
    fig.add_trace(trace3)
    fig.add_trace(trace4)
    fig.update_xaxes(title_text="Elapsed Time (sec)")
    fig.update_yaxes(title_text="Lower Bound")
    fig.write_image(lowerBoundVsElapsedTimeFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundVsElapsedTimeFigFilenamePattern.format("html"))
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
