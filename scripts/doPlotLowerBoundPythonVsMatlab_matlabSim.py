
import sys
import os
import torch
import pdb
import pickle
import argparse
import configparser
import matplotlib.pyplot as plt
from scipy.io import loadmat
import plotly.graph_objs as go
import plotly.offline
import plotly.io as pio
sys.path.append("../src")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("pEstNumber", help="Python's estimation number", type=int)
    parser.add_argument("--deviceName", help="name of device (cpu or cuda)", default="cpu")
    args = parser.parse_args()
    pEstNumber = args.pEstNumber
    deviceName = args.deviceName

    pEstimMetaDataFilename = "results/{:08d}_leasSimulation_estimation_metaData_{:s}.ini".format(pEstNumber, deviceName)
    pModelSaveFilename = "results/{:08d}_leasSimulation_estimatedModel_{:s}.pickle".format(pEstNumber, deviceName)
    pEstConfig = configparser.ConfigParser()
    pEstConfig.read(pEstimMetaDataFilename)
    mEstNumber = int(pEstConfig["data"]["mEstNumber"])

    mEstConfig = configparser.ConfigParser()
    mEstConfig.read("../../matlabCode/scripts/results/{:08d}-pointProcessEstimationParams.ini".format(mEstNumber))
    mModelSaveFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat".format(mEstNumber)
    mSimNumber = int(mEstConfig["data"]["simulationNumber"])
    ppSimulationFilename = os.path.join(os.path.dirname(__file__), "../../matlabCode/scripts/results/{:08d}-pointProcessSimulation.mat".format(mSimNumber))

    marker = 'x'
    lowerBoundVsIterNoFigFilenamePattern = "figures/{:08d}-lowerBoundVsIterNo.{{:s}}".format(pEstNumber)
    lowerBoundVsElapsedTimeFigFilenamePattern = "figures/{:08d}-lowerBoundVsRuntime.{{:s}}".format(pEstNumber)

    with open(pModelSaveFilename, "rb") as f: res = pickle.load(f)
    # pLowerBound = -torch.stack(res["lowerBoundHist"]).detach().numpy()
    pLowerBound = res["lowerBoundHist"]
    pElapsedTime = res["elapsedTimeHist"]

    loadRes = loadmat(mModelSaveFilename)
    mLowerBound = torch.cat(tuple(torch.from_numpy(loadRes["lowerBound"])))
    mElapsedTime = torch.cat(tuple(torch.from_numpy(loadRes["elapsedTime"])))

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
