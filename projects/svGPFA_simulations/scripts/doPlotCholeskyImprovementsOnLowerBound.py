
import sys
import os
import torch
import pdb
import pickle
import argparse
import configparser
import matplotlib.pyplot as plt
from scipy.io import loadmat
sys.path.append("../src")
import plotly.graph_objs as go
import plotly.offline

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pEstNumberCholRank1", help="Python's estimation number for rank one covariance representation and Choleksy method for matrix inverse", type=int, default=32807880)
    parser.add_argument("--pEstNumberPInvRank1", help="Python's estimation number for rank one covariance representation and pseudo-inverse method for matrix inverse", type=int, default=50477314)
    parser.add_argument("--pEstNumberCholChol", help="Python's estimation number for Cholesky covariance representation and Cholesky method for matrix inverse", type=int, default=82308174)
    parser.add_argument("--deviceName", help="name of device (cpu or cuda)", default="cpu")
    args = parser.parse_args()
    pEstNumberCholRank1 = args.pEstNumberCholRank1
    pEstNumberPInvRank1 = args.pEstNumberPInvRank1
    pEstNumberCholChol = args.pEstNumberCholChol
    deviceName = args.deviceName

    ylim = [-4810, -4640]
    pEstimMetaDataFilename = "results/{:08d}_leasSimulation_estimation_metaData_{:s}.ini".format(pEstNumberCholRank1, deviceName)
    pModelCholRank1SaveFilename = "results/{:08d}_leasSimulation_estimatedModel_cpu.pickle".format(pEstNumberCholRank1)
    pModelPInvRank1SaveFilename = "results/{:08d}_leasSimulation_estimatedModelPinv_cpu.pickle".format(pEstNumberPInvRank1)
    pModelCholCholSaveFilename = "results/{:08d}_leasSimulation_estimatedModelChol_cpu.pickle".format(pEstNumberCholChol)
    pEstConfig = configparser.ConfigParser()
    pEstConfig.read(pEstimMetaDataFilename)
    mEstNumber = int(pEstConfig["data"]["mEstNumber"])

    mEstConfig = configparser.ConfigParser()
    mEstConfig.read("../../matlabCode/scripts/results/{:08d}-pointProcessEstimationParams.ini".format(mEstNumber))
    mModelSaveFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat".format(mEstNumber)
    mSimNumber = int(mEstConfig["data"]["simulationNumber"])
    ppSimulationFilename = os.path.join(os.path.dirname(__file__), "../../matlabCode/scripts/results/{:08d}-pointProcessSimulation.mat".format(mSimNumber))

    marker = 'x'
    lowerBoundVsIterFigFilenamePattern = "figures/{:08d}-{:08d}-{:08d}-lowerBoundVsIter.{{:s}}".format(pEstNumberCholRank1, pEstNumberCholRank1, pEstNumberCholChol)
    lowerBoundVsElapsedTimeFigFilenamePattern = "figures/{:08d}-{:08d}-{:08d}-lowerBoundVsElapsedTime.{{:s}}".format(pEstNumberCholRank1, pEstNumberCholRank1, pEstNumberCholChol)

    with open(pModelCholRank1SaveFilename, "rb") as f: res = pickle.load(f)
    pLowerBoundCholRank1 = res["lowerBoundHist"]
    pElapsedTimeCholRank1 = res["elapsedTimeHist"]

    with open(pModelPInvRank1SaveFilename, "rb") as f: res = pickle.load(f)
    pLowerBoundPInvRank1 = res["lowerBoundHist"]
    pElapsedTimePInvRank1 = res["elapsedTimeHist"]

    with open(pModelCholCholSaveFilename, "rb") as f: res = pickle.load(f)
    pLowerBoundCholChol = res["lowerBoundHist"]
    pElapsedTimeCholChol = res["elapsedTimeHist"]

    loadRes = loadmat(mModelSaveFilename)
    mIter = torch.cat(tuple(torch.from_numpy(loadRes["lowerBound"])))
    mElapsedTime = torch.cat(tuple(torch.from_numpy(loadRes["elapsedTime"])))

    traceIterCholRank1 = go.Scatter(
        y=pLowerBoundCholRank1,
        line=dict(color='rgb(217,30,30)'),
        mode='lines+markers',
        name='P-Chol-Rank1',
        showlegend=True,
    )
    traceIterPInvRank1 = go.Scatter(
        y=pLowerBoundPInvRank1,
        line=dict(color='rgb(242,143,56)'),
        mode='lines+markers',
        name='P-PInv-Rank1',
        showlegend=True,
    )
    traceIterCholChol = go.Scatter(
        y=pLowerBoundCholChol,
        line=dict(color='rgb(242,211,56)'),
        mode='lines+markers',
        name='P-Chol-Chol',
        showlegend=True,
    )
    traceIterM = go.Scatter(
        y=mIter,
        line=dict(color='blue'),
        mode='lines+markers',
        name='M-PInv-Rank1',
        showlegend=True,
    )
    fig = go.Figure()
    fig.add_trace(traceIterCholRank1)
    fig.add_trace(traceIterPInvRank1)
    fig.add_trace(traceIterCholChol)
    fig.add_trace(traceIterM)
    fig.update_xaxes(title_text="Iteration Number")
    fig.update_yaxes(title_text="Lower Bound", range=ylim)
    fig.write_image(lowerBoundVsIterFigFilenamePattern.format("png"))
    plotly.offline.plot(fig, filename=lowerBoundVsIterFigFilenamePattern.format("html"))

    traceElapsedTimeCholRank1 = go.Scatter(
        x=pElapsedTimeCholRank1,
        y=pLowerBoundCholRank1,
        line=dict(color='rgb(217,30,30)'),
        mode='lines+markers',
        name='P-Chol-Rank1',
        showlegend=True,
    )
    traceElapsedTimePInvRank1 = go.Scatter(
        x=pElapsedTimePInvRank1,
        y=pLowerBoundPInvRank1,
        line=dict(color='rgb(242,143,56)'),
        mode='lines+markers',
        name='P-PInv-Rank1',
        showlegend=True,
    )
    traceElapsedTimeCholChol = go.Scatter(
        x=pElapsedTimeCholChol,
        y=pLowerBoundCholChol,
        line=dict(color='rgb(242,211,56)'),
        mode='lines+markers',
        name='P-Chol-Chol',
        showlegend=True,
    )
    traceElapsedTimeM = go.Scatter(
        x=mElapsedTime,
        y=mIter,
        # line=dict(color='rgb(0,100,80)'),
        line=dict(color='blue'),
        mode='lines+markers',
        name='M-PInv-Rank1',
        showlegend=True,
    )
    fig = go.Figure()
    fig.add_trace(traceElapsedTimeCholRank1)
    fig.add_trace(traceElapsedTimePInvRank1)
    fig.add_trace(traceElapsedTimeCholChol)
    fig.add_trace(traceElapsedTimeM)
    fig.update_xaxes(title_text="Elapsed Time (sec)")
    fig.update_yaxes(title_text="Lower Bound", range=ylim)
    fig.write_image(lowerBoundVsElapsedTimeFigFilenamePattern.format("png"))
    plotly.offline.plot(fig, filename=lowerBoundVsElapsedTimeFigFilenamePattern.format("html"))

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
