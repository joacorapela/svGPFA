
import sys
import os
import torch
import pdb
import pickle
import matplotlib.pyplot as plt
from scipy.io import loadmat
sys.path.append(os.path.expanduser("../src"))
import plotly.graph_objs as go
import plotly.offline

def main(argv):
    marker = 'x'
    pModelSaveFilename = "results/estimationResLeasSimulation.pickle"
    mLowerBoundFilename = "../../matlabCode/scripts/results/lowerBound.mat"
    lowerBoundVsIterNoStaticFigFilename = "figures/lowerBoundVsIterNo.png"
    lowerBoundVsIterNoDynamicFigFilename = "figures/lowerBoundVsIterNo.html"
    lowerBoundVsElapsedTimeStaticFigFilename = "figures/lowerBoundVsRuntime.png"
    lowerBoundVsElapsedTimeDynamicFigFilename = "figures/lowerBoundVsRuntime.html"

    with open(pModelSaveFilename, "rb") as f: res = pickle.load(f)
    pLowerBound = -torch.stack(res["lowerBoundHist"]).detach().numpy()
    pElapsedTime = res["elapsedTimeHist"]

    loadRes = loadmat(mLowerBoundFilename)
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
    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.update_xaxes(title_text="Iteration Number")
    fig.update_yaxes(title_text="Lower Bound")
    fig.write_image(lowerBoundVsIterNoStaticFigFilename)
    plotly.offline.plot(fig, filename=lowerBoundVsIterNoDynamicFigFilename)

    fig = go.Figure()
    fig.add_trace(trace3)
    fig.add_trace(trace4)
    fig.update_xaxes(title_text="Elapsed Time (sec)")
    fig.update_yaxes(title_text="Lower Bound")
    fig.write_image(lowerBoundVsElapsedTimeStaticFigFilename)
    plotly.offline.plot(fig, filename=lowerBoundVsElapsedTimeDynamicFigFilename)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
