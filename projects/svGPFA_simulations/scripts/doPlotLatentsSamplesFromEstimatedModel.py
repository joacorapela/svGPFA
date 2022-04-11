
import sys
import pdb
import argparse
import pickle
import torch
import plotly.graph_objs as go
sys.path.append("../src")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estNumber", help="Estimation number", type=int)
    parser.add_argument("nSamples", help="Number of samples to plot", type=int)
    parser.add_argument("--trialToPlot", default=0, help="Trial to plot", type=int)
    parser.add_argument("--latentToPlot", default=0, help="Latent to plot", type=int)
    parser.add_argument("--startTimeToPlot", help="Start time to plot", default=0.0, type=float)
    parser.add_argument("--endTimeToPlot", help="End time to plot", default=4.0, type=float)
    parser.add_argument("--sampleRateInPlot", help="Sample rate in plot", default=1e+2, type=float)
    parser.add_argument("--nodge", help="Kernel covariance matrix on inducing points nodge", default=1e-3, type=float)
    parser.add_argument("--modelFilenamePattern", default="results/{:08d}_estimatedModel.pickle", help="Estimated model filename pattern")
    parser.add_argument("--figFilenamePattern", default="figures/{:08d}_sampledLatents_trial{:d}_latent{:d}_nSamples{:d}.{:s}", help="Figure filename pattern")
    args = parser.parse_args()
    estNumber = args.estNumber
    nSamples = args.nSamples
    trialToPlot = args.trialToPlot
    latentToPlot = args.latentToPlot
    startTimeToPlot = args.startTimeToPlot
    endTimeToPlot = args.endTimeToPlot
    sampleRateInPlot = args.sampleRateInPlot
    nodge = args.nodge
    modelFilenamePattern = args.modelFilenamePattern
    figFilenamePattern = args.figFilenamePattern

    modelFilename = modelFilenamePattern.format(estNumber)
    with open(modelFilename, "rb") as f: modelRes = pickle.load(f)
    model = modelRes["model"]
    nTrials = model.getIndPointsLocs()[0].shape[0]
    timesOneTrial = torch.arange(start=startTimeToPlot, end=endTimeToPlot, step=1.0/sampleRateInPlot)
    times = torch.empty(nTrials, len(timesOneTrial), 1)
    for r in range(nTrials):
        times[r,:,0] = timesOneTrial
    samples = torch.empty(nSamples, nSamples, len(times))
    mean = torch.empty(len(times))
    std = torch.empty(len(times))
    # r for trial, k for latent, n for sample number, t for time
    # latentsSamples[r][k,n,t]
    # latentsMeans[r][k,t]
    # latentsSTDs[r][k,t]
    latentSamples, latentsMeans, latentsSTDs = model._eLL._svEmbeddingAllTimes._svPosteriorOnLatents.sample(times=times, nSamples=nSamples, regFactor=nodge)

    fig = go.Figure()
    for n in range(nSamples):
        trace = go.Scatter(
            x=times[trialToPlot,:,0],
            y=latentSamples[trialToPlot][latentToPlot, n, :],
            # line=dict(color='rgb(0,100,80)'),
            # line=dict(color='blue'),
            mode='lines+markers',
            name="sample {:03d}".format(n),
            showlegend=True,
        )
        fig.add_trace(trace)
    fig.update_xaxes(title_text="Time (sec)")
    fig.update_yaxes(title_text="Latent Value")

    fig.write_image(figFilenamePattern.format(estNumber, trialToPlot, latentToPlot, nSamples, "png"))
    fig.write_html(figFilenamePattern.format(estNumber, trialToPlot, latentToPlot, nSamples, "html"))
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
