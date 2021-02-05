
import sys
import os
import torch
import pdb
import pickle
import argparse
import configparser
import pandas
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
import scipy.io
import scipy.stats
import numpy as np
sys.path.append("../src")
import plot.svGPFA.plotUtilsPlotly
import stats.hypothesisTests.permutationTests
import utils.svGPFA.miscUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("pEstNumber", help="Python's estimation number", type=int)
    parser.add_argument("--deviceName", help="name of device (cpu or cuda)", default="cpu")
    args = parser.parse_args()
    pEstNumber = args.pEstNumber
    deviceName = args.deviceName

    marker = "x"
    tLabel = "True"
    ylim = [-6, 2]
    nResamples = 10000
    # pLabelPattern = "$\text{Python} (R^2={:.02f})$"
    # mLabelPattern = "$\text{Matlab} (R^2={:.02f})$"
    pLabelPattern = "Python (R<sup>2</sup>={:.02f})"
    mLabelPattern = "Matlab (R<sup>2</sup>={:.02f})"

    pEstimMetaDataFilename = "results/{:08d}_leasSimulation_estimation_metaData_{:s}.ini".format(pEstNumber, deviceName)
    pEstConfig = configparser.ConfigParser()
    pEstConfig.read(pEstimMetaDataFilename)
    mEstNumber = int(pEstConfig["data"]["mEstNumber"])

    mEstParamsFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessEstimationParams.ini".format(mEstNumber)
    mEstConfig = configparser.ConfigParser()
    mEstConfig.read(mEstParamsFilename)
    mSimNumber = int(mEstConfig["data"]["simulationNumber"])

    mSimFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessSimulation.mat".format(mSimNumber)
    mModelSaveFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat".format(mEstNumber)
    pModelSaveFilename = "results/{:08d}_leasSimulation_estimatedModel_cpu.pickle".format(pEstNumber)
    staticFigFilename = "figures/{:08d}_truePythonMatlabCIFsPointProcess.png".format(pEstNumber)
    dynamicFigFilename = "figures/{:08d}_truePythonMatlabCIFsPointProcess.html".format(pEstNumber)

    loadRes = scipy.io.loadmat(mSimFilename)
    nLatents = loadRes["trueLatents"].shape[1]
    nTrials = loadRes["trueLatents"].shape[0]
    nSamples = loadRes["testTimes"][:,0].shape[0]
    tTimes = torch.from_numpy(loadRes["testTimes"][:,0]).type(torch.DoubleTensor)
    tLatents_tmp = [[torch.from_numpy(loadRes["trueLatents"][t,l]).type(torch.DoubleTensor).squeeze() for l in range(nLatents)] for t in range(nTrials)]
    tLatents = torch.empty((nTrials, nSamples, nLatents), dtype=torch.double)
    for t in range(nTrials):
        for l in range(nLatents):
            tLatents[t,:,l] = tLatents_tmp[t][l]

    tC = torch.from_numpy(loadRes['prs'][0,0][0]).type(torch.DoubleTensor)
    nNeurons = tC.shape[0]
    td = torch.from_numpy(loadRes['prs'][0,0][1]).type(torch.DoubleTensor)
    tCIFs = utils.svGPFA.miscUtils.getCIFs(C=tC, d=td, latents=tLatents)

    loadRes = scipy.io.loadmat(mModelSaveFilename)
    mTimes = torch.from_numpy(loadRes["testTimes"][:,0]).type(torch.DoubleTensor).squeeze()
    mMeanLatents_tmp = torch.from_numpy(loadRes["meanEstimatedLatents"]).type(torch.DoubleTensor)
    mMeanLatents = torch.empty((nTrials, nSamples, nLatents), dtype=torch.double)
    for t in range(nTrials):
        for l in range(nLatents):
            mMeanLatents[t,:,l] = mMeanLatents_tmp[:,l,t]
    mVarLatents_tmp = torch.from_numpy(loadRes["varEstimatedLatents"]).type(torch.DoubleTensor)
    mVarLatents = torch.empty((nTrials, nSamples, nLatents), dtype=torch.double)
    for t in range(nTrials):
        for l in range(nLatents):
            mVarLatents[t,:,l] = mVarLatents_tmp[:,l,t]
    mC = torch.from_numpy(loadRes["m"]['prs'][0,0]["C"][0,0]).type(torch.DoubleTensor)
    md = torch.from_numpy(loadRes["m"]['prs'][0,0]["b"][0,0]).type(torch.DoubleTensor)
    mCIFs = utils.svGPFA.miscUtils.getCIFs(C=mC, d=md, latents=mMeanLatents)

    with open(pModelSaveFilename, "rb") as f: res = pickle.load(f)
    pModel = res["model"]
    embeddingParams = pModel.getSVEmbeddingParams()
    pC = embeddingParams[0]
    pd = torch.unsqueeze(embeddingParams[1], 1)

    with torch.no_grad():
        pTestMuK, _ = pModel.predictLatents(newTimes=mTimes)
    pCIFs = utils.svGPFA.miscUtils.getCIFs(C=pC, d=pd, latents=pTestMuK)
    pTimes = mTimes

    r2s = []
    methods = []
    trials = []
    neurons = []
    for r in range(nTrials):
        for n in range(nNeurons):
            tCIF = tCIFs[r,:,n]
            mCIF = mCIFs[r,:,n]
            pCIF = pCIFs[r,:,n]
            meanTCIF = torch.mean(tCIF)
            ssTot = torch.sum((tCIF-meanTCIF)**2)
            pSSRes = torch.sum((pCIF-tCIF)**2)
            mSSRes = torch.sum((mCIF-tCIF)**2)
            pR2 = (1-(pSSRes/ssTot)).item()
            mR2 = (1-(mSSRes/ssTot)).item()

            trials.append(r)
            neurons.append(n)
            methods.append("Python")
            r2s.append(pR2)

            trials.append(r)
            neurons.append(n)
            methods.append("Matlab")
            r2s.append(mR2)
    d = {"r2": r2s, "method": methods, "trial": trials, "neuron": neurons}
    dfm = pandas.DataFrame(data=d)
    dfu = dfm.set_index(['trial', 'neuron', 'method']).unstack(level=-1)
    dfuNumpy = dfu.to_numpy()
    # pTTestRes = scipy.stats.ttest_rel(a=dfuNumpy[:,0], b=dfuNumpy[:,1])
    # title = "Paired t-test, H<sub>0</sub>: R<sup>2</sup>(Matlab)=R<sup>2</sup>(Python); t={:.02f}, p={:.04f}".format(pTTestRes.statistic, pTTestRes.pvalue)
    validIndices = np.nonzero((np.abs(dfuNumpy)<6).all(1))[0]
    dfOutliersRemoved = dfuNumpy[validIndices,:]
    permRes = stats.hypothesisTests.permutationTests.permuteDiffPairedMeans(data1=dfOutliersRemoved[:,0], data2=dfOutliersRemoved[:,1], nResamples=nResamples)
    permResP = float(len(np.nonzero(np.absolute(permRes.t)>abs(permRes.t0))[0]))/len(permRes.t)
    title = "H<sub>0</sub>: R<sup>2</sup>(Matlab)=R<sup>2</sup>(Python); stat={:.02f}, p={:.04f}".format(permRes.t0, permResP)
    fig = px.box(dfm, x="method", y="r2", points="all", color="method", hover_data=["trial", "neuron"], title=title)
    fig.update_yaxes(title_text="R<sup>2</sup>", range=ylim)
    fig.update_xaxes(title_text="")

    fig.write_image(staticFigFilename)
    fig.write_html(dynamicFigFilename)
    pio.renderers.default = "browser"
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
