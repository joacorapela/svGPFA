
import sys
import argparse
import configparser
import pickle
import torch

sys.path.append("../../src")
import stats.pointProcess.sampler

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--from_time", help="starting spike analysis time",
                        type=float, default=0.0)
    parser.add_argument("--to_time", help="ending spike analysis time",
                        type=float, default=2500.0)
    parser.add_argument("--dtCIF", help="neuron to plot", type=float, default=1.0)
    parser.add_argument("--modelSaveFilenamePattern",
                        help="model save filename pattern",
                        default="results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--spikesFilenamePattern",
                        help="filename pattern for spikes sampled from the expected posterior CIF",
                        default="results/{:08d}_spikes_sampledCIF.pickle")
    args = parser.parse_args()

    estResNumber = args.estResNumber
    from_time = args.from_time
    to_time = args.to_time
    dtCIF = args.dtCIF
    modelSaveFilename = args.modelSaveFilenamePattern.format(estResNumber)
    spikesFilenamePattern = args.spikesFilenamePattern

    with open(modelSaveFilename, "rb") as f: estResults = pickle.load(f)
    model = estResults["model"]
    neurons_labels = estResults["neurons_labels"]
    T = to_time - from_time
    nTrials = model.getIndPointsLocs()[0].shape[0]
    times_one_trial = torch.arange(from_time, to_time, dtCIF)
    times = torch.outer(torch.ones(nTrials), times_one_trial)
    times = torch.unsqueeze(input=times, dim=2)
    sampledCIFs = model.sampleCIFs(times=times)
    nTrials = len(sampledCIFs)
    nNeurons = len(neurons_labels)
    spikesTimes = [[] for i in range(nTrials)]
    sampler = stats.pointProcess.sampler.Sampler()
    for r in range(nTrials):
        spikesTimes[r] = [[] for r in range(nNeurons)]
        for i in range(nNeurons):
            neuron_label = neurons_labels[i]
            print("Processing trial {:d} and neuron {:d}".format(r,
                                                                 neuron_label))
            spikesTimes[r][i] = torch.tensor(sampler.sampleInhomogeneousPP_timeRescaling(cifTimes=times[r,:], cifValues=sampledCIFs[r][i], T=T))

    results = {"spikesTimes": spikesTimes, "neurons_labels": neurons_labels}
    spikesFilename = spikesFilenamePattern.format(estResNumber)
    with open(spikesFilename, "wb") as f: pickle.dump(results, f)

if __name__=="__main__":
    main(sys.argv)
