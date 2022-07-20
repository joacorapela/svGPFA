
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
                        type=float, default=750.0)
    parser.add_argument("--to_time", help="ending spike analysis time",
                        type=float, default=2500.0)
    parser.add_argument("--dtCIF", help="neuron to plot", type=float, default=1.0)
    parser.add_argument("--modelSaveFilenamePattern",
                        help="model save filename pattern",
                        default="results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--spikesFilenamePattern",
                        help="filename pattern for spikes sampled from the expected posterior CIF",
                        default="results/{:08d}_spikesFromExpectedPosteriorCIF.pickle")
    args = parser.parse_args()

    estResNumber = args.estResNumber
    from_time = args.from_time
    to_time = args.to_time
    dtCIF = args.dtCIF
    modelSaveFilename = args.modelSaveFilenamePattern.format(estResNumber)
    spikesFilenamePattern = args.spikesFilenamePattern

    with open(modelSaveFilename, "rb") as f: estResults = pickle.load(f)
    model = estResults["model"]

    T = to_time - from_time
    times = torch.arange(from_time, to_time, dtCIF)
    epCIFs = model.computeExpectedPosteriorCIFs(times=times)
    nTrials = len(epCIFs)
    nNeurons = len(epCIFs[0])
    spikesTimes = [[] for n in range(nTrials)]
    sampler = stats.pointProcess.sampler.Sampler()
    for r in range(nTrials):
        spikesTimes[r] = [[] for r in range(nNeurons)]
        for n in range(nNeurons):
            print("Processing trial {:d} and neuron {:d}".format(r, n))
            spikesTimes[r][n] = torch.tensor(sampler.sampleInhomogeneousPP_timeRescaling(cifTimes=times, cifValues=epCIFs[r][n], T=T))

    results = {"spikesTimes": spikesTimes}
    spikesFilename = spikesFilenamePattern.format(estResNumber)
    with open(spikesFilename, "wb") as f: pickle.dump(results, f)

if __name__=="__main__":
    main(sys.argv)
