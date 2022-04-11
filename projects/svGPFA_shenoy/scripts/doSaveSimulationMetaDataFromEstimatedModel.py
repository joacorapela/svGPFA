
import sys
import argparse
import configparser
import pickle

sys.path.append("../../src")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--dtCIF", help="neuron to plot", type=float, default=1.0)
    parser.add_argument("--estInitConfigFilenamePattern",
                        help="estimation init configuration filename pattern",
                        default="data/{:08d}_estimation_metaData.ini")
    parser.add_argument("--estimResMetaDataFilenamePattern",
                        help="estimation result meta data filename pattern", 
                        default="results/{:08d}_estimation_metaData.ini")
    parser.add_argument("--modelSaveFilenamePattern",
                        help="model save filename pattern",
                        default="results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--simInitConfigFilenamePattern",
                        help="simulation initialization configuration filename patter",
                        default= "data/{:08d}_simulation_metaData.ini")
    args = parser.parse_args()

    estResNumber = args.estResNumber
    dtCIF = args.dtCIF
    estInitConfigFilename = args.estInitConfigFilenamePattern.format(estResNumber)
    estimResMetaDataFilename = args.estimResMetaDataFilenamePattern.format(estResNumber)
    modelSaveFilename = args.modelSaveFilenamePattern.format(estResNumber)
    simInitConfigFilenamePattern = args.simInitConfigFilenamePattern

    estInitConfig = configparser.ConfigParser()
    estInitConfig.read(estInitConfigFilename)

    simPrefixUsed = True
    while simPrefixUsed:
        simInitNumber = random.randint(0, 10**8)
        simInitConfigFilename = simInitConfigFilenamePattern.format(simInitNumber)
        if not os.path.exists(simInitConfigFilename):
           simPrefixUsed = False

    estimResConfig = configparser.ConfigParser()
    estimResConfig.read(estimResMetaDataFilename)
    indPointsLocsKMSRegEpsilon = estimResConfig["control_variables"]["indPointsLocsKMSRegEpsilon"]
    latentsCovRegEpsilon = estimResConfig["control_variables"]["latentsCovRegEpsilon"]
    from_time = float(estimResConfig["data_params"]["from_time"])
    to_time = float(estimResConfig["data_params"]["to_time"])
    trials_indices = [float(str) for str in estimResConfig["data_params"]["trials_indices"][1:-1].split(",")]
    nTrials = len(trials_indices)
    trial_lengths = [to_time-from_time for i in trials_indices]

    nTrials = len(trials_indices)
    trials_start_times = [from_time for i in range(nTrials)]
    trials_end_times = [to_time for i in range(nTrials)]
    trials_lengths = [trials_end_times[i]-trials_start_times[i] for i in range(nTrials)]
    with open(modelSaveFilename, "rb") as f: estResults = pickle.load(f)
    model = estResults["model"]
    C, d = model.getSVEmbeddingParams()

    import pdb; pdb.set_trace()

    # control_variables
    nNeurons, nLatents = C.shape
    trials_lengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]

    estInitConfig = configparser.ConfigParser()
    estInitConfig.read(estInitConfigFilename)

    estimResConfig = configparser.ConfigParser()
    estimResConfig["control_variables"] = {"nneurons": nNeurons,
                                           "nlatents": nLatents,
                                           "trials_lengths": trials_lengths,
                                           "dtCIF": dtCIF,
                                           "indPointsLocsKMSRegEpsilon":
                                            indPointsLocsKMSRegEpsilon,
                                           "latentsCovRegEpsilon":
                                            latentsCovRegEpsilon,
                                           "firstIndPointLoc": from_time,
                                           "estResNumber": estResNumber}

    # kernel_params
    kernels = model.getKernels()
    kenrels_params_dict = {}
    for k, kernel in enumerate(kernels):
        if type(kernel).__name__ == "ExponentialQuadraticKernel":
            kenrels_params_dict["kTypeLatent{:d}".format(k)] = "exponentialQuadratic"
            kenrels_params_dict["kScaleValueLatent{:d}".format(k)] = kernels[k]._scale
            kenrels_params_dict["kLengthScaledValueLatent{:d}".format(k)] = kernels[k]._params[0]
            kenrels_params_dict["kLengthScaleLatent{:d}".format(k)] = kernels[k]._lengthscaleScale
    estimResConfig["kernel_params"] = kenrels_params_dict

    # indPoints_params
    ind_points_locs = model.getIndPointsLocs()
    indPoints_params_dict = {}
    for r, trial_ind_points_locs in enumerate(ind_points_locs):
        nLatents = trial_ind_points_locs.shape[0]
        for k in range(nLatents):
            indPoints_params_dict["indPointsLocsLatent{:d}Trial{:d}".format(k,r)] = trial_ind_points_locs[k,:,0].tolist()
    estimResConfig["indPoints_params"] = indPoints_params_dict

    estimResConfig["optim_params"] = optimParams
    estimResConfig["estimation_params"] = {"estInitNumber": estInitNumber, "nIndPointsPerLatent": nIndPointsPerLatent}
    with open(estimResMetaDataFilename, "w") as f: estimResConfig.write(f)

if __name__=="__main__":
    main(sys.argv)
