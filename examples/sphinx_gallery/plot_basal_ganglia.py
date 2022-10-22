import sys
import os.path
import pdb
import random
import torch
import pickle
import argparse
import configparser
import pandas as pd

sys.path.append(os.path.expanduser("~/svGPFA/pythonCode/lib/"))
import gcnu_common.utils.neuralDataAnalysis
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.configUtils
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils


    parser = argparse.ArgumentParser()
    parser.add_argument("est_init_number", help="estimation init number",
                        type=int)
    parser.add_argument("--region", help="electrode region", type=str,
                        default="GPe")
    parser.add_argument("--save_partial", help="save partial model estimates",
                        action="store_true")
    parser.add_argument("--block_types_indices", help="block types indices",
                        default="[3]")
    parser.add_argument("--min_nSpikes_perNeuron_perTrial",
                        help="min number of spikes per neuron per trial",
                        type=int, default=1)
    parser.add_argument("--save_partial_filename_pattern_pattern",
                        help="pattern for save partial model filename pattern",
                        default="../../results/{:08d}_{{:s}}_estimatedModel.pickle")
    parser.add_argument("--region_spikes_times_filename_pattern",
                        help="region spikes times filename pattern",
                        type=str,
                        default="../../results/00000000_region{:s}_spikes_times_epochedaligned__last_center_out.{:s}")
    parser.add_argument("--events_times_filename",
                        help="events times filename",
                        type=str,
                        default="../../data/022822/s008_tab_m1113182_LR__20210516_173815__probabilistic_switching.df.csv")
    parser.add_argument("--est_init_config_filename_pattern",
                        help="estimation initialization filename pattern",
                        type=str,
                        default="../../init/{:08d}_estimation_metaData.ini")
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--estimRes_metadata_filename_pattern",
                        help="estimation result metadata filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimation_metaData.ini")
    parser.add_argument("--estim_data_for_matlab_filename_pattern",
                        help="estimation dation for matlab filename pattern",
                        type=str,
                        default="../../results/{:08d}_estimationDataForMatlab.mat")
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=str)
    args = parser.parse_args()

    region = args.region
    est_init_number = args.est_init_number
    save_partial = args.save_partial
    block_types_indices = [int(str) for str in args.block_types_indices[1:-1].split(",")]
    min_nSpikes_perNeuron_perTrial = args.min_nSpikes_perNeuron_perTrial
    save_partial_filename_pattern_pattern = args.save_partial_filename_pattern_pattern
    region_spikes_times_filename_pattern = args.region_spikes_times_filename_pattern
    events_times_filename = args.events_times_filename
    est_init_config_filename_pattern = args.est_init_config_filename_pattern
    model_save_filename_pattern = args.model_save_filename_pattern
    estimRes_metadata_filename_pattern = args.estimRes_metadata_filename_pattern
    estim_data_for_matlab_filename_pattern = args.estim_data_for_matlab_filename_pattern

    # get spike_times
    region_spikes_times_metadata_filename = region_spikes_times_filename_pattern.format(region, "ini")
    region_spikes_times_config = configparser.ConfigParser()
    region_spikes_times_config.read(region_spikes_times_metadata_filename)
    epoch_elapsed_time_before = float(region_spikes_times_config["epoch_info"]["epoch_elapsed_time_before"])
    epoch_elapsed_time_after = float(region_spikes_times_config["epoch_info"]["epoch_elapsed_time_after"])

    region_spikes_times_filename = region_spikes_times_filename_pattern.format(region, "pickle")
    with open(region_spikes_times_filename, "rb") as f:
        loadRes = pickle.load(f)
    spikes_times = loadRes["spikes_times"]
    events_times = pd.read_csv(events_times_filename)
    trials_indices = [r for r in range(len(events_times)) if events_times.iloc[r]["block_type_index"] in block_types_indices]
    spikes_times = [spikes_times[r] for r in trials_indices]
    spikes_times, neurons_indices = gcnu_common.utils.neuralDataAnalysis.removeUnitsWithLessSpikesThanThrInAnyTrial(
        spikes_times=spikes_times,
        min_nSpikes_perNeuron_perTrial=min_nSpikes_perNeuron_perTrial)
    spikes_times = [[torch.tensor(spikes_times[r][n])
                     for n in range(len(spikes_times[r]))]
                    for r in range(len(spikes_times))]
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])

    # get initial parameters
    est_init_config_filename = est_init_config_filename_pattern.format(est_init_number)
    est_init_config = configparser.ConfigParser()
    est_init_config.read(est_init_config_filename)

    initial_params, quad_params, kernels_types = \
        svGPFA.utils.initUtils.getInitialAndQuadParamsAndKernelsTypes(
            n_trials=n_trials, n_neurons=n_neurons, args=args,
            config=est_init_config)
    kernels_params0 = initial_params["svPosteriorOnLatents"]["kernelsMatricesStore"]["kernelsParams0"]

    # get optimization parameters
    optim_params = svGPFA.utils.initUtils.getOptimParams(
        args=args, config=est_init_config)
    optim_method = optim_params["optim_method"]
    prior_cov_reg_param = optim_params["prior_cov_reg_param"]

    # build modelSaveFilename
    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estimResMetaDataFilename = "../../results/{:08d}_estimation_metaData.ini".format(estResNumber)
        if not os.path.exists(estimResMetaDataFilename):
            estPrefixUsed = False
    modelSaveFilename = "../../results/{:08d}_estimatedModel.pickle".format(estResNumber)

    # save data for Matlab estimation
    estimationDataForMatlabFilename = estim_data_for_matlab_filename_pattern.format(estResNumber)

    dt_latents = 0.01
    trials_start_times, trials_end_times = svGPFA.utils.initUtils.getTrialsStartEndTimes(
        n_trials=n_trials, args=args, config=est_init_config)
    oneSetLatentsTrialTimes = torch.arange(trials_start_times[0],
                                           trials_end_times[0], dt_latents)
    qSVec0, qSDiag0 = svGPFA.utils.miscUtils.getQSVecsAndQSDiagsFromQSCholVecs(
        qsCholVecs=initial_params["svPosteriorOnLatents"]["svPosteriorOnIndPoints"]["cholVecs"])
    qMu0 = initial_params["svPosteriorOnLatents"]["svPosteriorOnIndPoints"]["mean"]
    C0 = initial_params["svEmbedding"]["C0"]
    d0 = initial_params["svEmbedding"]["d0"]
    Z0 = initial_params["svPosteriorOnLatents"]["kernelsMatricesStore"]["inducingPointsLocs0"]
    legQuadPoints = quad_params["legQuadPoints"]
    legQuadWeights = quad_params["legQuadWeights"]
    trials_start_times, trials_end_times = svGPFA.utils.initUtils.getTrialsStartEndTimes(
        n_trials=n_trials, args=args, config=est_init_config)
    trials_lengths = [trials_end_times[r] - trials_start_times[r]
                      for r in range(n_trials)]
    n_latents = len(Z0)
    latentsTrialsTimes = [oneSetLatentsTrialTimes for k in range(n_latents)]
    svGPFA.utils.miscUtils.saveDataForMatlabEstimations(
        qMu=qMu0, qSVec=qSVec0, qSDiag=qSDiag0,
        C=C0, d=d0,
        indPointsLocs=Z0,
        legQuadPoints=legQuadPoints,
        legQuadWeights=legQuadWeights,
        kernelsTypes=kernels_types,
        kernelsParams=kernels_params0,
        spikesTimes=spikes_times,
        indPointsLocsKMSRegEpsilon=prior_cov_reg_param,
        trialsLengths=torch.tensor(trials_lengths).reshape(-1,1),
        latentsTrialsTimes=latentsTrialsTimes,
        emMaxIter=optim_params["em_max_iter"],
        eStepMaxIter=optim_params["estep_optim_params"]["max_iter"],
        mStepEmbeddingMaxIter=optim_params["mstep_embedding_optim_params"]["max_iter"],
        mStepKernelsMaxIter=optim_params["mstep_kernels_optim_params"]["max_iter"],
        mStepIndPointsMaxIter=optim_params["mstep_indpointslocs_optim_params"]["max_iter"],
        saveFilename=estimationDataForMatlabFilename)

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(kernels_types=kernels_types,
                                                      kernels_params=kernels_params0)

    # create model
    kernelMatrixInvMethod = svGPFA.stats.svGPFAModelFactory.kernelMatrixInvChol
    indPointsCovRep = svGPFA.stats.svGPFAModelFactory.indPointsCovChol
    model = svGPFA.stats.svGPFAModelFactory.SVGPFAModelFactory.buildModelPyTorch(
        conditionalDist=svGPFA.stats.svGPFAModelFactory.PointProcess,
        linkFunction=svGPFA.stats.svGPFAModelFactory.ExponentialLink,
        embeddingType=svGPFA.stats.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels, kernelMatrixInvMethod=kernelMatrixInvMethod,
        indPointsCovRep=indPointsCovRep)

    model.setInitialParamsAndData(measurements=spikes_times,
                                  initialParams=initial_params,
                                  eLLCalculationParams=quad_params,
                                  priorCovRegParam=prior_cov_reg_param)

    # save estimated values
    estimResConfig = configparser.ConfigParser()
    estimResConfig["data_params"] = {"region": region,
                                     "trials_indices": trials_indices,
                                     "nLatents": n_latents,
                                     "from_time": trials_start_times[0].item(),
                                     "to_time": trials_end_times[0].item()}
    estimResConfig["optim_params"] = optim_params
    estimResConfig["estimation_params"] = {"est_init_number": est_init_number}
    with open(estimResMetaDataFilename, "w") as f: estimResConfig.write(f)

    # maximize lower bound
    def getKernelParams(model):
        kernelParams = model.getKernelsParams()[0]
        return kernelParams

    # maximize lower bound
    svEM = svGPFA.stats.svEM.SVEM_PyTorch()
    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
        svEM.maximize(model=model, optim_params=optim_params,
                      method=optim_method,
                      getIterationModelParamsFn=getKernelParams)

    resultsToSave = {"neurons_indices": neurons_indices, "lowerBoundHist": lowerBoundHist, "elapsedTimeHist": elapsedTimeHist, "terminationInfo": terminationInfo, "iterationModelParams": iterationsModelParams, "model": model}
    with open(modelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)
    print("Saved results to {:s}".format(modelSaveFilename))

