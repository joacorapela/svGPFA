
import configparser
import pandas as pd
import torch

import svGPFA.utils.initUtils
import gcnu_common.utils.config_dict


def test_getParam_0():
    ''' test get param from dynamic params'''

    true_n_latents = 3
    dynamic_params = {"model_structure_params":
                      {"n_latents": str(true_n_latents)}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    config_file_params = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    n_latents = svGPFA.utils.initUtils.getParam(
        section_name="model_structure_params",
        param_name="n_latents",
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=None,
        conversion_funct=int)
    assert true_n_latents == n_latents


def test_getParam_1():
    ''' test get param from config_file params'''
    true_trials_end_time = 1.0
    true_n_latents = 3
    dynamic_params = {"model_structure_params":
                      {"n_latents": str(true_n_latents)}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    config_file_params = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    trials_end_time = svGPFA.utils.initUtils.getParam(
        section_name="data_structure_params",
        param_name="trials_end_time",
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=None,
        conversion_funct=float)
    assert true_trials_end_time == trials_end_time


def test_getLinearEmbeddingParams_0(n_neurons=20, n_latents=7):
    true_C = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                          dtype=torch.double)
    true_d = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.double)
    dynamic_params = {"embedding_params": {"c": true_C, "d": true_d}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    config_file_params = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    C0, d0 = svGPFA.utils.initUtils.getLinearEmbeddingParams0(
        n_neurons=n_neurons, n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=None)
    assert torch.all(true_C == C0)
    assert torch.all(true_d == d0)


def test_getLinearEmbeddingParams_1(n_neurons=20, n_latents=7):
    true_n_latents = 3
    dynamic_params = {"model_structure_params":
                      {"n_latents": str(true_n_latents)}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_latents=n_latents)
    config_file_params = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    true_C_filename = config["embedding_params"]["c_filename"]
    true_C_df = pd.read_csv(true_C_filename, header=None)
    true_C = torch.from_numpy(true_C_df.values).type(torch.double)
    true_d_filename = config["embedding_params"]["d_filename"]
    true_d_df = pd.read_csv(true_d_filename, header=None)
    true_d = torch.from_numpy(true_d_df.values).type(torch.double)
    C0, d0 = svGPFA.utils.initUtils.getLinearEmbeddingParams0(
        n_neurons=n_neurons, n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    assert torch.all(true_C == C0)
    assert torch.all(true_d == d0)


def test_getTrialsStartEndTimes_0():
    n_trials = 15
    n_neurons = 100
    n_latents = 5
    dynamic_params = {"data_structure_params": {"trials_start_time": 3.0,
                                                "trials_end_time": 12.0}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    config_file_params = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_latents=n_latents)
    true_trials_start_times = torch.Tensor(
        [dynamic_params["data_structure_params"]["trials_start_time"] 
         for r in range(n_trials)])
    true_trials_end_times = torch.Tensor(
        [dynamic_params["data_structure_params"]["trials_end_time"] 
         for r in range(n_trials)])
    trials_start_times, trials_end_times = svGPFA.utils.initUtils.getTrialsStartEndTimes(
        n_trials=n_trials,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    assert torch.all(true_trials_start_times == trials_start_times)
    assert torch.all(true_trials_end_times == trials_end_times)


def test_getTrialsStartEndTimes_1():
    n_trials = 15
    n_neurons = 100
    n_latents = 5
    dynamic_params = {"model_structure_params":
                      {"n_latents": str(n_latents)}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    config_file_params = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_latents=n_latents)
    true_trials_start_times = torch.Tensor(
        [float(config_file_params["data_structure_params"]["trials_start_time"])
         for r in range(n_trials)])
    true_trials_end_times = torch.Tensor(
        [float(config_file_params["data_structure_params"]["trials_end_time"])
         for r in range(n_trials)])
    trials_start_times, trials_end_times = svGPFA.utils.initUtils.getTrialsStartEndTimes(
        n_trials=n_trials,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    assert torch.all(true_trials_start_times == trials_start_times)
    assert torch.all(true_trials_end_times == trials_end_times)


def test_getTrialsStartEndTimes_2():
    n_trials = 15
    n_neurons = 100
    n_latents = 5
    dynamic_params = {"model_structure_params":
                      {"n_latents": str(n_latents)}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    config_file_params = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    del config_file_params["data_structure_params"]
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_latents=n_latents)
    true_trials_start_times = torch.Tensor(
        [float(default_params["data_structure_params"]["trials_start_time"])
         for r in range(n_trials)])
    true_trials_end_times = torch.Tensor(
        [float(default_params["data_structure_params"]["trials_end_time"])
         for r in range(n_trials)])
    trials_start_times, trials_end_times = svGPFA.utils.initUtils.getTrialsStartEndTimes(
        n_trials=n_trials,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    assert torch.all(true_trials_start_times == trials_start_times)
    assert torch.all(true_trials_end_times == trials_end_times)


def test_getKernelsParams0AndTypes_0(nLatents=3, foreceKernelsUnitScale=True):
    kernelType = "exponentialQuadratic"
    lengthscaleValue = 2.0

    estInitConfig = configparser.ConfigParser()
    estInitConfig["kernels_params"] = \
        {"kTypeLatents": kernelType,
         "kLengthscaleValueLatents": lengthscaleValue}

    params0, kernels_types = svGPFA.utils.initUtils.getKernelsParams0AndTypes(
        nLatents=nLatents, foreceKernelsUnitScale=foreceKernelsUnitScale,
        config=estInitConfig)

    for k in range(nLatents):
        assert params0[k].item() == lengthscaleValue
        assert kernels_types[k] == kernelType


def test_getKernelsParams0AndTypes_1(nLatents=3, foreceKernelsUnitScale=True):
    kernelType = "periodic"
    lengthscaleValue = 2.0
    periodValue = 0.5

    estInitConfig = configparser.ConfigParser()
    estInitConfig["kernels_params"] = \
        {"kTypeLatents": kernelType,
         "kLengthscaleValueLatents": lengthscaleValue,
         "kPeriodValueLatents": periodValue}

    params0, kernels_types = svGPFA.utils.initUtils.getKernelsParams0AndTypes(
        nLatents=nLatents, foreceKernelsUnitScale=foreceKernelsUnitScale,
        config=estInitConfig)

    for k in range(nLatents):
        assert params0[k][0].item() == lengthscaleValue
        assert params0[k][1].item() == periodValue
        assert kernels_types[k] == kernelType


def test_getKernelsParams0AndTypes_2(foreceKernelsUnitScale=True):
    kernelTypes = ["exponentialQuadratic", "periodic", "periodic"]
    params0 = [torch.Tensor([1.0]), torch.Tensor([1.0, 0.5]),
               torch.Tensor([3.0, 2.5])]

    nLatents = len(kernelTypes)
    kernels_params = {}
    for k in range(nLatents):
        kernels_params[f"kTypeLatent{k}"] = kernelTypes[k]
        if kernelTypes[k] == "exponentialQuadratic":
            kernels_params[f"kLengthscaleValueLatent{k}"] = params0[k].item()
        if kernelTypes[k] == "periodic":
            kernels_params[f"kLengthscaleValueLatent{k}"] = \
                    params0[k][0].item()
            kernels_params[f"kperiodValueLatent{k}"] = params0[k][0].item()
    estInitConfig = configparser.ConfigParser()
    estInitConfig["kernels_params"] = kernels_params
    params0, kernels_types = svGPFA.utils.initUtils.getKernelsParams0AndTypes(
        nLatents=nLatents, foreceKernelsUnitScale=foreceKernelsUnitScale,
        config=estInitConfig)

    for k in range(nLatents):
        if kernelTypes[k] == "exponentialQuadratic":
            assert kernels_types[k] == "exponentialQuadratic"
            assert params0[k] == params0[k]


def test_getKernelsParams0AndTypes_3(nLatents=3, foreceKernelsUnitScale=True):
    kernelType = "exponentialQuadtric"
    lengthscaleValue = 2.0

    estInitConfig = None

    params0, kernels_types = svGPFA.utils.initUtils.getKernelsParams0AndTypes(
        nLatents=nLatents, foreceKernelsUnitScale=foreceKernelsUnitScale,
        config=estInitConfig,
        kernel_type_dft=kernelType,
        kernel_params_dft=torch.Tensor([lengthscaleValue]))

    for k in range(nLatents):
        assert kernels_types[k] == kernelType
        assert params0[k][0].item() == lengthscaleValue


if __name__ == "__main__":
    # test_getParam_0()
    # test_getParam_1()
    # test_getLinearEmbeddingParams_0()
    # test_getLinearEmbeddingParams_1()
    test_getTrialsStartEndTimes_0()
    # test_getTrialsStartEndTimes_1()
    # test_getTrialsStartEndTimes_2()
    # test_j_cholesky()
    # test_getKernelsParams0AndTypes_0()
    # test_getKernelsParams0AndTypes_1()
    # test_getKernelsParams0AndTypes_2()
    # test_getKernelsParams0AndTypes_3()
