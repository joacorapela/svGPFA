
import configparser
# import pandas as pd
import numpy as np
import torch
import pytest

import svGPFA.utils.initUtils
import gcnu_common.utils.config_dict


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_getParamsDictFromArgs_0(n_latents=7, n_trials=10,
                                 true_k_type="exponentialQuadratic",
                                 true_k_lengthscale=3.4):
    args = {"k_type": true_k_type, "k_lengthscale0": str(true_k_lengthscale)}
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    params_dict = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=args, args_info=args_info)
    assert true_k_type == params_dict["kernels_params0"]["k_type"]
    assert true_k_lengthscale == params_dict["kernels_params0"]["k_lengthscale0"]


def test_getParamsDictFromArgs_1(n_latents=3, n_trials=10,
                                 true_k_type_latent0="exponentialQuadratic",
                                 true_k_lengthscale_latent0=3.4,
                                 true_k_type_latent1="exponentialQuadratic",
                                 true_k_lengthscale_latent1=7.4,
                                 true_k_type_latent2="periodic",
                                 true_k_lengthscale_latent2=2.9,
                                 true_k_period_latent2=1.3,
                                ):
    args = {"k_type_latent0": true_k_type_latent0,
            "k_lengthscale0_latent0": str(true_k_lengthscale_latent0),
            "k_type_latent1": true_k_type_latent1,
            "k_lengthscale0_latent1": str(true_k_lengthscale_latent1),
            "k_type_latent2": true_k_type_latent2,
            "k_lengthscale0_latent2": str(true_k_lengthscale_latent2),
            "k_period0_latent2": str(true_k_period_latent2),
           }
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    params_dict = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=args, args_info=args_info)
    assert true_k_type_latent0 == params_dict["kernels_params0"]["k_type_latent0"]
    assert true_k_lengthscale_latent0 == params_dict["kernels_params0"]["k_lengthscale0_latent0"]
    assert true_k_type_latent1 == params_dict["kernels_params0"]["k_type_latent1"]
    assert true_k_lengthscale_latent1 == params_dict["kernels_params0"]["k_lengthscale0_latent1"]
    assert true_k_type_latent2 == params_dict["kernels_params0"]["k_type_latent2"]
    assert true_k_lengthscale_latent2 == params_dict["kernels_params0"]["k_lengthscale0_latent2"]
    assert true_k_period_latent2 == params_dict["kernels_params0"]["k_period0_latent2"]


def test_getParamsDictFromArgs_2(n_latents=3, n_trials=2,
                                 true_variational_means_str="1.0 2.0 3.0",
                                 true_variational_covs_str="1.0 0.0 0.0; "
                                                           "0.0 1.0 0.0; "
                                                           "0.0 0.0 1.0"
                                ):
    args = {"variational_means0": true_variational_means_str,
            "variational_covs0": true_variational_covs_str}
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    params_dict = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=args, args_info=args_info)
    true_variational_means = svGPFA.utils.initUtils.strTo1DDoubleTensor(aString=true_variational_means_str)
    true_variational_covs = svGPFA.utils.initUtils.strTo2DDoubleTensor(aString=true_variational_covs_str)
    assert torch.all(true_variational_means == params_dict["variational_params0"]["variational_means0"])
    assert torch.all(true_variational_covs == params_dict["variational_params0"]["variational_covs0"])


def test_getParamsDictFromStringsDict_1(n_latents=3, n_trials=10,
                                        true_k_type_latent0="exponentialQuadratic",
                                        true_k_lengthscale_latent0=3.4,
                                        true_k_type_latent1="exponentialQuadratic",
                                        true_k_lengthscale_latent1=7.4,
                                        true_k_type_latent2="periodic",
                                        true_k_lengthscale_latent2=2.9,
                                        true_k_period_latent2=1.3,
                                       ):
    strings_dict = {"kernels_params0": {"k_type_latent0": true_k_type_latent0,
                                       "k_lengthscale0_latent0": str(true_k_lengthscale_latent0),
                                       "k_type_latent1": true_k_type_latent1,
                                       "k_lengthscale0_latent1": str(true_k_lengthscale_latent1),
                                       "k_type_latent2": true_k_type_latent2,
                                       "k_lengthscale0_latent2": str(true_k_lengthscale_latent2),
                                       "k_period0_latent2": str(true_k_period_latent2),
                                      }
                  }
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    params_dict = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    assert true_k_type_latent0 == params_dict["kernels_params0"]["k_type_latent0"]
    assert true_k_lengthscale_latent0 == params_dict["kernels_params0"]["k_lengthscale0_latent0"]
    assert true_k_type_latent1 == params_dict["kernels_params0"]["k_type_latent1"]
    assert true_k_lengthscale_latent1 == params_dict["kernels_params0"]["k_lengthscale0_latent1"]
    assert true_k_type_latent2 == params_dict["kernels_params0"]["k_type_latent2"]
    assert true_k_lengthscale_latent2 == params_dict["kernels_params0"]["k_lengthscale0_latent2"]
    assert true_k_period_latent2 == params_dict["kernels_params0"]["k_period0_latent2"]


def test_getParamsDictFromStringsDict_2(n_latents=2, n_trials=15,
                                        estInitConfigFilename = "data/99999998_estimation_metaData.ini"):

    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    params_dict = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    for k in range(n_latents):
        for r in range(n_trials):
            true_ind_points_locs_filename = strings_dict["ind_points_locs_params0"][f"ind_points_locs0_filename_latent{k}_trial{r}"]
            params_ind_points_locs_filename = params_dict["ind_points_locs_params0"][f"ind_points_locs0_filename_latent{k}_trial{r}"]
            assert true_ind_points_locs_filename == params_ind_points_locs_filename


def test_getParam_0(true_n_latents=3, n_trials=15, n_neurons=100,
                    n_ind_points=(10, 10, 10), diag_var_cov0_value=1e-2):
    ''' test get param from dynamic params'''

    dynamic_params = {"model_structure_params":
                      {"n_latents": true_n_latents}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=true_n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=true_n_latents, diag_var_cov0_value=diag_var_cov0_value)
    n_latents = svGPFA.utils.initUtils.getParam(
        section_name="model_structure_params",
        param_name="n_latents",
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    assert true_n_latents == n_latents


def test_getParam_1(n_latents=3, n_trials=20, n_neurons=100,
                    n_ind_points=(10, 10, 10), diag_var_cov0_value=1e-2):
    ''' test get param from config_file params'''
    dynamic_params = {"model_structure_params":
                      {"n_latents": n_latents}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    true_trials_end_time = float(strings_dict["data_structure_params"]["trials_end_time"])
    trials_end_time = svGPFA.utils.initUtils.getParam(
        section_name="data_structure_params",
        param_name="trials_end_time",
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    assert true_trials_end_time == trials_end_time


def test_getLinearEmbeddingParams0_0(n_neurons=20, n_latents=7, n_trials=10,
                                     n_ind_points=(10, 10, 10, 10, 10, 10, 10),
                                     diag_var_cov0_value=1e-2):
    true_C0 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                          dtype=torch.double)
    true_d0 = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.double)
    dynamic_params = {"embedding_params0": {"c0": true_C0, "d0": true_d0}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    C0, d0 = svGPFA.utils.initUtils.getLinearEmbeddingParams0(
        n_neurons=n_neurons, n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    assert torch.all(true_C0 == C0)
    assert torch.all(true_d0 == d0)


def test_getLinearEmbeddingParams0_1(n_neurons=20, n_latents=7, n_trials=20,
                                     n_ind_points=(10, 10, 10, 10, 10, 10, 10),
                                     diag_var_cov0_value=1e-2):
    dynamic_params = {"model_structure_params":
                      {"n_latents": str(n_latents)}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    true_C0_filename = config["embedding_params0"]["c0_filename"]
    true_C0_np = np.genfromtxt(true_C0_filename, delimiter=",")
    true_C0 = torch.from_numpy(true_C0_np).type(torch.double)
    true_d0_filename = config["embedding_params0"]["d0_filename"]
    true_d0_np = np.genfromtxt(true_d0_filename, delimiter=",")
    true_d0 = torch.from_numpy(true_d0_np).type(torch.double)
    C0, d0 = svGPFA.utils.initUtils.getLinearEmbeddingParams0(
        n_neurons=n_neurons, n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    assert torch.all(true_C0 == C0)
    assert torch.all(true_d0 == d0)


def test_getLinearEmbeddingParams0_2(n_neurons=20, n_latents=7, n_trials=10,
                                     n_ind_points=(10, 10, 10, 10, 10, 10, 10),
                                     diag_var_cov0_value=1e-2):
    c_random_seed = 102030
    d_random_seed = 203040

    torch.manual_seed(c_random_seed)
    true_C0 = torch.normal(mean=0.0, std=1.0, size=(n_neurons, n_latents))
    torch.manual_seed(d_random_seed)
    true_d0 = torch.normal(mean=0.0, std=1.0, size=(n_neurons, 1))
    torch.seed()

    dynamic_params = {"embedding_params0": {"c0_distribution": "Normal",
                                           "c0_loc": 0.0,
                                           "c0_scale": 1.0,
                                           "c0_random_seed": c_random_seed,
                                           "d0_distribution": "Normal",
                                           "d0_loc": 0.0,
                                           "d0_scale": 1.0,
                                           "d0_random_seed": d_random_seed}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    C0, d0 = svGPFA.utils.initUtils.getLinearEmbeddingParams0(
        n_neurons=n_neurons, n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    assert torch.all(true_C0 == C0)
    assert torch.all(true_d0 == d0)


def test_getTrialsStartEndTimes_0(n_trials=15, n_neurons=100, n_latents=5,
                                  n_ind_points=(10, 10, 10, 10, 10),
                                  diag_var_cov0_value=1e-2,
                                  trials_start_time=3.0, trials_end_time=12.0):
    dynamic_params = {"data_structure_params":
                      {"trials_start_time": trials_start_time,
                       "trials_end_time": trials_end_time}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
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


def test_getTrialsStartEndTimes_1( n_trials=15, n_neurons=100, n_latents=5,
                                  n_ind_points=(10, 10, 10, 10, 10),
                                  diag_var_cov0_value=1e-2):
    dynamic_params = {"model_structure_params":
                      {"n_latents": str(n_latents)}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
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


def test_getTrialsStartEndTimes_2(n_trials=15, n_neurons=100, n_latents=5,
                                  n_ind_points=(10, 10, 10, 10, 10),
                                  diag_var_cov0_value=1e-2):
    dynamic_params = {"model_structure_params":
                      {"n_latents": str(n_latents)}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    del config_file_params["data_structure_params"]
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
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


def test_getKernelsParams0AndTypes_0(n_neurons=20, n_latents=3, n_trials=50,
                                     n_ind_points=(10, 10, 10),
                                     diag_var_cov0_value=1e-2,
                                     kernel_type="exponentialQuadratic",
                                     lengthscale0=2.0):
    # dynamic_params, short format, exponential quadratic kernel
    dynamic_params = {"kernels_params0": {"k_types": kernel_type,
                                          "k_lengthscales0": lengthscale0}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    params0, kernels_types = svGPFA.utils.initUtils.getKernelsParams0AndTypes(
        n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    for k in range(n_latents):
        assert params0[k].item() == lengthscale0
        assert kernels_types[k] == kernel_type


def test_getKernelsParams0AndTypes_1(n_neurons=20, n_latents=3, n_trials=50,
                                     n_ind_points=(10, 10, 10),
                                     diag_var_cov0_value=1e-2,
                                     kernel_types="periodic",
                                     lengthscales0=2.0,
                                     periods0=3.0):
    # dynamic_params, short format, periodic kernel
    dynamic_params = {"kernels_params0": {"k_types": kernel_types,
                                          "k_lengthscales0": lengthscales0,
                                          "k_periods0": periods0}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    params0, kernels_types = svGPFA.utils.initUtils.getKernelsParams0AndTypes(
        n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    for k in range(n_latents):
        assert params0[k][0].item() == lengthscales0
        assert params0[k][1].item() == periods0
        assert kernels_types[k] == kernel_types


def test_getKernelsParams0AndTypes_2(n_neurons=20, n_latents=3, n_trials=50,
                                     n_ind_points=(10, 10, 10),
                                     diag_var_cov0_value=1e-2,
                                     section_name="kernels_params0",
                                     kernel_type_param_name="k_types",
                                     lengthscale_param_name="k_lengthscales0"):
    # config_params, short format
    dynamic_params = {"model_structure_params": {"n_latents": n_latents}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    true_lengthscale0 = config_file_params[section_name][lengthscale_param_name]
    true_kernel_type = config_file_params[section_name][kernel_type_param_name]
    params0, kernels_types = svGPFA.utils.initUtils.getKernelsParams0AndTypes(
        n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    for k in range(n_latents):
        assert params0[k].item() == true_lengthscale0
        assert kernels_types[k] == true_kernel_type


def test_getKernelsParams0AndTypes_3(
    n_neurons=20, n_latents=3, n_trials=50,
    n_ind_points=(10, 10, 10),
    diag_var_cov0_value=1e-2,
    true_kernels_types=["exponentialQuadratic",
                        "exponentialQuadratic",
                        "periodic"],
    true_params0=[torch.DoubleTensor([1.0]),
                  torch.DoubleTensor([2.3]),
                  torch.DoubleTensor([1.7, 0.25])]):
    # dynamic_params, binary format
    dynamic_params = {"kernels_params0": {"k_types": true_kernels_types,
                                         "k_params0": true_params0}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    params0, kernels_types = svGPFA.utils.initUtils.getKernelsParams0AndTypes(
        n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    for k in range(n_latents):
        assert kernels_types[k] == true_kernels_types[k]
        assert torch.all(params0[k] == true_params0[k])


def test_getIndPointsLocs0_0(n_neurons=20, n_latents=3, n_trials=50,
                             n_ind_points=(10, 10, 10),
                             diag_var_cov0_value=1e-2,
                             section_name="ind_points_params0",
                             ind_points_locs_filename="data/equispacedValuesBtw0and1_len09.csv",
                             ind_points_locs_filename_param_name="ind_points_locs0_filename",
                             estInitConfigFilename="data/99999999_estimation_metaData.ini",
                             delimiter=",",
                            ):

    dynamic_params = {section_name: {ind_points_locs_filename_param_name:
                                     ind_points_locs_filename}}
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    true_ind_points_locs0_np = np.genfromtxt(ind_points_locs_filename,
                                             delimiter=delimiter)
    true_ind_points_locs0 = torch.from_numpy(true_ind_points_locs0_np).flatten()
    ind_points_locs0 = svGPFA.utils.initUtils.getIndPointsLocs0(
        n_latents=n_latents, n_trials=n_trials,
        n_ind_points=None,
        trials_start_times=None,
        trials_end_times=None,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)

    for r in range(n_trials):
        for k in range(n_latents):
            assert torch.all(ind_points_locs0[k][r, :, 0] ==
                             true_ind_points_locs0)


def test_getIndPointsLocs0_1(n_neurons=20, n_latents=2, n_trials=15,
                             n_ind_points=(10, 10),
                             diag_var_cov0_value=1e-2,
                             section_name="ind_points_params0",
                             ind_points_locs0_filename="data/equispacedValuesBtw0and1_len09.csv",
                             estInitConfigFilename="data/99999998_estimation_metaData.ini",
                             delimiter=",",
                            ):

    dynamic_params = {"model_structure_params":
                      {"n_latents": str(n_latents)}}
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    true_ind_points_locs0_np = np.genfromtxt(
        ind_points_locs0_filename, delimiter=delimiter)
    true_ind_points_locs0 = torch.from_numpy(true_ind_points_locs0_np).flatten()
    ind_points_locs0 = svGPFA.utils.initUtils.getIndPointsLocs0(
        n_latents=n_latents, n_trials=n_trials,
        n_ind_points=None,
        trials_start_times=None,
        trials_end_times=None,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)

    for r in range(n_trials):
        for k in range(n_latents):
            assert torch.all(ind_points_locs0[k][r, :, 0] ==
                             true_ind_points_locs0)


def test_getVariationalMean0_0(n_neurons=20, n_latents=3,
                               n_ind_points=(9, 9, 9),
                               n_trials=15, diag_var_cov0_value=1e-2,
                               section_name="variational_params0",
                               param_name="variational_mean0",
                               estInitConfigFilename="data/99999999_estimation_metaData.ini",
                              ):
    # dynamic_params, binary format
    true_variational_mean0 = [torch.normal(mean=0.0, std=1.0,
                                           size=(n_trials, n_ind_points[k], 1))
                              for k in range(n_latents)]
    dynamic_params = {section_name: {param_name: true_variational_mean0}}
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    variational_mean0 = svGPFA.utils.initUtils.getVariationalMean0(
        n_latents=n_latents, n_trials=n_trials, n_ind_points=n_ind_points,
        dynamic_params=dynamic_params, config_file_params=config_file_params,
        default_params=default_params)

    for k in range(n_latents):
        assert torch.all(true_variational_mean0[k] == variational_mean0[k])


def test_getVariationalMean0_1(n_neurons=20, n_latents=2,
                               n_ind_points=(10, 10),
                               n_trials=15, diag_var_cov0_value=1e-2,
                               section_name="variational_params0",
                               variational_mean0_filename_param_name_pattern="variational_mean0_filename_latent{:d}_trial{:d}",
                               estInitConfigFilename="data/99999998_estimation_metaData.ini",
                               delimiter=",",
                              ):
    # config_params, long format
    dynamic_params = {"model_structure_params":
                      {"n_latents": str(n_latents)}}
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    variational_mean0 = svGPFA.utils.initUtils.getVariationalMean0(
        n_latents=n_latents, n_trials=n_trials, n_ind_points=n_ind_points,
        dynamic_params=dynamic_params, config_file_params=config_file_params,
        default_params=default_params)

    for k in range(n_latents):
        for r in range(n_trials):
            true_variational_mean0_filename_kr = config_file_params[section_name][variational_mean0_filename_param_name_pattern.format(k, r)]
            true_variational_mean0_kr_np = np.genfromtxt(
                true_variational_mean0_filename_kr, delimiter=delimiter)
            true_variational_mean0_kr = \
                torch.from_numpy(true_variational_mean0_kr_np).flatten()
            assert torch.all(variational_mean0[k][r, :, 0] ==
                             true_variational_mean0_kr)


def test_getVariationalCov0_0(n_neurons=20, n_latents=3,
                              n_ind_points=(9, 9, 9),
                              n_trials=15, diag_var_cov0_value=1e-2,
                              section_name="variational_params0",
                              param_name="variational_cov0",
                              estInitConfigFilename="data/99999999_estimation_metaData.ini",
                             ):
    true_variational_cov0 = [torch.normal(mean=0.0, std=1.0,
                                          size=(n_trials, n_ind_points[k],
                                                n_ind_points[k]))
                             for k in range(n_latents)]
    dynamic_params = {section_name: {param_name: true_variational_cov0}}
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    variational_cov0 = svGPFA.utils.initUtils.getVariationalCov0(
        n_latents=n_latents, n_trials=n_trials, n_ind_points=n_ind_points,
        dynamic_params=dynamic_params, config_file_params=config_file_params,
        default_params=default_params)

    for k in range(n_latents):
        assert torch.all(true_variational_cov0[k] == variational_cov0[k])


def test_getVariationalCov0_1(n_neurons=20, n_latents=2, n_ind_points=(10, 10),
                              n_trials=15, diag_var_cov0_value=1e-2,
                              section_name="variational_params0",
                              variational_cov0_filename_param_name_pattern="variational_cov0_filename_latent{:d}_trial{:d}",
                              estInitConfigFilename="data/99999998_estimation_metaData.ini",
                              delimiter=",",
                             ):
    # config_params, long format
    dynamic_params = {"model_structure_params":
                      {"n_latents": str(n_latents)}}
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    variational_cov0 = svGPFA.utils.initUtils.getVariationalCov0(
        n_latents=n_latents, n_trials=n_trials, n_ind_points=n_ind_points,
        dynamic_params=dynamic_params, config_file_params=config_file_params,
        default_params=default_params)

    for k in range(n_latents):
        for r in range(n_trials):
            true_variational_cov0_filename_kr = config_file_params[section_name][variational_cov0_filename_param_name_pattern.format(k, r)]
            true_variational_cov0_kr_np = \
                np.genfromtxt(true_variational_cov0_filename_kr,
                              delimiter=delimiter)
            true_variational_cov0_kr = torch.from_numpy(
                true_variational_cov0_kr_np)
            assert torch.all(variational_cov0[k][r, :, :] ==
                             true_variational_cov0_kr)

def test_getOptimParams_0(n_neurons=100, n_trials=15, n_latents=8,
                          n_ind_points=[10]*8,
                          diag_var_cov0_value=1e-2):
    # extracting all optim params from default_params
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    flat_true_optim_params = default_params["optim_params"]
    hier_true_optim_params = \
        svGPFA.utils.initUtils.flatToHierarchicalOptimParams(
            flat_optim_params=flat_true_optim_params)
    hier_optim_params = svGPFA.utils.initUtils.getOptimParams(
        dynamic_params=None, config_file_params=None,
        default_params=default_params)
    for true_param_name in hier_true_optim_params:
        if not type(hier_true_optim_params[true_param_name]).__name__ == "dict":
            assert hier_true_optim_params[true_param_name] == \
                hier_optim_params[true_param_name]
        else:
            for true_param2_name in hier_true_optim_params[true_param_name]:
                assert hier_true_optim_params[true_param_name][true_param2_name] == hier_optim_params[true_param_name][true_param2_name]


def test_getOptimParams_1(n_neurons=100, n_trials=15, n_latents=8,
                          n_ind_points=[10]*8, diag_var_cov0_value=1e-2):
    # extracting all optim params from default_params except max_iter that
    # comes from dynamic params
    dynamic_params = {"optim_params": {"em_max_iter": 1000}}
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    flat_true_optim_params = default_params["optim_params"]
    flat_true_optim_params["em_max_iter"] = \
        dynamic_params["optim_params"]["em_max_iter"]
    hier_true_optim_params = \
        svGPFA.utils.initUtils.flatToHierarchicalOptimParams(
            flat_optim_params=flat_true_optim_params)
    hier_optim_params = svGPFA.utils.initUtils.getOptimParams(
        dynamic_params=dynamic_params, config_file_params=None,
        default_params=default_params)
    for true_param_name in hier_true_optim_params:
        if not type(hier_true_optim_params[true_param_name]).__name__ == "dict":
            assert hier_true_optim_params[true_param_name] == \
                hier_optim_params[true_param_name]
        else:
            for true_param2_name in hier_true_optim_params[true_param_name]:
                assert hier_true_optim_params[true_param_name][true_param2_name] == hier_optim_params[true_param_name][true_param2_name]


def test_getOptimParams_2(n_neurons=100, n_latents=8, n_trials=15,
                          n_ind_points=[10]*8, diag_var_cov0_value=1e-2,
                          estInitConfigFilename="data/99999998_estimation_metaData.ini",
                         ):
    # extracting all optim params from default_params except max_iter that
    # comes from dynamic params
    dynamic_params = {"optim_params": {"em_max_iter": 1000}}
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_ind_points=n_ind_points,
        n_latents=n_latents, diag_var_cov0_value=diag_var_cov0_value)
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    strings_dict = gcnu_common.utils.config_dict.GetDict(config=config).get_dict()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    flat_true_optim_params = config_file_params["optim_params"]
    flat_true_optim_params["em_max_iter"] = dynamic_params["optim_params"]["em_max_iter"]
    hier_true_optim_params = \
        svGPFA.utils.initUtils.flatToHierarchicalOptimParams(
            flat_optim_params=flat_true_optim_params)
    hier_optim_params = svGPFA.utils.initUtils.getOptimParams(
        dynamic_params=dynamic_params, config_file_params=config_file_params,
        default_params=default_params)
    for true_param_name in hier_true_optim_params:
        if not type(hier_true_optim_params[true_param_name]).__name__ == "dict":
            assert hier_true_optim_params[true_param_name] == \
                hier_optim_params[true_param_name]
        else:
            for true_param2_name in hier_true_optim_params[true_param_name]:
                assert hier_true_optim_params[true_param_name][true_param2_name] == hier_optim_params[true_param_name][true_param2_name]


if __name__ == "__main__":
    test_getParamsDictFromArgs_0()
    test_getParamsDictFromArgs_1()
    test_getParamsDictFromArgs_2()
    test_getParamsDictFromStringsDict_1()
    test_getParamsDictFromStringsDict_2()
    test_getParam_0()
    test_getParam_1()
    test_getLinearEmbeddingParams0_0()
    test_getLinearEmbeddingParams0_1()
    test_getLinearEmbeddingParams0_2()
    test_getTrialsStartEndTimes_0()
    test_getTrialsStartEndTimes_1()
    test_getTrialsStartEndTimes_2()
    test_getKernelsParams0AndTypes_0()
    test_getKernelsParams0AndTypes_1()
    test_getKernelsParams0AndTypes_2()
    test_getKernelsParams0AndTypes_3()
    test_getIndPointsLocs0_0()
    test_getIndPointsLocs0_1()
    test_getVariationalMean0_0()
    test_getVariationalMean0_1()
    test_getVariationalCov0_0()
    test_getVariationalCov0_1()
    test_getOptimParams_0()
    test_getOptimParams_1()
    test_getOptimParams_2()
