
import configparser
import pandas as pd
import torch

import svGPFA.utils.initUtils
import gcnu_common.utils.config_dict


def test_getParamsDictFromArgs_0(n_latents=7, n_trials=10,
                                 true_k_type="exponentialQuadratic",
                                 true_k_lengthscale=3.4):
    args = {"k_type": true_k_type, "k_lengthscale": str(true_k_lengthscale)}
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    params_dict = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=args, args_info=args_info)
    assert true_k_type == params_dict["kernels_params"]["k_type"]
    assert true_k_lengthscale == params_dict["kernels_params"]["k_lengthscale"]


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
            "k_lengthscale_latent0": str(true_k_lengthscale_latent0),
            "k_type_latent1": true_k_type_latent1,
            "k_lengthscale_latent1": str(true_k_lengthscale_latent1),
            "k_type_latent2": true_k_type_latent2,
            "k_lengthscale_latent2": str(true_k_lengthscale_latent2),
            "k_period_latent2": str(true_k_period_latent2),
           }
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    params_dict = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=args, args_info=args_info)
    assert true_k_type_latent0 == params_dict["kernels_params"]["k_type_latent0"]
    assert true_k_lengthscale_latent0 == params_dict["kernels_params"]["k_lengthscale_latent0"]
    assert true_k_type_latent1 == params_dict["kernels_params"]["k_type_latent1"]
    assert true_k_lengthscale_latent1 == params_dict["kernels_params"]["k_lengthscale_latent1"]
    assert true_k_type_latent2 == params_dict["kernels_params"]["k_type_latent2"]
    assert true_k_lengthscale_latent2 == params_dict["kernels_params"]["k_lengthscale_latent2"]
    assert true_k_period_latent2 == params_dict["kernels_params"]["k_period_latent2"]


def test_getParamsDictFromArgs_2(n_latents=3, n_trials=2,
                                 true_variational_means_str="1.0 2,0 3.0",
                                 true_variational_covs_str="1.0 0.0 0.0; "
                                                           "0.0 1.0 0.0; "
                                                           "0.0 0.0 1.0"
                                ):
    args = {"variational_means": true_variational_means_str,
            "variational_covs": true_variational_covs_str}
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    params_dict = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=args, args_info=args_info)
    true_variational_means = svGPFA.utils.initUtils.strTo1DDoubleTensor(aString=true_variational_means_str)
    true_variational_covs = svGPFA.utils.initUtils.strTo2DDoubleTensor(aString=true_variational_covs_str)
    assert torch.all(true_variational_means == params_dict["variational_params"]["variational_means"])
    assert torch.all(true_variational_covs == params_dict["variational_params"]["variational_covs"])


def test_getParamsDictFromStringsDict_1(n_latents=3, n_trials=10,
                                        true_k_type_latent0="exponentialQuadratic",
                                        true_k_lengthscale_latent0=3.4,
                                        true_k_type_latent1="exponentialQuadratic",
                                        true_k_lengthscale_latent1=7.4,
                                        true_k_type_latent2="periodic",
                                        true_k_lengthscale_latent2=2.9,
                                        true_k_period_latent2=1.3,
                                       ):
    strings_dict = {"kernels_params": {"k_type_latent0": true_k_type_latent0,
                                       "k_lengthscale_latent0": str(true_k_lengthscale_latent0),
                                       "k_type_latent1": true_k_type_latent1,
                                       "k_lengthscale_latent1": str(true_k_lengthscale_latent1),
                                       "k_type_latent2": true_k_type_latent2,
                                       "k_lengthscale_latent2": str(true_k_lengthscale_latent2),
                                       "k_period_latent2": str(true_k_period_latent2),
                                      }
                  }
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    params_dict = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    assert true_k_type_latent0 == params_dict["kernels_params"]["k_type_latent0"]
    assert true_k_lengthscale_latent0 == params_dict["kernels_params"]["k_lengthscale_latent0"]
    assert true_k_type_latent1 == params_dict["kernels_params"]["k_type_latent1"]
    assert true_k_lengthscale_latent1 == params_dict["kernels_params"]["k_lengthscale_latent1"]
    assert true_k_type_latent2 == params_dict["kernels_params"]["k_type_latent2"]
    assert true_k_lengthscale_latent2 == params_dict["kernels_params"]["k_lengthscale_latent2"]
    assert true_k_period_latent2 == params_dict["kernels_params"]["k_period_latent2"]


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
            true_ind_points_locs_filename = strings_dict["ind_points_params"][f"ind_points_locs_filename_latent{k}_trial{r}"]
            params_ind_points_locs_filename = params_dict["ind_points_params"][f"ind_points_locs_filename_latent{k}_trial{r}"]
            assert true_ind_points_locs_filename == params_ind_points_locs_filename


def test_getParam_0(true_n_latents = 3, n_trials=15, n_neurons=100):
    ''' test get param from dynamic params'''

    dynamic_params = {"model_structure_params":
                      {"n_latents": str(true_n_latents)}}
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
        n_neurons=n_neurons, n_latents=true_n_latents)
    n_latents = svGPFA.utils.initUtils.getParam(
        section_name="model_structure_params",
        param_name="n_latents",
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params,
        conversion_funct=int)
    assert true_n_latents == n_latents


def test_getParam_1(n_latents=3, n_trials=20, n_neurons=100):
    ''' test get param from config_file params'''
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
        n_neurons=n_neurons, n_latents=n_latents)
    true_trials_end_time = float(strings_dict["data_structure_params"]["trials_end_time"])
    trials_end_time = svGPFA.utils.initUtils.getParam(
        section_name="data_structure_params",
        param_name="trials_end_time",
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params,
        conversion_funct=float)
    assert true_trials_end_time == trials_end_time


def test_getLinearEmbeddingParams_0(n_neurons=20, n_latents=7, n_trials=10):
    true_C = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                          dtype=torch.double)
    true_d = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.double)
    dynamic_params = {"embedding_params": {"c": true_C, "d": true_d}}
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
        n_neurons=n_neurons, n_latents=n_latents)
    C0, d0 = svGPFA.utils.initUtils.getLinearEmbeddingParams0(
        n_neurons=n_neurons, n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    assert torch.all(true_C == C0)
    assert torch.all(true_d == d0)


def test_getLinearEmbeddingParams_1(n_neurons=20, n_latents=7, n_trials=20):
    dynamic_params = {"model_structure_params":
                      {"n_latents": str(n_latents)}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_latents=n_latents)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
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


def test_getLinearEmbeddingParams_2(n_neurons=20, n_latents=7, n_trials=10):
    c_random_seed = 102030
    d_random_seed = 203040

    torch.manual_seed(c_random_seed)
    true_C = torch.normal(mean=0.0, std=1.0, size=(n_neurons, n_latents))
    torch.manual_seed(d_random_seed)
    true_d = torch.normal(mean=0.0, std=1.0, size=(n_neurons, 1))
    torch.seed()

    dynamic_params = {"embedding_params": {"c_distribution": "Normal",
                                           "c_loc": 0.0,
                                           "c_scale": 1.0,
                                           "c_random_seed": c_random_seed,
                                           "d_distribution": "Normal",
                                           "d_loc": 0.0,
                                           "d_scale": 1.0,
                                           "d_random_seed": d_random_seed}}
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
        n_neurons=n_neurons, n_latents=n_latents)
    C0, d0 = svGPFA.utils.initUtils.getLinearEmbeddingParams0(
        n_neurons=n_neurons, n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    assert torch.all(true_C == C0)
    assert torch.all(true_d == d0)


def test_getTrialsStartEndTimes_0(n_trials=15, n_neurons=100, n_latents=5,
                                  trials_start_time=3.0, trials_end_time=12.0):
    dynamic_params = {"data_structure_params":
                      {"trials_start_time": trials_start_time,
                       "trials_end_time": trials_end_time}
                     }
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


def test_getTrialsStartEndTimes_1( n_trials=15, n_neurons=100, n_latents=5):
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


def test_getTrialsStartEndTimes_2(n_trials=15, n_neurons=100, n_latents=5):
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


def test_getKernelsParams0AndTypes_0(n_neurons=20, n_latents=3, n_trials=50,
                                     kernel_type = "exponentialQuadratic",
                                     lengthscale = 2.0):

    dynamic_params = {"kernels_params": {"k_type": kernel_type,
                                         "k_lengthscale": lengthscale}}
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
        n_neurons=n_neurons, n_latents=n_latents)
    params0, kernels_types = svGPFA.utils.initUtils.getKernelsParams0AndTypes(
        n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    for k in range(n_latents):
        assert params0[k].item() == lengthscale
        assert kernels_types[k] == kernel_type


def test_getKernelsParams0AndTypes_1(n_neurons=20, n_latents=3, n_trials=50,
                                     kernel_type = "periodic",
                                     lengthscale = 2.0,
                                     period=3.0):

    dynamic_params = {"kernels_params": {"k_type": kernel_type,
                                         "k_lengthscale": lengthscale,
                                         "k_period": period}}
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
        n_neurons=n_neurons, n_latents=n_latents)
    params0, kernels_types = svGPFA.utils.initUtils.getKernelsParams0AndTypes(
        n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    for k in range(n_latents):
        assert params0[k][0].item() == lengthscale
        assert params0[k][1].item() == period
        assert kernels_types[k] == kernel_type


def test_getKernelsParams0AndTypes_2(n_neurons=20, n_latents=3, n_trials=50,
                                     section_name = "kernels_params",
                                     kernel_type_param_name = "k_type",
                                     lengthscale_param_name = "k_lengthscale"):

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
        n_neurons=n_neurons, n_latents=n_latents)
    true_lengthscale = config_file_params[section_name][lengthscale_param_name]
    true_kernel_type = config_file_params[section_name][kernel_type_param_name]
    params0, kernels_types = svGPFA.utils.initUtils.getKernelsParams0AndTypes(
        n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    for k in range(n_latents):
        assert params0[k].item() == true_lengthscale
        assert kernels_types[k] == true_kernel_type


def test_getIndPointsLocs0_0(n_neurons=20, n_latents=3, n_trials=50,
                              section_name = "ind_points_params",
                              ind_points_locs_filename="data/equispacedValuesBtw0and1_len09.csv",
                              ind_points_locs_filename_param_name="ind_points_locs_filename",
                              estInitConfigFilename = "data/99999999_estimation_metaData.ini",
                             ):

    dynamic_params = {section_name: {ind_points_locs_filename_param_name: ind_points_locs_filename}}
    config = configparser.ConfigParser()
    config.read(estInitConfigFilename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=config).get_dict()
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_latents=n_latents)
    true_ind_points_locs0 = torch.from_numpy(pd.read_csv(ind_points_locs_filename, header=None).to_numpy()).flatten()
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
            assert torch.all(ind_points_locs0[k][r, :, 0] == true_ind_points_locs0)


def test_getIndPointsLocs0_1(n_neurons=20, n_latents=2, n_trials=15,
                             section_name="ind_points_params",
                             ind_points_locs_filename="data/equispacedValuesBtw0and1_len09.csv",
                             estInitConfigFilename="data/99999998_estimation_metaData.ini",
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
        n_neurons=n_neurons, n_latents=n_latents)
    true_ind_points_locs0 = torch.from_numpy(pd.read_csv(
        ind_points_locs_filename, header=None).to_numpy()).flatten()
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


def test_variationalMean0_0(n_neurons=20, n_latents=3, n_ind_points=9,
                            n_trials=15,
                            section_name="variational_params",
                            param_name="variational_mean",
                            estInitConfigFilename="data/99999999_estimation_metaData.ini"):
    variational_mean0_k = torch.normal(mean=0.0, std=1.0,  size=(n_trials, n_ind_points, 1))
    true_variational_mean0 = [variational_mean0_k for k in range(n_latents)]
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
        n_neurons=n_neurons, n_latents=n_latents)
    variational_mean0 = svGPFA.utils.initUtils.getVariationalMean0(
        n_latents=n_latents, n_trials=n_trials, n_ind_points=n_ind_points,
        dynamic_params=dynamic_params, config_file_params=config_file_params,
        default_params=default_params)

    for k in range(n_latents):
        assert torch.all(true_variational_mean0[k] == variational_mean0[k])


def test_variationalMean0_1(n_neurons=20, n_latents=2, n_ind_points=None,
                            n_trials=15,
                            section_name="variational_params",
                            variational_mean0_filename_param_name_pattern="variational_mean_filename_latent{:d}_trial{:d}",
                            estInitConfigFilename="data/99999998_estimation_metaData.ini"):
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
        n_neurons=n_neurons, n_latents=n_latents)
    variational_mean0 = svGPFA.utils.initUtils.getVariationalMean0(
        n_latents=n_latents, n_trials=n_trials, n_ind_points=n_ind_points,
        dynamic_params=dynamic_params, config_file_params=config_file_params,
        default_params=default_params)

    for k in range(n_latents):
        for r in range(n_trials):
            true_variational_mean0_filename_kr = config_file_params[section_name][variational_mean0_filename_param_name_pattern.format(k, r)]
            true_variational_mean0_kr = torch.from_numpy(pd.read_csv(true_variational_mean0_filename_kr, header=None).to_numpy()).flatten()
            assert torch.all(variational_mean0[k][r, :, 0] ==
                             true_variational_mean0_kr)


def test_variationalMean0_2(n_neurons=20, n_latents=2, n_ind_points=None,
                            n_trials=15,
                            section_name="variational_params",
                            variational_mean0_filename_param_name_pattern="variational_mean_filename_latent{:d}_trial{:d}",
                            estInitConfigFilename="data/99999998_estimation_metaData.ini"):
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
    del config_file_params[section_name]
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_latents=n_latents)
    variational_mean0 = svGPFA.utils.initUtils.getVariationalMean0(
        n_latents=n_latents, n_trials=n_trials, n_ind_points=n_ind_points,
        dynamic_params=dynamic_params, config_file_params=config_file_params,
        default_params=default_params)

    true_variational_mean0 = default_params[section_name]["variational_means"]
    for k in range(n_latents):
        for r in range(n_trials):
            true_variational_mean0_kr = torch.from_numpy(pd.read_csv(true_variational_mean0_filename_kr, header=None).to_numpy()).flatten()
            assert torch.all(variational_mean0[k][r, :, 0] ==
                             true_variational_mean0_kr)


if __name__ == "__main__":
    # test_getParamsDictFromArgs_0()
    # test_getParamsDictFromArgs_1()
    # test_getParamsDictFromArgs_2()
    # test_getParamsDictFromStringsDict_1()
    # test_getParamsDictFromStringsDict_2()
    # test_getParam_0()
    # test_getParam_1()
    # test_getLinearEmbeddingParams_0()
    # test_getLinearEmbeddingParams_1()
    test_getLinearEmbeddingParams_2()
    # test_getTrialsStartEndTimes_0()
    # test_getTrialsStartEndTimes_1()
    # test_getTrialsStartEndTimes_2()
    # test_getKernelsParams0AndTypes_0()
    # test_getKernelsParams0AndTypes_1()
    # test_getKernelsParams0AndTypes_2()
    # test_getIndPointsLocs0_0()
    # test_getIndPointsLocs0_1()
    # test_variationalMean0_0() 
    # test_variationalMean0_1() 
