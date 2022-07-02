
import configparser
import torch

import svGPFA.utils.initUtils
import gcnu_common.utils.config_dict


def test_getParam_0():
    true_n_latents = 3
    dynamic_params = {"model_structure_params":
                      {"n_latents": str(true_n_latents)}}
    estInitConfigFilename = "data/99999999_estimation_metadata.ini"
    config_file_params = gcnu_common.utils.config_dict.GetDict(
        config=estInitConfigFilename).get_dict()
    n_latents = svGPFA.utils.initUtils.getParam(
        section_name="model_structure_params",
        param_name="n_latents",
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=None,
        conversion_funct=int)
    assert(true_n_latents == n_latents)


def test_getParam_1():
    true_trials_end_time = 1.0
    true_n_latents = 3
    dynamic_params = {"model_structure_params":
                      {"n_latents": str(true_n_latents)}}
    estInitConfigFilename = "data/99999999_estimation_metaData.ini"
    config_file_params = gcnu_common.utils.config_dict.GetDict(
        config=estInitConfigFilename).get_dict()
    trials_end_time = svGPFA.utils.initUtils.getParam(
        section_name="data_structure_params",
        param_name="trials_end_time",
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=None,
        conversion_funct=float)
    assert(true_trials_end_time == trials_end_time)


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
        assert(params0[k].item() == lengthscaleValue)
        assert(kernels_types[k] == kernelType)


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
        assert(params0[k][0].item() == lengthscaleValue)
        assert(params0[k][1].item() == periodValue)
        assert(kernels_types[k] == kernelType)


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
            assert(kernels_types[k] == "exponentialQuadratic")
            assert(params0[k] == params0[k])


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
        assert(kernels_types[k] == kernelType)
        assert(params0[k][0].item() == lengthscaleValue)


if __name__ == "__main__":
    # test_getParam_0()
    test_getParam_1()
    # test_j_cholesky()
    # test_getKernelsParams0AndTypes_0()
    # test_getKernelsParams0AndTypes_1()
    # test_getKernelsParams0AndTypes_2()
    # test_getKernelsParams0AndTypes_3()
