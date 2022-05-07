
import sys
import configparser
import torch

import svGPFA.utils.initUtils

def test_getKernelsParams0AndTypes_0(nLatents=3, foreceKernelsUnitScale=True):
    kernelType = "exponentialQuadratic"
    lengthscaleValue = 2.0

    estInitConfig = configparser.ConfigParser()
    estInitConfig["kernels_params"] = {"kTypeLatents": kernelType,
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
    estInitConfig["kernels_params"] = {"kTypeLatents": kernelType,
                                       "kLengthscaleValueLatents": lengthscaleValue,
                                       "kPeriodValueLatents": periodValue,
                                      }

    params0, kernels_types = svGPFA.utils.initUtils.getKernelsParams0AndTypes(
        nLatents=nLatents, foreceKernelsUnitScale=foreceKernelsUnitScale,
        config=estInitConfig)

    for k in range(nLatents):
        assert(params0[k][0].item() == lengthscaleValue)
        assert(params0[k][1].item() == periodValue)
        assert(kernels_types[k] == kernelType)

def test_getKernelsParams0AndTypes_2(foreceKernelsUnitScale=True):
    kernelTypes = ["exponentialQuadratic", "periodic", "periodic"]
    params0 = [torch.Tensor([1.0]), torch.Tensor([1.0, 0.5]), torch.Tensor([3.0, 2.5])]

    nLatents = len(kernelTypes)
    kernels_params = {}
    for k in range(nLatents):
        kernels_params[f"kTypeLatent{k}"] = kernelTypes[k]
        if kernelTypes[k] == "exponentialQuadratic":
            kernels_params[f"kLengthscaleValueLatent{k}"] = params0[k].item()
        if kernelTypes[k] == "periodic":
            kernels_params[f"kLengthscaleValueLatent{k}"] = params0[k][0].item()
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

def test_getPropSamplesCovered():
    N = 100
    tol = .1

    mean = torch.rand(size=(N,))*2-1
    std = torch.rand(size=(N,))*0.3
    sample = torch.normal(mean=mean, std=std)
    propSamplesCovered = utils.svGPFA.miscUtils.getPropSamplesCovered(sample=sample, mean=mean, std=std, percent=.95)
    assert(.95-tol<propSamplesCovered and propSamplesCovered<tol+.95)

def test_getDiagIndicesIn3DArray():
    N = 3
    M = 2
    trueDiagIndices = torch.tensor([0, 4, 8, 9, 13, 17])

    diagIndices = utils.svGPFA.miscUtils.getDiagIndicesIn3DArray(N=N, M=M)
    assert(((trueDiagIndices-diagIndices)**2).sum()==0)

def test_build3DdiagFromDiagVector():
    N = 3
    M = 2
    v = torch.arange(M*N, dtype=torch.double)
    D = utils.svGPFA.miscUtils.build3DdiagFromDiagVector(v=v, N=N, M=M)
    trueD = torch.tensor([[[0,0,0],[0,1,0],[0,0,2]],[[3,0,0],[0,4,0],[0,0,5]]], dtype=torch.double)
    assert(((trueD-D)**2).sum()==0)

# def test_j_cholesky():
#     tol = 1e-3
# 
#     A = torch.randn((3, 4))
#     K = torch.mm(A, A.T)
#     trueY = torch.unsqueeze(torch.tensor([1.0, 2.0, 3.0]), 1)
#     b = torch.mm(K, trueY)
#     KChol = torch.cholesky(K)
#     yTorch = torch.cholesky_solve(b, KChol)
#     yJ = stats.svGPFA.utils.j_cholesky_solve(b, KChol)
#     error = ((yTorch-yJ)**2).sum()
#     assert(error<tol)
# 
if __name__=="__main__":
    # test_getDiagIndicesIn3DArray()
    # test_build3DdiagFromDiagVector()
    # test_j_cholesky()
    # test_getPropSamplesCovered()
    test_getKernelsParams0AndTypes_0()
    test_getKernelsParams0AndTypes_1()
    test_getKernelsParams0AndTypes_2()
    test_getKernelsParams0AndTypes_3()
