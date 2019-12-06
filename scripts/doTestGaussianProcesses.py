
import sys
import pdb
import torch
import matplotlib.pyplot as plt
sys.path.append("../src")
sys.path.append("/home/rapela/dev/research/programs/src/python")
import kernels
import stats.gaussianProcesses.core

def test_GaussianProcess_eval():
    nSamples = 2000
    samplesLow = -20
    samplesHigh = 20
    k0Scale, k0LenghtScale, k0Period= .1, 1/2.5, 1.5
    k1Scale, k1LenghtScale = .1, 1
    nTraces = 5


    mean = torch.sin
    kernel0 = kernels.PeriodicKernel()
    kernel0.setParams(params=[k0Scale, k0LenghtScale, k0Period])
    kernel1 = kernels.ExponentialQuadraticKernel()
    kernel1.setParams(params=[k1Scale, k1LenghtScale])

    latent0 = stats.gaussianProcesses.core.GaussianProcess(mean=mean,
                                                           kernel=kernel0)
    latent1 = stats.gaussianProcesses.core.GaussianProcess(mean=mean,
                                                           kernel=kernel1)
    f, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    t = torch.rand(nSamples)*(samplesHigh-samplesLow)+samplesLow
    t, _ = t.sort()

    axs[0].plot(t, latent0(t), marker="x")
    axs[0].set_title("Periodic Kernel (scale={:.2f}, lenght scale={:.2f}, period={:.2f})".format(k0Scale, k0LenghtScale, k0Period))

    axs[1].plot(t, latent1(t), marker="x")
    axs[1].set_title("Periodic Kernel (scale={:.2f}, lenght scale={:.2f}".format(k1Scale, k1LenghtScale))

    plt.show()

    pdb.set_trace()

if __name__=="__main__":
    test_GaussianProcess_eval()
