
import sys
import os
import pdb
import torch
import matplotlib.pyplot as plt
sys.path.append("..")
import stats.kernels
import stats.gaussianProcesses.eval

def test_GaussianProcess_eval():
    nSamples = 2000
    samplesLow = 0
    samplesHigh = 20
    k0Scale, k0LengthScale, k0Period = 1, 1, 1/2.5
    k1Scale, k1LengthScale = 1, 1
    epsilon = 1e-3
    nTraces = 5

    mean = torch.sin
    kernel0 = stats.kernels.PeriodicKernel()
    kernel0.setParams(params=[k0Scale, k0LengthScale, k0Period])
    kernel1 = stats.kernels.ExponentialQuadraticKernel()
    kernel1.setParams(params=[k1Scale, k1LengthScale])

    latent0 = stats.gaussianProcesses.eval.GaussianProcess(mean=mean,
                                                           kernel=kernel0)
    latent1 = stats.gaussianProcesses.eval.GaussianProcess(mean=mean,
                                                           kernel=kernel1)
    f, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    t = torch.rand(nSamples)*(samplesHigh-samplesLow)+samplesLow
    t, _ = t.sort()

    axs[0].plot(t, latent0.mean(t), color="gray", linestyle="-")
    axs[0].fill_between(t, latent0.mean(t)-1.96*latent0.std(t), latent0.mean(t)+1.96*latent0.std(t), color="lightgray")
    axs[0].plot(t, latent0(t, epsilon=epsilon), color="blue", marker="x")
    axs[0].set_title("Periodic Kernel (scale={:.2f}, length scale={:.2f}, period={:.2f})".format(k0Scale, k0LengthScale, k0Period))
    axs[0].set_xlabel("Time (sec)")

    axs[1].plot(t, latent1.mean(t), color="gray", linestyle="-")
    axs[1].fill_between(t, latent1.mean(t)-1.96*latent1.std(t), latent1.mean(t)+1.96*latent1.std(t), color="lightgray")
    axs[1].plot(t, latent1(t, epsilon=epsilon), color="blue", marker="x")
    axs[1].set_title("Exponential Quadratic Kernel (scale={:.2f}, length scale={:.2f})".format(k1Scale, k1LengthScale))
    axs[1].set_xlabel("Time (sec)")

    plt.show()

    pdb.set_trace()

if __name__=="__main__":
    test_GaussianProcess_eval()
