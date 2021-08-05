import torch

def getTrueVariationalMean(t, latent_mean, inducing_points_locs, kernel):
    Ktz = kernel.buildKernelMatrix(X1=t, X2=inducing_points_locs)
    Kzz = kernel.buildKernelMatrix(X1=inducing_points_locs, X2=inducing_points_locs)
    res = torch.linalg.lstsq(Ktz, latent_mean)
    vMean = torch.matmul(Kzz, res.solution)
    return vMean
