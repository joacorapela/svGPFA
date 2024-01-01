
import sys
import os
import math
from scipy.io import loadmat
import jax
import jax.numpy as jnp
import svGPFA.utils.miscUtils

jax.config.update("jax_enable_x64", True)

def test_getPropSamplesCovered():
    N = 100
    tol = .1

    seed = 1234
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    mean = jax.random.uniform(key, shape=(N,))*2-1
    key, subkey = jax.random.split(key)
    std = jax.random.uniform(key, shape=(N,))*0.3
    key, subkey = jax.random.split(key)
    std = jax.random.uniform(key, shape=(N,))*0.3
    key, subkey = jax.random.split(key)
    sample = jax.random.normal(key, shape=mean.shape) * std + mean
    propSamplesCovered = svGPFA.utils.miscUtils.getPropSamplesCovered(sample=sample, mean=mean, std=std, percent=.95)
    assert(.95-tol<propSamplesCovered and propSamplesCovered<tol+.95)

def test_getDiagIndicesIn3DArray():
    N = 3
    M = 2
    trueDiagIndices = jnp.array([0, 4, 8, 9, 13, 17])

    diagIndices = svGPFA.utils.miscUtils.getDiagIndicesIn3DArray(N=N, M=M)
    assert(((trueDiagIndices-diagIndices)**2).sum()==0)

def test_build3DdiagFromDiagVector():
    N = 3
    M = 2
    v = jnp.arange(M*N, dtype=jnp.double)
    D = svGPFA.utils.miscUtils.build3DdiagFromDiagVector(v=v, N=N, M=M)
    trueD = jnp.array([[[0,0,0],[0,1,0],[0,0,2]],[[3,0,0],[0,4,0],[0,0,5]]],
                      dtype=jnp.double)
    assert(((trueD-D)**2).sum()==0)

def test_cholVecs(tol=1e-6, n_latents=3, n_trials=5, n_ind_points=10):
    M = int(n_ind_points * (n_ind_points + 1) / 2)
    tril_indices = jnp.tril_indices(n_ind_points)
    chols = [None] * n_latents
    covs = [None] * n_latents
    key = jax.random.PRNGKey(0)
    for k in range(n_latents):
        chols[k] = jnp.zeros((n_trials, n_ind_points, n_ind_points))
        for r in range(n_trials):
            key, subkey = jax.random.split(key)
            chols[k] = chols[k].at[r, tril_indices[0],
                                   tril_indices[1]].set(jax.random.normal(subkey,
                                                                          shape=(M,)))
        covs[k] = jnp.matmul(chols[k], jnp.transpose(chols[k], (0, 2, 1)))
    chol_vecs = svGPFA.utils.miscUtils.getVectorRepOfLowerTrianMatrices(
        lt_matrices=chols)
    covs_reconstructed = svGPFA.utils.miscUtils.buildCovsFromCholVecs(chol_vecs)
    for k in range(n_latents):
        error = jnp.mean((covs[k] - covs_reconstructed[k])**2)
        assert(error < tol)


# def test_j_cholesky():
#     tol = 1e-3
# 
#     A = np.randn((3, 4))
#     K = np.mm(A, A.T)
#     trueY = np.unsqueeze(np.tensor([1.0, 2.0, 3.0]), 1)
#     b = np.mm(K, trueY)
#     KChol = np.cholesky(K)
#     yTorch = np.cholesky_solve(b, KChol)
#     yJ = stats.svGPFA.utils.j_cholesky_solve(b, KChol)
#     error = ((yTorch-yJ)**2).sum()
#     assert(error<tol)
# 

if __name__=="__main__":
    # test_getDiagIndicesIn3DArray()
    # test_build3DdiagFromDiagVector()
    # # test_j_cholesky()
    test_getPropSamplesCovered()
    # test_cholVecs()
