import sys
import os
from scipy.io import loadmat
import jax
import jax.numpy as jnp
import svGPFA.stats.variationalDist
import svGPFA.utils.miscUtils

jax.config.update("jax_enable_x64", True)

# def test_get_flattened_params():
#     nTrials = 2
#     nIndPoints = [2, 2, 2]
# 
#     nLatents = len(nIndPoints)
#     qMu0 = [torch.rand((nTrials, nIndPoints[k], 1), dtype=torch.double) for k in range(nLatents)]
#     chol0Vecs = [torch.rand((nTrials, int(((nIndPoints[k]+1)*nIndPoints[k])/2), 1), dtype=torch.double) for k in range(nLatents)]
#     initial_params = {"qMu0": qMu0, "chol0Vecs": chol0Vecs}
# 
#     true_flattened_params = []
#     for k in range(nLatents):
#         true_flattened_params.extend(qMu0[k].flatten().tolist())
#     for k in range(nLatents):
#         true_flattened_params.extend(chol0Vecs[k].flatten().tolist())
# 
#     variationalDist = stats.svGPFA.variationalDist.VariationalDistChol()
#     variationalDist.setInitialParams(initial_params=initial_params)
#     flattened_params = variationalDist.get_flattened_params()
# 
#     assert(flattened_params==true_flattened_params)
# 
# def test_set_flattened_params():
#     nTrials = 2
#     nIndPoints = [2, 2, 2]
# 
#     nLatents = len(nIndPoints)
# 
#     qMu0_1 = [torch.rand((nTrials, nIndPoints[k], 1), dtype=torch.double) for k in range(nLatents)]
#     chol0Vecs_1 = [torch.rand((nTrials, int(((nIndPoints[k]+1)*nIndPoints[k])/2), 1), dtype=torch.double) for k in range(nLatents)]
#     initial_params_1 = {"qMu0": qMu0_1, "chol0Vecs": chol0Vecs_1}
#     variationalDist_1 = stats.svGPFA.variationalDist.VariationalDistChol()
#     variationalDist_1.setInitialParams(initial_params=initial_params_1)
#     flattened_params_1 = variationalDist_1.get_flattened_params()
# 
#     qMu0_2 = [torch.rand((nTrials, nIndPoints[k], 1), dtype=torch.double) for k in range(nLatents)]
#     chol0Vecs_2 = [torch.rand((nTrials, int(((nIndPoints[k]+1)*nIndPoints[k])/2), 1), dtype=torch.double) for k in range(nLatents)]
#     initial_params_2 = {"qMu0": qMu0_2, "chol0Vecs": chol0Vecs_2}
#     variationalDist_2 = stats.svGPFA.variationalDist.VariationalDistChol()
#     variationalDist_2.setInitialParams(initial_params=initial_params_2)
#     variationalDist_2.set_params_from_flattened(flattened_params=flattened_params_1)
#     flattened_params_2 = variationalDist_2.get_flattened_params()
# 
#     assert(flattened_params_1==flattened_params_2)
# 
# def test_set_params_requires_grad():
#     nTrials = 2
#     nIndPoints = [2, 2, 2]
# 
#     nLatents = len(nIndPoints)
# 
#     qMu0 = [torch.rand((nTrials, nIndPoints[k], 1), dtype=torch.double) for k in range(nLatents)]
#     chol0Vecs = [torch.rand((nTrials, int(((nIndPoints[k]+1)*nIndPoints[k])/2), 1), dtype=torch.double) for k in range(nLatents)]
#     initial_params = {"qMu0": qMu0, "chol0Vecs": chol0Vecs}
#     variationalDist = stats.svGPFA.variationalDist.VariationalDistChol()
#     variationalDist.setInitialParams(initial_params=initial_params)
#     variationalDist.set_params_requires_grad(requires_grad=True)
#     params = variationalDist.getParams()
#     for param in params:
#         assert(param.requires_grad)
# 
#     variationalDist.set_params_requires_grad(requires_grad=False)
#     params = variationalDist.getParams()
#     for param in params:
#         assert(not param.requires_grad)

def test_buildCov():
    tol = 1e-5
    dataFilename = os.path.join(os.path.dirname(__file__), "data/get_full_from_lowplusdiag.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['q_sqrt'].shape[0]
    nTrials = mat['q_sqrt'][(0,0)].shape[2]
    true_qSigma = [jax.device_put(mat['q_sigma'][(0,k)].astype("float64").transpose(2,0,1)) for k in range(nLatents)]
    chol_vecs = svGPFA.utils.miscUtils.getCholVecsFromCov(cov=true_qSigma)
    est_qSigma = svGPFA.utils.miscUtils.buildCovsFromCholVecs(
        chol_vecs=chol_vecs)

    error = jnp.array([jnp.linalg.norm(est_qSigma[k]-true_qSigma[k])
                       for k in range(len(est_qSigma))]).sum()

    assert(error<tol)

if __name__=="__main__":
    test_buildCov()
