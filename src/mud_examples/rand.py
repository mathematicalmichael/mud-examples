import numpy as np
from typing import List


def createRandomLinearMap(dim_input, dim_output, dist='normal', repeated=False):
    """
    Create random linear map from P dimensions to S dimensions.
    """
    if dist == 'normal':
        M     = np.random.randn(dim_output, dim_input)
    else:
        M     = np.random.rand(dim_output, dim_input)
    if repeated:  # just use first row
        M     = np.array(list(M[0, :]) * dim_output).reshape(dim_output, dim_input)

    return M


def createNoisyReferenceData(M, reference_point, std):
    dim_input  = len(reference_point)
    dim_output = M.shape[0]
    assert M.shape[1] == dim_input, "Mperator/Data dimension mismatch"
    if isinstance(std, int) or isinstance(std, float):
        std    = np.array([std] * dim_output)

    ref_input  = np.array(list(reference_point)).reshape(-1, 1)
    ref_data   = M @ ref_input
    noise      = np.diag(std) @ np.random.randn(dim_output, 1)
    data       = ref_data + noise
    return data


def createRandomLinearPair(reference_point, num_observations, std,
                           dist='normal', repeated=False):
    """
    data will come from a normal distribution centered at zero
    with standard deviation given by `std`
    QoI map will come from standard uniform or normal if dist=normal
    if `repeated` is True, the map will be rank 1.
    """
    dim_input = len(reference_point)
    M         = createRandomLinearMap(dim_input, num_observations, dist, repeated)
    data      = createNoisyReferenceData(M, reference_point, std)
    return M, data


def createRandomLinearProblem(reference_point, num_qoi,
                              num_obs_list, std_list,
                              dist='normal', repeated=False):
    """
    Wrapper around `createRandomLinearQoI` to generalize to multiple QoI maps.
    """
    if isinstance(std_list, int) or isinstance(std_list, float):
        std_list                = [std_list] * num_qoi
    else:
        assert len(std_list) == num_qoi

    if isinstance(num_obs_list, int) or isinstance(num_obs_list, float):
        num_obs_list   = [num_obs_list] * num_qoi
    else:
        assert len(num_obs_list) == num_qoi

    assert len(std_list) == len(num_obs_list)
    results       = [createRandomLinearPair(reference_point, n, s, dist, repeated)
                     for n, s in zip(num_obs_list, std_list)]
    operator_list = [r[0] for r in results]
    data_list     = [r[1] for r in results]
    return operator_list, data_list, std_list


def randA_outer(dim_output, dim_input=None, seed=None):
    """
    Generate `dimension` rank-1 matrices using Gaussian entries
    to generate a vector `x` and then take outer-product with self.
    """
    if seed is not None:
        np.random.seed(seed)
    A = []
    for i in range(dim_output):
        _x = randA_gauss(dim_output, 1)
#         _x = _x / np.linalg.norm(_x) # unit vector
        _a = np.outer(_x,_x)
        A.append(_a)
    return A


def randA_list_svd(dim_output, dim_input=None, seed=None) -> List:
    """
    Generate random square Gaussian matrix, perform SVD, and
    construct rank-1 matrices from components. Return list of them.
    Sum `R` entries of this returned list to generate a rank-R matrix.
    """
    A = []
    _A = randA_gauss(dim_output, dim_input, seed=seed)
    u, s, v = np.linalg.svd(_A)
    for i in range(dim_output):
        _a = s[i]*(u[:,i].reshape(-1,1))@v[:,i].reshape(1,-1)
        A.append(_a)
    return A


def randA_qr(dim_output, dim_input=None, seed=None):
    """
    Generate random Gaussian matrix, perform QR,
    and returns the resulting (orthogonal) Q
    """
    A = randA_gauss(dim_output, dim_input, seed)
    A, _ = np.linalg.qr(A)
    return A


def randA_gauss(dim_output, dim_input=None, seed=None):
    """
    Generate random Gaussian matrix, perform QR,
    and returns the resulting (orthogonal) Q
    """
    if seed is not None:
        np.random.seed(seed)
    if dim_input is None:
        dim_input = dim_output
    A = np.random.randn(dim_output, dim_input)
    return A


def randP(dim_output, dim_input=None, randA=randA_gauss, seed=None):
    """
    Constructs problem set
    """
    A = randA(dim_output, dim_input, seed=seed)
    b = np.random.randn(dim_output).reshape(-1,1)
    return A, b

