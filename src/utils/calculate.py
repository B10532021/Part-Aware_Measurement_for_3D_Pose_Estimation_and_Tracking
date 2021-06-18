import cv2
import torch
import numpy as np
import numpy.linalg as la
from numba import vectorize, float32, float64, jit, boolean
from math import sqrt

def get_believe(points2d):
    believe = []
    for point in points2d:
        w = point[2]
        if w >= 0:
            believe.append(w)
    return np.mean(believe)

@vectorize([float64(float64,float64,float64,float64,float64)])
def line_to_point_distance(a,b,c,x,y):
    return abs(a*x + b*y + c) / sqrt(a**2 + b**2)

def line2line_distance_3D(pt1, directions1, pt2, directions2):
    n = np.cross(directions1, directions2)
    n = n / np.linalg.norm(n, axis=1).reshape(-1, 1)
    distances = np.abs(np.sum(n * (pt1-pt2), axis=1))
    return distances

def line2point_distance_3D(camera_position, directions, points3d):
    x0 = points3d.astype(np.float)
    x1 = camera_position
    x2 = camera_position + directions
    cross = np.cross(x2-x1, x1-x0)
    distances = la.norm(cross, axis=1) / la.norm(x2-x1, axis=1)
    return distances

def euclidean_distance(a, b):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    distances = np.clip(r2, 0., float(np.inf))
    return np.maximum(0.0, distances.min(axis=0))

def cosine_distance(a, b, data_is_normalized=False):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    distances = 1. - np.dot(a, b.T)
    return distances.min(axis=0)

def transform_closure(X_bin):
    """
    Convert binary relation matrix to permutation matrix
    :param X_bin: torch.tensor which is binarized by a threshold
    :return:
    """
    temp = torch.zeros_like ( X_bin )
    N = X_bin.shape[0]
    for k in range ( N ):
        for i in range ( N ):
            for j in range ( N ):
                temp[i][j] = X_bin[i, j] or (X_bin[i, k] and X_bin[k, j])
    vis = torch.zeros ( N )
    match_mat = torch.zeros_like ( X_bin )
    for i, row in enumerate ( temp ):
        if vis[i]:
            continue
        for j, is_relative in enumerate ( row ):
            if is_relative:
                vis[j] = 1
                match_mat[j, i] = 1
    return match_mat

def proj2dpam(Y, tol=1e-4):
    X0 = Y
    X = Y
    I2 = 0

    for iter_ in range ( 10 ):

        X1 = projR ( X0 + I2 )
        I1 = X1 - (X0 + I2)
        X2 = projC ( X0 + I1 )
        I2 = X2 - (X0 + I1)

        chg = torch.sum ( torch.abs ( X2[:] - X[:] ) ) / X.numel ()
        X = X2
        if chg < tol:
            return X
    return X

def projR(X):
    for i in range ( X.shape[0] ):
        X[i, :] = proj2pav ( X[i, :] )
    return X

def projC(X):
    for j in range ( X.shape[1] ):
        X[:, j] = proj2pav ( X[:, j] )
    return X

def proj2pav(y):
    y[y < 0] = 0
    x = torch.zeros_like ( y )
    if torch.sum ( y ) < 1:
        x += y
    else:
        u, _ = torch.sort ( y, descending=True )
        sv = torch.cumsum ( u, 0 )
        to_find = u > (sv - 1) / (torch.arange ( 1, len ( u ) + 1, device=u.device, dtype=u.dtype ))
        rho = torch.nonzero ( to_find.reshape ( -1 ) )[-1]
        theta = torch.max ( torch.tensor ( 0, device=sv.device, dtype=sv.dtype ), (sv[rho] - 1) / (rho.float () + 1) )
        x += torch.max ( y - theta, torch.tensor ( 0, device=sv.device, dtype=y.dtype ) )
    return x