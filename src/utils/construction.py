import cv2
import numpy as np
import numpy.linalg as la
from math import exp

from default_limbs import DEFAULT_SENSIBLE_LIMB_LENGTH
from matching import epipolar_distance, BIP_matching

def top_down_pose_kernel(cameras, poses2d, weight2d=None):
    poses3d = []
    weight3d = []
    reproj_error = []
    for i in range(len(poses2d)):
        for j in range(i + 1, len(poses2d)):
            projmat_i, projmat_j = cameras[i].P, cameras[j].P
            pose2d_i, pose2d_j = poses2d[i], poses2d[j]
            pose3d_homo = cv2.triangulatePoints ( projmat_i, projmat_j, pose2d_i.T, pose2d_j.T )
            pose3d_ij = pose3d_homo[:3] / pose3d_homo[3]
            poses3d.append(pose3d_ij)
            weight3d.append((weight2d[i] + weight2d[j]) / 2)
            this_error = 0
            for camera, pk in zip(cameras, poses2d):
                projmat_k = camera.P
                projected_pose_k_homo = projmat_k @ pose3d_homo
                projected_pose_k = projected_pose_k_homo[:2] / (projected_pose_k_homo[2] + 10e-6)
                this_error += np.linalg.norm ( projected_pose_k.T - pk )
            reproj_error.append(this_error)
        idx = np.argmin(reproj_error)
    pose3d = poses3d[idx]

    return pose3d.T, weight3d[idx]

def hybrid_pose_kernel(cameras, poses2d, joint_num):
    if len(poses2d) < 2:
        return
    
    candidates = np.zeros((joint_num, len(poses2d) * (len(poses2d) - 1) // 2, 3))
    cnt = 0
    for i in range(len(poses2d)):
        for j in range(i + 1, len(poses2d)):
            projmat_i, projmat_j = cameras[i].P, cameras[j].P
            pose2d_i, pose2d_j = poses2d[i], poses2d[j]
            pose3d_homo = cv2.triangulatePoints ( projmat_i, projmat_j, pose2d_i.T, pose2d_j.T )
            pose3d_ij = pose3d_homo[:3] / pose3d_homo[3]
            candidates[:, cnt] += pose3d_ij.T
            cnt += 1

    joint_num = len(candidates)
    point_num = len(candidates[0])
    unary = np.log10(np.ones((joint_num,point_num)) * 10e-6)
    coco_2_skel = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    candidates = np.array ( candidates )[coco_2_skel]
    unary = unary[coco_2_skel]
    skel = getskel ()
    edges = getPictoStruct(skel, load_distribution('Unified'))
    xp = inferPict3D_MaxProd ( unary, edges, candidates )
    human = np.array ( [candidates[i][j] for i, j in zip ( range ( xp.shape[0] ), xp )] )
    human_coco = np.zeros ( (17, 3) )
    human_coco[[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] = human
    human_coco[[1, 2, 3, 4]] = human_coco[0]

    return human_coco

def SVD_pose_kernel(cameras, Ts, joints, remains, lambda_t, next_pose=None):
    pose3d = []
    for jid, (joint, remain) in enumerate(zip(joints, remains)):
        if len(remain) <= 1:
            pose3d.append(np.array([None, None, None]) if next_pose is None else next_pose[jid])
            continue

        C = []
        W = []
        for i, j in enumerate(remain):
            x = joint[j][0]
            y = joint[j][1]
            P = cameras[j].P
            C.append(y * P[2] - P[0])
            W.append(exp(-lambda_t * Ts[j]) / la.norm(C[2 * i]))
            C.append(x * P[2] - P[1])
            W.append(exp(-lambda_t * Ts[j]) / la.norm(C[2 * i + 1]))
        A = np.multiply(np.array(W).reshape(-1,1), C)
        U, sigma, VT = la.svd(A)
        V = np.transpose(VT)
        X = np.transpose(V[:, -1])
        X /= X[3]
        pose3d.append(X[:3])
    return pose3d

def SVD_pose_kernel_jf(cameras, Ts, pose_mat, lambda_t, remains, joints_views, next_pose=None):
    A = []
    for camera, poses, T in zip(cameras, pose_mat, Ts):
        poses = np.flip(poses[:,:2], axis=1)
        C = np.multiply(poses.reshape(-1,1), np.repeat([camera.P[2]], poses.size, 0))
        C = C - np.tile(camera.P[:2], (len(poses), 1))
        C = C / np.linalg.norm(C, axis=1).reshape(-1, 1)
        W = np.repeat([exp(-lambda_t * T)], poses.size, 0)
        A_ = np.multiply(W.reshape(-1,1), C).reshape(-1,2,4)
        A.append(A_)
    A = np.concatenate(A, axis=1)

    pose3d = np.zeros((17, 3))
    remains = remains == 1
    for i, views in enumerate(joints_views):
        if len(views) == 0:
            continue
        if i == 0:
            pose3d[views] = next_pose[views]
        else:
            A_ = A[views][remains[views]].reshape(len(views),-1,4)
            U, sigma, VT = la.svd(A_)
            V = np.transpose(VT, (0,2,1))
            X = V[:,:,-1]
            pose3d[views] = X[:,:3] / X[:,3].reshape(-1,1)
    return pose3d

def SVD_pose_kernel_parallel(cameras, Ts, pose_mat, lambda_t):
    A = []
    for camera, poses, T in zip(cameras, pose_mat, Ts):
        poses = np.flip(poses[:,:2], axis=1)
        C = np.multiply(poses.reshape(-1,1), np.repeat([camera.P[2]], poses.size, 0))
        C = C - np.tile(camera.P[:2], (len(poses), 1))
        C = C / np.linalg.norm(C, axis=1).reshape(-1, 1)
        W = np.repeat([exp(-lambda_t * T)], poses.size, 0)
        A_ = np.multiply(W.reshape(-1,1), C).reshape(-1,2,4)
        A.append(A_)
    A = np.concatenate(A, axis=1)
    U, sigma, VT = la.svd(A)
    V = np.transpose(VT, (0,2,1))
    X = V[:,:,-1]
    pose3d = X[:,:3] / X[:,3].reshape(-1,1)
    return pose3d

def correct_limbs(human, scale_to_mm):
    # --- remove limbs with bad length ---
    ua_range = DEFAULT_SENSIBLE_LIMB_LENGTH[2]
    la_range = DEFAULT_SENSIBLE_LIMB_LENGTH[3]
    ul_range = DEFAULT_SENSIBLE_LIMB_LENGTH[7]
    ll_range = DEFAULT_SENSIBLE_LIMB_LENGTH[8]

    # check left arm
    if test_distance(human, scale_to_mm, 5, 6, *ua_range):
        human[6] = None
        human[7] = None  # we need to disable hand too
    elif test_distance(human, scale_to_mm, 6, 7, *la_range):
        human[7] = None

    # check right arm
    if test_distance(human, scale_to_mm, 2, 3, *ua_range):
        human[3] = None
        human[4] = None  # we need to disable hand too
    elif test_distance(human, scale_to_mm, 3, 4, *la_range):
        human[4] = None

    # check left leg
    if test_distance(human, scale_to_mm, 11, 12, *ua_range):
        human[12] = None
        human[13] = None  # we need to disable foot too
    elif test_distance(human, scale_to_mm, 12, 13, *la_range):
        human[13] = None

    # check right leg
    if test_distance(human, scale_to_mm, 8, 9, *ua_range):
        human[9] = None
        human[10] = None  # we need to disable foot too
    elif test_distance(human, scale_to_mm, 9, 10, *la_range):
        human[10] = None

def test_distance(human, scale_to_mm, jid1, jid2, lower, higher):
    """
    :param human: [ (x, y, z) ] * J
    :param scale_to_mm:
    :param jid1:
    :param jid2:
    :param lower:
    :param higher:
    :return:
    """
    a = human[jid1]
    b = human[jid2]
    if a is None or b is None:
        return False
    distance = la.norm(a - b) * scale_to_mm
    if lower <= distance <= higher:
        return False
    else:
        return True

def getskel():
    skel = {}
    skel['tree'] = [{} for i in range ( 13 )]
    skel['tree'][0]['name'] = 'Nose'
    skel['tree'][0]['children'] = [1, 2, 7, 8]
    skel['tree'][1]['name'] = 'LSho'
    skel['tree'][1]['children'] = [3]
    skel['tree'][2]['name'] = 'RSho'
    skel['tree'][2]['children'] = [4]
    skel['tree'][3]['name'] = 'LElb'
    skel['tree'][3]['children'] = [5]
    skel['tree'][4]['name'] = 'RElb'
    skel['tree'][4]['children'] = [6]
    skel['tree'][5]['name'] = 'LWri'
    skel['tree'][5]['children'] = []
    skel['tree'][6]['name'] = 'RWri'
    skel['tree'][6]['children'] = []
    skel['tree'][7]['name'] = 'LHip'
    skel['tree'][7]['children'] = [9]
    skel['tree'][8]['name'] = 'RHip'
    skel['tree'][8]['children'] = [10]
    skel['tree'][9]['name'] = 'LKne'
    skel['tree'][9]['children'] = [11]
    skel['tree'][10]['name'] = 'RKne'
    skel['tree'][10]['children'] = [12]
    skel['tree'][11]['name'] = 'LAnk'
    skel['tree'][11]['children'] = []
    skel['tree'][12]['name'] = 'RAnk'
    skel['tree'][12]['children'] = []
    return skel

def load_distribution(dataset='Shelf'):
    # please give a detailed description for "distribution"
    joints2edges = {(0, 1): 0,
                    (1, 0): 0,
                    (0, 2): 1,
                    (2, 0): 1,
                    (0, 7): 2,
                    (7, 0): 2,
                    (0, 8): 3,
                    (8, 0): 3,
                    (1, 3): 4,
                    (3, 1): 4,
                    (2, 4): 5,
                    (4, 2): 5,
                    (3, 5): 6,
                    (5, 3): 6,
                    (4, 6): 7,
                    (6, 4): 7,
                    (7, 9): 8,
                    (9, 7): 8,
                    (8, 10): 9,
                    (10, 8): 9,
                    (9, 11): 10,
                    (11, 9): 10,
                    (10, 12): 11,
                    (12, 10): 11}

    distribution_dict = {
        'Shelf': {'mean': np.array ( [0.30280354, 0.30138756, 0.79123502, 0.79222949, 0.28964179,
                                    0.30393598, 0.24479075, 0.24903801, 0.40435882, 0.39445121,
                                    0.3843522, 0.38199836] ),
                'std': np.array ( [0.0376412, 0.0304385, 0.0368604, 0.0350577, 0.03475468,
                                0.03876828, 0.0353617, 0.04009757, 0.03974647, 0.03696424,
                                0.03008979, 0.03143456] ) * 2, 'joints2edges': joints2edges
                },
        'Campus':{'mean': np.array ( [0.29567343, 0.28090078, 0.89299809, 0.88799211, 0.32651703,
                                    0.33454941, 0.29043165, 0.29932416, 0.43846395, 0.44881553,
                                    0.46952846, 0.45528477] ),
                'std': np.array ( [0.01731019, 0.0226062, 0.06650426, 0.06009805, 0.04606478,
                                0.04059899, 0.05868499, 0.06553948, 0.04129285, 0.04205624,
                                0.03633746, 0.02889456] ) * 2, 'joints2edges': joints2edges},
        'Unified': {'mean': np.array ( [0.29743698, 0.28764493, 0.86562234, 0.86257052, 0.31774172,
                                0.32603399, 0.27688682, 0.28548218, 0.42981244, 0.43392589,
                                0.44601327, 0.43572195] ),
                    'std': np.array ( [0.02486281, 0.02611557, 0.07588978, 0.07094158, 0.04725651,
                                0.04132808, 0.05556177, 0.06311393, 0.04445206, 0.04843436,
                                0.0510811, 0.04460523] ) * 16, 'joints2edges': joints2edges}
                    }

    return distribution_dict[dataset]

class Edge:
    def __init__(self):
        child = None
        parent = None
        bone_mean = None
        bone_std = None

def getPictoStruct(skel, distribution):
    """to get the pictorial structure"""
    graph = skel['tree']
    level = np.zeros ( len ( graph ) )
    for i in range ( len ( graph ) ):
        queue = np.array ( graph[i]['children'], dtype=np.int32 )
        for j in range ( queue.shape[0] ):
            graph[queue[j]]['parent'] = i
        while queue.shape[0] != 0:
            level[queue[0]] = level[queue[0]] + 1
            queue = np.append ( queue, graph[queue[0]]['children'] )
            queue = np.delete ( queue, 0 )
            queue = np.array ( queue, dtype=np.int32 )
    trans_order = np.argsort ( -level )
    edges = [Edge() for i in range(len(trans_order) - 1)]
    for i in range ( len ( trans_order )-1):
        edges[i].child = trans_order[i]
        edges[i].parent = graph[edges[i].child]['parent']
        edge_id = distribution['joints2edges'][(edges[i].child, edges[i].parent)]
        edges[i].bone_mean = distribution['mean'][edge_id]
        edges[i].bone_std = distribution['std'][edge_id]
    return edges

def get_prior(i, n, p, j, edges, X):
    """calculate the probability p(si,sj)"""
    edges_2_joint[13]
    edges_2_joint[:] = [-1, 8, 9, 4, 5, 0, 1, 10, 11, 6, 7, 2, 3]
    bone_std = edges[edges_2_joint[i]].bone_std
    bone_mean = edges[edges_2_joint[i]].bone_mean
    distance = np.linalg.norm ( X[i][n] - X[p][j] )
    relative_error = np.abs ( distance - bone_mean ) / bone_std
    prior = scipy.stats.norm.sf ( relative_error ) * 2
    return prior

def get_max(i, p, j, unary, candidateNum, edges, X):
    unary_sum = np.zeros ( candidateNum )
    for n in range ( candidateNum ):
        prior = get_prior ( i, n, p, j, edges, X )
        unary_sum[n] = prior + unary[i][n]
    this_max = np.max ( unary_sum )
    indextemp = np.where ( unary_sum == np.max ( unary_sum ) )
    if len(indextemp[0]) == 0:
        index = 0
    else:
        index = indextemp[0][0]
    return this_max, index

def inferPict3D_MaxProd(unary, edges, X):
    """
    To inference the pictorial structure in parallel
    """
    unary_c = unary
    X_c = X
    jointNum = unary.shape[0]
    candidateNum = unary[0].shape[0]

    for curJoint in range ( jointNum - 1, 0, -1 ):
        parentJoint = get_pa[curJoint]
        for parentCandidate in prange ( candidateNum, nogil=True):
            maxUnary = -100000 # very negative value
            for curCandidate in range ( candidateNum ):
                # Begin of get prior
                bone_std = edges[edges2Joint[curJoint]].bone_std
                bone_mean = edges[edges2Joint[curJoint]].bone_mean
                distance = c_sqrt((X_c[curJoint][curCandidate][0] - X_c[parentJoint][parentCandidate][0])**2+
                                  (X_c[curJoint][curCandidate][1] - X_c[parentJoint][parentCandidate][1])**2+
                                  (X_c[curJoint][curCandidate][2] - X_c[parentJoint][parentCandidate][2])**2)
                # relative_error = (distance - bone_mean) / bone_std
                prior = c_exp(-(distance-bone_mean)**2/(2*bone_std**2))/bone_std
                # end of get prior
                if prior + unary_c[curJoint][curCandidate] > maxUnary:
                    maxUnary = prior + unary_c[curJoint][curCandidate]
            unary_c[parentJoint][parentCandidate] += maxUnary
    # get the max index

    values = unary[0]
    xpk = np.zeros ( unary.shape[0], dtype=np.int64 )  # Also change from original implementation
    xpk[0] = values.argmax ()
    for curJoint in range ( 1, jointNum ):
        parentJoint = get_pa[curJoint]
        xn = get_max ( curJoint, parentJoint, xpk[parentJoint], unary, candidateNum, edges, X )
        xpk[curJoint] = xn[1]
    return xpk