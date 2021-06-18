import cv2
import time
import torch
import numpy as np
import numpy.linalg as la
from scipy.optimize import linear_sum_assignment

from calculate import get_believe, transform_closure, proj2dpam
from calculate import line2point_distance_3D, line_to_point_distance, line2line_distance_3D
def back_project_ray(RK_INV, camera_position, points):
    joints = len(points)
    projected_points = np.concatenate([points[:,:2], np.ones((joints, 1))], axis=1)
    Xs = np.repeat([RK_INV], joints, axis=0) @ projected_points.reshape(joints,3,1)
    Xs = Xs[:,:3]
    directions = Xs.reshape(joints,3)
    directions = directions / np.linalg.norm(directions, axis=1).reshape(joints, 1)
    return directions

def back_project_distance(points0, POS0, RK_INV0, points1, POS1, RK_INV1):
    assert len(points0) == len(points1)
    camera_position0 = POS0
    camera_position1 = POS1
    distances = []
    directions0 = back_project_ray_(RK_INV0, camera_position0, points0)
    directions1 = back_project_ray_(RK_INV1, camera_position1, points1)
    distances = line2line_distance_3D_(camera_position0,directions0, camera_position1, directions1)
    return np.mean(distances)

def back_project_affinity(points_set, cams, sub_imgid2cam):
    M, _, _ = points_set.shape
    distance_matrix = np.ones ( (M, M), dtype=np.float32 )
    np.fill_diagonal(distance_matrix, 0)
    for i in range(M - 1):
        for j in range(i, M):
            if sub_imgid2cam[i] == sub_imgid2cam[j]:
                continue
            cam_id0 = sub_imgid2cam[i]
            cam_id1 = sub_imgid2cam[j]
            pose_id0 = points_set[i]
            pose_id1 = points_set[j]
            distance = back_project_distance(pose_id0, cams[cam_id0].POS, cams[cam_id0].RK_INV, pose_id1, cams[cam_id1].POS, cams[cam_id1].RK_INV)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    affinity_matrix = - (distance_matrix - distance_matrix.mean ()) / distance_matrix.std ()
    # TODO: add flexible factor
    affinity_matrix = 1 / (1 + np.exp ( -5 * affinity_matrix ))
    # print(affinity_matrix)
    return affinity_matrix

def epipolar_distance(cam1, person1, cam2, person2):
    """ calculate the epipolar distance between two humans
    :param cam1:
    :param person1:
    :param cam2:
    :param person2:
    :return:
    """
    n_pairs = len(person1)
    F = cam1.F[cam2.cid]
    pts1 = np.flip(person1[:,:2], axis=1)
    pts2 = np.flip(person2[:,:2], axis=1)


    if len(pts1) == 0:
        return []
    
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    epilines_1to2 = np.squeeze(
        cv2.computeCorrespondEpilines(pts1, 2, F))
    
    epilines_2to1 = np.squeeze(
        cv2.computeCorrespondEpilines(pts2, 1, F))
    

    if n_pairs == 1:
        epilines_1to2 = np.expand_dims(epilines_1to2, axis=0)
        epilines_2to1 = np.expand_dims(epilines_2to1, axis=0)
    
    pts1 = np.concatenate([pts1, np.ones((n_pairs, 1))], axis=1)
    pts2 = np.concatenate([pts2, np.ones((n_pairs, 1))], axis=1)
    d1 = np.abs(np.sum(pts1 * epilines_2to1, axis=1)) / np.sqrt(np.sum(epilines_2to1[:,:2] ** 2, axis=1))
    d2 = np.abs(np.sum(pts2 * epilines_1to2, axis=1)) / np.sqrt(np.sum(epilines_1to2[:,:2] ** 2, axis=1))
    distances = np.hstack([d1.reshape(-1,1),d2.reshape(-1,1)])

    # print(epilines_1to2)
    # test = np.transpose(F.T @ pts1.T)
    # test = test / np.linalg.norm(test[:, :2], axis=1).reshape(-1,1)
    

    return distances

def epipolar_affinity(cameras, sub_imgid2cam, pose_mat, num_joints):
    M = len(pose_mat)
    pose_mat = np.array(pose_mat)
    distances_matrix = np.zeros ((M, M, num_joints), dtype=np.float32)
    affinity_matrix = np.ones ((M, M), dtype=np.float32) * 25
    np.fill_diagonal(affinity_matrix, 0)
    for i in range(M - 1):
        for j in range(i+1, M):
            if sub_imgid2cam[i] == sub_imgid2cam[j]:
                continue
            cam_id0 = sub_imgid2cam[i]
            cam_id1 = sub_imgid2cam[j]
            pose_id0 = pose_mat[i]
            pose_id1 = pose_mat[j]
            distances = epipolar_distance(cameras[cam_id0], pose_id0, cameras[cam_id1], pose_id1)
            distances_matrix[i, j] = [(dis[0] + dis[1]) /2 for dis in distances]
            distances_matrix[j, i] = [(dis[0] + dis[1]) /2 for dis in distances]
            distance = np.mean([(dis[0] + dis[1]) /2 for dis in distances])
            affinity_matrix[i, j] = distance
            affinity_matrix[j, i] = distance
    return affinity_matrix, distances_matrix

def epipolar_affinity_parallel(cameras, sub_imgid2cam, pose_mat, num_joints):
    M = len(pose_mat)
    pose_mat = np.array(pose_mat)
    pose_mat = np.concatenate([np.flip(pose_mat[:,:,:2], axis=2), np.ones((M,num_joints,1))], axis=2)
    poses0 = np.transpose(np.repeat(pose_mat, M, 0), (0,2,1))
    poses1 = np.tile(pose_mat, (M,1,1))
    # poses0 = []
    # poses1 = []
    Fs = []
    for i in range(M):
        for j in range(M):
            # pose0 = np.concatenate([np.flip(pose_mat[i, :, :2], axis=1), np.ones((num_joints,1))], axis=1)
            # pose1 = np.concatenate([np.flip(pose_mat[j, :, :2], axis=1), np.ones((num_joints,1))], axis=1)
            # poses0.append(pose0.T)
            # poses1.append(pose1)

            cam_id0 = sub_imgid2cam[i]
            cam_id1 = sub_imgid2cam[j]
            if cam_id0 == cam_id1:
                Fs.append(np.zeros((3,3)))
            else:
                Fs.append(cameras[cam_id0].F[cameras[cam_id1].cid].T)
    # poses0 = np.array(poses0)
    # poses1 = np.array(poses1)
    Fs = np.array(Fs)
    lines = np.transpose(Fs @ poses0, (0, 2, 1))
    nu = np.linalg.norm(lines[:,:,:2], axis=2).reshape(-1,num_joints,1)
    nu[nu == 0] = 1
    lines /= nu
    norm = np.sum(lines[:,:,:2] ** 2, axis=2)
    norm[norm==0] = 1
    distances = np.abs(np.sum(poses1 * lines, axis=2)) / np.sqrt(norm)
    distances_matrix = distances.reshape(M,M,-1)
    distances_matrix = (distances_matrix + np.transpose(distances_matrix, (1,0,2))) / 2
    affinity_matrix = np.mean(distances_matrix, axis=2)

    return affinity_matrix, distances_matrix

def geometry_distance(pts_0, pts_1, F):
    lines = cv2.computeCorrespondEpilines ( pts_0.reshape ( -1, 1, 2 ), 2, F )
    lines = lines.reshape(-1, 17, 1, 3)
    lines = lines.transpose(0, 2, 1, 3)
    points_1 = np.ones([1, pts_1.shape[0], 17, 3])
    points_1[0, :, :, :2] = pts_1
    dist = np.sum(lines * points_1 , axis=3) #/ np.linalg.norm(lines[:, :, :, :2], axis=3)
    dist = np.abs(dist)
    dist = np.mean(dist, axis=2)
    return dist

def geometry_affinity(points_set, cameras, dimGroup):
    M, _, _ = points_set.shape
    # distance_matrix = np.zeros ( (M, M), dtype=np.float32 )
    distance_matrix = np.ones ( (M, M), dtype=np.float32 ) * 25
    np.fill_diagonal(distance_matrix, 0)
    for cam_id0, h in enumerate ( range ( len ( dimGroup ) - 1 ) ):
        for cam_add, k in enumerate ( range ( cam_id0+1, len(dimGroup)-1 ) ):
            cam_id1 = cam_id0 + cam_add + 1
            if dimGroup[h] == dimGroup[h+1] or dimGroup[k] == dimGroup[k+1]:
                continue
            F0 = cameras[cam_id0]
            F1 = cameras[cam_id1]
            pose_id0 = points_set[dimGroup[h]:dimGroup[h + 1]]
            pose_id1 = points_set[dimGroup[k]:dimGroup[k + 1]]
            distance_matrix[dimGroup[h]:dimGroup[h + 1], dimGroup[k]:dimGroup[k + 1]] = \
                (geometry_distance(pose_id0, pose_id1, F0[cam_id1]) + geometry_distance(pose_id1, pose_id0, F1[cam_id0]).T) / 2

            distance_matrix[dimGroup[k]:dimGroup[k+1], dimGroup[h]:dimGroup[h+1]] = distance_matrix[dimGroup[h]:dimGroup[h + 1], dimGroup[k]:dimGroup[k + 1]].T
    affinity_matrix = - (distance_matrix - distance_matrix.mean ()) / (distance_matrix.std () + 10e-6)
    affinity_matrix = 1 / (1 + np.exp ( -5 * affinity_matrix ))
    return affinity_matrix

def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    # import pdb; pdb.set_trace()
    x = torch.cat ( [query_features[f].unsqueeze ( 0 ) for f, _, _ in query], 0 )
    y = torch.cat ( [gallery_features[f].unsqueeze ( 0 ) for f, _, _ in gallery], 0 )
    m, n = x.size ( 0 ), y.size ( 0 )
    x = x.view ( m, -1 )
    y = y.view ( n, -1 )
    dist = torch.pow ( x, 2 ).sum ( dim=1, keepdim=True ).expand ( m, n ) + \
           torch.pow ( y, 2 ).sum ( dim=1, keepdim=True ).expand ( n, m ).t ()
    dist.addmm_ ( 1, -2, x, y.t () )
    return dist

def pairwise_affinity(query_features, gallery_features, query=None, gallery=None):
    # import pdb; pdb.set_trace()
    x = torch.cat ( [query_features[f].unsqueeze ( 0 ) for f, _, _ in query], 0 )
    y = torch.cat ( [gallery_features[f].unsqueeze ( 0 ) for f, _, _ in gallery], 0 )
    m, n = x.size ( 0 ), y.size ( 0 )
    x = x.view ( m, -1 )
    y = y.view ( n, -1 )
    dist = torch.pow ( x, 2 ).sum ( dim=1, keepdim=True ).expand ( m, n ) + \
           torch.pow ( y, 2 ).sum ( dim=1, keepdim=True ).expand ( n, m ).t ()
    dist.addmm_ ( 1, -2, x, y.t () )
    normalized_affinity = - (dist - dist.mean ()) / dist.std ()
    affinity = torch.sigmoid ( normalized_affinity * torch.tensor ( 5. ) )  # x5 to match 1->1
    # pro = x @ y.t ()
    # norms = x.norm ( dim=1 ).unsqueeze ( 1 ) @ y.norm ( dim=1 ).unsqueeze ( 0 )
    # affinity = (pro / norms + 1) / 2  # map from (-1, 1) to (0, 1)
    # affinity = torch.sigmoid ( pro / norms ) #  map to (0, 1)
    return affinity

def embedding_affinity(query_features, gallery_features, query=None, gallery=None, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    x = [query_features[f].numpy() for f, _, _ in query]
    y = [gallery_features[f].numpy() for f, _, _ in gallery]
    cost_matrix = np.zeros((len(x), len(y)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    cost_matrix = np.maximum(0.0, cdist(x, y, metric))  # Nomalized features
    cost_matrix = 1 - cost_matrix
    return torch.from_numpy(cost_matrix).float().cuda()


def BIP_matching(model, cameras, dimGroup, pose_mat=None, num_joints=17, threshold=40):
    sub_imgid2cam = np.zeros(dimGroup[-1] if dimGroup[-1] - 1 >= 0 else 0, dtype=np.int32)
    for idx, i in enumerate(range(len(dimGroup)-1)):
        sub_imgid2cam[dimGroup[i]:dimGroup[i+1]]=idx
    affinity_mat, _ = epipolar_affinity(cameras, sub_imgid2cam, pose_mat, num_joints)
    affinity_mat = 1 - affinity_mat / threshold
    matched_list = model.solve(affinity_mat.astype(np.double))
    return matched_list, sub_imgid2cam

def Greedy_matching(cameras, pose_mat=None, affinity_mat=None, costs=None, next_pose=None, mode='update'):
    n_nodes = affinity_mat.shape[0]
    matched_list = np.arange(n_nodes)
    binary_list = np.ones(n_nodes * 2, dtype=np.int)

    rows, cols = np.where(np.triu(affinity_mat) < 0)
    back_project_distances = np.zeros(n_nodes)
    for row, col in zip(rows, cols):
        if row not in matched_list or col not in matched_list:
            continue
        if mode == 'update':
            if  back_project_distances[row] == 0:
                RK_INV = cameras[row].RK_INV
                position = cameras[row].position
                point = np.array([np.flip(pose_mat[row,0,:2])])

                direction = back_project_ray(RK_INV, position, point)
                cost = line2point_distance_3D(position, direction, np.array([next_pose]))
                back_project_distances[row] = cost[0]

            if back_project_distances[col] == 0:
                RK_INV = cameras[col].RK_INV
                position = cameras[col].position
                point = np.array([np.flip(pose_mat[col,0,:2])])

                direction = back_project_ray(RK_INV, position, point)
                cost = line2point_distance_3D(position, direction, np.array([next_pose]))
                back_project_distances[col] = cost[0]

            if back_project_distances[row] > back_project_distances[col]:
                matched_list = matched_list[matched_list != row]
                binary_list[row*2:row*2+2] = 0
            else:
                matched_list = matched_list[matched_list != col]
                binary_list[col*2:col*2+2] = 0


            # if costs[row] > costs[col]:
            #     matched_list = matched_list[matched_list != row]
            #     binary_list[row*2:row*2+2] = 0
            # else:
            #     matched_list = matched_list[matched_list != col]
            #     binary_list[col*2:col*2+2] = 0
        else:
            sum1 = np.sum(affinity_mat[row])
            sum2 = np.sum(affinity_mat[col])
            if sum1 > sum2:
                matched_list = matched_list[matched_list != col]
                binary_list[col*2:col*2+2] = 0
            else:
                matched_list = matched_list[matched_list != row]
                binary_list[row*2:row*2+2] = 0
    return matched_list, binary_list, affinity_mat

def distance_between_3Dposes(pose1, weight1, pose2, weight2, z_axis):
    """
    :param pose1: dict('3dpose', '3dweight)
    :param pose2: dict('3dpose', '3dweight)
    :param z_axis: some datasets are rotated around one axis
    :return:
    """
    assert len(pose1) == len(pose2) and len(weight1) == len(pose1)and len(weight2) == len(pose2)
    distances = []
    dis = []
    wei = []
    for jid in range(len(pose1)):
        if np.isnan(pose2[jid]).any() or np.isnan(pose1[jid]).any():
            continue

        d = la.norm(pose2[jid] - pose1[jid])
        w = (1 - abs(weight1[jid] - weight2[jid])) * min(weight1[jid], weight2[jid])
        distances.append(d * w)
    if len(distances) == 0:
        # TODO check this heuristic
        # take the centre distance in x-y coordinates
        valid1 = []
        valid2 = []
        for jid in range(len(pose1)):
            if pose1[jid] is not None:
                valid1.append(pose1[jid])
            if pose2[jid] is not None:
                valid2.append(pose2[jid])

        assert len(valid1) > 0
        assert len(valid2) > 0
        mean1 = np.mean(valid1, axis=0)
        mean2 = np.mean(valid2, axis=0)
        assert len(mean1) == 3
        assert len(mean2) == 3

        # we only care about xy coordinates
        mean1[z_axis] = 0
        mean2[z_axis] = 0

        return la.norm(mean1 - mean2)
    else:
        return np.mean(distances)  # TODO try different versions
