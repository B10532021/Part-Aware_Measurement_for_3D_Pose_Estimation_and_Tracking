import cv2
import torch
import time as t
import numpy as np
import numpy.linalg as la
from math import exp
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from torchvision import transforms as T

from hypothesis import Hypothesis
from binary_integer_programming import GLPKSolver

from calculate import get_believe, line2point_distance_3D, cosine_distance
from construction import top_down_pose_kernel, hybrid_pose_kernel, SVD_pose_kernel, SVD_pose_kernel_parallel, SVD_pose_kernel_jf
from matching import back_project_ray, BIP_matching, Greedy_matching, epipolar_affinity, epipolar_affinity_parallel
from OneEuroFilter import OneEuroFilter
from KalmanFilter import KalmanFilter

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class IterativeTracker(object):
    BIPSolver = GLPKSolver()
    def __init__(self, args):
        self.args = args
        self.cam_num = 0
        self.conf_threshold = self.args.conf_threshold
        self.num_joints = self.args.num_joints
        self.epi_threshold=self.args.epi_threshold
        self.unmatched = dict()
        self.tracks = list()
        self.tracks_ids = set()
        self.build3D = None

    def track_restart(self):
        self.unmatched = dict()
        self.tracks = list()
        self.tracks_ids = set()

    def init_target_GD(self, time):
        if len(self.unmatched.keys()) < 2:
            return
        
        for cid, value in self.unmatched.items():
            _detections = list()
            for detection in value['detections']:
                if get_believe(detection) > self.conf_threshold:
                    _detections.append(detection)
            value['detections'] = np.array(_detections)
        
        H = []
        for idx, (key, value) in enumerate(self.unmatched.items()):
            if idx == 0:
                H = [Hypothesis(value['camera'], detection, self.epi_threshold) for detection in value['detections']]
            else:
                n_hyp = len(H)
                n_det = len(value['detections'])
                C = np.zeros((n_hyp, n_det))
                Mask = np.zeros_like(C).astype('int32')

                for hid, hyp in enumerate(H):
                    for pid, detection in enumerate(value['detections']):
                        pose_cost, veto = hyp.calculate_cost(value['camera'], detection)
                        C[hid, pid] = pose_cost
                        if veto:
                            Mask[hid, pid] = 1
                rows, cols = linear_sum_assignment(C)
                handled_pids = set()
                for hid, pid in zip(rows, cols):
                    is_masked = Mask[hid, pid] == 1
                    handled_pids.add(pid)
                    if is_masked:
                        H.append(
                            Hypothesis(
                            value['camera'],
                            value['detections'][pid],
                            self.epi_threshold)
                        )
                    else:
                        H[hid].merge(value['camera'], value['detections'][pid])
                
                for pid, detection in enumerate(value['detections']):
                    if pid not in handled_pids:
                        H.append(
                            Hypothesis(
                            value['camera'],
                            value['detections'][pid],
                            self.epi_threshold)
                        )
        for hid, hyp in enumerate(H):
            if hyp.size() > 1:
                # cameras, poses2d, pose3d, joints_views, succeed = hyp.get_3dpose(self.args.lambda_t)
                cameras, poses2d, pose3d, joints_views, succeed = hyp.get_3dpose_jf(self.args.init_threshold, self.args.lambda_t)
                if not succeed:
                    continue
                if len(self.tracks_ids) == 0:
                    track_id = 0
                else:
                    track_id = max(self.tracks_ids) + 1
                self.tracks.append(IterTrack(track_id, time, cameras, poses2d, pose3d, joints_views, self.args, self.build3D))
                self.tracks_ids.add(track_id)
       
    def init_target_BIP(self, time):
        if len(self.unmatched.keys()) < 2:
            return
        cameras = list()
        poses = list()

        fnames = list()
        pids   = list()
        cids   = list()
        dimGroup = [0]
        for cid, value in self.unmatched.items():
            cameras.append(value['camera'])
            count = 0
            for pid, det in enumerate(value['detections']):
                if get_believe(det) > 0.4:
                    count += 1
                    poses.append(det)
            dimGroup.append(dimGroup[-1] + count)
        if dimGroup[-1] == 0:
            return
    
        matched_list, sub_imgid2cam = BIP_matching(IterativeTracker.BIPSolver, cameras, dimGroup, pose_mat=poses,
                            num_joints=self.num_joints, threshold=self.epi_threshold)
        for match in matched_list:
            if len(match) < 2:
                continue
            cams = []
            poses2d = []
            for pid in match:
                cid = sub_imgid2cam[pid]
                cams.append(self.unmatched[cid]['camera'])
                poses2d.append(poses[pid])

            pose3d, _ = top_down_pose_kernel(cams, np.flip(np.array(poses2d)[:,:,:2], axis=2), np.array(poses2d)[:,:,2])
            if len(self.tracks_ids) == 0:
                    track_id = 0
            else:
                track_id = max(self.tracks_ids) + 1
            self.tracks.append(IterTrack(track_id, time, cams, poses2d, pose3d, self.args, self.build3D))
            self.tracks_ids.add(track_id)

    def tracking(self, frame_id, camera_list, frame_list, boxes_list, detections_list, build3D='TopDown'):
        """
        :param calib_per_frame: [ [cam1, ... ], ... ] * frames
        :param poses_per_frame: [ [pose1, ...], ... ] * frames
        :return:
        """
        self.frame_list = frame_list
        self.build3D=build3D
        self.cam_num = len(camera_list)
        tracks_pose = []
        tracks_time_interval = []
        for track in self.tracks:
            track.add_age()
            tracks_pose.append(track.poses3d[-1]['pose3d'])
            tracks_time_interval.append(frame_id - track.poses3d[-1]['time'])
        
        asso_time = 0
        for camera, boxes, detections in zip(camera_list, boxes_list, detections_list):
            n = len(self.tracks)
            m = len(detections)
            if n > 0 and m > 0:
                start = t.time()
                tracks_reprojection = camera.projectPoints_parallel(np.array(tracks_pose))
                detections = np.array(detections)
                Ts = np.repeat(tracks_reprojection, m, 0)
                Ds = np.tile(detections[:,:,:2], (n,1,1))

                costs_2d = np.linalg.norm(Ts-Ds, axis=2).reshape(n, m, -1)
                costs_2d = 1 - np.transpose(costs_2d.T / (self.args.alpha2d * np.array(tracks_time_interval)))

                remain = np.sum(costs_2d > 0, axis=2) > 10 # Shelf 10 Campus 14
                affinity_matrix = np.sum(costs_2d, where=costs_2d > 0, axis=2) / np.sum(costs_2d > 0, axis=2)
                affinity_matrix[~remain] = 0
                affinity_matrix = np.transpose(affinity_matrix.T / np.exp(self.args.lambda_a * np.array(tracks_time_interval)))
                affinity_matrix[np.isnan(affinity_matrix)] = 0
                rows, cols = linear_sum_assignment(-affinity_matrix)
                asso_time += t.time() - start

                handled_pids = set()
                match_ids = []
                for tid, pid in zip(rows, cols):
                    if affinity_matrix[tid, pid] > 0:                  
                        detection = detections[pid]
                        self.tracks[tid].add_pose(camera, frame_id, detection)
                        handled_pids.add(pid)
                        match_ids.append(tid)


                detections = np.delete(detections, list(handled_pids), axis=0)
                boxes = np.delete(boxes, list(handled_pids), axis=0)
                self.unmatched[camera.cid] = {'camera':camera, 'time':frame_id, 'bboxes':boxes, 'detections':detections}
            else:
                self.unmatched[camera.cid] = {'camera':camera, 'time':frame_id, 'bboxes':boxes, 'detections':detections}

        start = t.time()  
        for track in self.tracks:
            track.update(frame_id)
        update_time =  t.time() - start

        start = t.time()
        eval('self.init_target_' + self.args.init_method)(frame_id)
        init_time = t.time()- start

        self.tracks = [track for track in self.tracks if not track.is_deleted()]

        return asso_time, update_time, init_time

class IterTrack:
    def __init__(self, track_id, time, cameras, poses2d, pose3d, joints_views, args, build3D='SVD'):
        # for campus w2d=0.4, alpha2d=25, w3d=0.6, alpha3d=0.1, lambda_a=5, lambda_t=10
        # for shelf and panoptic w2d=0.4, alpha2d=60, w3d=0.6, alpha3d=0.15, lambda_a=5, lambda_t=10
        # for 5FLobby w2d=0.4, alpha2d=70, w3d=0.6, alpha3d=0.25, lambda_a=3, lambda_t=5
        """
        :param track_id: track's id
        :param hypothesis: 3d pose, included camera, 2d poses
        :param z_axis: some datasets are rotated around one axis
        :param max_age: maximum time hasn't been updated
        :param n_init: a confirmed threshold that need to be updated
        """
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.already_update = False

        self.joints = len(pose3d)
        self.poses2d = self.init_poses2d(time, cameras, poses2d)
        self.poses3d = [{'time':time, 'pose3d':np.array(pose3d), 'joints_views':joints_views}]
        self.next_pose3d = np.array(pose3d)
        self.velocity_3d = np.array([[0., 0., 0.] for i in range(self.joints)])

        self.state = TrackState.Tentative
        self._n_init = args.n_init
        self._max_age = args.max_age
        self.alpha2d = args.alpha2d
        self.alpha3d = args.alpha3d
        self.lambda_a = args.lambda_a
        self.lambda_t = args.lambda_t
        self.joint_threshold = args.joint_threshold
        self.sigma = args.sigma
        self.arm_sigma = args.arm_sigma
        self.build3D = build3D

         # initializing one euro filters for all the joints
        filter_config_2d = {
            'freq': 25,        # system frequency about 25 Hz
            'mincutoff': 1.7,  # value refer to the paper
            'beta': 0.3,       # value refer to the paper
            'dcutoff': 0.4     # not mentioned, empirically set
        }
        filter_config_3d = {
            'freq': 25,        # system frequency about 25 Hz
            'mincutoff': 0.8,  # value refer to the paper
            'beta': 0.4,       # value refer to the paper
            'dcutoff': 0.4     # not mentioned, empirically set
        }
        self.filter_2d = [(OneEuroFilter(**filter_config_2d),
                           OneEuroFilter(**filter_config_2d))
                          for _ in range(self.joints)]
        self.filter_3d = [(OneEuroFilter(**filter_config_3d),
                           OneEuroFilter(**filter_config_3d),
                           OneEuroFilter(**filter_config_3d))
                          for _ in range(self.joints)]
        # Kalman Filter            
        # self.kalman_3d = [KalmanFilter(pose3d[i]) for i in range(self.joints)]

    def init_poses2d(self, time, cameras, poses2d):
        init_poses2d = dict()
        for cam, pose in zip(cameras, poses2d):
            init_poses2d.setdefault(cam.cid, dict())
            init_poses2d[cam.cid] = {'time':time, 'camera':cam, 'pose':pose}
        return init_poses2d

    def add_age(self):
        self.already_update = False
        self.age += 1
        self.time_since_update += 1
    
    def update(self, time):
        if self.update_3dpose(time):
            
            self.update_motion(time)
            self.hits += 1
            self.time_since_update = 0
            if self.state == TrackState.Tentative and self.hits >= self._n_init:
                self.state = TrackState.Confirmed
            # for j in range(self.joints):
            #     self.next_pose3d[j] = self.kalman_3d[j].predict(self.poses3d[-1]['pose3d'][j])
        else:
            self.mark_missed()
            # for j in range(self.joints):
            #     self.next_pose3d[j] = self.kalman_3d[j].predict()

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative and not self.already_update:
            self.state = TrackState.Deleted
        elif self.time_since_update >= self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def add_pose(self, camera, time, pose):
        """ add pose
        :param pose:
        :return:
        """
        if not self.already_update:
            self.already_update = True

        self.poses2d.setdefault(camera.cid, dict())
        self.poses2d[camera.cid]= {'time':time, 'camera':camera, 'pose':pose}
        # self.poses2d.setdefault(camera.cid, list())
        # self.poses2d[camera.cid].append({'time':time, 'camera':camera, 'pose':pose})
        # if time - self.poses2d[camera.cid][0]['time'] > self._max_age:
        #     del self.poses2d[camera.cid][0]
    
    # using time weight
    def update_3dpose(self, time):
        assert self.build3D == 'SVD', "Please modify BUILD3D to SVD when PERSON_MATCHER == Iterative"
        if not self.already_update:
            return False
        
        Ts = []
        cameras = []
        pose_mat = []
        costs3d = []
        count = 0
        for poses in self.poses2d.values():
            time_interval = time - poses['time']
            if time_interval <= 3:
                Ts.append(time_interval)
                cameras.append(poses['camera'])
                last_pose = poses['pose']
                pose_mat.append(last_pose)
                # costs3d.append(self.cost_to_3d(poses['camera'], time, poses['pose']))
                count += 1
        if count < 2:
            return False
        
        pose3d, joints_views, succeed = self.get_3dpose(time, cameras, Ts, np.array(pose_mat)) #, np.array(costs3d))
        if succeed:
            pose3d = self.smooth_3dpose(time, np.array(pose3d), self.sigma, self.arm_sigma)
            self.poses3d.append({'time':time, 'pose3d':pose3d, 'joints_views':joints_views})
            if time - self.poses3d[0]['time'] > self._max_age:
                del self.poses3d[0]
            return True
        else:
            return False

    def get_3dpose(self, time, cameras, Ts, pose_mat, costs3d=None):
        last_pose3d_info = self.poses3d[-1]
        last_time = last_pose3d_info['time']
        last_pose3d = last_pose3d_info['pose3d']
        next_pose3d = last_pose3d + self.velocity_3d * (time - last_time)

        # start = t.time()
        # _, distances_mat = epipolar_affinity(cameras, np.arange(len(cameras)), pose_mat, num_joints=self.joints)
        _, distances_mat = epipolar_affinity_parallel(cameras, np.arange(len(cameras)), pose_mat, num_joints=self.joints)
        distances_mat = 1 - distances_mat / self.joint_threshold
        # print('distances matrix:', t.time() - start)

        remains = []
        fail_match = 0
        # start = t.time()
        joints_views = [[] for i in range(len(cameras))]
        binary_lists = np.ones((self.joints, len(cameras)*2) ,dtype=np.int)
        for j, pose in enumerate(np.transpose(pose_mat, (1,0,2))):
            # costs = costs3d[:,j]
            matched_list, binary_lists[j], _ = Greedy_matching(cameras, pose_mat=pose.reshape(-1,1,3),affinity_mat=distances_mat[:,:,j], next_pose=next_pose3d[j])
            remains.append(matched_list)
            joints_views[len(matched_list)-1].append(j)

            if len(matched_list) < 2:
                fail_match += 1
        # print('joint filter:', t.time() - start)

        # start = t.time()
        # pose3d = SVD_pose_kernel(cameras, Ts, joints, remains, self.lambda_t, next_pose3d)
        pose3d = SVD_pose_kernel_jf(cameras, Ts, pose_mat, self.lambda_t, binary_lists, joints_views, next_pose3d)
        # pose3d = SVD_pose_kernel_parallel(cameras, Ts, pose_mat, self.lambda_t)
        # print('SVD:', t.time() - start)
        return pose3d, joints_views, False if fail_match > self.joints/3 else True

    def smooth_3dpose(self, time, pose3d=None, sigma=0.3, arm_sigma=0.8):
        # One Euro Filter
        # for i in range(self.joints):
        #     pose3d[i][0] = self.filter_3d[i][0](pose3d[i][0], time)
        #     pose3d[i][1] = self.filter_3d[i][1](pose3d[i][1], time)
        #     pose3d[i][2] = self.filter_3d[i][2](pose3d[i][2], time)

        # gaussian filter
        poses3d = np.array([p3d['pose3d'] for p3d in self.poses3d] + [pose3d])
        not_arm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
        pose3d[not_arm] = gaussian_filter1d(poses3d[:,not_arm,:].T, sigma=sigma, mode='reflect')[:,:,-1].T
        pose3d[[9,10]] = gaussian_filter1d(poses3d[:,[9,10],:].T, sigma=arm_sigma, mode='reflect')[:,:,-1].T
        return pose3d

    def update_motion(self, time):
        if len(self.poses3d) < 2:
            return
        
        motions = []
        for idx in range(len(self.poses3d)-1, 0, -1):
            motions.append(self.poses3d[idx]['pose3d'].astype(np.float32) - self.poses3d[idx-1]['pose3d'].astype(np.float32))
            if len(motions) > 4:
                break
        motions = np.mean(motions, axis=0)
        self.velocity_3d = motions
    
    def cost_to_last(self, camera, time, pose):
        body_parts = [[0,1,2,3,4], [5,7,9], [6,8,10], [11,13,15], [12,14,16]]
        # last 3D pose info
        last_pose3d_info = self.poses3d[-1]
        last_time = last_pose3d_info['time']
        last_pose3d = last_pose3d_info['pose3d']
        time_interval_3d = time - last_time

        # last 3D pose project to 2D view
        last_pose2d = camera.projectPoints(last_pose3d)
        cost2d = la.norm(pose[:,:2] - last_pose2d[:,:2], axis=1)
        cost2d = cost2d[cost2d < self.alpha2d * time_interval_3d]
        total_cost = np.sum((1 - cost2d / (self.alpha2d * time_interval_3d)) / exp(self.lambda_a * time_interval_3d))

        return total_cost / len(cost2d) if len(cost2d) > 14 else 0 # Shelf 10 Campus 14 
    
    def cost_to_3d(self, camera, time, pose):
        # camera info
        RK_INV = camera.RK_INV
        position = camera.position

        # Geometry
        last_pose3d = self.poses3d[-1]['pose3d']
        time_interval = time - self.poses3d[-1]['time']

        pose = np.flip(pose[:,:2], axis=1)
        poses3d = last_pose3d + self.velocity_3d * time_interval
        directions = back_project_ray(RK_INV, position, pose)
        cost3d = line2point_distance_3D(position, directions, poses3d)
        return cost3d
