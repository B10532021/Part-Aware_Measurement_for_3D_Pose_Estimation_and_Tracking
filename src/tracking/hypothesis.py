import cv2
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from matching import epipolar_distance, epipolar_affinity, Greedy_matching
from construction import SVD_pose_kernel, SVD_pose_kernel_parallel, SVD_pose_kernel_jf
from calculate import get_believe

class Hypothesis:

    def __init__(self, cam, pts, epi_threshold=40):
        """
        """
        self.joints = len(pts)
        self.pose3d = None
        self.poses = [pts]
        self.cams = [cam]
        self.threshold = epi_threshold

    def size(self):
        return len(self.poses)

    def get_3dpose_jf(self, init_threshold, lambda_t):
        Ts = []
        joints = [[] for i in range(self.joints)]
        for pts in self.poses:
            Ts.append(0)
            for j in range(self.joints):
                joints[j].append(pts[j])
        
        _, distances_mat = epipolar_affinity(self.cams, np.arange(len(self.cams)), self.poses, num_joints=self.joints)
        distances_mat = 1 - distances_mat / init_threshold

        joints_views = [[] for i in range(len(self.cams))]
        binary_lists = np.ones((self.joints, len(self.cams)*2) ,dtype=np.int)
        for j, (joint) in enumerate(joints):
            matched_list, binary_lists[j], _ = Greedy_matching(self.cams, affinity_mat=distances_mat[:,:,j], mode='init')
            joints_views[len(matched_list)-1].append(j)

            if len(matched_list) < 2:
                return [],[],[],[], False
        
        human3d = SVD_pose_kernel_jf(self.cams, Ts, self.poses, lambda_t, binary_lists, joints_views)
        return self.cams, self.poses, human3d, joints_views, True

    def get_3dpose(self, lambda_t):
        assert self.size() > 1
        joints_views = [[] for i in range(len(self.cams))]
        joints_views[len(self.cams) - 1] = list(range(self.joints))
        SVD_pose_kernel_parallel(self.cams, Ts, self.poses, lambda_t)
        return self.cams, self.poses, human3d, joints_views, True

    def calculate_cost(self, o_cam, o_pose):
        """
        :param o_pose: other poses * J
        :param o_cam: other camera
        :return:
        """
        veto = False  # if true we cannot join {other} with this
        pose_cost = 0
        for person, cam in zip(self.poses, self.cams):
            distances = epipolar_distance(cam, person, o_cam, o_pose)
            p_cost = np.mean([(dis[0] * j0[2] + dis[1] * j1[2]) / 2 for dis, j0, j1, in zip(distances, person, o_pose)]) / self.threshold
            pose_cost += p_cost

            if p_cost > 1 and get_believe(o_pose) > 0.5:
                veto = True
        return pose_cost / len(self.poses), veto

    def merge(self, o_cam, o_pose):
        """ integrate {other} into our hypothesis
        :param o_pose:
        :param o_cam:
        :return:
        """
        self.cams.append(o_cam)
        self.poses.append(o_pose)

class Person2d:
    def __init__(self, cid, cam, points2d, noundistort=False):
        """
        :param cid
        :param cam: {Camera}
        :param points2d: distorted points
        :param noundistort: if True do not undistort
        """
        self.cid = cid
        self.cam = cam
        self.believe = get_believe(points2d)

        if noundistort:
            self.points2d = points2d
        else:
            # ~~~ undistort ~~~
            valid_points2d = []
            jids = []
            for jid, pt2d in enumerate(points2d):
                if pt2d[0] < 0:
                    continue
                jids.append(jid)
                valid_points2d.append(pt2d)
            valid_points2d = np.array(valid_points2d, np.float32)
            points2d_undist = cam.undistort_points(valid_points2d)
            self.points2d = points2d.copy()
            for idx, jid in enumerate(jids):
                self.points2d[jid] = points2d_undist[idx]
            # ~~~~~~~~~~~~~~~~~~~~

    def __len__(self):
        return len(self.points2d)

    def triangulate(self, other):
        """
        :param other: {Person2d}
        :return:
        """
        Pts1 = []
        Pts2 = []
        jids = []
        W1 = []
        W2 = []

        J = len(other)
        assert J == len(self.points2d)
        assert J == len(self)

        for jid in range(J):
            if self.points2d[jid, 2] > 0 and other.points2d[jid, 2] > 0:
                Pts1.append([self.points2d[jid, 1], self.points2d[jid, 0]])
                Pts2.append([other.points2d[jid, 1], other.points2d[jid, 0]])
                jids.append(jid)
                W1.append(self.points2d[jid, 2])
                W2.append(other.points2d[jid, 2])

        Pts1 = np.transpose(Pts1)
        Pts2 = np.transpose(Pts2)

        Points3d = [None] * J
        w = [-1] * J
        if len(Pts1) > 0:
            pose3d_homo = cv2.triangulatePoints(self.cam.P, other.cam.P, Pts1, Pts2)
            Pts3d = np.transpose(pose3d_homo[:3]/(pose3d_homo[3]+10e-6))
            # 下面是用來調整pose axis
            R = np.array ( [[1, 0, 0], [0, 1, 0], [0, 0, 1]] )
            Pts3d = [R @ i for i in Pts3d]

            for jid, pt3d, w1, w2 in zip(jids, Pts3d, W1, W2):
                Points3d[jid] = pt3d
                w[jid] = min(w1, w2)

        return Points3d, np.array(w)

def get_single_human3d(humans3d):
    J = len(humans3d[0][0])
    total_cost = 0
    count = 0
    for person3d in humans3d:
        total_cost += person3d[2]
        count += 1
    
    human3d = [None] * J  # single 3d person
    weight3d = [None] * J
    for jid in range(J):
        pts3d = []
        w3d = []
        for person3d in humans3d:
            if person3d[0][jid] is not None:
                if count == 1:
                    pts3d.append(person3d[0][jid])
                else:
                    pts3d.append(person3d[0][jid] * float((total_cost - person3d[2])) / float((total_cost * (count - 1))))
                w3d.append(person3d[1][jid])

        if len(pts3d) > 0:
            pt3d = np.sum(pts3d, axis=0)
            human3d[jid] = pt3d
            w3d = np.mean(w3d)
            weight3d[jid] = w3d

    return human3d, weight3d


