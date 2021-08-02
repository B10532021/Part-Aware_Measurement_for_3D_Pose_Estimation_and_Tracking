# -*- coding: utf-8 -*-
#"""
#.. class:: ivclabpose
#"""

#!/usr/bin/python

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import cv2
import time
import pickle
import argparse
import numpy as np
import torch
torch.multiprocessing.set_start_method('forkserver', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
from torch import nn
from torch.utils.data import Dataset
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from pdb import set_trace as bp
from easydict import EasyDict as edict

from backend.YOLOv3 import YOLOv3
from backend.HRPose.SimpleHRNet import HRNetPose
from tracking.IterativeTracker import IterativeTracker

class Camera(object):
	def __init__(self, cid, P, K, RT, F, w=640, h=480):
		self.cid = cid
		self.P = P
		self.K = K
		self.RT = RT
		self.RK_INV = np.linalg.inv(self.RT[:, :3]) @ np.linalg.inv(self.K)
		self.F = F
		self.w = w
		self.h = h
		RT_inv = np.linalg.inv(np.vstack([self.RT, [0,0,0,1]]))
		self.position = RT_inv[:3, 3]
	
	def undistort(self, im):
		""" undistorts the image
		:param im: {h x w x c}
		:return:
		"""
		return im
	
	def undistort_points(self, points2d):
		"""
		:param points2d: [ (x,y,w), ...]
		:return:
		"""
		return points2d
	
	def projectPoints_undist(self, points3d):
		"""
			projects 3d points into 2d ones with
			no distortion
		:param points3d: {n x 3}
		:return:
		"""

		points2d = np.zeros((len(points3d), 2))
		for i, (x,y,z) in enumerate(points3d):
			p3d = np.array([x, y, z, 1])
			a, b, c = self.P @ p3d
			# assert c != 0 , print(self.P, p3d,self.P @ p3d)
			c = 10e-6 if c == 0 else c
			points2d[i, 1] = a/c if a is not None else None
			points2d[i, 0] = b/c if b is not None else None
		return points2d

	def projectPoints(self, points3d):
		"""
			projects 3d points into 2d with
			distortion being considered
		:param points3d: {n x 3}
		:param withmask: {boolean} if True return mask that tells if a point is in the view or not
		:return:
		"""
		pts2d = self.projectPoints_undist(points3d)
		return pts2d

	def projectPoints_parallel(self ,points3d):
		points3d = np.concatenate([points3d, np.ones((points3d.shape[0], points3d.shape[1], 1))], axis=2)
		homo_reproj = np.transpose(self.P @ points3d.reshape(-1, 4).T)
		reproj = homo_reproj[:,:2] / homo_reproj[:,2].reshape(-1,1)
		reproj = np.flip(reproj, axis=1)
		reproj = reproj.reshape(-1, 17, 2)

		return reproj

class ivclabpose:
	def __init__(self, person_detector=None, pose_detector=None, person_matcher=None, conf_threshold=0.4):
		self.person_detector  = person_detector if person_detector.NAME != '' else None
		self.pose_detector    = pose_detector
		self.person_matcher   = person_matcher
		self.conf_threshold   = conf_threshold
		
		gpu_args = edict()
		gpu_args.gpus = []
		for i in range(torch.cuda.device_count()):
			gpu_args.gpus.append(i)
		gpu_args.device = torch.device('cuda:' + str(gpu_args.gpus[0]) if len(gpu_args.gpus) > 0 else 'cpu')

		# Detect Models
		if self.person_detector == None:
			print ("Person Detector : Close.")
		elif self.person_detector.NAME == 'YOLOv3':
			self.bbox_detector = YOLOv3(self.person_detector.CFG, self.person_detector.WEIGHT, self.person_detector.CLASS_NAMES, 
					score_thresh=self.person_detector.SCORE_THRESH, nms_thresh=self.person_detector.NMS_THRESH, 
					use_cuda=True if len(gpu_args.gpus) > 0 else False)
			print("Pose Detector : ", self.person_detector.NAME)

		# Pose Models
		if self.pose_detector == None:
			print ("Pose Detector : Close.")
		elif self.pose_detector.NAME == 'HRPose':
			c               = self.pose_detector.C
			num_joints      = self.pose_detector.NUM_JOINTS
			checkpoint_file = self.pose_detector.CHECKPOINT_FILE
			model_name      = self.pose_detector.MODEL_NAME
			resolution      = tuple(self.pose_detector.RESOLUTION)			
			self.pose_model = HRNetPose(c, num_joints, checkpoint_file,
				model_name=model_name, resolution=resolution,  hrpose_args=deepcopy(gpu_args))

			print ("Pose Detector : ", self.pose_detector.NAME)
		
		# Matching and Tracking Methods
		if self.person_matcher == None:
			print ("Person Matcher : Close.")
		elif self.person_matcher.NAME == 'Iterative':
			iter_args = edict()
			iter_args.conf_threshold      = self.conf_threshold
			iter_args.epi_threshold       = self.person_matcher.EPI_THRESHOLD
			iter_args.init_threshold      = self.person_matcher.INIT_THRESHOLD
			iter_args.joint_threshold     = self.person_matcher.JOINT_THRESHOLD
			iter_args.num_joints          = self.person_matcher.NUM_JOINTS
			iter_args.init_method         = self.person_matcher.INIT_METHOD
			iter_args.n_init              = self.person_matcher.N_INIT
			iter_args.max_age             = self.person_matcher.MAX_AGE
			iter_args.w2d                 = self.person_matcher.W2D
			iter_args.alpha2d             = self.person_matcher.ALPHA2D
			iter_args.w3d                 = self.person_matcher.W3D
			iter_args.alpha3d             = self.person_matcher.ALPHA3D
			iter_args.lambda_a            = self.person_matcher.LAMBDA_A
			iter_args.lambda_t            = self.person_matcher.LAMBDA_T
			iter_args.sigma               = self.person_matcher.SIGMA
			iter_args.arm_sigma           = self.person_matcher.ARM_SIGMA
			
			self.tracker = IterativeTracker(iter_args)

			print ("Person Matcher : ", self.person_matcher.NAME)

	def GetCameraParameters(self, camera_parameter, im_width, im_height):
		P  = camera_parameter['P'].astype(np.float32)
		K  = camera_parameter['K'].astype(np.float32)
		RT = camera_parameter['RT'].astype(np.float32)
		skew_op = lambda x: torch.tensor ( [[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]] )
		fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1: torch.inverse ( K_0 ).t () @ (R_0 @ R_1.t ()) @ K_1.t () @ skew_op ( K_1 @ R_1 @ R_0.t () @ (T_0 - R_0 @ R_1.t () @ T_1) )
		fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1: fundamental_op ( K_0, RT_0[:, :3], RT_0[:, 3], K_1, RT_1[:, :3], RT_1[:, 3] )
		camera_num = len(P)
		F = torch.zeros(camera_num,camera_num,3,3) # NxNx3x3 matrix
		# TODO: optimize this stupid nested for loop
		for x in range(camera_num):
			for y in range(camera_num):
				F[x,y]+=fundamental_RT_op(torch.tensor(K[x]), torch.tensor(RT[x]), torch.tensor(K[y]), torch.tensor(RT[y]))
				if F[x,y].sum() ==0:
					F[x,y] += 1e-12  # to avoid nan
		F = F.numpy()
		self.cameras = []
		for j in range(camera_num):
			self.cameras.append(Camera(j, P[j], K[j], RT[j], F[j], w=im_width, h=im_height))
		return self.cameras

	def PersonDetect(self, imglist, image_id):
		if self.person_detector.NAME == 'YOLOv3':
			person_bbox_list = list()
			results = self.bbox_detector(imglist)
			for idx, result in enumerate(results):
				person_temps = []
				h, w, _ = imglist[idx].shape
				for ret in result:
					x1 = max(0, ret[0])
					y1 = max(0, ret[1])
					x2 = min(ret[2], w)
					y2 = min(ret[3], h)
					person_temp = dict(
						image_id=image_id, 
						category_id=1,
						score=float(round(ret[4], 4)),
						bbox=[x1, y1, x2 - x1, y2 - y1],
						data=imglist[idx],
						feature=[])
					person_temps.append(person_temp)
				person_bbox_list.append(person_temps)
			return person_bbox_list
		else:
			return None		

	def PersonPoseDetect(self, imagelist=None, person_bbox_list=None, batch_size=20, image_id=None):
		if self.pose_detector.NAME == "HRPose":
			dump_results = self.pose_model.predict(person_bbox_list, batch_size, self.conf_threshold)

			return dump_results
		else:
			return None		

	def PersonTrack_Project3DPose(self, frame_id, person_bbox_list=None, dump_results=None, build3D='SVD'):
		frames = []
		poses = []
		features = []
		boxes = []
		for f_id, sub_person_bbox_list in enumerate(person_bbox_list):
			bxs = []
			pts   = []
			fts   = []
			if len(sub_person_bbox_list) == 0:
				person_ids  = np.array((), dtype=np.int32)
				frame = []
				bxs= np.array(bxs)
				pts  = np.array(pts)
				fts  = np.array(fts)
			else:
				frame = sub_person_bbox_list[0]['data']
				for item in dump_results[f_id]:
					bxs.append([item['bbox'][0],item['bbox'][1],item['bbox'][2],item['bbox'][3]])
					keypoints=item['keypoints']
					keypoints=np.array(keypoints)
					keypoints=keypoints.reshape(17, 3)
					keypoints_y = keypoints[:,0].copy() # 這樣後面x y是不是不用掉換
					keypoints_x = keypoints[:,1].copy()						
					keypoints_score=item['keypoints_score']
					keypoints_score=np.array(keypoints_score)
					keypoints[:,2]=keypoints_score
					keypoints[:,0]=keypoints_x
					keypoints[:,1]=keypoints_y
					pts.append(keypoints)
					fts.append(item['feature'])
				bxs= np.array(bxs)
				pts  = np.array(pts)
				fts  = np.array(fts)
			
			frames.append(frame)
			boxes.append(bxs)
			poses.append(pts)
			features.append(fts)
		

		asso_time, update_time, init_time = self.tracker.tracking(frame_id, self.cameras, frames, boxes, poses, build3D)

		camera_ids = []
		pts = []
		person_ids = []
		pts3d = []
		pts3d_joints_views = []
		person3d_ids = []
		for track in self.tracker.tracks:
			if track.time_since_update > 0 or not track.is_confirmed():
				continue

			poses2d, pose3d, joints_views = track.poses2d, track.poses3d[-1]['pose3d'], track.poses3d[-1]['joints_views']
			pts3d.append(np.transpose(pose3d))
			pts3d_joints_views.append(joints_views)
			person3d_ids.append(track.track_id)
			person_ids.append([track.track_id for i in range(len(poses2d))])
			cams = []
			poses = []
			for cid, pose_2d in poses2d.items():
				if pose_2d['time'] == frame_id:
					cams.append(cid)
					poses.append(pose_2d['pose'])
			camera_ids.append(cams)
			pts.append(poses)
			
		camera_ids = np.array(camera_ids, dtype='object')
		pts = np.array(pts, dtype='object')
		pts3d = np.array(pts3d)
		person3d_ids = np.array(person3d_ids)
		return camera_ids, pts, person_ids, pts3d, pts3d_joints_views, person3d_ids, asso_time, update_time, init_time
