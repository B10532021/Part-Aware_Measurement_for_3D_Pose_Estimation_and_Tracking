import _init_path 
import cv2
import os
import glob
import natsort
import sys
import argparse
import torch
import time
import yaml
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from tqdm import tqdm
from pdb import set_trace as bp
from easydict import EasyDict as edict
from dataset import Testdatast, GetConfig, LoadFilenames, LoadImages

def test_ivclabpose_PersonTrack_Project3DPose(cfg, inputs):
	output = cfg.OUTPUT
	os.makedirs(output, exist_ok=True)

	import pickle
	import copy

	dataset = cfg.DATASET
	storecameradir = output+os.sep+dataset.TEST_DATASET+os.sep+'Images'
	os.makedirs(storecameradir, exist_ok=True)
	with open(dataset.ROOT+os.sep+dataset.CALIBRATION_FILE, 'rb') as f:
		camera_parameter = pickle.load(f)

	from ivclabpose import ivclabpose
	# from visualization import plot3DPose
	from HRPose.misc.visualization import joints_dict, draw_points_and_skeleton

	pipeline       = cfg.PIPELINE_COMBINATION
	det_model      = cfg.DETECT_MODELS[pipeline['DETECT_MODEL'].upper()]
	pose_model     = cfg.POSE_MODELS[pipeline['POSE_MODEL'].upper()]
	person_matcher = cfg.PERSON_MATCHERS[pipeline['PERSON_MATCHER'].upper()]
	conf_threshold = pipeline['CONF_THRESHOLD']
	build3D        = pipeline['BUILD_3D']
	
	ivclabpose_model = ivclabpose(person_detector=det_model,
								pose_detector=pose_model, person_matcher=person_matcher, conf_threshold=conf_threshold)

	totaltimeperson = 0
	totaltimepose = 0
	totaltimetrack = 0
	test_start = dataset.TEST_RANGE[0]
	test_end = dataset.TEST_RANGE[1]
	for i, frame_id in enumerate(tqdm(range(test_start, test_end))):
		imagelist, _ = LoadImages(dataset.TEST_DATASET, inputs[frame_id])
		image3d = None
		if i == 0:
			cameras = ivclabpose_model.GetCameraParameters(camera_parameter, imagelist[0].shape[0], imagelist[0].shape[1])


		personstarttime = time.time()
		person_bbox_list = ivclabpose_model.PersonDetect(imagelist, 3, frame_id)
		personendtime = time.time()

		posestarttime = time.time()
		dump_result_list = ivclabpose_model.PersonPoseDetect(imagelist=None, person_bbox_list=person_bbox_list, batch_size=20)
		poseendtime = time.time()

		if np.array(dump_result_list, dtype='object').size > 0:
			trackstarttime = time.time()
			camera_ids, pts, person_ids, pts3d, pts3d_joints_views, person3d_ids, asso_time, update_time, init_time = ivclabpose_model.PersonTrack_Project3DPose(frame_id=frame_id,
															person_bbox_list=person_bbox_list,dump_results=dump_result_list, build3D=build3D)

			trackendtime = time.time()
			for cids, poses_2d, pids, in zip(camera_ids, pts, person_ids):
				for cid, pose_2d, pid in zip(cids, poses_2d, pids):
					imagelist[cid] = draw_points_and_skeleton(imagelist[cid], pose_2d, joints_dict()["coco"]['skeleton'], person_index=pid,
															points_color_palette='gist_rainbow', skeleton_color_palette='tab20',
															points_palette_samples=17, confidence_threshold=0.0)
			# you can design your own method to visualize 3D poses
			# below is our method, but we are't releasing our visualize code yet
			#if len(pts3d) != 0:
				# plot3DPose(dataset.TEST_DATASET, storecameradir, frame_id, cameras, imagelist[0].shape, person3d_ids, pts3d, pts3d_joints_views)
				# image3d = cv2.imread(storecameradir+os.sep+str(frame_id)+'.png')
		else:
			trackstarttime = 0
			trackendtime = 0

		if frame_id > test_start + 10:
			totaltimeperson = totaltimeperson+ (personendtime-personstarttime)
			totaltimepose   = totaltimepose+ (poseendtime-posestarttime)
			totaltimetrack  = totaltimetrack+ (trackendtime-trackstarttime)


	avgpersondetecttime = totaltimeperson/((test_end - test_start -10))
	avgposedetecttime = totaltimepose/((test_end - test_start -10))
	avgtracktime = totaltimetrack/((test_end - test_start -10))
	print ("Person Detect Processing time (s/f): %f" %(avgpersondetecttime))
	print ("Pose Detect Processing time (s/f): %f" %(avgposedetecttime))
	print ("Track Processing time (s/f): %f" %(avgtracktime))
	print ("fps: %f" %(1 / ((avgpersondetecttime + avgposedetecttime) / len(imagelist) + avgtracktime)))
	print ("tracking fps: %f"%(1 / avgtracktime))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', help='Three options:CampusSeq, Shelf, Panoptic', type=str, default='CampusSeq1')
	opt = parser.parse_args()
	cfg = GetConfig('./configs/{}/model_configs.yaml'.format(opt.dataset))
	datas = LoadFilenames(cfg.DATASET)
	eval('test_ivclabpose_' + cfg.TEST_FUNCTION)(cfg, datas)

	