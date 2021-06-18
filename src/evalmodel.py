"""
Evaluate model for Campus Dataset(CampusSeq1)
"""
import _init_path
import cv2, os, glob, sys
import natsort
import numpy as np
import pandas as pd
import os.path as osp
import torch
import time
import yaml
import json
import pickle
import copy
import argparse
import scipy.io as scio
import matplotlib.pyplot as plt
import motmetrics as mm
from tqdm import tqdm
from collections import OrderedDict
from pdb import set_trace as bp
from copy import deepcopy
from prettytable import PrettyTable
from PIL import Image 
from torch.utils.data import DataLoader

from transformation import coco2shelf3D
from numeric import vectorize_distance
from dataset import Testdatast, GetConfig, LoadFilenames, LoadImages

def eval_ivclabpose_PersonTrack_Project3DPose(cfg, inputs):
	output = cfg.OUTPUT
	os.makedirs(output, exist_ok=True)

	dataset = cfg.DATASET
	store_dir = output+os.sep+dataset.TEST_DATASET
	os.makedirs(store_dir+os.sep+'Images', exist_ok=True)
	with open(dataset.ROOT+os.sep+dataset.CALIBRATION_FILE, 'rb') as f:
		camera_parameter = pickle.load(f)

	from ivclabpose import ivclabpose
	from visualization import plot3DPose
	from HRPose.misc.visualization import joints_dict, draw_points_and_skeleton
	
	pipeline       = cfg.PIPELINE_COMBINATION
	det_model      = cfg.DETECT_MODELS[pipeline['DETECT_MODEL'].upper()]
	pose_model     = cfg.POSE_MODELS[pipeline['POSE_MODEL'].upper()]
	person_matcher = cfg.PERSON_MATCHERS[pipeline['PERSON_MATCHER'].upper()]
	conf_threshold = pipeline['CONF_THRESHOLD']
	build3D        = pipeline['BUILD_3D']
	ivclabpose_model = ivclabpose(person_detector=det_model,
								pose_detector=pose_model, person_matcher=person_matcher, conf_threshold=conf_threshold)
	

	totaltimeperson=0
	totaltimepose=0
	totaltimeasso = 0
	totaltimeupdate = 0
	totaltimeinit = 0
	totaltimetrack=0
	multi_poses3d = dict()
	annotations = []
	test_start = dataset.TEST_RANGE[0]
	test_end = dataset.TEST_RANGE[1]
	for i, frame_id in enumerate(tqdm(range(test_start, test_end))):
		imagelist, timestamp = LoadImages(dataset.TEST_DATASET, inputs[frame_id])
		if i == 0:
			cameras = ivclabpose_model.GetCameraParameters(camera_parameter, imagelist[0].shape[0], imagelist[0].shape[1])

		personstarttime = time.time()
		person_bbox_list = ivclabpose_model.PersonDetect(imagelist, 3, frame_id)
		personendtime = time.time()

		posestarttime = time.time()
		dump_result_list = ivclabpose_model.PersonPoseDetect(imagelist=None,person_bbox_list=person_bbox_list, batch_size=20)
		poseendtime = time.time()
		
		if np.array(dump_result_list, dtype='object').size > 0:
			trackstarttime = time.time()
			camera_ids, pts, person_ids, pts3d, pts3d_joints_views, person3d_ids, asso_time, update_time, init_time = ivclabpose_model.PersonTrack_Project3DPose(frame_id=frame_id,
															person_bbox_list=person_bbox_list,dump_results=dump_result_list, build3D=build3D)
			trackendtime = time.time()
			multi_poses3d[timestamp if dataset.TEST_DATASET == 'Panoptic' else frame_id] = pts3d
			for cids, poses_2d, pids, in zip(camera_ids, pts, person_ids):
				for cid, pose_2d, pid in zip(cids, poses_2d, pids):
					annotations.append({'timestamp':timestamp, 'cid':cid, 'pid':pid, 'pose':pose_2d[:, 0:2], 'scores':pose_2d[:, 2]})
		else:
			trackstarttime = 0
			trackendtime = 0
			asso_time, update_time, init_time = 0, 0, 0
			multi_poses3d[timestamp if dataset.TEST_DATASET == 'Panoptic' else frame_id] = []

		totaltimeperson = totaltimeperson+ (personendtime-personstarttime)
		totaltimepose   = totaltimepose+ (poseendtime-posestarttime)
		totaltimeasso   = totaltimeasso + asso_time
		totaltimeupdate = totaltimeupdate + update_time
		totaltimeinit   = totaltimeinit + init_time
		totaltimetrack  = totaltimetrack+ (trackendtime-trackstarttime)

	
	filepath = store_dir+os.sep+'logs'+os.sep+'{}_{}_{}_{}.pkl'.format(
			pipeline['DETECT_MODEL'], pipeline['POSE_MODEL'], pipeline['PERSON_MATCHER'], dataset.ROOT.split('/')[-1])
	Write3DResult(multi_poses3d, filepath)
	if dataset.TEST_DATASET == 'Panoptic':
		EvaluatePanoptic(dataset.EVAL_RANGE, filepath, dataset.TEST_DATASET, seqs=dataset.FOLDERS_ORDER, data_root=dataset.ROOT)
	else:
		Evaluate3DPose_PCP(dataset.EVAL_RANGE, filepath, gt_path=dataset.ROOT, dataset_name=dataset.TEST_DATASET)
	avgpersondetecttime = totaltimeperson/(test_end - test_start)
	avgposedetecttime = totaltimepose/(test_end - test_start)
	avgassotime = totaltimeasso / (test_end - test_start)
	avgupdatetime = totaltimeupdate / (test_end - test_start)
	avginittime = totaltimeinit / (test_end - test_start)
	avgtracktime = totaltimetrack /(test_end - test_start)
	print ("Person Detect Processing time (s/f): %f" %(avgpersondetecttime))
	print ("Pose Detect Processing time (s/f): %f" %(avgposedetecttime))
	print ("Association Processing time (s/f): %f" %(avgassotime))
	print ("Update 3D Processing time (s/f): %f" %(avgupdatetime))
	print ("Initiate 3D Processing time (s/f): %f" %(avginittime))
	print ("Track Processing time (s/f): %f" %(avgtracktime))
	print ("fps: %f" %(1 / ((avgpersondetecttime + avgposedetecttime) / 3 + avgtracktime)))

def Evaluate3DPose_PCP(eval_ranges, pred_path, gt_path='CatchImage/CampusSeq1', dataset_name='CampusSeq1'):
	'''
	coco17 
	kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',  # 5
                'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',  # 10
                'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
	'''
	def is_right(model_start_point, model_end_point, gt_start_point, gt_end_point, alpha=0.5):
		bone_lenth = np.linalg.norm ( gt_end_point - gt_start_point )
		start_difference = np.linalg.norm ( gt_start_point - model_start_point )
		end_difference = np.linalg.norm ( gt_end_point - model_end_point )
		return ((start_difference + end_difference) / 2) <= alpha * bone_lenth

	with open(pred_path, 'rb') as f:
		multi_poses3d = pickle.load(f)

	actorsGT = scio.loadmat (osp.join (gt_path, 'actorsGT.mat'))
	actorsGT = actorsGT['actor3D'][0]
	if dataset_name == 'Panoptic':
		actorsGT /= 100 # mm -> m

	check_result = np.zeros ( (len(actorsGT[0]), len(actorsGT), 10), dtype=np.int32 )
	accuracy_cnt = 0
	error_cnt = 0
	for eval_range in eval_ranges:
		eval_start = eval_range[0]
		eval_end = eval_range[1]
		for frame_id in range(eval_start, eval_end):
			poses3d = multi_poses3d[frame_id].astype(np.float)
			for pid in range(len(actorsGT)):
				if actorsGT[pid][frame_id][0].shape == (1, 0) or actorsGT[pid][frame_id][0].shape == (0, 0):
					continue
				if len(poses3d) == 0:
					check_result[frame_id, pid,:] = -1
					print('Cannot get any pose in frame:{}'.format(frame_id))
					continue
				model_poses = np.stack([coco2shelf3D(i) for i in deepcopy(poses3d)])
				gt_pose = actorsGT[pid][frame_id][0]
				dist = vectorize_distance ( np.expand_dims ( gt_pose, 0 ), model_poses )
				model_pose = model_poses[np.argmin ( dist[0] )]
				bones = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13]]
				for i, bone in enumerate ( bones ):
					start_point, end_point = bone
					if is_right ( model_pose[start_point], model_pose[end_point], gt_pose[start_point],
								gt_pose[end_point] ):
						check_result[frame_id, pid, i] = 1
						accuracy_cnt += 1
					else:
						check_result[frame_id, pid, i] = -1
						error_cnt += 1
				gt_hip = (gt_pose[2] + gt_pose[3]) / 2
				model_hip = (model_pose[2] + model_pose[3]) / 2
				if is_right (model_hip, model_pose[12], gt_hip, gt_pose[12] ):
					check_result[frame_id, pid, -1] = 1
					accuracy_cnt += 1
				else:
					check_result[frame_id, pid, -1] = -1
					error_cnt += 1

	bone_group = OrderedDict (
		[('Head', np.array ( [8] )), ('Torso', np.array ( [9] )), ('Upper arms', np.array ( [5, 6] )),
		('Lower arms', np.array ( [4, 7] )), ('Upper legs', np.array ( [1, 2] )),
		('Lower legs', np.array ( [0, 3] ))] )

	total_avg = np.sum ( check_result > 0 ) / np.sum ( np.abs ( check_result ) )
	person_wise_avg = np.sum ( check_result > 0, axis=(0, 2) ) / np.sum ( np.abs ( check_result ), axis=(0, 2) )

	bone_wise_result = OrderedDict()
	bone_person_wise_result = OrderedDict()
	for k, v in bone_group.items():
		bone_wise_result[k] = np.sum ( check_result[:, :, v] > 0 ) / np.sum ( np.abs ( check_result[:, :, v] ) )
		bone_person_wise_result[k] = np.sum ( check_result[:, :, v] > 0, axis=(0, 2) ) / np.sum (
				np.abs ( check_result[:, :, v] ), axis=(0, 2) )
	# np.save('ours.npy', np.sum(check_result[500:666,1,:] < 0, axis=1))
	tb = PrettyTable ()
	tb.field_names = ['Bone Group'] + ['Actor {}'.format(i) for i in range(3)] + ['Average']
	list_tb = [tb.field_names]
	for k, v in bone_person_wise_result.items():
		this_row = [k] + [np.char.mod ( '%.2f', i * 100 ) for i in v[:3]] + [np.char.mod ( '%.2f', np.sum ( v[:3] ) * 100 / len ( v[:3] ) )]
		list_tb.append ( [float ( i ) if isinstance ( i, type ( np.array ( [] ) ) ) else i for i in this_row] )
		tb.add_row ( this_row )

	this_row = ['Total'] + [np.char.mod ( '%.2f', i * 100 ) for i in person_wise_avg[:3]] + [np.char.mod ( '%.2f', np.sum ( person_wise_avg[:3] ) * 100 / len ( person_wise_avg[:3] ) )]
	tb.add_row ( this_row )
	list_tb.append ( [float ( i ) if isinstance ( i, type ( np.array ( [] ) ) ) else i for i in this_row] )
	print ( tb )
	return check_result, list_tb

def EvaluatePanoptic(eval_ranges, pred_path, dataset='Panoptic', seqs=['Camera0', 'Camera1', 'Camera2'], data_root='CatchImage/Panoptic/160906_pizza1'):
	JOINTS_DEF = {'neck': 0, 'nose': 1,'mid-hip': 2, 'l-shoulder': 3,'l-elbow': 4,'l-wrist': 5,'l-hip': 6,
	'l-knee': 7,'l-ankle': 8,'r-shoulder': 9,'r-elbow': 10,'r-wrist': 11,'r-hip': 12,'r-knee': 13,'r-ankle': 14,
	'l-eye': 15,'l-ear': 16,'r-eye': 17,'r-ear': 18}
	def getGT(interval=12):
		curr_anno = osp.join(data_root, 'hdPose3d_stage1_coco19')
		anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))
		human_gt = dict()
		for i, filename in enumerate(anno_files):
			timestamp = filename.split('/')[-1][:-5].replace('body3DScene_', '')
			if i % interval == 0:
				with open(filename) as dfile:
					bodies = json.load(dfile)['bodies']
				if len(bodies) == 0:
					continue
				
				all_poses_3d = []
				all_poses_vis_3d = []
				for body in bodies:
					pose3d = np.array(body['joints19']).reshape((-1, 4))
					pose3d = pose3d[1:15]

					joints_vis = pose3d[:, -1] > 0.1

					if not joints_vis[2]:
						continue

					# Coordinate transformation
					M = np.array([[1.0, 0.0, 0.0],
									[0.0, 0.0, -1.0],
									[0.0, 1.0, 0.0]])
					pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

					all_poses_3d.append(pose3d[:, 0:3] * 10.0)
					all_poses_vis_3d.append(
						np.repeat(
							np.reshape(joints_vis, (-1, 1)), 3, axis=1))
				
				human_gt[int(timestamp)] = {'joints_3d': all_poses_3d,
							'joints_3d_vis': all_poses_vis_3d}
		return human_gt

	def eval_list_to_ap(eval_list, total_gt, threshold):
		total_num = len(eval_list)

		tp = np.zeros(total_num)
		fp = np.zeros(total_num)
		gt_det = []
		for i, item in enumerate(eval_list):
			if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
				tp[i] = 1
				gt_det.append(item["gt_id"])
			else:
				fp[i] = 1
		tp = np.cumsum(tp)
		fp = np.cumsum(fp)
		recall = tp / (total_gt + 1e-5)
		precise = tp / (tp + fp + 1e-5)
		for n in range(total_num - 2, -1, -1):
			precise[n] = max(precise[n], precise[n + 1])

		precise = np.concatenate(([0], precise, [0]))
		recall = np.concatenate(([0], recall, [1]))
		index = np.where(recall[1:] != recall[:-1])[0]
		ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

		return ap, recall[-2]

	def eval_list_to_mpjpe(eval_list, threshold=500):
		gt_det = []
		mpjpes = []
		for i, item in enumerate(eval_list):
			if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
				mpjpes.append(item["mpjpe"])
				gt_det.append(item["gt_id"])

		return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

	def eval_list_to_recall(eval_list, total_gt, threshold=500):
		gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]

		return len(np.unique(gt_ids)) / total_gt

	def evaluate(gts, preds):
		eval_list = []
		total_gt = 0
		ranges = []
		for eval_range in eval_ranges:
			eval_start = eval_range[0]
			eval_end = eval_range[1]
			ranges += [i for i in range(eval_start, eval_end)]

		# for timestamp, pred in preds.items():
		for timestamp, gt in gts.items():
			# gt = gts[timestamp]
			joints_3d = gt['joints_3d']
			joints_3d_vis = gt['joints_3d_vis']

			if len(joints_3d) == 0:
				continue
			
			pred = preds[timestamp].copy()
			for pose in pred:
				pose = pose.T * 1000.
				pelvis = (pose[11] + pose[12]) / 2
				pose = pose[[0, 5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16]]
				pose = np.insert(pose, 1*3, pelvis).reshape(-1, 3)
				mpjpes = []
				for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
					vis = gt_vis[:, 0] > 0
					mpjpe = np.mean(np.sqrt(np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
					mpjpes.append(mpjpe)
				min_gt = np.argmin(mpjpes)
				min_mpjpe = np.min(mpjpes)
				eval_list.append({
					"mpjpe": float(min_mpjpe),
					"gt_id": int(total_gt + min_gt)
				})

			total_gt += len(joints_3d)

		mpjpe_threshold = np.arange(25, 155, 25)
		aps = []
		recs = []
		for t in mpjpe_threshold:
			ap, rec = eval_list_to_ap(eval_list, total_gt, t)
			aps.append(ap)
			recs.append(rec)

		return aps, recs, eval_list_to_mpjpe(eval_list), eval_list_to_recall(eval_list, total_gt)

	with open(pred_path, 'rb') as f:
		multi_poses3d = pickle.load(f)

	human_gts = getGT()
	tb = PrettyTable()
	mpjpe_threshold = np.arange(25, 155, 25)
	aps, recs, mpjpe, _ = evaluate(human_gts, multi_poses3d)
	tb.field_names = ['Threshold/mm'] + [f'{i}' for i in mpjpe_threshold]
	tb.add_row(['AP'] + [f'{ap * 100:.2f}' for ap in aps])
	tb.add_row(['Recall'] + [f'{re * 100:.2f}' for re in recs])
	print(tb)
	print(f'MPJPE: {mpjpe:.2f}mm')

def Write2DResult(image_wh, annotations, save_dir='TrackResult'):
		import json
		os.makedirs(save_dir, exist_ok=True)
		Cameras = dict()
		for annotation in annotations:
			# {'timestamp':timestamp, 'cid':cid, 'pid':pid, 'pose':pose_2d[:, 0:2], 'scores':pose_2d[:, 2]}
			camera = "Camera"+str(annotation['cid'])
			timestamp = annotation['timestamp']
			frame_name = camera+os.sep+timestamp+'.jpg'
			Cameras.setdefault(camera, {'image_wh':[image_wh[1], image_wh[0]], 'frames':dict()})
			Cameras[camera]['frames'].setdefault(frame_name, {'camera':camera, 'timestamp':float(timestamp),'poses':list()})

			target_id = annotation['pid']
			pose = np.flip(annotation['pose'], axis=1).tolist()
			scores = annotation['scores'].tolist()
			Cameras[camera]['frames'][frame_name]['poses'].append({'id':target_id, 'points_2d':pose, 'scores':scores})
		
		for key, value in Cameras.items():
			with open(save_dir+os.sep+key+'.json', 'w') as fp:
				json.dump(value, fp)

def Write3DResult(multi_poses3d, filepath):
	os.makedirs('/'.join(filepath.split('/')[:-1]), exist_ok=True)
	annotation_3d = open(filepath, "wb")
	pickle.dump(multi_poses3d, annotation_3d)
	annotation_3d.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', help='Three options:CampusSeq, Shelf, Panoptic', type=str, default='CampusSeq1')
	opt = parser.parse_args()
	cfg = GetConfig('./configs/{}/model_configs.yaml'.format(opt.dataset))
	datas = LoadFilenames(cfg.DATASET)
	eval('eval_ivclabpose_' + cfg.TEST_FUNCTION)(cfg, datas)
	# Evaluate3DPose_PCP([[300,600]], '../results/PersonPoseDetectResult/Shelf/logs/YOLOv3_HRPose_Iterative_Shelf_nofilter.pkl', gt_path='../CatchImage/Shelf', dataset_name='Shelf')
	
