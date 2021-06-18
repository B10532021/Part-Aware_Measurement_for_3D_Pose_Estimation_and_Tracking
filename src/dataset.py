import os
import cv2
import copy
import glob
import yaml
import natsort
import numpy
import torch
from easydict import EasyDict as edict
from copy import deepcopy
from torch.utils.data import Dataset

def GetConfig(config_file):
	exp_config = None
	with open(config_file) as f:
		exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
	return exp_config

def LoadFilenames(dataset):
	print('------start load files------')
	root = dataset.ROOT
	folders = dataset.FOLDERS_ORDER
	ext = dataset.DATA_FORMAT
	files = []
	for folder in folders:
		data_path = os.path.join(root, folder, ext)
		f = glob.glob(data_path)
		files.append(natsort.natsorted(f, reverse=False))

	datas = []
	for i in range(len(files[0])):
		datas.append([f[i] for f in files])
	print('------finish load files------')
	return datas

def LoadImages(dataset, files):
	if dataset == 'Panoptic':
		timestamp = int(files[0].split('/')[-1].split('_')[-1].split('.')[0])
	else:
		timestamp = files[0].split('/')[-1].split('.')[0]

	datas = []
	for f in files:
		datas.append(cv2.imread(f))
	return datas, timestamp

class Testdatast(Dataset):
	def __init__(self, dataset, is_train=False):
		self.is_train = is_train
		self.dataset = dataset.TEST_DATASET
		self.root = dataset.ROOT
		self.folders = dataset.FOLDERS_ORDER
		self.ext = dataset.DATA_FORMAT
		self.files = self._get_filenames()
		test_start = dataset.TEST_RANGE[0]
		test_end = dataset.TEST_RANGE[1]
		self.test_range = [i for i in range(test_start, test_end)]
		
	def _get_filenames(self):
		files = []
		for folder in self.folders:
			data_path = os.path.join(self.root, folder, self.ext)
			f = glob.glob(data_path)
			files.append(natsort.natsorted(f, reverse=False))
		return files

	def __len__(self):
		return len(self.test_range)

	def __getitem__(self, idx):
		datas = []
		datas.append([cv2.imread(f[idx]) for f in self.files])
		if self.dataset == 'Panoptic':
			timestamp = int(self.files[0][idx].split('/')[-1].split('_')[-1].split('.')[0])
		else:
			timestamp = self.files[0][idx].split('/')[-1].split('.')[0]
		
		return datas, timestamp