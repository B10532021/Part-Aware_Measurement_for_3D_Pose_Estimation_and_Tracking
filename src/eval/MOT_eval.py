import os
import numpy as np
import copy
import motmetrics as mm
mm.lap.default_solver = 'lap'

from MOT_io import read_results, unzip_objs, Convert_Campus_or_Shelf_Format


class Evaluator(object):

    def __init__(self, data_root, seq_name, data_type):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type

        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        assert self.data_type == 'mot'

        gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
        self.gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)
        self.gt_ignore_frame_dict = read_results(gt_filename, self.data_type, is_ignore=True)
    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]

        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]
        # match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
        # match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
        # match_ious = iou_distance[match_is, match_js]

        # match_js = np.asarray(match_js, dtype=int)
        # match_js = match_js[np.logical_not(np.isnan(match_ious))]
        # keep[match_js] = False
        # trk_tlwhs = trk_tlwhs[keep]
        # trk_ids = trk_ids[keep]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)
        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_file(self, filename):
        self.reset_accumulator()

        result_frame_dict = read_results(filename, self.data_type, is_gt=False)
        #frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
        frames = sorted(list(set(result_frame_dict.keys())))
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        return self.acc

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()

class Campus_Shelf_Evaluator(object):

    def __init__(self, eval_range, data_type, seq_name, data_root):
        self.data_type = data_type
        self.seq_name = seq_name
        self.gt_filename = data_root

        self.load_annotations(eval_range)
        self.reset_accumulator()
    

    def load_annotations(self, eval_range):
        assert self.data_type == 'CampusSeq1' or self.data_type == 'Shelf' or self.data_type == 'Panoptic'

        self.gt_frame_dict = read_results(self.gt_filename, self.data_type)
        self.gt_compute_frame = dict()
        for key in self.gt_frame_dict.keys():
            self.gt_compute_frame[key] = [i * 0.04 for i in eval_range]

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, seq, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict[seq].get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)
        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_file(self, filename, seq):
        self.reset_accumulator()

        result_frame_dict = read_results(filename, self.data_type, is_gt=False)[seq]
        #frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
        frames = sorted(list(set(result_frame_dict.keys())))
        for i, frame_id in enumerate(frames):
            if i in self.gt_compute_frame[seq]:
                trk_objs = result_frame_dict.get(frame_id, [])
                trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
                self.eval_frame(frame_id, trk_tlwhs, trk_ids, seq, rtn_events=False)

        return self.acc

    def eval_files(self, fileroot, seqs):
        self.reset_accumulator()

        for seq in seqs:
            filename = os.path.join(fileroot, '{}.json'.format(seq))
            result_frame_dict = read_results(filename, self.data_type, is_gt=False)[seq]
            #frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))
            frames = sorted(list(set(result_frame_dict.keys())))
            for i, frame_id in enumerate(frames):
                if i in self.gt_compute_frame[seq]:
                    trk_objs = result_frame_dict.get(frame_id, [])
                    trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
                    self.eval_frame(frame_id, trk_tlwhs, trk_ids, seq, rtn_events=False)

        return self.acc
    
    def eval_results(self, results, seqs):
        self.reset_accumulator()
        # results_frame_dict = Convert_Campus_or_Shelf_Format(results)
        for seq in seqs:
            result_frame_dict = Convert_Campus_or_Shelf_Format(results[seq])[seq]
            frames = sorted(list(set(result_frame_dict.keys())))
            for i, frame_id in enumerate(frames):
                if i in self.gt_compute_frame[seq]:
                    trk_objs = result_frame_dict.get(frame_id, [])
                    trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
                    self.eval_frame(frame_id, trk_tlwhs, trk_ids, seq, rtn_events=False)
        return self.acc

    def eval_result(self, results, seq):
        self.reset_accumulator()
        result_frame_dict = Convert_Campus_or_Shelf_Format(results[seq])[seq]
        frames = sorted(list(set(result_frame_dict.keys())))
        for i, frame_id in enumerate(frames):
            if frame_id in self.gt_compute_frame[seq]:
                trk_objs = result_frame_dict.get(frame_id, [])
                trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
                self.eval_frame(frame_id, trk_tlwhs, trk_ids, seq, rtn_events=False)
        return self.acc

    @staticmethod
    def get_summary(accs, names, metrics=('mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()