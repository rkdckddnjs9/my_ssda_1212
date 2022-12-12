#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch.utils.data as torch_data

from pcdet.datasets import *

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'KittiDatasetSSL': KittiDatasetSSL,
    'NuScenesDataset': NuScenesDataset,
    'NuScenesDatasetSSL': NuScenesDatasetSSL,
    'WaymoDataset': WaymoDataset,
    'OmegaDataset': OmegaDataset,
    'OmegaDatasetSSL': OmegaDatasetSSL,
    'OmegaDataset_CenterPoint_SSL' : OmegaDataset_CenterPoint_SSL
}

class DADatasetSSDA(torch_data.Dataset):
    """
    """
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.logger = logger
        self.repeat = self.dataset_cfg.REPEAT
        self.class_names = class_names

        self.src_dataset = __all__[dataset_cfg.SRC_DATASET.DATASET](
            dataset_cfg=dataset_cfg.SRC_DATASET,
            class_names=class_names,
            root_path=Path(self.dataset_cfg.SRC_DATASET.DATA_PATH),
            training=training,
            logger=logger,
        )
        self.trg_dataset = __all__[dataset_cfg.TRG_DATASET.DATASET](
            dataset_cfg=dataset_cfg.TRG_DATASET,
            class_names=class_names,
            root_path=Path(self.dataset_cfg.TRG_DATASET.DATA_PATH),
            training=training,
            logger=logger,
        )
        self.point_feature_encoder = self.trg_dataset.point_feature_encoder
        self.point_cloud_range = self.trg_dataset.point_cloud_range
        self.grid_size = self.trg_dataset.grid_size
        self.voxel_size = self.trg_dataset.voxel_size

        self.src_index = [False for _ in range(len(self.src_dataset))]

    def __len__(self):
        if self.training:
            return len(self.trg_dataset.labeled_infos) * self.repeat
        else:
            return len(self.trg_dataset.labeled_infos)

    def add_ema_key(self, data_dict):
        # for dummy key
        if self.training:
            data_dict['points_ema'] = data_dict['points'].copy()
            data_dict['gt_boxes_ema'] = data_dict['gt_boxes'].copy()
            try:
                data_dict['voxels_ema'] = data_dict['voxels'].copy()
                data_dict['voxel_coords_ema'] = data_dict['voxel_coords'].copy()
                data_dict['voxel_num_points_ema'] = data_dict[
                    'voxel_num_points'].copy()
            except:
                pass
        return data_dict

    def get_src_index(self):
        # assumme len(src_dataset) > len(trg_dataset), get random idx
        src_len = len(self.src_dataset)
        if sum(self.src_index) == src_len:
            self.src_index = [False for _ in range(src_len)]
        idx = np.random.randint(src_len)
        while self.src_index[idx] == True:
            idx = np.random.randint(src_len)
        self.src_index[idx] = True
        return idx

    def __getitem__(self, index):
        trg_item = self.trg_dataset.__getitem__(index)
        if self.training:
            # SL, TL, TU
            src_idx = self.get_src_index()
            src_item = self.src_dataset.__getitem__(src_idx)
            src_item = self.add_ema_key(src_item)
            return [src_item, trg_item[0], trg_item[1]]
        else:
            # TL
            return trg_item

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        if isinstance(batch_list[0], list):
            for cur_sample in batch_list:
                for key, val in cur_sample[0].items():
                    data_dict[key].append(val)
                data_dict['mask'].append(np.ones([len(batch_list)]))
                for key, val in cur_sample[1].items():
                    data_dict[key].append(val)
                data_dict['mask'].append(np.ones([len(batch_list)]))
                for key, val in cur_sample[2].items():
                    data_dict[key].append(val)
                data_dict['mask'].append(np.zeros([len(batch_list)]))
            batch_size = len(batch_list) * 3
        else:
            for cur_sample in batch_list:
                for key, val in cur_sample.items():
                    data_dict[key].append(val)
            batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in [
                        'voxels', 'voxel_num_points', 'voxels_ema',
                        'voxel_num_points_ema'
                ]:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in [
                        'points', 'voxel_coords', 'points_ema',
                        'voxel_coords_ema'
                ]:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)),
                                          mode='constant',
                                          constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes', 'gt_boxes_ema']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros(
                        (batch_size, max_gt, val[0].shape[-1]),
                        dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k][:, :8]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

    def generate_prediction_dicts(self,
                                  batch_dict,
                                  pred_dicts,
                                  class_names,
                                  output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples),
                'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]),
                'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict, shift_coor=None):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            if shift_coor:
                pred_boxes[:, 0:3] -= shift_coor

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict, self.trg_dataset.shift_coor)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if kwargs['eval_metric'] == 'kitti':
            eval_det_annos = copy.deepcopy(det_annos)
            eval_gt_annos = copy.deepcopy(self.trg_dataset.labeled_infos)
            return self.trg_dataset.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
        elif kwargs['eval_metric'] == 'nuscenes':
            return self.trg_dataset.nuscene_eval(det_annos, class_names, **kwargs)
        else:
            raise NotImplementedError

class DADatasetDANN(DADatasetSSDA):
    """ Made for DANN training
    target dataset only use on domain adversarial training (only input)
    """
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg,
                         class_names=class_names,
                         training=training,
                         root_path=root_path,
                         logger=logger)
        self.point_feature_encoder = self.src_dataset.point_feature_encoder
        self.point_cloud_range = self.src_dataset.point_cloud_range
        self.grid_size = self.src_dataset.grid_size
        self.voxel_size = self.src_dataset.voxel_size
        self.target = dataset_cfg.TARGET

    def __len__(self):
        if self.training:
            return len(self.src_dataset.infos) * self.repeat
        else:
            if self.target:
                return len(self.trg_dataset.infos)
            else:
                return len(self.src_dataset.infos)

    def __getitem__(self, index):
        if self.training:
            # SL, TL(for only DANN)
            src_item = self.src_dataset.__getitem__(index)
            trg_item = self.trg_dataset.__getitem__(index)
            return [src_item, trg_item]
        else:
            # whether target or not
            if self.target:
                trg_item = self.trg_dataset.__getitem__(index)
                return trg_item
            else:
                src_item = self.src_dataset.__getitem__(index)
                return src_item

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        if isinstance(batch_list[0], list):
            for cur_sample in batch_list:
                for key, val in cur_sample[0].items():
                    data_dict[key].append(val)
                for key, val in cur_sample[1].items():
                    data_dict[key].append(val)
            batch_size = len(batch_list) * 2
        else:
            for cur_sample in batch_list:
                for key, val in cur_sample.items():
                    data_dict[key].append(val)
            batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in [
                        'voxels', 'voxel_num_points', 'voxels_ema',
                        'voxel_num_points_ema'
                ]:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in [
                        'points', 'voxel_coords', 'points_ema',
                        'voxel_coords_ema'
                ]:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)),
                                          mode='constant',
                                          constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes', 'gt_boxes_ema']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros(
                        (batch_size, max_gt, val[0].shape[-1]),
                        dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret

    def evaluation(self, det_annos, class_names, **kwargs):
        if kwargs['eval_metric'] == 'kitti':
            eval_det_annos = copy.deepcopy(det_annos)
            if self.target:
                eval_gt_annos = copy.deepcopy(self.trg_dataset.infos)
                return self.trg_dataset.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
            else:
                eval_gt_annos = copy.deepcopy(self.src_dataset.infos)
                return self.src_dataset.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
        elif kwargs['eval_metric'] == 'nuscenes':
            return self.src_dataset.nuscene_eval(det_annos, class_names, **kwargs)
        else:
            raise NotImplementedError

    def generate_prediction_dicts(self,
                                  batch_dict,
                                  pred_dicts,
                                  class_names,
                                  output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples),
                'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]),
                'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict, shift_coor=None):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            if shift_coor:
                pred_boxes[:, 0:3] -= shift_coor

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            if self.target:
                single_pred_dict = generate_single_sample_dict(box_dict, self.trg_dataset.shift_coor)
            else:
                single_pred_dict = generate_single_sample_dict(box_dict, self.src_dataset.shift_coor)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos
