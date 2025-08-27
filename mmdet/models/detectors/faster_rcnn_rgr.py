import os
from collections import OrderedDict

import torch
from mmengine import Config
from mmengine.runner import load_checkpoint, load_state_dict

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector


@MODELS.register_module()
class FasterRCNNRGR(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 ori_setting: ConfigType,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
        self.ori_num_classes = None
        self.load_base_detector(ori_setting)
        torch.save(self.state_dict(), ori_setting['load_from_weight'])

    def load_base_detector(self, ori_setting):
        assert os.path.isfile(ori_setting['ori_checkpoint_file']), '{} is not a valid file'.format(
            ori_setting['ori_checkpoint_file'])
        # ##### init original branches of new model #####
        self.ori_num_classes = ori_setting.ori_num_classes
        self._load_checkpoint_for_new_model(ori_setting.ori_checkpoint_file, strict=False)
        print('======> load base checkpoint for new model from {}'.format(ori_setting.ori_checkpoint_file))

    def _load_checkpoint_for_new_model(self, checkpoint_file, map_location=None, strict=True, logger=None):
        # load ckpt
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        # get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(
                'No state_dict found in checkpoint file {}'.format(checkpoint_file))
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k,
            v in checkpoint['state_dict'].items()}
        # modify cls head size of state_dict
        added_branch_weight = self.roi_head.bbox_head.fc_cls.weight[self.ori_num_classes:, ...]
        added_branch_bias = self.roi_head.bbox_head.fc_cls.bias[self.ori_num_classes:]
        state_dict['roi_head.bbox_head.fc_cls.weight'] = torch.cat(
            (state_dict['roi_head.bbox_head.fc_cls.weight'][:self.ori_num_classes], added_branch_weight), dim=0)
        state_dict['roi_head.bbox_head.fc_cls.bias'] = torch.cat(
            (state_dict['roi_head.bbox_head.fc_cls.bias'][:self.ori_num_classes], added_branch_bias), dim=0)

        # modify reg head size of state_dict
        added_branch_weight = self.roi_head.bbox_head.fc_reg.weight[self.ori_num_classes * 4:, ...]
        added_branch_bias = self.roi_head.bbox_head.fc_reg.bias[self.ori_num_classes * 4:]
        state_dict['roi_head.bbox_head.fc_reg.weight'] = torch.cat(
            (state_dict['roi_head.bbox_head.fc_reg.weight'], added_branch_weight), dim=0)
        state_dict['roi_head.bbox_head.fc_reg.bias'] = torch.cat(
            (state_dict['roi_head.bbox_head.fc_reg.bias'], added_branch_bias), dim=0)

        # load state_dict
        if hasattr(self, 'module'):
            load_state_dict(self.module, state_dict, strict, logger)
        else:
            load_state_dict(self, state_dict, strict, logger)
