# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
from torchdata.datapipes.iter import IterDataPipe
from pycocotools.coco import COCO

from .coco import build as build_coco
from .coco_data_pipe import build_datapipe, build_web_datapipe

def find_coco(s):
    """
    Traverses a datastructure to find a COCO dataset structure
    """
    for v in s.__dict__.values():
        if isinstance(v, COCO):
            return v
        elif isinstance(v, IterDataPipe):
            ret = find_coco(v)
            if ret:
                return ret

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    if isinstance(dataset, IterDataPipe):
        return find_coco(dataset)

def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
