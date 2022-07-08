# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO datapipe 

"""
import os
import re
from pathlib import Path
from PIL import Image

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torchdata.datapipes.iter import FileLister, HttpReader, IterDataPipe, Batcher

import util.misc as utils
from datasets.coco import make_coco_transforms, ConvertCocoPolysToMask


    
class ObtainAnnotationAndImage(IterDataPipe):
    def __init__(self, source_dp, anno_file, return_masks=False):
        self.source_dp = source_dp
        self.coco = COCO(anno_file)
        self.prepare = ConvertCocoPolysToMask(return_masks)
        
    def __iter__(self):
        for img_file in self.source_dp:
            img_id = self.id_from_filename(img_file)

            coco = self.coco

            ann_ids = coco.getAnnIds(imgIds=img_id)
            
            target = coco.loadAnns(ann_ids)

            target = {'image_id': img_id, 'annotations': target}

            img = Image.open(img_file).convert('RGB')

            img, target = self.prepare(img, target)

            yield img, target

    def id_from_filename(self, filename):
        expr = "^0+(?!$)"
        filename = filename.split(os.sep)[-1].split(".")[0]
        return int(re.sub(expr, "", filename))


class WithLength(IterDataPipe):
    def __init__(self, sdp, length):
        self.sdp = sdp
        self.length = length

    def __iter__(self):
        for x in self.sdp:
            yield x

    def __len__(self):
        return self.length


def build_datapipe(image_set, args):
    root_dir = args.coco_path
    image_dir = Path(root_dir) / f"{image_set}2017"
    anno_file = Path(root_dir) / "annotations" / f"instances_{image_set}2017.json"
    
    assert image_dir.exists(), f"Image folder not found: {image_dir}"
    assert anno_file.exists(), f"Annotation file not found: {anno_file}"

    dp = FileLister(str(image_dir)).shuffle().sharding_filter()

    dp = ObtainAnnotationAndImage(dp, anno_file, args.masks)

    num_samples = len(dp.coco.imgs)

    transforms = make_coco_transforms(image_set)
    if transforms:
        dp = dp.map(lambda x: transforms(*x))

    dp = Batcher(dp, args.batch_size)

    dp = dp.map(utils.collate_fn)

    dp = WithLength(dp, max(1, num_samples // args.batch_size // args.world_size))

    return dp


