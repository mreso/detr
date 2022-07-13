# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO datapipe 

"""
import math
import os
import re
import requests
import tempfile
from pathlib import Path
from PIL import Image, UnidentifiedImageError

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

class ApplyTransform(IterDataPipe):
    def __init__(self, sdp, transforms):
        self.sdp = sdp
        self.transforms = transforms

    def __iter__(self):
        for x in self.sdp:
            yield self.transforms(*x)

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
        dp = ApplyTransform(dp, transforms)

    if image_set == "train":
        dp = Batcher(dp, args.batch_size, drop_last=True)
    else:
        dp = Batcher(dp, args.batch_size, drop_last=False)

    dp = dp.map(utils.collate_fn)

    if image_set == "train":
        dp = WithLength(dp, max(1, num_samples // args.batch_size // args.world_size))
    else:
        dp = WithLength(dp, max(1, math.ceil(num_samples / args.batch_size / args.world_size)))

    return dp



class WebCocoImgUrlLister(IterDataPipe):
    """
    This datapipe takes the image ids of the coco dataset and return the image
    urls based on the given base_url
    """
    def __init__(self, base_url, image_set, coco, return_masks=False):
        self.base_url = base_url.rstrip('/')
        self.image_set = image_set
        self.coco = coco
        
    def __iter__(self):
        for img_id in self.coco.getImgIds():
            yield f"{self.base_url}/{self.image_set}2017/{self.coco.imgs[img_id]['file_name']}"



class WebCocoAnnotationLoader(IterDataPipe):
    """
    This datapipe combines the output of a HttpReader with the annotations from the coco dataset structure.
    Based on detr.datasets.coco.CocoDetection
    """
    def __init__(self, sdp, base_url, image_set, coco, return_masks=False):
        self.sdp = sdp
        self.base_url = base_url.rstrip('/')
        self.image_set = image_set

        self.coco = coco
        self.prepare = ConvertCocoPolysToMask(return_masks)
        
    def __iter__(self):

        for url, io_stream in self.sdp:
            img_id = self.id_from_filename(url)

            coco = self.coco

            ann_ids = coco.getAnnIds(imgIds=img_id)
            
            target = coco.loadAnns(ann_ids)

            target = {'image_id': img_id, 'annotations': target}

            try:
                # skip sample if image is not readable
                img = Image.open(io_stream).convert('RGB')
            except UnidentifiedImageError:
                print(f"Error while reading image {url}")
                continue

            img, target = self.prepare(img, target)

            yield img, target

    def id_from_filename(self, filename):
        expr = "^0+(?!$)"
        filename = filename.split(os.sep)[-1].split(".")[0]
        return int(re.sub(expr, "", filename))


def build_web_datapipe(image_set, args):
    """
    Builds a datapipe that expects ms coco to reside on a blob storage like S3 with http endpoint.
    """
    base_url = args.coco_path

    coco = load_annotation_from_url(base_url, image_set)

    num_samples = len(coco.imgs)
    
    dp = WebCocoImgUrlLister(base_url, image_set, coco)

    dp = dp.shuffle().sharding_filter()
    # TODO mreso: combine with build_datapipe because the only difference is the HttpReader component
    dp = HttpReader(dp)
    dp = WebCocoAnnotationLoader(dp, base_url, image_set, coco)

    transforms = make_coco_transforms(image_set)
    if transforms:
        # Here we use a custom class to avoid an unpicklable warning when using a lambda function
        dp = ApplyTransform(dp, transforms)

    if image_set == "train":
        dp = Batcher(dp, args.batch_size, drop_last=True)
    else:
        dp = Batcher(dp, args.batch_size, drop_last=False)

    dp = dp.map(utils.collate_fn)

    if image_set == "train":
        dp = WithLength(dp, max(1, num_samples // args.batch_size // args.world_size))
    else:
        dp = WithLength(dp, max(1, math.ceil(num_samples / args.batch_size / args.world_size)))

    return dp

def load_annotation_from_url(base_url, image_set):
    url = f"{base_url.rstrip('/')}/annotations/instances_{image_set}2017.json"
    # We actually save this to disk here to avoid editing COCO class which requires a filename and not an open file object
    with tempfile.TemporaryDirectory() as tmp:
        r = requests.get(url)
        fname = Path(tmp) / url.split("/")[-1]
        with open(fname, "wb") as f:
            f.write(r.content)
        return COCO(fname)
        